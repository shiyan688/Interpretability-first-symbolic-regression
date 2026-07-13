#!/usr/bin/env python3
"""End-to-end pilot for IF-SR (development_only surrogate SR engine).

Runs the full pipeline on one known-formula pilot task:
  data gen -> split -> feature/factor build -> 4 conditions -> IF-SR select
  -> expand + inverse-standardize consistency -> blind export -> fake judges
  -> aggregate -> ExprSim.

Emits a machine-readable summary that docs/pilot_report.md references.
Nothing here enters a confirmatory main table.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from factor_pysr_llm.expression_similarity import expression_similarity_report
from factor_pysr_llm.ifsr_selector import Candidate, select_if_sr
from factor_pysr_llm.interpretability_eval import (
    export_blind_ratings,
    run_llm_judge,
)
from factor_pysr_llm.known_data import generate_known_formula_dataset, write_known_formula_task
from factor_pysr_llm.lineage import FactorCard, build_card_index, check_numeric_consistency
from factor_pysr_llm.pilot import ablation_is_nontrivial, pilot_conditions

RUBRIC = "configs/interpretability_rubric.json"
DIMS = ["domain_meaning", "structural_plausibility", "readability_generalizability", "hypothesis_support"]


def _fake_judge(prompt: str) -> str:
    m = re.search(r"item_id `([^`]+)`", prompt)
    item_id = m.group(1)
    # deterministic pseudo-score based on formula length (shorter -> higher)
    formula = prompt.split("```")[1] if "```" in prompt else ""
    base = max(1, min(5, 6 - len(formula) // 20))
    return json.dumps(
        {
            "item_id": item_id,
            "ratings": {d: base for d in DIMS},
            "overall_judgment": "surrogate",
            "confidence": 0.5,
        }
    )


def run_selection_divergence_task(out_root: Path) -> dict:
    """A second pilot task where accuracy-first and interpretability-first
    selection genuinely diverge on the SAME candidate pool (mine_pysr vs
    if_sr_no_domain), proving the selection-rule ablation is not a no-op."""
    from factor_pysr_llm.pilot import (
        interpretability_first_selection,
        standard_selection,
        surrogate_search,
    )

    truth = "v1 + 0.05*v2*v3"
    ds = generate_known_formula_dataset("SEL_DIVERGE", truth, 3, n_train=400, n_validation=200, n_test=200, n_exprsim=100, seed=20260709, var_low=1.0, var_high=2.0)
    comb = pd.concat([ds["train"], ds["validation"], ds["test"]], ignore_index=True)
    roles = ["train"] * 400 + ["validation"] * 200 + ["test"] * 200
    y = comb["y"].to_numpy(dtype=float)
    tr = np.array([r == "train" for r in roles])
    va = np.array([r == "validation" for r in roles])
    ff = comb[ds["variables"] + ds["irrelevant_variables"]].copy()
    ff["mine_prod"] = comb["v2"] * comb["v3"]
    cols = ds["variables"] + ds["irrelevant_variables"] + ["mine_prod"]
    cands = surrogate_search(ff, y, tr, va, cols)
    std = standard_selection(cands)
    iff = interpretability_first_selection(cands)
    return {
        "truth": truth,
        "candidates": [(c["expression"], round(c["r2_val"], 4)) for c in cands],
        "mine_pysr_selection": std,
        "if_no_domain_selection": iff,
        "selection_diverges": std != iff,
    }


def main() -> None:
    out_root = Path("outputs/pilot_run")
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- one known-formula pilot task ----
    task_id = "F_I_12_11"
    expression = "v1 * (v2 + v3)"   # simple readable pilot truth
    ds = generate_known_formula_dataset(
        task_id=task_id,
        expression=expression,
        n_variables=3,
        n_train=200,
        n_validation=100,
        n_test=200,
        n_exprsim=200,
        seed=20260709,
    )
    task_dir = out_root / task_id
    meta = write_known_formula_task(task_dir, ds)

    combined = pd.read_csv(task_dir / "data.csv")
    roles = pd.read_csv(task_dir / "predefined_roles.csv")["role"].tolist()
    y = combined["y"].to_numpy(dtype=float)
    train_mask = np.array([r == "train" for r in roles])
    val_mask = np.array([r == "validation" for r in roles])
    test_mask = np.array([r == "test" for r in roles])

    variables = ds["variables"]
    irrelevant = ds["irrelevant_variables"]
    feature_frame = combined[variables + irrelevant].copy()
    # mined columns (interaction) and an approved domain factor column
    feature_frame["mine_sum"] = combined["v2"] + combined["v3"]
    feature_frame["domain_prod"] = combined["v1"] * (combined["v2"] + combined["v3"])

    raw_cols = variables + irrelevant
    mined_cols = ["mine_sum"]
    domain_cols = ["domain_prod"]

    archives = pilot_conditions(
        raw_cols, mined_cols, domain_cols, feature_frame, y, train_mask, val_mask
    )
    ablation = ablation_is_nontrivial(archives)

    # ---- IF-SR selection on the if_sr condition ----
    cards = [
        FactorCard("mine_sum", "mine_sum", "(v2 + v3)", ["v2", "v3"], source="mined", unit_status="screening_only"),
        FactorCard("domain_prod", "domain_prod", "(v1 * (v2 + v3))", ["v1", "v2", "v3"], source="expert", unit_status="valid", approved_for_final_formula=True),
    ]
    card_index = build_card_index(cards)
    ifsr_cands = []
    for c in archives["if_sr"]["candidates"]:
        uses_domain = any(dc in c["columns"] for dc in domain_cols)
        ifsr_cands.append(
            Candidate(
                candidate_id=c["candidate_id"],
                expression=c["expression"],
                r2_val=c["r2_val"],
                uses_domain_factor=uses_domain,
            )
        )
    decision = select_if_sr(ifsr_cands, delta=0.02, cards=cards)

    # ---- expansion + numeric consistency of the selected formula ----
    consistency = None
    if decision["selected"]:
        sel_expr = decision["selected"]["expression"]
        consistency = check_numeric_consistency(sel_expr, card_index, feature_frame)

    # ---- blind export + two fake judges + aggregate ----
    # build candidate formulas from each condition's best (validation) candidate
    rating_candidates = []
    for name, arc in archives.items():
        if not arc["candidates"]:
            continue
        best = max(arc["candidates"], key=lambda c: c["r2_val"])
        rating_candidates.append(
            {
                "formula": best["expression"],
                "variables": [{"name": v, "definition": f"variable {v}", "unit": "-", "allowed_range": "[1,3]"} for v in variables],
                "task_context": "Pilot known-formula task (development_only).",
                "dataset_label": "pilot_dataset",
                "method": name,
                "seed": 20260709,
                "r2_val": best["r2_val"],
                "dataset": task_id,
                "target": "y",
            }
        )
    export_info = export_blind_ratings(
        rating_candidates,
        out_manifest=out_root / "rating_manifest.jsonl",
        out_private_map=out_root / "rating_private_map.json",
        seed=20260709,
    )
    judge_a = run_llm_judge(out_root / "rating_manifest.jsonl", RUBRIC, out_root / "judge", _fake_judge, "fake_llm_a", seed=1)
    judge_b = run_llm_judge(out_root / "rating_manifest.jsonl", RUBRIC, out_root / "judge", _fake_judge, "fake_llm_b", seed=999)

    from factor_pysr_llm.rating_aggregate import aggregate_ratings

    agg = aggregate_ratings(
        human_csv=None,
        llm_result_paths={
            "llm_a": out_root / "judge" / "judge_fake_llm_a_seed1.jsonl",
            "llm_b": out_root / "judge" / "judge_fake_llm_b_seed999.jsonl",
        },
        private_map_path=out_root / "rating_private_map.json",
        out_path=out_root / "rating_aggregate.json",
    )

    # ---- ExprSim of the selected formula vs truth ----
    exprsim = None
    if decision["selected"]:
        # substitute the surrogate column names back to raw variable expressions
        sel = decision["selected"]["expression"]
        sel_expanded = check_numeric_consistency(sel, card_index, feature_frame)["expanded_expression"]
        exprsim = expression_similarity_report(sel_expanded, expression, variables=variables, seed=42)

    sel_diverge = run_selection_divergence_task(out_root)

    summary = {
        "development_only": True,
        "engine": "surrogate_linear_additive",
        "task": meta,
        "ablation": ablation,
        "selection_divergence_task": sel_diverge,
        "ifsr_decision": {
            "selected_id": decision["selected"]["candidate_id"] if decision["selected"] else None,
            "selected_expression": decision["selected"]["expression"] if decision["selected"] else None,
            "n_survivors": decision["n_survivors"],
            "threshold": decision["threshold"],
        },
        "numeric_consistency": consistency,
        "blind_export": export_info,
        "judge_a_summary": judge_a["summary"],
        "judge_b_summary": judge_b["summary"],
        "aggregate_groups": list(agg["groups"].keys()),
        "exprsim": exprsim,
    }
    (out_root / "pilot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({
        "task_a_selections": ablation["selections"],
        "task_b_selection_diverges": sel_diverge["selection_diverges"],
        "task_b_mine_vs_ifnodomain": [sel_diverge["mine_pysr_selection"], sel_diverge["if_no_domain_selection"]],
        "selected": summary["ifsr_decision"]["selected_expression"],
        "consistency": consistency["consistent"] if consistency else None,
        "judges_ok": judge_a["summary"]["n_errors"] == 0 and judge_b["summary"]["n_errors"] == 0,
        "exprsim": exprsim["expr_sim"] if exprsim else None,
        "summary_path": str(out_root / "pilot_summary.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
