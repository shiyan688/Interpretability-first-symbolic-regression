from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .expr import eval_expr
from .expression_similarity import expression_similarity_report
from .ifsr_selector import Candidate, select_if_sr
from .lineage import FactorCard, build_card_index, complexity_stats, expand_expression
from .llmsrbench_loader import load_task

# Run all methods (raw_pysr, mine_pysr, if_sr, direct_llm, llm_sr) on REAL
# LLM-SRBench tasks (downloaded via the ModelScope mirror), using each task's
# own train/valid/id_test splits and its ground-truth formula for ExprSim.

RESERVED = {"S", "N", "O", "E", "I", "Q", "beta", "gamma", "zeta", "Order"}


def _sanitize(task: dict[str, Any]) -> dict[str, Any]:
    import re
    rename = {v: f"var_{v}" for v in task["variables"] if v in RESERVED}
    if task["target"] in RESERVED:
        rename[task["target"]] = f"var_{task['target']}"
    if not rename:
        return task
    frame = task["frame"].rename(columns=rename)
    expr = task["expression"]
    for old, new in rename.items():
        expr = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, expr)
    out = dict(task)
    out["frame"] = frame
    out["expression"] = expr
    out["variables"] = [rename.get(v, v) for v in task["variables"]]
    out["target"] = rename.get(task["target"], task["target"])
    return out


def _r2_masked(pred, y, mask):
    ok = mask & np.isfinite(pred) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return float("nan")
    return float(r2_score(y[ok], pred[ok]))


def _mined_columns(frame: pd.DataFrame, variables: list[str]) -> dict[str, str]:
    factors = {}
    for i, a in enumerate(variables):
        factors[f"sq_{a}"] = f"{a}**2"
        factors[f"inv_{a}"] = f"1.0/{a}"
        for b in variables[i + 1:]:
            factors[f"mul_{a}_{b}"] = f"{a}*{b}"
            factors[f"div_{a}_{b}"] = f"{a}/{b}"
            factors[f"div_{b}_{a}"] = f"{b}/{a}"
    return factors


def _pysr_fit(X, y, train_mask, seed, budget):
    from pysr import PySRRegressor
    import os
    os.environ.setdefault("PYSR_USE_BEARTYPE", "0")
    model = PySRRegressor(
        niterations=int(budget.get("niterations", 60)),
        populations=int(budget.get("populations", 8)),
        population_size=int(budget.get("population_size", 40)),
        maxsize=int(budget.get("maxsize", 24)),
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["cos", "sin", "exp", "sqrt_abs(x)=sqrt(abs(x))", "inv(x)=1/x"],
        extra_sympy_mappings={"inv": lambda x: 1 / x, "sqrt_abs": lambda x: (abs(x)) ** 0.5},
        elementwise_loss="loss(x, y) = (x - y)^2",
        model_selection="best",
        parsimony=float(budget.get("parsimony", 1e-4)),
        timeout_in_seconds=int(budget.get("timeout_seconds", 120)),
        deterministic=True, parallelism="serial", random_state=int(seed),
        verbosity=0, progress=False, temp_equation_file=True,
    )
    model.fit(X.to_numpy(dtype=float)[train_mask], np.asarray(y, dtype=float)[train_mask], variable_names=list(X.columns))
    return model


def _archive(model, X, y, masks):
    eqs = model.equations_
    out = []
    for i, row in eqs.iterrows():
        try:
            pred = np.asarray(model.predict(X.to_numpy(dtype=float), i), dtype=float)
        except Exception:
            continue
        try:
            cx = complexity_stats(str(row["equation"]), {})["expanded_node_count"]
        except Exception:
            cx = int(row.get("complexity", 999))
        out.append({"candidate_id": f"eq_{i}", "expression": str(row["equation"]),
                    "expanded_node_count": int(cx),
                    "r2_val": _r2_masked(pred, y, masks["validation"]),
                    "r2_test": _r2_masked(pred, y, masks["test"])})
    return out


def _standard_select(archive):
    v = [c for c in archive if np.isfinite(c["r2_val"])]
    return max(v, key=lambda c: (c["r2_val"], -c["expanded_node_count"])) if v else None


def run_llmsrbench_task(
    task_dir: str,
    call_fn: Callable[[str], str] | None,
    seed: int,
    methods: list[str],
    budget: dict[str, Any],
    llmsr_iterations: int = 10,
    log_fn=None,
) -> dict[str, Any]:
    task = _sanitize(load_task(Path(task_dir)))
    frame = task["frame"]
    y = frame[task["target"]].to_numpy(dtype=float)
    roles = task["roles"]
    masks = {r: (roles == r) for r in ("train", "validation", "test")}
    variables = task["variables"]
    truth = task["expression"]
    var_meanings = {v: "" for v in variables}
    md_feats = task.get("metadata", {}).get("dataset", {}).get("features", [])
    for fdef in md_feats:
        nm = fdef.get("name")
        if nm in var_meanings:
            var_meanings[nm] = str(fdef.get("description", "")).strip()

    def eval_expression(expr: str, card_index=None) -> dict[str, Any]:
        try:
            expanded = expand_expression(expr, card_index) if card_index else expr
        except Exception:
            expanded = expr
        with np.errstate(all="ignore"):
            pred = eval_expr(expanded, frame)
        try:
            cx = complexity_stats(expanded, {})["expanded_node_count"]
        except Exception:
            cx = None
        es = expression_similarity_report(expanded, truth, variables=variables, seed=seed + 7, n_points=300)
        return {"expression": expr, "expanded_expression": expanded,
                "r2_val": _r2_masked(pred, y, masks["validation"]),
                "r2_test": _r2_masked(pred, y, masks["test"]),
                "expanded_node_count": cx, "expr_sim": es["expr_sim"],
                "algebraic_equivalence": es["separate_metrics"]["algebraic_equivalence"],
                "numeric_equivalence": es["separate_metrics"]["numeric_equivalence"],
                "support_f1": es["separate_metrics"]["support_f1"]}

    conditions: dict[str, Any] = {}

    # ---- PySR conditions ----
    if any(m in methods for m in ("raw_pysr", "mine_pysr", "if_sr")):
        X_raw = frame[variables].copy()
        raw_model = _pysr_fit(X_raw, y, masks["train"], seed, budget)
        raw_arc = _archive(raw_model, X_raw, y, masks)
        if "raw_pysr" in methods:
            sel = _standard_select(raw_arc)
            conditions["raw_pysr"] = {"condition": "raw_pysr", **(eval_expression(sel["expression"]) if sel else {"selected": None})}

        factors = _mined_columns(frame, variables)
        X_mine = X_raw.copy()
        kept = {}
        for name, fx in factors.items():
            with np.errstate(all="ignore"):
                vals = eval_expr(fx, frame)
            if np.isfinite(vals).mean() > 0.98 and np.std(vals[np.isfinite(vals)]) > 1e-12:
                X_mine[name] = np.where(np.isfinite(vals), vals, 0.0)
                kept[name] = fx
        cards = [FactorCard(n, n, e, source="mined", unit_status="screening_only") for n, e in kept.items()]
        cidx = build_card_index(cards)
        mine_model = _pysr_fit(X_mine, y, masks["train"], seed, budget)
        mine_arc = _archive(mine_model, X_mine, y, masks)
        if "mine_pysr" in methods:
            sel = _standard_select(mine_arc)
            conditions["mine_pysr"] = {"condition": "mine_pysr", **(eval_expression(sel["expression"], cidx) if sel else {"selected": None})}
        if "if_sr" in methods:
            cands = [Candidate(candidate_id=c["candidate_id"], expression=c["expression"], r2_val=c["r2_val"], r2_test=c["r2_test"])
                     for c in mine_arc if np.isfinite(c["r2_val"])]
            dec = select_if_sr(cands, delta=float(budget.get("delta", 0.02))) if cands else {"selected": None}
            if dec.get("selected"):
                sid = dec["selected"]["candidate_id"]
                sel = next((c for c in mine_arc if c["candidate_id"] == sid), None)
                conditions["if_sr"] = {"condition": "if_sr", **eval_expression(sel["expression"], cidx)}
            else:
                conditions["if_sr"] = {"condition": "if_sr", "selected": None}

    # ---- LLM conditions ----
    if call_fn is not None and any(m in methods for m in ("direct_llm", "llm_sr")):
        cols = {v: frame[v].to_numpy(dtype=float) for v in variables}
        from .llm_sr_baselines import direct_llm_expression, run_llmsr
        if "direct_llm" in methods:
            try:
                expr = direct_llm_expression(variables, var_meanings, cols, y, masks["train"], call_fn)
                conditions["direct_llm"] = {"condition": "direct_llm", **eval_expression(expr)}
            except Exception as exc:
                conditions["direct_llm"] = {"condition": "direct_llm", "selected": None, "error": repr(exc)[:200]}
        if "llm_sr" in methods:
            try:
                out = run_llmsr(variables, var_meanings, cols, y, masks["train"], masks["validation"], call_fn,
                                n_iterations=llmsr_iterations, seed=seed, log_fn=log_fn)
                if out["status"] == "success":
                    conditions["llm_sr"] = {"condition": "llm_sr", **eval_expression(out["best_expression"])}
                else:
                    conditions["llm_sr"] = {"condition": "llm_sr", "selected": None, "error": out["status"]}
            except Exception as exc:
                conditions["llm_sr"] = {"condition": "llm_sr", "selected": None, "error": repr(exc)[:200]}

    return {"task_id": task["task_id"], "truth": truth, "seed": seed,
            "category": task["task_id"].split("__")[1] if "__" in task["task_id"] else "",
            "conditions": conditions}
