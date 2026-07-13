from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Merge PySR/IF-SR results and LLM-SR baseline results into a single comparison
# across all methods, computing per-method means (test R2, ExprSim, complexity,
# equivalence rates) and building a unified blind rating candidate set.

PYSR_CONDITIONS = ["raw_pysr", "mine_pysr", "if_sr"]
LLM_CONDITIONS = ["direct_llm", "llm_sr"]


def _load(path: Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [r for r in json.loads(p.read_text(encoding="utf-8")) if r.get("status") == "success"]


def _collect(results: list[dict[str, Any]], conditions: list[str]) -> dict[str, list[dict[str, Any]]]:
    by_method: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}
    for r in results:
        for cond, c in r.get("conditions", {}).items():
            if cond in by_method and c.get("expression"):
                rec = dict(c)
                rec["task_id"] = r["task_id"]
                rec["seed"] = r.get("seed")
                by_method[cond].append(rec)
    return by_method


def _agg(records: list[dict[str, Any]]) -> dict[str, Any]:
    def vals_of(key):
        return [r[key] for r in records if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
    def mean(key):
        v = vals_of(key)
        return float(np.mean(v)) if v else None
    def median(key):
        v = vals_of(key)
        return float(np.median(v)) if v else None
    n = len(records)
    aeq = sum(1 for r in records if r.get("algebraic_equivalence"))
    neq = sum(1 for r in records if r.get("numeric_equivalence"))
    return {
        "n": n,
        "test_r2_mean": mean("r2_test"),
        "test_r2_median": median("r2_test"),
        "expr_sim_mean": mean("expr_sim"),
        "node_count_mean": mean("expanded_node_count"),
        "support_f1_mean": mean("support_f1"),
        "algebraic_equiv_rate": (aeq / n) if n else None,
        "numeric_equiv_rate": (neq / n) if n else None,
    }


def build_comparison(
    pysr_results_path: Path,
    llm_results_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    pysr = _load(pysr_results_path)
    llm = _load(llm_results_path)
    by_method = {}
    by_method.update(_collect(pysr, PYSR_CONDITIONS))
    by_method.update(_collect(llm, LLM_CONDITIONS))
    summary = {m: _agg(recs) for m, recs in by_method.items()}
    report = {
        "n_pysr_runs": len(pysr),
        "n_llm_runs": len(llm),
        "methods": summary,
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_unified_rating_candidates(
    pysr_results_path: Path,
    llm_results_path: Path,
    catalog_path: Path,
    out_path: Path,
    first_seed_only: bool = True,
) -> dict[str, Any]:
    """One blind rating item per (task, method) using the first seed, so human /
    LLM judges rate all 5 methods on equal footing."""
    from .experiment2 import _rename_variables

    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    byid = {f["task_id"]: f for f in catalog["formulas"]}
    pysr = _load(pysr_results_path)
    llm = _load(llm_results_path)

    # index: (task, method) -> record, preferring earliest seed
    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    for results in (pysr, llm):
        for r in sorted(results, key=lambda x: (x["task_id"], x.get("seed", 0))):
            for cond, c in r.get("conditions", {}).items():
                if not c.get("expression"):
                    continue
                key = (r["task_id"], cond)
                if key not in chosen:
                    rec = dict(c)
                    rec["task_id"] = r["task_id"]
                    rec["method"] = cond
                    rec["seed"] = r.get("seed")
                    chosen[key] = rec

    cands = []
    for (task_id, method), rec in chosen.items():
        f, _ = _rename_variables(byid[task_id])
        vardefs = [{"name": v["name"], "definition": v.get("meaning", ""), "unit": "-",
                    "allowed_range": f"[{v['low']},{v['high']}]"} for v in f["variables"]]
        formula = rec.get("expanded_expression") or rec["expression"]
        cands.append({
            "formula": formula,
            "variables": vardefs,
            "task_context": f"Recover a scientific relationship for target y. Domain: {byid[task_id].get('name','')}.",
            "dataset_label": f"task_{task_id[:3]}",
            "method": method, "seed": rec.get("seed"), "r2_test": rec.get("r2_test"),
            "dataset": task_id, "target": "y",
        })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(cands, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"n_candidates": len(cands), "output_path": str(out_path)}
