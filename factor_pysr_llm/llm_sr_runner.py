from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .experiment2 import _rename_variables, build_task_data
from .expression_similarity import expression_similarity_report
from .lineage import complexity_stats
from .llm_sr_baselines import direct_llm_expression, run_llmsr

# Run LLM-SR-style baselines on the SAME experiment-2 tasks, with the SAME
# no-leakage protocol and evaluation, so they are directly comparable to
# IF-SR / PySR results.


def _cols_from_frame(frame, names: list[str]) -> dict[str, np.ndarray]:
    return {n: frame[n].to_numpy(dtype=float) for n in names}


def run_baselines_task(
    formula: dict[str, Any],
    gen: dict[str, Any],
    call_fn: Callable[[str], str],
    seed: int,
    methods: list[str],
    llmsr_iterations: int = 10,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    formula, _ = _rename_variables(formula)
    td = build_task_data(formula, gen)
    y = td.frame["y"].to_numpy(dtype=float)
    masks = {r: (td.roles == r) for r in ("train", "validation", "test")}
    all_names = td.variables + td.irrelevant
    cols = _cols_from_frame(td.frame, all_names)
    var_meanings = {v["name"]: v.get("meaning", "") for v in formula["variables"]}
    for z in td.irrelevant:
        var_meanings[z] = "irrelevant variable"

    from sklearn.metrics import r2_score

    def eval_expression(expr: str) -> dict[str, Any]:
        from .expr import eval_expr
        with np.errstate(all="ignore"):
            pred = eval_expr(expr, td.frame)
        def r2_on(mask):
            ok = mask & np.isfinite(pred) & np.isfinite(y)
            if int(ok.sum()) < 3:
                return float("nan")
            return float(r2_score(y[ok], pred[ok]))
        try:
            cx = complexity_stats(expr, {})["expanded_node_count"]
        except Exception:
            cx = None
        es = expression_similarity_report(expr, td.expression, variables=all_names, seed=seed + 7, n_points=int(gen.get("n_exprsim", 300)))
        return {
            "expression": expr,
            "r2_val": r2_on(masks["validation"]),
            "r2_test": r2_on(masks["test"]),
            "expanded_node_count": cx,
            "expr_sim": es["expr_sim"],
            "algebraic_equivalence": es["separate_metrics"]["algebraic_equivalence"],
            "numeric_equivalence": es["separate_metrics"]["numeric_equivalence"],
            "support_f1": es["separate_metrics"]["support_f1"],
        }

    conditions: dict[str, Any] = {}

    if "direct_llm" in methods:
        try:
            expr = direct_llm_expression(all_names, var_meanings, cols, y, masks["train"], call_fn)
            conditions["direct_llm"] = {"condition": "direct_llm", **eval_expression(expr)}
        except Exception as exc:
            conditions["direct_llm"] = {"condition": "direct_llm", "selected": None, "error": repr(exc)[:200]}

    if "llm_sr" in methods:
        try:
            out = run_llmsr(all_names, var_meanings, cols, y, masks["train"], masks["validation"], call_fn,
                            n_iterations=llmsr_iterations, seed=seed, log_fn=log_fn)
            if out["status"] == "success":
                conditions["llm_sr"] = {"condition": "llm_sr", **eval_expression(out["best_expression"]),
                                        "n_candidates": out["n_candidates"]}
            else:
                conditions["llm_sr"] = {"condition": "llm_sr", "selected": None, "error": out["status"]}
        except Exception as exc:
            conditions["llm_sr"] = {"condition": "llm_sr", "selected": None, "error": repr(exc)[:200]}

    return {
        "task_id": formula["task_id"],
        "name": formula.get("name"),
        "truth": td.expression,
        "seed": seed,
        "conditions": conditions,
    }


def run_baselines(
    catalog_path: Path,
    out_dir: Path,
    call_fn: Callable[[str], str],
    task_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
    llmsr_iterations: int = 10,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    gen = catalog["meta"]["data_generation"]
    seeds = seeds or [int(gen.get("seed", 20260709))]
    methods = methods or ["direct_llm", "llm_sr"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "llm_sr_baseline_results.json"

    formulas = [f for f in catalog["formulas"] if f.get("regressible", True)]
    if task_ids:
        formulas = [f for f in formulas if f["task_id"] in set(task_ids)]

    done = {}
    if results_path.exists():
        for r in json.loads(results_path.read_text(encoding="utf-8")):
            done[(r["task_id"], r["seed"])] = r
    results = list(done.values())

    for f in formulas:
        for seed in seeds:
            if (f["task_id"], seed) in done:
                continue
            t0 = time.time()
            try:
                res = run_baselines_task(f, gen, call_fn, seed, methods, llmsr_iterations, log_fn)
                res["status"] = "success"
            except Exception as exc:
                import traceback
                res = {"task_id": f["task_id"], "seed": seed, "status": "error",
                       "error": repr(exc), "traceback": traceback.format_exc()}
            res["wall_time_seconds"] = round(time.time() - t0, 1)
            results.append(res)
            results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            if log_fn:
                log_fn(f"  baseline {f['task_id']} seed={seed} -> {res.get('status')} ({res['wall_time_seconds']}s)")
    return {"n_runs": len(results), "results": results, "out_dir": str(out_dir)}
