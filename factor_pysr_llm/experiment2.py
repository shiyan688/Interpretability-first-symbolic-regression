from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .expr import eval_expr
from .expression_similarity import expression_similarity_report
from .ifsr_selector import Candidate, select_if_sr
from .lineage import FactorCard, build_card_index, complexity_stats, expand_expression

# Experiment 2 harness: known-formula recovery with a no-leakage protocol.
#
#   - synthetic data per formula (train/val/test + independent ExprSim points,
#     5% noise + irrelevant vars)
#   - PySR fits on TRAIN rows only; the Pareto archive is scored on validation
#   - three real conditions from the same runs:
#       raw_pysr   : PySR over raw variables, standard (best-loss) selection
#       mine_pysr  : PySR over raw + mined factors, standard selection
#       if_sr      : interpretability-first selection over the raw+mined archive
#   - test R^2 is read exactly once, after each condition locks its formula
#   - ExprSim compares the locked formula to ground truth on independent points


@dataclass
class TaskData:
    task_id: str
    expression: str
    variables: list[str]
    irrelevant: list[str]
    frame: pd.DataFrame          # all rows, raw feature columns + y
    roles: np.ndarray            # 'train'/'validation'/'test'
    exprsim_frame: pd.DataFrame  # independent points, noise-free y


# Names that sympy/PySR reserve as functions or singletons and therefore cannot
# be used as variable names. Colliding variables are renamed consistently across
# the data, the ground-truth expression, mined factors and ExprSim.
_RESERVED_NAMES = {"S", "N", "O", "E", "I", "Q", "beta", "gamma", "zeta", "Order"}


def _rename_variables(formula: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Return a copy of the formula with reserved variable names made safe."""
    import re as _re

    rename: dict[str, str] = {}
    for v in formula["variables"]:
        name = v["name"]
        if name in _RESERVED_NAMES:
            rename[name] = f"var_{name}"
    if not rename:
        return formula, {}
    new = dict(formula)
    new_vars = []
    for v in formula["variables"]:
        nv = dict(v)
        nv["name"] = rename.get(v["name"], v["name"])
        new_vars.append(nv)
    new["variables"] = new_vars
    expr = formula["expression"]
    for old, sub in rename.items():
        expr = _re.sub(rf"(?<![A-Za-z0-9_]){_re.escape(old)}(?![A-Za-z0-9_])", sub, expr)
    new["expression"] = expr
    return new, rename


def build_task_data(formula: dict[str, Any], gen: dict[str, Any]) -> TaskData:
    formula, _renamed = _rename_variables(formula)
    variables = [v["name"] for v in formula["variables"]]
    n_irr = int(gen.get("n_irrelevant", 5))
    irrelevant = [f"z{i}" for i in range(1, n_irr + 1)]
    seed = int(gen.get("seed", 20260709))
    noise = float(gen.get("target_noise", 0.05))
    n_tr, n_va, n_te = int(gen["n_train"]), int(gen["n_validation"]), int(gen["n_test"])
    n_es = int(gen.get("n_exprsim", n_te))
    ranges = {v["name"]: (float(v["low"]), float(v["high"])) for v in formula["variables"]}
    # irrelevant vars sampled on a generic positive range
    irr_range = (1.0, 3.0)

    def sample(n: int, sub_seed: int, noisy: bool) -> pd.DataFrame:
        r = np.random.default_rng(sub_seed)
        cols = {v: r.uniform(*ranges[v], n) for v in variables}
        for z in irrelevant:
            cols[z] = r.uniform(*irr_range, n)
        frame = pd.DataFrame(cols)
        with np.errstate(all="ignore"):
            signal = eval_expr(formula["expression"], frame)
        signal = np.where(np.isfinite(signal), signal, 0.0)
        if noisy:
            scale = float(np.std(signal)) or 1.0
            frame["y"] = signal + r.normal(0.0, noise * scale, n)
        else:
            frame["y"] = signal
        return frame

    train = sample(n_tr, seed + 1, True)
    val = sample(n_va, seed + 2, True)
    test = sample(n_te, seed + 3, True)
    exprsim = sample(n_es, seed + 4, False)
    frame = pd.concat([train, val, test], axis=0, ignore_index=True)
    roles = np.array(["train"] * n_tr + ["validation"] * n_va + ["test"] * n_te, dtype=object)
    return TaskData(formula["task_id"], formula["expression"], variables, irrelevant, frame, roles, exprsim)


def _mined_factor_columns(frame: pd.DataFrame, variables: list[str]) -> dict[str, str]:
    """Generic, answer-agnostic domain-style factors: pairwise products, ratios,
    squares and inverses of raw variables. These are the same for every task and
    do NOT encode the ground truth."""
    factors: dict[str, str] = {}
    for i, a in enumerate(variables):
        factors[f"sq_{a}"] = f"{a}**2"
        factors[f"inv_{a}"] = f"1.0/{a}"
        for b in variables[i + 1 :]:
            factors[f"mul_{a}_{b}"] = f"{a}*{b}"
            factors[f"div_{a}_{b}"] = f"{a}/{b}"
            factors[f"div_{b}_{a}"] = f"{b}/{a}"
    return factors


def _run_pysr(
    X: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    seed: int,
    budget: dict[str, Any],
) -> pd.DataFrame:
    """Fit PySR on train rows only. Return the equations_ dataframe."""
    from pysr import PySRRegressor

    import os
    os.environ.setdefault("PYSR_USE_BEARTYPE", "0")
    model = PySRRegressor(
        niterations=int(budget.get("niterations", 40)),
        populations=int(budget.get("populations", 8)),
        population_size=int(budget.get("population_size", 40)),
        maxsize=int(budget.get("maxsize", 22)),
        binary_operators=list(budget.get("binary_operators", ["+", "-", "*", "/"])),
        unary_operators=list(budget.get("unary_operators", ["cos", "sin", "exp", "sqrt_abs(x)=sqrt(abs(x))", "inv(x)=1/x"])),
        extra_sympy_mappings={"inv": lambda x: 1 / x, "sqrt_abs": lambda x: (abs(x)) ** 0.5},
        elementwise_loss="loss(x, y) = (x - y)^2",
        model_selection=str(budget.get("model_selection", "best")),
        parsimony=float(budget.get("parsimony", 1e-4)),
        timeout_in_seconds=int(budget.get("timeout_seconds", 90)),
        deterministic=True,
        parallelism="serial",
        random_state=int(seed),
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    model.fit(X.to_numpy(dtype=float)[train_mask], np.asarray(y, dtype=float)[train_mask], variable_names=list(X.columns))
    eqs = model.equations_.copy()
    eqs["_sympy"] = [model.sympy(i) for i in range(len(eqs))]
    return eqs, model


def _archive_from_equations(
    eqs: pd.DataFrame,
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    card_index: dict[str, "FactorCard"] | None = None,
) -> list[dict[str, Any]]:
    """Score every Pareto equation on VALIDATION only (P0-7: test is NOT read
    here; it is evaluated once after the formula is locked, see lock_test_metric).

    ``expanded_node_count`` is measured on the FULLY EXPANDED expression (P0-1):
    mined-factor column names (e.g. ``sq_t``, ``div_q1_r``) are expanded back to
    raw variables via ``card_index`` before counting, so aliases cannot hide
    complexity.
    """
    card_index = card_index or {}
    archive = []
    for i, row in eqs.iterrows():
        expr = str(row["equation"])
        try:
            pred = np.asarray(model.predict(X.to_numpy(dtype=float), i), dtype=float)
        except Exception:
            continue
        def r2_on(mask):
            m = mask & np.isfinite(pred) & np.isfinite(y)
            if int(m.sum()) < 3:
                return float("nan")
            return float(r2_score(y[m], pred[m]))
        try:
            stats = complexity_stats(expr, card_index)
            cx = stats["expanded_node_count"]
            expanded = stats["expanded_expression"]
        except Exception:
            cx = int(row.get("complexity", 999))
            expanded = expr
        archive.append({
            "candidate_id": f"eq_{i}",
            "expression": expr,
            "expanded_expression": expanded,
            "pysr_complexity": int(row.get("complexity", -1)),
            "expanded_node_count": int(cx),
            "loss": float(row.get("loss", float("nan"))),
            "r2_val": r2_on(masks["validation"]),
            # test intentionally NOT computed here (locked-once policy)
            "eq_index": int(i),
        })
    return archive


def standard_select(archive: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best validation R^2, ties broken by lower complexity."""
    valid = [c for c in archive if np.isfinite(c["r2_val"])]
    if not valid:
        return None
    return max(valid, key=lambda c: (c["r2_val"], -c["expanded_node_count"]))


def run_task(
    formula: dict[str, Any],
    gen: dict[str, Any],
    budget: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    td = build_task_data(formula, gen)
    y = td.frame["y"].to_numpy(dtype=float)
    masks = {r: (td.roles == r) for r in ("train", "validation", "test")}

    def lock_test_metric(model, X, eq_index):
        """P0-7: compute test R^2 ONCE, only after the formula is locked."""
        try:
            pred = np.asarray(model.predict(X.to_numpy(dtype=float), int(eq_index)), dtype=float)
        except Exception:
            return float("nan")
        m = masks["test"] & np.isfinite(pred) & np.isfinite(y)
        if int(m.sum()) < 3:
            return float("nan")
        return float(r2_score(y[m], pred[m]))

    # ---- raw condition (no mined cards) ----
    X_raw = td.frame[td.variables + td.irrelevant].copy()
    raw_eqs, raw_model = _run_pysr(X_raw, y, masks["train"], seed, budget)
    raw_archive = _archive_from_equations(raw_eqs, raw_model, X_raw, y, masks, card_index={})

    # ---- mined condition (raw + generic factor columns) ----
    factors = _mined_factor_columns(td.frame, td.variables)
    X_mine = X_raw.copy()
    kept_factors: dict[str, str] = {}
    for name, fexpr in factors.items():
        with np.errstate(all="ignore"):
            vals = eval_expr(fexpr, td.frame)
        if np.isfinite(vals).mean() > 0.98 and np.std(vals[np.isfinite(vals)]) > 1e-12:
            X_mine[name] = np.where(np.isfinite(vals), vals, 0.0)
            kept_factors[name] = fexpr
    # factor cards so mined-column formulas expand back to raw variables
    mine_cards = [FactorCard(n, n, e, source="mined", unit_status="screening_only") for n, e in kept_factors.items()]
    mine_card_index = build_card_index(mine_cards)
    mine_eqs, mine_model = _run_pysr(X_mine, y, masks["train"], seed, budget)
    # P0-1: pass the card index so expanded_node_count counts EXPANDED nodes
    mine_archive = _archive_from_equations(mine_eqs, mine_model, X_mine, y, masks, card_index=mine_card_index)

    # ---- selections (validation-only) ----
    raw_sel = standard_select(raw_archive)
    mine_sel = standard_select(mine_archive)

    # IF-SR: interpretability-first over the mined archive.
    # P0-1: pass cards=mine_cards so the selector minimizes EXPANDED complexity.
    ifsr_cands = [
        Candidate(candidate_id=c["candidate_id"], expression=c["expression"], r2_val=c["r2_val"],
                  stability=0.0)
        for c in mine_archive if np.isfinite(c["r2_val"])
    ]
    ifsr_decision = select_if_sr(ifsr_cands, delta=float(budget.get("delta", 0.02)),
                                 cards=mine_cards) if ifsr_cands else {"selected": None}
    ifsr_sel = None
    if ifsr_decision.get("selected"):
        sid = ifsr_decision["selected"]["candidate_id"]
        ifsr_sel = next((c for c in mine_archive if c["candidate_id"] == sid), None)

    # ---- pack: test locked once here; ExprSim on independent points ----
    def pack(name, sel, model, X):
        if not sel:
            return {"condition": name, "selected": None}
        expanded = sel.get("expanded_expression") or sel["expression"]
        r2_test = lock_test_metric(model, X, sel["eq_index"])  # locked-once test read
        es = expression_similarity_report(
            expanded, td.expression, variables=td.variables + td.irrelevant,
            seed=seed + 7, n_points=int(gen.get("n_exprsim", 300)),
        )
        return {
            "condition": name,
            "expression": sel["expression"],
            "expanded_expression": expanded,
            "r2_val": sel["r2_val"],
            "r2_test": r2_test,
            "expanded_node_count": sel["expanded_node_count"],
            "expr_sim": es["expr_sim"],
            "algebraic_equivalence": es["separate_metrics"]["algebraic_equivalence"],
            "numeric_equivalence": es["separate_metrics"]["numeric_equivalence"],
            "support_f1": es["separate_metrics"]["support_f1"],
        }

    return {
        "task_id": formula["task_id"],
        "name": formula.get("name"),
        "truth": td.expression,
        "seed": seed,
        "n_raw_candidates": len(raw_archive),
        "n_mine_candidates": len(mine_archive),
        "conditions": {
            "raw_pysr": pack("raw_pysr", raw_sel, raw_model, X_raw),
            "mine_pysr": pack("mine_pysr", mine_sel, mine_model, X_mine),
            "if_sr": pack("if_sr", ifsr_sel, mine_model, X_mine),
        },
        "ifsr_threshold": ifsr_decision.get("threshold"),
    }


def run_experiment2(
    catalog_path: Path,
    out_dir: Path,
    task_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    gen = catalog["meta"]["data_generation"]
    budget = budget or {}
    seeds = seeds or [int(gen.get("seed", 20260709))]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    formulas = [f for f in catalog["formulas"] if f.get("regressible", True)]
    if task_ids:
        formulas = [f for f in formulas if f["task_id"] in set(task_ids)]

    results = []
    for f in formulas:
        for seed in seeds:
            t0 = time.time()
            try:
                res = run_task(f, gen, budget, seed)
                res["wall_time_seconds"] = round(time.time() - t0, 1)
                res["status"] = "success"
            except Exception as exc:  # keep going; record failures
                import traceback
                res = {
                    "task_id": f["task_id"], "seed": seed, "status": "error",
                    "error": repr(exc), "traceback": traceback.format_exc(),
                    "wall_time_seconds": round(time.time() - t0, 1),
                }
            results.append(res)
            # incremental save
            (out_dir / "experiment2_results.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    return {"n_runs": len(results), "results": results, "out_dir": str(out_dir)}
