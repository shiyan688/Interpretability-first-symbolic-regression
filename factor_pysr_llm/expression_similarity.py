from __future__ import annotations

import re
from typing import Any

import numpy as np

from .expr import eval_expr, to_python_expr

# Expression similarity (ExprSim) for experiment 2
# ------------------------------------------------
# Frozen weights (see configs/paper_scope.yaml, exprsim_v1):
#   ExprSim = 0.50 * numeric similarity on independently sampled points
#           + 0.20 * variable-set F1
#           + 0.20 * operator-set F1
#           + 0.10 * normalized tree-structure similarity
# All sub-scores lie in [0, 1]. Algebraic / numeric equivalence and support F1
# are reported separately (not folded into the composite).

WEIGHTS = {
    "numeric_similarity": 0.50,
    "variable_set_f1": 0.20,
    "operator_set_f1": 0.20,
    "tree_structure_similarity": 0.10,
}

_OPERATOR_FUNCS = {
    "sin", "cos", "tan", "exp", "log", "sqrt", "inv", "square", "cube",
    "cbrt", "abs", "asin", "acos", "atan", "tanh", "pow",
}


import pandas as pd


def _identifiers(expr: str) -> set[str]:
    text = to_python_expr(expr)
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    call_names = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text))
    return {n for n in names if n not in _OPERATOR_FUNCS and n not in call_names and n != "pi"}


def _operators(expr: str) -> set[str]:
    text = to_python_expr(expr)
    ops: set[str] = set()
    for sym in ("+", "-", "*", "/", "**"):
        if sym in text:
            ops.add(sym)
    for fn in _OPERATOR_FUNCS:
        if re.search(rf"(?<![A-Za-z0-9_]){fn}\s*\(", text):
            ops.add(fn)
    return ops


def _f1(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    tp = len(a & b)
    if tp == 0:
        return 0.0
    precision = tp / len(b)
    recall = tp / len(a)
    return float(2 * precision * recall / (precision + recall))


def _sympy_tree(expr: str):
    import sympy as sp

    local = {
        "inv": lambda x: 1 / x,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
        "sqrt_abs": lambda x: sp.sqrt(sp.Abs(x)),
        "log_abs": lambda x: sp.log(sp.Abs(x)),
        "abs": sp.Abs,
    }
    return sp.sympify(to_python_expr(expr), locals=local)


def _node_count(expr_tree) -> int:
    import sympy as sp

    return sum(1 for _ in sp.preorder_traversal(expr_tree))


def tree_structure_similarity(pred: str, truth: str) -> float:
    """Normalized structural similarity based on node counts of canonical forms."""
    try:
        import sympy as sp

        p = sp.simplify(_sympy_tree(pred))
        t = sp.simplify(_sympy_tree(truth))
        np_nodes = _node_count(p)
        nt_nodes = _node_count(t)
        if np_nodes == 0 and nt_nodes == 0:
            return 1.0
        # difference of the simplified difference: if algebraically equal, diff==0
        diff = sp.simplify(p - t)
        if diff == 0:
            return 1.0
        denom = max(np_nodes, nt_nodes)
        return float(1.0 - abs(np_nodes - nt_nodes) / denom) if denom else 0.0
    except Exception:
        pa = len(re.findall(r"[A-Za-z0-9_.]+|[+\-*/()]", to_python_expr(pred)))
        ta = len(re.findall(r"[A-Za-z0-9_.]+|[+\-*/()]", to_python_expr(truth)))
        denom = max(pa, ta)
        return float(1.0 - abs(pa - ta) / denom) if denom else 0.0


def _sample_points(
    variables: list[str],
    n_points: int,
    seed: int,
    low: float = -2.0,
    high: float = 2.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    data = {v: rng.uniform(low, high, n_points) for v in variables}
    return pd.DataFrame(data)


def numeric_similarity(
    pred: str,
    truth: str,
    variables: list[str],
    n_points: int = 400,
    seed: int = 20260709,
) -> dict[str, Any]:
    """Numeric similarity on INDEPENDENTLY sampled points (not training data)."""
    if not variables:
        variables = sorted(_identifiers(pred) | _identifiers(truth))
    if not variables:
        # both constant expressions
        variables = ["__dummy__"]
    frame = _sample_points(variables, n_points, seed)
    with np.errstate(all="ignore"):
        p = eval_expr(pred, frame)
        t = eval_expr(truth, frame)
    ok = np.isfinite(p) & np.isfinite(t)
    n_ok = int(ok.sum())
    if n_ok < 3:
        return {"score": 0.0, "n_valid_points": n_ok, "note": "insufficient defined points"}
    pv = p[ok]
    tv = t[ok]
    # scale-robust similarity: normalized RMSE mapped to (0,1]
    denom = float(np.std(tv)) if np.std(tv) > 1e-12 else (float(np.mean(np.abs(tv))) or 1.0)
    nrmse = float(np.sqrt(np.mean((pv - tv) ** 2)) / denom)
    score = float(1.0 / (1.0 + nrmse))
    return {"score": score, "n_valid_points": n_ok, "nrmse": nrmse}


def numeric_equivalence(
    pred: str,
    truth: str,
    variables: list[str],
    n_points: int = 400,
    seed: int = 20260709,
    tol: float = 1e-6,
) -> bool:
    if not variables:
        variables = sorted(_identifiers(pred) | _identifiers(truth))
    if not variables:
        variables = ["__dummy__"]
    frame = _sample_points(variables, n_points, seed)
    with np.errstate(all="ignore"):
        p = eval_expr(pred, frame)
        t = eval_expr(truth, frame)
    ok = np.isfinite(p) & np.isfinite(t)
    if int(ok.sum()) < 3:
        return False
    return bool(np.allclose(p[ok], t[ok], atol=tol, rtol=tol))


def algebraic_equivalence(pred: str, truth: str) -> bool:
    try:
        import sympy as sp

        diff = sp.simplify(_sympy_tree(pred) - _sympy_tree(truth))
        return bool(diff == 0)
    except Exception:
        return False


def support_f1(pred: str, truth: str) -> float:
    """F1 over the variable support (variables actually used)."""
    return _f1(_identifiers(truth), _identifiers(pred))


def expression_similarity_report(
    predicted: str,
    truth: str,
    variables: list[str] | None = None,
    n_points: int = 400,
    seed: int = 20260709,
) -> dict[str, Any]:
    variables = variables or sorted(_identifiers(predicted) | _identifiers(truth))
    num = numeric_similarity(predicted, truth, variables, n_points=n_points, seed=seed)
    var_f1 = _f1(_identifiers(truth), _identifiers(predicted))
    op_f1 = _f1(_operators(truth), _operators(predicted))
    tree_sim = tree_structure_similarity(predicted, truth)

    subscores = {
        "numeric_similarity": float(num["score"]),
        "variable_set_f1": float(var_f1),
        "operator_set_f1": float(op_f1),
        "tree_structure_similarity": float(tree_sim),
    }
    # all subscores must be in [0,1]
    for k, v in subscores.items():
        subscores[k] = float(min(1.0, max(0.0, v)))
    expr_sim = float(sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS))
    return {
        "predicted": predicted,
        "truth": truth,
        "variables": variables,
        "weights": WEIGHTS,
        "subscores": subscores,
        "expr_sim": expr_sim,
        "separate_metrics": {
            "algebraic_equivalence": algebraic_equivalence(predicted, truth),
            "numeric_equivalence": numeric_equivalence(predicted, truth, variables, n_points=n_points, seed=seed),
            "support_f1": float(support_f1(predicted, truth)),
        },
        "numeric_detail": num,
    }
