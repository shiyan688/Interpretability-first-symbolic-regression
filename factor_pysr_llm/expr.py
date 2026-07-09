from __future__ import annotations

import ast
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

EPS = 1.0e-12


def bars_to_abs(expr: str) -> str:
    out: list[str] = []
    open_abs = False
    for ch in str(expr):
        if ch == "|":
            out.append(")" if open_abs else "abs(")
            open_abs = not open_abs
        else:
            out.append(ch)
    if open_abs:
        out.append(")")
    return "".join(out)


def to_python_expr(expr: str) -> str:
    text = bars_to_abs(str(expr))
    return text.replace("^", "**")


def safe_inv(x: Any) -> Any:
    return 1.0 / x


def safe_log(x: Any) -> Any:
    return np.log(np.abs(x))


def safe_sqrt(x: Any) -> Any:
    return np.sqrt(np.abs(x))


def namespace_from_frame(frame: pd.DataFrame) -> dict[str, Any]:
    ns: dict[str, Any] = {
        "abs": np.abs,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": safe_log,
        "sqrt": safe_sqrt,
        "inv": safe_inv,
        "square": lambda x: x * x,
        "cube": lambda x: x * x * x,
        "cbrt": np.cbrt,
        "pow": np.power,
        "pi": math.pi,
    }
    for col in frame.columns:
        ns[str(col)] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    return ns


def eval_expr(expr: str, frame: pd.DataFrame) -> np.ndarray:
    with np.errstate(all="ignore"):
        value = eval(to_python_expr(expr), {"__builtins__": {}}, namespace_from_frame(frame))
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        arr = np.full(len(frame), float(arr), dtype=float)
    return arr


def finite_array(values: Any, n: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape == () and n is not None:
        arr = np.full(n, float(arr), dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def metric_dict(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(pred, dtype=float)
    ok = np.isfinite(yy) & np.isfinite(pp)
    if int(ok.sum()) < 3:
        return {"r2_verified": None, "rmse_verified": None, "n_finite": int(ok.sum())}
    return {
        "r2_verified": float(r2_score(yy[ok], pp[ok])),
        "rmse_verified": float(math.sqrt(mean_squared_error(yy[ok], pp[ok]))),
        "n_finite": int(ok.sum()),
    }


def identifier_names(expr: str) -> set[str]:
    """Return variable-like identifiers in an expression.

    This is used for diagnostics only; expression execution still happens in a
    restricted namespace.
    """

    try:
        tree = ast.parse(to_python_expr(expr), mode="eval")
    except SyntaxError:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(expr)))
    funcs = {
        "abs",
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "inv",
        "square",
        "cube",
        "cbrt",
        "pow",
        "pi",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return {x for x in names if x not in funcs}

