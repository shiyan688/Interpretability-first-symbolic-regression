from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .expr import eval_expr, metric_dict
from .features import finite_frame, safe_read_csv


def read_feature_dir(feature_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    x_path = feature_dir / "hybrid_features.csv"
    if not x_path.exists():
        x_path = feature_dir / "features.csv"
    y_path = feature_dir / "y.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"feature_dir must contain features/hybrid_features and y.csv: {feature_dir}")
    X = finite_frame(safe_read_csv(x_path))
    y_df = safe_read_csv(y_path)
    y_col = "target" if "target" in y_df.columns else y_df.columns[0]
    y = pd.to_numeric(y_df[y_col], errors="coerce").to_numpy(dtype=float)
    return X, y


def equation_from_result(result_path: Path) -> str:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    for key in ("best_equation", "equation", "expression"):
        value = str(data.get(key, "")).strip()
        if value:
            return value
    raise ValueError(f"no equation found in result file: {result_path}")


def verify_expression(feature_dir: Path, expression: str) -> dict[str, Any]:
    X, y = read_feature_dir(feature_dir)
    pred = eval_expr(expression, X)
    out = metric_dict(y, pred)
    out.update(
        {
            "feature_dir": str(feature_dir),
            "expression": expression,
            "n_features": int(X.shape[1]),
        }
    )
    return out


def verify_result(feature_dir: Path, result_path: Path) -> dict[str, Any]:
    expr = equation_from_result(result_path)
    out = verify_expression(feature_dir, expr)
    out["result_path"] = str(result_path)
    return out


def verify_hof(feature_dir: Path, hof_path: Path) -> dict[str, Any]:
    hof = safe_read_csv(hof_path)
    eq_col = "equation" if "equation" in hof.columns else "Equation"
    if eq_col not in hof.columns:
        raise ValueError(f"no equation column in {hof_path}")
    if "loss" in hof.columns:
        row = hof.sort_values("loss", ascending=True).iloc[0]
    elif "Loss" in hof.columns:
        row = hof.sort_values("Loss", ascending=True).iloc[0]
    elif "score" in hof.columns:
        row = hof.sort_values("score", ascending=False).iloc[0]
    elif "Score" in hof.columns:
        row = hof.sort_values("Score", ascending=False).iloc[0]
    else:
        row = hof.iloc[-1]
    expr = str(row[eq_col])
    out = verify_expression(feature_dir, expr)
    out["hof_path"] = str(hof_path)
    for key in ("loss", "Loss", "complexity", "Complexity", "score", "Score"):
        if key in row.index:
            try:
                out[f"hof_{key}"] = float(row[key])
            except Exception:
                out[f"hof_{key}"] = str(row[key])
    return out


def write_verify_json(output_path: Path, data: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

