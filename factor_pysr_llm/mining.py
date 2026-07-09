from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .features import safe_read_csv


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _target_from_path(path: Path) -> str:
    if path.parent.name:
        return path.parent.name
    return "unknown"


def _best_result_rows(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    expr = str(data.get("best_equation") or data.get("expression") or data.get("equation") or "").strip()
    if not expr:
        return []
    return [
        {
            "target": str(data.get("target") or _target_from_path(path)),
            "source_kind": "best_result",
            "source_file": str(path),
            "rank_in_file": 0,
            "expression": expr,
            "r2": data.get("best_r2", data.get("r2_verified")),
            "rmse": data.get("best_rmse", data.get("rmse_verified")),
            "complexity": data.get("best_complexity", data.get("complexity")),
            "loss": data.get("loss"),
            "score": data.get("score"),
            "status": data.get("status"),
        }
    ]


def _equation_rows(path: Path) -> list[dict[str, Any]]:
    try:
        df = safe_read_csv(path)
    except Exception:
        return []
    eq_col = "equation" if "equation" in df.columns else "Equation"
    if eq_col not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    target = _target_from_path(path)
    for i, row in df.iterrows():
        expr = str(row.get(eq_col, "")).strip()
        if not expr:
            continue
        rows.append(
            {
                "target": target,
                "source_kind": "equation_snapshot",
                "source_file": str(path),
                "rank_in_file": int(i),
                "expression": expr,
                "r2": row.get("r2", row.get("R2")),
                "rmse": row.get("rmse", row.get("RMSE")),
                "complexity": row.get("complexity", row.get("Complexity")),
                "loss": row.get("loss", row.get("Loss")),
                "score": row.get("score", row.get("Score")),
                "status": "",
            }
        )
    return rows


def mine_expression_list(
    roots: list[Path],
    output_path: Path,
    targets: list[str] | None = None,
    top_k_per_target: int | None = None,
) -> pd.DataFrame:
    target_filter = set(targets or [])
    rows: list[dict[str, Any]] = []
    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue
        for path in root.rglob("best_result.json"):
            rows.extend(_best_result_rows(path))
        for path in root.rglob("model_equations_snapshot.csv"):
            rows.extend(_equation_rows(path))
    df = pd.DataFrame(rows)
    if df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df
    if target_filter:
        df = df[df["target"].astype(str).isin(target_filter)].copy()
    df["_r2_sort"] = pd.to_numeric(df.get("r2"), errors="coerce")
    df["_score_sort"] = pd.to_numeric(df.get("score"), errors="coerce")
    df["_loss_sort"] = pd.to_numeric(df.get("loss"), errors="coerce")
    df["_complexity_sort"] = pd.to_numeric(df.get("complexity"), errors="coerce")
    df = df.sort_values(
        ["target", "_r2_sort", "_score_sort", "_loss_sort", "_complexity_sort"],
        ascending=[True, False, False, True, True],
        na_position="last",
    )
    df = df.drop_duplicates(["target", "expression"], keep="first")
    if top_k_per_target and top_k_per_target > 0:
        df = df.groupby("target", group_keys=False).head(int(top_k_per_target)).copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

