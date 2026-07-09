from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from .config import WorkflowConfig
from .expr import EPS, eval_expr


def safe_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return text or "source"


def finite_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def target_vector(input_csv: Path, target: str) -> np.ndarray:
    df = safe_read_csv(input_csv)
    if target not in df.columns:
        raise KeyError(f"target not found in input csv: {target}")
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        bad = int((~np.isfinite(y)).sum())
        raise ValueError(f"target {target} has {bad} non-finite values")
    return y


def raw_feature_frame_from_manifest(input_csv: Path, target: str, manifest_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = read_json(manifest_path)
    df = safe_read_csv(input_csv)
    names = manifest["safe_feature_names"]
    fmap = manifest["feature_name_map"]
    means = np.asarray(manifest["x_mean"], dtype=float)
    scales = np.asarray(manifest["x_scale"], dtype=float)
    cols: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        orig = str(fmap[name])
        if orig.endswith("__is_missing"):
            base = orig[: -len("__is_missing")]
            raw = pd.to_numeric(df.get(base, pd.Series([np.nan] * len(df))), errors="coerce")
            values = raw.isna().astype(float).to_numpy(dtype=float)
        else:
            if orig not in df.columns:
                raise KeyError(f"raw feature {orig} from manifest missing in {input_csv}")
            values = pd.to_numeric(df[orig], errors="coerce").to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, 0.0)
        scale = scales[i] if abs(float(scales[i])) > EPS else 1.0
        cols[str(name)] = (values - means[i]) / scale
    y_raw = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    y_scale = float(manifest.get("y_scale", 1.0)) or 1.0
    y_z = (y_raw - float(manifest.get("y_mean", 0.0))) / y_scale
    return pd.DataFrame(cols), y_z


def add_frame(
    parts: list[pd.DataFrame],
    sources: dict[str, str],
    df: pd.DataFrame,
    source: str,
    dedupe_exact_collision_values: bool = True,
) -> dict[str, Any]:
    clean = finite_frame(df)
    keep: dict[str, np.ndarray] = {}
    used = set(sources)
    stats: dict[str, Any] = {
        "source": source,
        "input_columns": int(clean.shape[1]),
        "kept_columns": 0,
        "dropped_exact_duplicates": 0,
        "renamed_collisions": 0,
    }
    existing = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=clean.index)
    for col in clean.columns:
        base = str(col)
        values = clean[col].to_numpy(dtype=float)
        name = base
        if name in used:
            old = existing[name].to_numpy(dtype=float) if name in existing.columns else None
            if (
                dedupe_exact_collision_values
                and old is not None
                and old.shape == values.shape
                and np.allclose(old, values, rtol=1e-10, atol=1e-10)
            ):
                stats["dropped_exact_duplicates"] += 1
                continue
            prefix = safe_slug(source)
            name = f"{prefix}__{base}"
            i = 2
            while name in used:
                name = f"{prefix}__{base}_{i}"
                i += 1
            stats["renamed_collisions"] += 1
        keep[name] = values
        sources[name] = source
        used.add(name)
    if keep:
        parts.append(pd.DataFrame(keep))
    stats["kept_columns"] = int(len(keep))
    return stats


def read_equation_from_best_result(best_result_path: Path) -> str:
    best = read_json(best_result_path)
    for key in ("best_equation", "equation", "expression"):
        value = str(best.get(key, "")).strip()
        if value:
            return value
    raise ValueError(f"no equation found in {best_result_path}")


def best_prediction_from_result(best_result_path: Path, frame: pd.DataFrame) -> np.ndarray:
    expr = read_equation_from_best_result(best_result_path)
    return eval_expr(expr, frame)


def formula_feature_name(expr: str, prefix: str) -> str:
    body = re.sub(r"[^A-Za-z0-9_]+", "_", str(expr)).strip("_")
    if not body:
        body = "expr"
    return f"{safe_slug(prefix)}_{body[:88]}"


def sorted_equations(equations: pd.DataFrame) -> pd.DataFrame:
    out = equations.copy()
    for col in ("score", "Score"):
        if col in out.columns:
            return out.sort_values(col, ascending=False)
    for col in ("loss", "Loss"):
        if col in out.columns:
            return out.sort_values(col, ascending=True)
    for col in ("complexity", "Complexity"):
        if col in out.columns:
            return out.sort_values(col, ascending=True)
    return out


def equation_snapshot_predictions(equations_path: Path, frame: pd.DataFrame, source: str, top_n: int) -> pd.DataFrame:
    if top_n <= 0 or not equations_path.exists():
        return pd.DataFrame()
    equations = sorted_equations(safe_read_csv(equations_path))
    eq_col = "equation" if "equation" in equations.columns else "Equation"
    if eq_col not in equations.columns:
        return pd.DataFrame()
    cols: dict[str, np.ndarray] = {}
    for _, row in equations.head(top_n).iterrows():
        expr = str(row.get(eq_col, "")).strip()
        if not expr:
            continue
        try:
            pred = eval_expr(expr, frame)
        except Exception:
            continue
        if len(pred) != len(frame):
            continue
        if not np.isfinite(pred).any():
            continue
        pred = np.where(np.isfinite(pred), pred, 0.0)
        name = formula_feature_name(expr, f"{source}_eq")
        base = name
        i = 2
        while name in cols:
            name = f"{base}_{i}"
            i += 1
        cols[name] = pred
    return pd.DataFrame(cols)


def load_raw_pysr_source(cfg: WorkflowConfig, target: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_cfg = dict(cfg.data.get("raw_pysr") or {})
    if not raw_cfg:
        return pd.DataFrame(), pd.DataFrame(), {}
    run_root = cfg.resolve_path(raw_cfg["run_root"])
    manifest_path = run_root / target / "input_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"raw PySR manifest missing: {manifest_path}")
    raw_z, _ = raw_feature_frame_from_manifest(cfg.input_csv, target, manifest_path)
    meta_cols: dict[str, np.ndarray] = {}
    if raw_cfg.get("include_best_prediction", True):
        confirmed_path = cfg.resolve_path(raw_cfg["confirmed_csv"])
        confirmed = safe_read_csv(confirmed_path)
        if target in set(confirmed.get("target", pd.Series(dtype=str)).astype(str)):
            expr = str(confirmed.loc[confirmed["target"].astype(str).eq(target), "expression"].iloc[0])
            manifest = read_json(manifest_path)
            pred_z = eval_expr(expr, raw_z)
            y_mean = float(manifest.get("y_mean", 0.0))
            y_scale = float(manifest.get("y_scale", 1.0)) or 1.0
            meta_cols["old_raw_pysr_best_zpred"] = pred_z
            meta_cols["old_raw_pysr_best_rawpred"] = y_mean + y_scale * pred_z
    if not raw_cfg.get("include_z_features", True):
        raw_z = pd.DataFrame(index=range(len(target_vector(cfg.input_csv, target))))
    meta = pd.DataFrame(meta_cols)
    summary = {
        "name": raw_cfg.get("name", "raw_pysr"),
        "manifest_path": str(manifest_path),
        "n_z_features": int(raw_z.shape[1]),
        "n_meta_predictions": int(meta.shape[1]),
    }
    return raw_z, meta, summary


def source_feature_frame(cfg: WorkflowConfig, source_cfg: dict[str, Any], target: str) -> pd.DataFrame:
    path = cfg.resolve_path(str(source_cfg["feature_path"]).format(target=target))
    if not path.exists():
        raise FileNotFoundError(f"feature source missing: {path}")
    return finite_frame(safe_read_csv(path))


def build_union(cfg: WorkflowConfig, target: str) -> dict[str, Any]:
    out_dir = cfg.output_root / "feature_tables" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    y = target_vector(cfg.input_csv, target)

    parts: list[pd.DataFrame] = []
    sources: dict[str, str] = {}
    part_stats: list[dict[str, Any]] = []
    dedupe_exact = bool(cfg.data.get("dedupe_exact_collision_values", True))

    raw_z, raw_meta, raw_summary = load_raw_pysr_source(cfg, target)
    if raw_z.shape[1]:
        part_stats.append(add_frame(parts, sources, raw_z, str(raw_summary.get("name", "raw_pysr_z_features")), dedupe_exact))
    if raw_meta.shape[1]:
        part_stats.append(add_frame(parts, sources, raw_meta, f"{raw_summary.get('name', 'raw_pysr')}_best_prediction", dedupe_exact))

    for source_cfg in cfg.data.get("feature_sources", []):
        name = str(source_cfg.get("name", "source"))
        frame = source_feature_frame(cfg, source_cfg, target)
        if len(frame) != len(y):
            raise ValueError(f"{name} rows {len(frame)} != target rows {len(y)} for {target}")
        if source_cfg.get("include_features", True):
            part_stats.append(add_frame(parts, sources, frame, name, dedupe_exact))
        best_path = cfg.format_path(source_cfg.get("best_result_path"), target)
        if source_cfg.get("include_best_prediction", True) and best_path and best_path.exists():
            try:
                pred = best_prediction_from_result(best_path, frame)
                pred_col = str(source_cfg.get("best_prediction_column") or f"{name}_best_pred")
                part_stats.append(add_frame(parts, sources, pd.DataFrame({pred_col: pred}), f"{name}_best_prediction", dedupe_exact))
            except Exception as exc:
                part_stats.append({"source": f"{name}_best_prediction", "error": repr(exc)})
        equations_path = cfg.format_path(source_cfg.get("equations_path"), target)
        if source_cfg.get("include_equation_predictions", False) and equations_path:
            preds = equation_snapshot_predictions(equations_path, frame, name, int(source_cfg.get("top_equations", 12)))
            if preds.shape[1]:
                part_stats.append(add_frame(parts, sources, preds, f"{name}_equation_snapshot_predictions", dedupe_exact))

    if not parts:
        raise RuntimeError(f"no feature parts were built for target {target}")
    X = finite_frame(pd.concat(parts, axis=1))
    X = X.loc[:, ~X.columns.duplicated()].copy()
    zero_var = [c for c in X.columns if float(np.nanstd(X[c].to_numpy(dtype=float))) <= 0.0]
    if zero_var:
        X = X.drop(columns=zero_var)
        for col in zero_var:
            sources.pop(col, None)

    lin = LinearRegression().fit(X.to_numpy(dtype=float), y)
    pred = lin.predict(X.to_numpy(dtype=float))
    linear_r2 = float(r2_score(y, pred))
    linear_rmse = float(math.sqrt(mean_squared_error(y, pred)))

    X.to_csv(out_dir / "features.csv", index=False)
    X.to_csv(out_dir / "hybrid_features.csv", index=False)
    pd.DataFrame({"target": y}).to_csv(out_dir / "y.csv", index=False)
    manifest: dict[str, Any] = {
        "target": target,
        "input_csv": str(cfg.input_csv),
        "config_path": str(cfg.path),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "linear_r2": linear_r2,
        "linear_rmse": linear_rmse,
        "raw_pysr": raw_summary,
        "dedupe_exact_collision_values": dedupe_exact,
        "part_stats": part_stats,
        "dropped_zero_variance": zero_var,
        "feature_source": sources,
        "notes": [
            "Feature union combines historical high-signal factors and formula predictions.",
            "Formula prediction columns are meta-factors and must be expanded before final interpretation.",
            "No target residual/leakage columns are intentionally added.",
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    return {
        "target": target,
        "feature_dir": str(out_dir),
        "n_features": int(X.shape[1]),
        "linear_r2": linear_r2,
        "linear_rmse": linear_rmse,
    }
