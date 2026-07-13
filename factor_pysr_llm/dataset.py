from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from .config import WorkflowConfig

DEFAULT_TARGET_PREFIXES = ("Ads_", "N2_", "H2_", "Hydrogenation_", "NH3_")
DEFAULT_ID_COLUMNS = ("structure_id", "design_index", "id", "ID", "sample_id", "Sample", "Task")
EPS = 1.0e-12


@dataclass(frozen=True)
class DatasetGrammar:
    """Declarative rules for turning a CSV into SR-ready X/y tables."""

    format: str
    encoding: str
    id_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    target_prefixes: tuple[str, ...]
    target_regex: tuple[str, ...]
    exclude_target_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    feature_regex: tuple[str, ...]
    exclude_feature_columns: tuple[str, ...]
    exclude_feature_regex: tuple[str, ...]
    numeric_only: bool
    add_missing_indicators: bool
    missing_fill: str
    scaling: str
    safe_prefix: str
    max_name_length: int

    @classmethod
    def from_config(cls, cfg: WorkflowConfig) -> "DatasetGrammar":
        data = dict(cfg.data.get("dataset") or {})
        targets = data.get("targets", data.get("target_columns", cfg.targets))
        target_rules = data.get("target_rules", {})
        feature_rules = data.get("feature_rules", {})
        missing = data.get("missing", {})
        naming = data.get("naming", {})
        return cls(
            format=str(data.get("format", "tabular_csv_v1")),
            encoding=str(data.get("encoding", "utf-8-sig")),
            id_columns=tuple(str(x) for x in data.get("id_columns", DEFAULT_ID_COLUMNS)),
            target_columns=tuple(str(x) for x in targets),
            target_prefixes=tuple(str(x) for x in target_rules.get("prefixes", DEFAULT_TARGET_PREFIXES)),
            target_regex=tuple(str(x) for x in target_rules.get("regex", [])),
            exclude_target_columns=tuple(str(x) for x in target_rules.get("exclude", [])),
            feature_columns=tuple(str(x) for x in feature_rules.get("include", [])),
            feature_regex=tuple(str(x) for x in feature_rules.get("regex", [])),
            exclude_feature_columns=tuple(str(x) for x in feature_rules.get("exclude", [])),
            exclude_feature_regex=tuple(str(x) for x in feature_rules.get("exclude_regex", [])),
            numeric_only=bool(feature_rules.get("numeric_only", True)),
            add_missing_indicators=bool(missing.get("add_indicators", True)),
            missing_fill=str(missing.get("fill", "median")),
            scaling=str(data.get("scaling", "zscore")),
            safe_prefix=str(naming.get("safe_prefix", "raw_")),
            max_name_length=int(naming.get("max_length", 96)),
        )


def read_input_csv(cfg: WorkflowConfig, grammar: DatasetGrammar | None = None) -> pd.DataFrame:
    grammar = grammar or DatasetGrammar.from_config(cfg)
    df = pd.read_csv(cfg.input_csv, encoding=grammar.encoding)
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def _match_any_regex(value: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, value) for pattern in patterns)


def infer_target_columns(df: pd.DataFrame, grammar: DatasetGrammar) -> list[str]:
    if grammar.target_columns:
        targets = [c for c in grammar.target_columns if c in df.columns]
    else:
        targets = []
        for col in df.columns:
            if col in grammar.exclude_target_columns:
                continue
            if grammar.target_prefixes and col.startswith(grammar.target_prefixes):
                targets.append(col)
                continue
            if grammar.target_regex and _match_any_regex(col, grammar.target_regex):
                targets.append(col)
    return list(dict.fromkeys(targets))


def infer_feature_columns(df: pd.DataFrame, target: str, grammar: DatasetGrammar, all_targets: list[str]) -> list[str]:
    blocked = set(grammar.id_columns) | set(all_targets) | {target}
    blocked.update(grammar.exclude_feature_columns)
    cols: list[str] = []
    explicit = set(grammar.feature_columns)
    for col in df.columns:
        if col in blocked:
            continue
        if grammar.exclude_feature_regex and _match_any_regex(col, grammar.exclude_feature_regex):
            continue
        if explicit and col not in explicit:
            continue
        if grammar.feature_regex and not _match_any_regex(col, grammar.feature_regex):
            continue
        if grammar.numeric_only:
            series = pd.to_numeric(df[col], errors="coerce")
            if not series.notna().any():
                continue
        cols.append(col)
    return cols


def safe_feature_name(name: str, used: set[str], grammar: DatasetGrammar) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip().lstrip("\ufeff")).strip("_")
    if not clean:
        clean = "x"
    if clean[0].isdigit():
        clean = f"x_{clean}"
    clean = f"{grammar.safe_prefix}{clean}"
    if len(clean) > grammar.max_name_length:
        clean = clean[: grammar.max_name_length].rstrip("_")
    base = clean
    out = base
    i = 2
    while out in used:
        suffix = f"_{i}"
        out = f"{base[: max(1, grammar.max_name_length - len(suffix))]}{suffix}"
        i += 1
    used.add(out)
    return out


def fill_numeric(series: pd.Series, policy: str) -> tuple[np.ndarray, float]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = numeric.to_numpy(dtype=float)
    if policy == "zero":
        fill_value = 0.0
    elif policy == "mean":
        fill_value = float(np.nanmean(values)) if np.isfinite(np.nanmean(values)) else 0.0
    elif policy == "median":
        fill_value = float(np.nanmedian(values)) if np.isfinite(np.nanmedian(values)) else 0.0
    else:
        raise ValueError(f"unsupported missing fill policy: {policy}")
    filled = np.where(np.isfinite(values), values, fill_value)
    return filled, fill_value


def scale_values(values: np.ndarray, mode: str) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=float)
    if mode == "none":
        return arr, 0.0, 1.0
    if mode != "zscore":
        raise ValueError(f"unsupported scaling mode: {mode}")
    mean = float(np.mean(arr))
    scale = float(np.std(arr))
    if not math.isfinite(scale) or abs(scale) <= EPS:
        scale = 1.0
    return (arr - mean) / scale, mean, scale


def feature_scores(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    scores: dict[str, float] = {}
    yy = np.asarray(y, dtype=float)
    for col in X.columns:
        xx = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(xx) & np.isfinite(yy)
        if int(ok.sum()) < 3 or float(np.std(xx[ok])) <= EPS:
            scores[col] = 0.0
            continue
        corr = np.corrcoef(xx[ok], yy[ok])[0, 1]
        scores[col] = abs(float(corr)) if np.isfinite(corr) else 0.0
    return pd.Series(scores).sort_values(ascending=False)


def apply_feature_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    selection_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not selection_cfg or selection_cfg.get("mode", "all") == "all":
        return X, {"mode": "all", "n_before": int(X.shape[1]), "n_after": int(X.shape[1])}
    mode = str(selection_cfg.get("mode", "top_k_abs_corr"))
    if mode != "top_k_abs_corr":
        raise ValueError(f"unsupported feature selection mode: {mode}")
    scores = feature_scores(X, y)
    top_k = selection_cfg.get("top_k")
    top_fraction = selection_cfg.get("top_fraction")
    if top_k is None and top_fraction is not None:
        top_k = max(1, int(math.ceil(float(top_fraction) * X.shape[1])))
    if top_k is None:
        top_k = X.shape[1]
    keep = list(scores.head(max(0, int(top_k))).index)
    return X[keep].copy(), {
        "mode": mode,
        "n_before": int(X.shape[1]),
        "n_after": int(len(keep)),
        "top_k": int(top_k),
        "score": "abs_pearson",
        "kept_features": keep,
    }


def build_raw_feature_table(
    cfg: WorkflowConfig,
    target: str,
    split_manifest_path: "Path | None" = None,
) -> dict[str, Any]:
    """Build the raw feature table for a target.

    If ``split_manifest_path`` is provided, all missing-value fill, scaling and
    correlation-based feature selection are fit on TRAIN rows only and then
    applied deterministically to all rows (no-leakage path). The resulting
    feature table still contains all target-finite rows; downstream code uses
    the split manifest to restrict training/selection/test evaluation.
    """
    grammar = DatasetGrammar.from_config(cfg)
    df = read_input_csv(cfg, grammar)
    targets = infer_target_columns(df, grammar)
    if target not in df.columns:
        raise KeyError(f"target not found: {target}")
    if target not in targets:
        targets = list(dict.fromkeys([target, *targets]))

    y_series = pd.to_numeric(df[target], errors="coerce")
    ok = y_series.notna() & np.isfinite(y_series.to_numpy(dtype=float))
    if int(ok.sum()) < 3:
        raise ValueError(f"target {target} has fewer than 3 finite rows")
    work = df.loc[ok].reset_index(drop=True)
    y = pd.to_numeric(work[target], errors="coerce").to_numpy(dtype=float)
    feature_cols = infer_feature_columns(work, target, grammar, targets)

    if split_manifest_path is not None:
        return _build_raw_feature_table_split(
            cfg, target, grammar, df, work, y, feature_cols, split_manifest_path
        )

    used: set[str] = set()
    cols: dict[str, np.ndarray] = {}
    feature_name_map: dict[str, str] = {}
    fill_values: dict[str, float] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    dropped: dict[str, str] = {}
    for col in feature_cols:
        filled, fill_value = fill_numeric(work[col], grammar.missing_fill)
        if not np.isfinite(filled).all():
            dropped[col] = "nonfinite_after_fill"
            continue
        if float(np.std(filled)) <= EPS:
            dropped[col] = "constant"
            continue
        safe = safe_feature_name(col, used, grammar)
        scaled, mean, scale = scale_values(filled, grammar.scaling)
        cols[safe] = scaled
        feature_name_map[safe] = col
        fill_values[safe] = fill_value
        means[safe] = mean
        scales[safe] = scale
        missing_mask = pd.to_numeric(work[col], errors="coerce").isna().astype(float).to_numpy(dtype=float)
        if grammar.add_missing_indicators and float(np.std(missing_mask)) > EPS:
            miss_safe = safe_feature_name(f"{col}__is_missing", used, grammar)
            cols[miss_safe] = missing_mask
            feature_name_map[miss_safe] = f"{col}__is_missing"
            fill_values[miss_safe] = 0.0
            means[miss_safe] = 0.0
            scales[miss_safe] = 1.0

    X = pd.DataFrame(cols)
    X, selection_manifest = apply_feature_selection(X, y, dict(cfg.data.get("raw_feature_selection") or {}))
    if X.shape[1] == 0:
        raise RuntimeError(f"no usable raw features for target {target}")

    lin = LinearRegression().fit(X.to_numpy(dtype=float), y)
    pred = lin.predict(X.to_numpy(dtype=float))
    linear_r2 = float(r2_score(y, pred))
    linear_rmse = float(math.sqrt(mean_squared_error(y, pred)))

    out_dir = cfg.output_root / "feature_tables" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_dir / "features.csv", index=False)
    X.to_csv(out_dir / "hybrid_features.csv", index=False)
    pd.DataFrame({"target": y}).to_csv(out_dir / "y.csv", index=False)
    manifest: dict[str, Any] = {
        "target": target,
        "input_csv": str(cfg.input_csv),
        "config_path": str(cfg.path),
        "builder": "raw_dataset_grammar",
        "n_input_rows": int(len(df)),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_candidate_feature_columns": int(len(feature_cols)),
        "linear_r2": linear_r2,
        "linear_rmse": linear_rmse,
        "dataset_grammar": grammar.__dict__,
        "feature_selection": selection_manifest,
        "feature_source": {col: "raw_dataset" for col in X.columns},
        "feature_name_map": {k: v for k, v in feature_name_map.items() if k in X.columns},
        "fill_values": {k: v for k, v in fill_values.items() if k in X.columns},
        "x_mean": {k: v for k, v in means.items() if k in X.columns},
        "x_scale": {k: v for k, v in scales.items() if k in X.columns},
        "dropped_features": dropped,
        "notes": [
            "Built from declarative dataset grammar; no project-specific target prefixes are required.",
            "Target-missing rows are dropped per target before feature construction.",
            "Feature names are sanitized and recorded in feature_name_map.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "target": target,
        "feature_dir": str(out_dir),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "linear_r2": linear_r2,
        "linear_rmse": linear_rmse,
    }


def _build_raw_feature_table_split(
    cfg: WorkflowConfig,
    target: str,
    grammar: DatasetGrammar,
    df: pd.DataFrame,
    work: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    split_manifest_path: "Path",
) -> dict[str, Any]:
    """No-leakage variant: fit preprocessing on train rows, transform all rows."""
    from .preprocess import fit_feature_selection, fit_preprocess, transform_preprocess
    from .splits import load_split_manifest, role_masks_for_frame

    manifest_obj = load_split_manifest(Path(split_manifest_path))
    masks = role_masks_for_frame(manifest_obj, work)
    train_mask = masks["train"]
    if int(train_mask.sum()) < 3:
        raise ValueError(f"split has fewer than 3 train rows for target {target}")

    # pre-compute safe names for every candidate column and its missing indicator
    used: set[str] = set()
    safe_names: dict[str, str] = {}
    for col in feature_cols:
        safe_names[col] = safe_feature_name(col, used, grammar)
        safe_names[f"{col}__is_missing"] = safe_feature_name(f"{col}__is_missing", used, grammar)

    state = fit_preprocess(
        work,
        feature_cols,
        train_mask,
        safe_names,
        missing_fill=grammar.missing_fill,
        scaling=grammar.scaling,
        add_missing_indicators=grammar.add_missing_indicators,
    )
    X_all = transform_preprocess(work, state)
    if X_all.shape[1] == 0:
        raise RuntimeError(f"no usable raw features for target {target}")

    keep, selection_manifest = fit_feature_selection(
        X_all, y, train_mask, dict(cfg.data.get("raw_feature_selection") or {})
    )
    X_all = X_all[keep].copy()
    state.selected_features = keep
    state.selection_manifest = selection_manifest

    # linear R2 reported on TRAIN rows only (no leakage into a headline number)
    Xtr = X_all.to_numpy(dtype=float)[train_mask]
    ytr = np.asarray(y, dtype=float)[train_mask]
    lin = LinearRegression().fit(Xtr, ytr)
    pred_tr = lin.predict(Xtr)
    linear_r2 = float(r2_score(ytr, pred_tr))
    linear_rmse = float(math.sqrt(mean_squared_error(ytr, pred_tr)))

    out_dir = cfg.output_root / "feature_tables" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    X_all.to_csv(out_dir / "features.csv", index=False)
    X_all.to_csv(out_dir / "hybrid_features.csv", index=False)
    pd.DataFrame({"target": y}).to_csv(out_dir / "y.csv", index=False)
    role = np.array(["unassigned"] * len(work), dtype=object)
    role[masks["train"]] = "train"
    role[masks["validation"]] = "validation"
    role[masks["test"]] = "test"
    pd.DataFrame({"role": role}).to_csv(out_dir / "row_roles.csv", index=False)

    feature_name_map = {k: v for k, v in state.feature_name_map.items() if k in X_all.columns}
    manifest: dict[str, Any] = {
        "target": target,
        "input_csv": str(cfg.input_csv),
        "config_path": str(cfg.path),
        "builder": "raw_dataset_grammar_train_fit",
        "split_manifest": str(split_manifest_path),
        "split_sha256": manifest_obj.content_sha256(),
        "n_input_rows": int(len(df)),
        "n_rows": int(len(X_all)),
        "n_train_rows": int(train_mask.sum()),
        "n_validation_rows": int(masks["validation"].sum()),
        "n_test_rows": int(masks["test"].sum()),
        "n_features": int(X_all.shape[1]),
        "n_candidate_feature_columns": int(len(feature_cols)),
        "train_linear_r2": linear_r2,
        "train_linear_rmse": linear_rmse,
        "dataset_grammar": grammar.__dict__,
        "feature_selection": selection_manifest,
        "feature_source": {col: "raw_dataset" for col in X_all.columns},
        "feature_name_map": feature_name_map,
        "fill_values": {k: v for k, v in state.fill_values.items() if k in X_all.columns},
        "x_mean": {k: v for k, v in state.means.items() if k in X_all.columns},
        "x_scale": {k: v for k, v in state.scales.items() if k in X_all.columns},
        "dropped_features": state.dropped,
        "preprocess_state": state.to_dict(),
        "notes": [
            "No-leakage build: fill/scale/selection fit on train rows only.",
            "Feature table contains all target-finite rows; use row_roles.csv / split manifest downstream.",
            "train_linear_r2 is computed on train rows only.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "target": target,
        "feature_dir": str(out_dir),
        "n_rows": int(len(X_all)),
        "n_train_rows": int(train_mask.sum()),
        "n_features": int(X_all.shape[1]),
        "linear_r2": linear_r2,
        "linear_rmse": linear_rmse,
        "split_sha256": manifest_obj.content_sha256(),
    }


def inspect_dataset(cfg: WorkflowConfig) -> dict[str, Any]:
    grammar = DatasetGrammar.from_config(cfg)
    df = read_input_csv(cfg, grammar)
    targets = infer_target_columns(df, grammar)
    feature_counts = {}
    for target in targets:
        feature_counts[target] = len(infer_feature_columns(df, target, grammar, targets))
    return {
        "input_csv": str(cfg.input_csv),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "targets": targets,
        "n_targets": int(len(targets)),
        "feature_counts_by_target": feature_counts,
        "id_columns_present": [c for c in grammar.id_columns if c in df.columns],
        "grammar": grammar.__dict__,
    }

