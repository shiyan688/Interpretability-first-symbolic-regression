from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

EPS = 1.0e-12


@dataclass
class PreprocessState:
    """Fitted preprocessing state.

    All statistics (fill values, means, scales, selected features) are computed
    ONLY from the training rows and then applied deterministically to
    validation/test rows. This is the core no-leakage guarantee.
    """

    missing_fill: str
    scaling: str
    fill_values: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    feature_name_map: dict[str, str] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)
    selection_manifest: dict[str, Any] = field(default_factory=dict)
    fit_row_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_fill": self.missing_fill,
            "scaling": self.scaling,
            "fill_values": self.fill_values,
            "means": self.means,
            "scales": self.scales,
            "feature_name_map": self.feature_name_map,
            "dropped": self.dropped,
            "selected_features": self.selected_features,
            "selection_manifest": self.selection_manifest,
            "fit_row_count": int(self.fit_row_count),
        }


def _fill_value(values: np.ndarray, policy: str) -> float:
    finite = values[np.isfinite(values)]
    if policy == "zero":
        return 0.0
    if finite.size == 0:
        return 0.0
    if policy == "mean":
        v = float(np.mean(finite))
    elif policy == "median":
        v = float(np.median(finite))
    else:
        raise ValueError(f"unsupported missing fill policy: {policy}")
    return v if math.isfinite(v) else 0.0


def fit_preprocess(
    df: pd.DataFrame,
    feature_columns: list[str],
    train_mask: np.ndarray,
    safe_names: dict[str, str],
    missing_fill: str = "median",
    scaling: str = "zscore",
    add_missing_indicators: bool = True,
) -> PreprocessState:
    """Fit fill/scale statistics using ONLY training rows.

    df: target-finite frame (all rows).
    feature_columns: original column names to process.
    train_mask: boolean mask over df rows selecting training rows.
    safe_names: mapping original column -> safe feature name (and __is_missing).
    """
    state = PreprocessState(missing_fill=missing_fill, scaling=scaling)
    state.fit_row_count = int(train_mask.sum())
    for col in feature_columns:
        raw = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw_values = raw.to_numpy(dtype=float)
        train_values = raw_values[train_mask]
        fill_val = _fill_value(train_values, missing_fill)
        filled_train = np.where(np.isfinite(train_values), train_values, fill_val)
        if not np.isfinite(filled_train).all():
            state.dropped[col] = "nonfinite_after_fill"
            continue
        if float(np.std(filled_train)) <= EPS:
            state.dropped[col] = "constant_on_train"
            continue
        safe = safe_names[col]
        if scaling == "none":
            mean, scale = 0.0, 1.0
        elif scaling == "zscore":
            mean = float(np.mean(filled_train))
            scale = float(np.std(filled_train))
            if not math.isfinite(scale) or abs(scale) <= EPS:
                scale = 1.0
        else:
            raise ValueError(f"unsupported scaling mode: {scaling}")
        state.fill_values[safe] = fill_val
        state.means[safe] = mean
        state.scales[safe] = scale
        state.feature_name_map[safe] = col
        if add_missing_indicators:
            miss_train = raw.isna().to_numpy(dtype=bool)[train_mask].astype(float)
            if float(np.std(miss_train)) > EPS:
                miss_safe = safe_names.get(f"{col}__is_missing")
                if miss_safe:
                    state.fill_values[miss_safe] = 0.0
                    state.means[miss_safe] = 0.0
                    state.scales[miss_safe] = 1.0
                    state.feature_name_map[miss_safe] = f"{col}__is_missing"
    return state


def transform_preprocess(df: pd.DataFrame, state: PreprocessState) -> pd.DataFrame:
    """Apply fitted state deterministically to ALL rows of df."""
    cols: dict[str, np.ndarray] = {}
    inv_map = state.feature_name_map
    for safe, orig in inv_map.items():
        if orig.endswith("__is_missing"):
            base = orig[: -len("__is_missing")]
            raw = pd.to_numeric(df.get(base, pd.Series([np.nan] * len(df))), errors="coerce")
            values = raw.isna().astype(float).to_numpy(dtype=float)
            cols[safe] = values
            continue
        raw = pd.to_numeric(df[orig], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = raw.to_numpy(dtype=float)
        fill_val = state.fill_values.get(safe, 0.0)
        filled = np.where(np.isfinite(values), values, fill_val)
        mean = state.means.get(safe, 0.0)
        scale = state.scales.get(safe, 1.0)
        if abs(scale) <= EPS:
            scale = 1.0
        cols[safe] = (filled - mean) / scale
    return pd.DataFrame(cols, index=df.index)


def feature_scores_train(X: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> pd.Series:
    """Absolute Pearson correlation computed on TRAIN rows only."""
    scores: dict[str, float] = {}
    yy = np.asarray(y, dtype=float)[train_mask]
    for col in X.columns:
        xx = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)[train_mask]
        ok = np.isfinite(xx) & np.isfinite(yy)
        if int(ok.sum()) < 3 or float(np.std(xx[ok])) <= EPS:
            scores[col] = 0.0
            continue
        corr = np.corrcoef(xx[ok], yy[ok])[0, 1]
        scores[col] = abs(float(corr)) if np.isfinite(corr) else 0.0
    return pd.Series(scores).sort_values(ascending=False)


def fit_feature_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    selection_cfg: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Select features using TRAIN-only correlation. Returns kept column names."""
    if not selection_cfg or selection_cfg.get("mode", "all") == "all":
        return list(X.columns), {"mode": "all", "n_before": int(X.shape[1]), "n_after": int(X.shape[1])}
    mode = str(selection_cfg.get("mode", "top_k_abs_corr"))
    if mode != "top_k_abs_corr":
        raise ValueError(f"unsupported feature selection mode: {mode}")
    scores = feature_scores_train(X, y, train_mask)
    top_k = selection_cfg.get("top_k")
    top_fraction = selection_cfg.get("top_fraction")
    if top_k is None and top_fraction is not None:
        top_k = max(1, int(math.ceil(float(top_fraction) * X.shape[1])))
    if top_k is None:
        top_k = X.shape[1]
    keep = list(scores.head(max(0, int(top_k))).index)
    return keep, {
        "mode": mode,
        "n_before": int(X.shape[1]),
        "n_after": int(len(keep)),
        "top_k": int(top_k),
        "score": "abs_pearson_train_only",
        "kept_features": keep,
    }
