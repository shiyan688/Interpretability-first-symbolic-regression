"""Leakage-aware cross-validation for the factor-mining pipeline.

Motivation
----------
The pipeline selects factors by ``|corr(factor, y)|`` (see ``factor_miner``).
That selection uses labels. If a K-fold split happens *after* the factor set is
fixed on the full dataset, every test fold's labels have already influenced
which factors exist -- a form of feature-selection leakage that makes CV scores
optimistically biased. The bias grows with noise and with the number of
enumerated candidates (winner's curse / multiple comparisons).

This module offers two evaluation modes so the effect is measurable:

- ``honest``: mine + select *inside each training fold only*; score on the
  held-out fold. This is the correct, leakage-free estimate.
- ``leaky``:  mine + select once on the full data, then cross-validate only the
  final linear fit. This reproduces the optimistic behaviour of running
  selection outside the CV loop.

The gap ``leaky_r2 - honest_r2`` quantifies the leakage.

The default per-fold model is a plain ``LinearRegression`` surrogate. The
leakage question is fundamentally about *feature selection*, and a linear
surrogate answers it directly and cheaply -- running full PySR K times (each a
long search) is optional and gated behind ``model="pysr"``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from .factor_miner import FeatureSpec, build_design_matrix, select_factor_expressions


def _fit_score_linear(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Fit a linear surrogate on the train design matrix and score the test fold.

    Columns are aligned by name; any column present at train but missing at test
    (should not happen, specs are shared) is treated as zeros.
    """

    cols = list(X_train.columns)
    Xtr = X_train[cols].to_numpy(dtype=float)
    Xte = X_test.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)
    lin = LinearRegression().fit(Xtr, y_train)
    pred = lin.predict(Xte)
    ok = np.isfinite(pred) & np.isfinite(y_test)
    if int(ok.sum()) < 2:
        return {"r2": float("nan"), "rmse": float("nan"), "n_test": int(ok.sum())}
    return {
        "r2": float(r2_score(y_test[ok], pred[ok])),
        "rmse": float(math.sqrt(mean_squared_error(y_test[ok], pred[ok]))),
        "n_test": int(ok.sum()),
    }


def _summary(fold_scores: list[dict[str, float]]) -> dict[str, Any]:
    r2 = np.array([s["r2"] for s in fold_scores], dtype=float)
    rmse = np.array([s["rmse"] for s in fold_scores], dtype=float)
    r2 = r2[np.isfinite(r2)]
    rmse = rmse[np.isfinite(rmse)]
    return {
        "mean_r2": float(np.mean(r2)) if r2.size else float("nan"),
        "std_r2": float(np.std(r2)) if r2.size else float("nan"),
        "mean_rmse": float(np.mean(rmse)) if rmse.size else float("nan"),
        "std_rmse": float(np.std(rmse)) if rmse.size else float("nan"),
        "n_folds_scored": int(r2.size),
        "per_fold": fold_scores,
    }


def evaluate_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    mining_opts: dict[str, Any] | None = None,
    selection_opts: dict[str, Any] | None = None,
    k: int = 5,
    mode: str = "both",
    model: str = "linear",
    seed: int = 20260714,
) -> dict[str, Any]:
    """Cross-validate the mining pipeline.

    Parameters
    ----------
    X, y            : raw feature frame and target (as produced by build-raw).
    mining_opts     : forwarded to mine_factors_from_frame.
    selection_opts  : raw_top_k / raw_top_fraction / factor_top_k.
    k               : number of folds.
    mode            : "honest", "leaky", or "both".
    model           : "linear" (default) or "pysr" (per-fold PySR; expensive).
    seed            : KFold shuffle seed.
    """

    if model != "linear":
        raise NotImplementedError(
            "Only the linear surrogate is implemented; per-fold PySR is a planned "
            "opt-in and intentionally left out of the default path."
        )

    X = X.reset_index(drop=True)
    yy = np.asarray(y, dtype=float)
    n = len(X)
    k = max(2, min(int(k), n))
    kf = KFold(n_splits=k, shuffle=True, random_state=int(seed))
    splits = list(kf.split(np.arange(n)))

    out: dict[str, Any] = {
        "n_rows": int(n),
        "n_raw_features": int(X.shape[1]),
        "k": int(k),
        "model": model,
        "seed": int(seed),
        "mode": mode,
    }

    if mode in ("honest", "both"):
        honest_folds: list[dict[str, float]] = []
        honest_n_specs: list[int] = []
        for tr, te in splits:
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = yy[tr], yy[te]
            specs = select_factor_expressions(X_tr, y_tr, mining_opts, selection_opts)
            honest_n_specs.append(len(specs))
            if not specs:
                honest_folds.append({"r2": float("nan"), "rmse": float("nan"), "n_test": len(te)})
                continue
            D_tr = build_design_matrix(X_tr, specs)
            D_te = build_design_matrix(X_te, specs)
            honest_folds.append(_fit_score_linear(D_tr, y_tr, D_te, y_te))
        out["honest"] = _summary(honest_folds)
        out["honest"]["mean_n_selected"] = float(np.mean(honest_n_specs)) if honest_n_specs else 0.0

    if mode in ("leaky", "both"):
        # Select ONCE on the full data (this is the leak), then CV only the fit.
        full_specs: list[FeatureSpec] = select_factor_expressions(X, yy, mining_opts, selection_opts)
        D_full = build_design_matrix(X, full_specs)
        leaky_folds: list[dict[str, float]] = []
        for tr, te in splits:
            leaky_folds.append(
                _fit_score_linear(D_full.iloc[tr], yy[tr], D_full.iloc[te], yy[te])
            )
        out["leaky"] = _summary(leaky_folds)
        out["leaky"]["n_selected"] = int(len(full_specs))

    if mode == "both":
        h = out["honest"]["mean_r2"]
        l = out["leaky"]["mean_r2"]
        if math.isfinite(h) and math.isfinite(l):
            out["leakage_gap_r2"] = float(l - h)
    return out
