from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .expr import eval_expr
from .ifsr_selector import Candidate, select_if_sr
from .lineage import FactorCard, build_card_index, check_numeric_consistency, complexity_stats

# End-to-end pilot harness.
#
# NOTE: A real confirmatory run uses PySR (and an external baseline) as the
# search engine. PySR/Julia are optional and not present in every environment,
# so this harness accepts an injectable ``search_fn`` and ships a deterministic
# linear-combination surrogate for CI/pilot smoke. Any run using the surrogate
# is marked ``development_only`` and must never enter a confirmatory main table.


def _fit_r2(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    if X.shape[1] == 0 or int(mask.sum()) < 3:
        return -math.inf
    lin = LinearRegression().fit(X[mask], y[mask])
    pred = lin.predict(X[mask])
    return float(r2_score(y[mask], pred))


def surrogate_search(
    feature_frame: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    columns: list[str],
    max_terms: int = 3,
) -> list[dict[str, Any]]:
    """Deterministic surrogate: build additive candidate expressions from the
    top train-correlated columns. Returns candidate records with r2_val.

    This is a development_only stand-in for a genuine SR search.
    """
    # rank columns by train correlation
    scored = []
    ytr = y[train_mask]
    for col in columns:
        x = feature_frame[col].to_numpy(dtype=float)[train_mask]
        if np.std(x) <= 1e-12:
            continue
        c = abs(float(np.corrcoef(x, ytr)[0, 1])) if np.std(ytr) > 0 else 0.0
        scored.append((c, col))
    scored.sort(reverse=True)
    ranked = [col for _, col in scored]

    candidates: list[dict[str, Any]] = []
    yv = y

    def eval_candidate(cid: str, expr: str, cols: list[str], design: np.ndarray) -> dict[str, Any]:
        lin = LinearRegression().fit(design[train_mask], y[train_mask])
        pred = lin.predict(design)
        r2v = float(r2_score(yv[val_mask], pred[val_mask])) if int(val_mask.sum()) >= 3 else -math.inf
        return {
            "candidate_id": cid,
            "expression": expr,
            "columns": cols,
            "r2_val": r2v,
            "coef": [float(c) for c in lin.coef_],
            "intercept": float(lin.intercept_),
        }

    for k in range(1, min(max_terms, len(ranked)) + 1):
        cols = ranked[:k]
        expr = " + ".join(cols)
        X = feature_frame[cols].to_numpy(dtype=float)
        candidates.append(eval_candidate(f"surrogate_add_{k}", expr, cols, X))

    # Also emit a higher-accuracy but higher-complexity multiplicative candidate
    # from the top-2 columns. This creates a genuine accuracy/complexity
    # tradeoff so accuracy-first and interpretability-first selection diverge.
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        prod = (feature_frame[a].to_numpy(dtype=float) * feature_frame[b].to_numpy(dtype=float)).reshape(-1, 1)
        candidates.append(eval_candidate(f"surrogate_prod_{a}_{b}", f"{a} * {b}", [a, b], prod))
    return candidates


def run_condition(
    name: str,
    feature_frame: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    columns: list[str],
    search_fn: Callable | None = None,
) -> dict[str, Any]:
    """Run one method condition, returning its candidate archive."""
    search = search_fn or surrogate_search
    candidates = search(feature_frame, y, train_mask, val_mask, columns)
    return {
        "condition": name,
        "columns_available": list(columns),
        "candidates": candidates,
        "engine": "surrogate_linear_additive (development_only)" if search_fn is None else "injected",
    }


def pilot_conditions(
    raw_columns: list[str],
    mined_columns: list[str],
    domain_columns: list[str],
    feature_frame: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    search_fn: Callable | None = None,
) -> dict[str, Any]:
    """Run the four internal conditions with different column availability so
    the ablation is genuinely non-trivial.

    - raw_pysr:            raw columns only
    - mine_pysr:           raw + mined columns
    - if_sr_no_domain:     raw + mined columns (interpretability selection, no domain)
    - if_sr:               raw + mined + approved domain columns
    """
    conditions = {
        "raw_pysr": list(raw_columns),
        "mine_pysr": list(raw_columns) + list(mined_columns),
        "if_sr_no_domain": list(raw_columns) + list(mined_columns),
        "if_sr": list(raw_columns) + list(mined_columns) + list(domain_columns),
    }
    archives = {}
    for name, cols in conditions.items():
        cols = [c for c in cols if c in feature_frame.columns]
        archives[name] = run_condition(name, feature_frame, y, train_mask, val_mask, cols, search_fn)
    return archives


def standard_selection(candidates: list[dict[str, Any]]) -> str | None:
    """Standard PySR-style selection: best validation R2, ties by fewest terms."""
    if not candidates:
        return None
    best = max(candidates, key=lambda c: (c["r2_val"], -len(c.get("columns", []))))
    return best["expression"]


def interpretability_first_selection(
    candidates: list[dict[str, Any]], delta: float = 0.02
) -> str | None:
    """IF-style: within validation tolerance band, prefer lowest expanded complexity."""
    if not candidates:
        return None
    best_r2 = max(c["r2_val"] for c in candidates)
    acceptable = [c for c in candidates if c["r2_val"] >= best_r2 - delta]

    def node_count(c: dict[str, Any]) -> int:
        try:
            return complexity_stats(c["expression"], {})["expanded_node_count"]
        except Exception:
            return 10**9

    chosen = min(acceptable, key=lambda c: (node_count(c), c["expression"]))
    return chosen["expression"]


def ablation_is_nontrivial(archives: dict[str, Any]) -> dict[str, Any]:
    """Check the four conditions differ by candidate pool OR by selection.

    raw_pysr / mine_pysr / if_sr differ by pool (column availability).
    mine_pysr vs if_sr_no_domain share a pool but differ by SELECTION rule.
    """
    pools = {name: {c["expression"] for c in arc["candidates"]} for name, arc in archives.items()}
    selections = {
        "raw_pysr": standard_selection(archives["raw_pysr"]["candidates"]),
        "mine_pysr": standard_selection(archives["mine_pysr"]["candidates"]),
        "if_sr_no_domain": interpretability_first_selection(archives["if_sr_no_domain"]["candidates"]),
        "if_sr": interpretability_first_selection(archives["if_sr"]["candidates"]),
    }
    distinct = {}
    names = list(pools)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pool_diff = pools[a] != pools[b]
            sel_diff = selections[a] != selections[b]
            distinct[f"{a}__vs__{b}"] = {
                "pool_differs": pool_diff,
                "selection_differs": sel_diff,
                "differs": pool_diff or sel_diff,
            }
    return {
        "pools": {k: sorted(v) for k, v in pools.items()},
        "selections": selections,
        "distinct_pairs": distinct,
        "all_pairs_differ": all(v["differs"] for v in distinct.values()),
    }
