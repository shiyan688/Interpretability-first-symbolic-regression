from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .interpretability_eval import RUBRIC_DIMENSIONS

# Rating aggregation
# ------------------
# Human, LLM-A and LLM-B ratings are aggregated SEPARATELY. Human and LLM scores
# are never averaged into a single overall number. We report per-dimension and
# overall means, bootstrap 95% CIs, inter-rater agreement, and LLM-human
# Spearman / pairwise preference agreement.


def load_human_csv(path: Path) -> pd.DataFrame:
    """Load a human rating CSV (item_id, four dims, rater_id)."""
    df = pd.read_csv(path)
    required = {"item_id", *RUBRIC_DIMENSIONS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"human CSV missing columns: {sorted(missing)}")
    for dim in RUBRIC_DIMENSIONS:
        df[dim] = pd.to_numeric(df[dim], errors="coerce")
    df = df.dropna(subset=list(RUBRIC_DIMENSIONS))
    for dim in RUBRIC_DIMENSIONS:
        bad = df[(df[dim] < 1) | (df[dim] > 5)]
        if len(bad):
            raise ValueError(f"human ratings out of range in {dim}")
    if "rater_id" not in df.columns:
        df["rater_id"] = "human_1"
    return df


def load_llm_results(path: Path) -> pd.DataFrame:
    """Load LLM judge results (jsonl cache) into a flat frame."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        row = {"item_id": rec["item_id"], "rater_id": rec.get("model_id", "llm")}
        for dim in RUBRIC_DIMENSIONS:
            row[dim] = rec["ratings"][dim]
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 20260709) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = values.size
    for i in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        means[i] = float(np.mean(sample))
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return (lo, hi)


def summarize_group(df: pd.DataFrame, label: str, seed: int = 20260709) -> dict[str, Any]:
    """Per-dimension and overall means + bootstrap CI for one rater group."""
    out: dict[str, Any] = {"group": label, "n_ratings": int(len(df)), "dimensions": {}}
    per_item_overall = df[list(RUBRIC_DIMENSIONS)].mean(axis=1)
    for dim in RUBRIC_DIMENSIONS:
        vals = df[dim].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(vals, seed=seed)
        out["dimensions"][dim] = {
            "mean": float(np.mean(vals)) if vals.size else float("nan"),
            "ci95": [lo, hi],
        }
    overall = per_item_overall.to_numpy(dtype=float)
    lo, hi = _bootstrap_ci(overall, seed=seed)
    out["overall"] = {
        "mean": float(np.mean(overall)) if overall.size else float("nan"),
        "ci95": [lo, hi],
    }
    return out


def inter_rater_agreement(df: pd.DataFrame) -> dict[str, Any]:
    """Agreement across raters for a group with multiple raters.

    Uses per-item overall score; reports mean pairwise Pearson correlation and
    average absolute difference between raters.
    """
    if "rater_id" not in df.columns or df["rater_id"].nunique() < 2:
        return {"n_raters": int(df["rater_id"].nunique()) if "rater_id" in df else 1, "note": "single rater"}
    df = df.copy()
    df["_overall"] = df[list(RUBRIC_DIMENSIONS)].mean(axis=1)
    pivot = df.pivot_table(index="item_id", columns="rater_id", values="_overall", aggfunc="mean")
    raters = list(pivot.columns)
    corrs = []
    absdiffs = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            a = pivot[raters[i]].to_numpy(dtype=float)
            b = pivot[raters[j]].to_numpy(dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            if int(ok.sum()) >= 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                corrs.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
            if int(ok.sum()) >= 1:
                absdiffs.append(float(np.mean(np.abs(a[ok] - b[ok]))))
    return {
        "n_raters": len(raters),
        "mean_pairwise_pearson": float(np.mean(corrs)) if corrs else float("nan"),
        "mean_pairwise_abs_diff": float(np.mean(absdiffs)) if absdiffs else float("nan"),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 2:
        return float("nan")
    ar = pd.Series(a[ok]).rank().to_numpy()
    br = pd.Series(b[ok]).rank().to_numpy()
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def _item_overall(df: pd.DataFrame) -> pd.Series:
    tmp = df.copy()
    tmp["_overall"] = tmp[list(RUBRIC_DIMENSIONS)].mean(axis=1)
    return tmp.groupby("item_id")["_overall"].mean()


def llm_human_correlation(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> dict[str, Any]:
    """Spearman + pairwise preference agreement between human and LLM overall."""
    h = _item_overall(human_df)
    m = _item_overall(llm_df)
    common = sorted(set(h.index) & set(m.index))
    if len(common) < 2:
        return {"n_common_items": len(common), "spearman": float("nan"), "pairwise_agreement": float("nan")}
    hv = h.loc[common].to_numpy(dtype=float)
    mv = m.loc[common].to_numpy(dtype=float)
    spearman = _spearman(hv, mv)
    # pairwise preference agreement
    agree = 0
    total = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            dh = hv[i] - hv[j]
            dm = mv[i] - mv[j]
            if dh == 0 or dm == 0:
                continue
            total += 1
            if (dh > 0) == (dm > 0):
                agree += 1
    pairwise = float(agree / total) if total else float("nan")
    return {
        "n_common_items": len(common),
        "spearman": spearman,
        "pairwise_agreement": pairwise,
        "n_pairs": total,
    }


def aggregate_ratings(
    human_csv: Path | None,
    llm_result_paths: dict[str, Path],
    private_map_path: Path | None = None,
    out_path: Path | None = None,
    seed: int = 20260709,
) -> dict[str, Any]:
    """Full aggregation. llm_result_paths maps group label (e.g. 'llm_a') -> jsonl.

    Human and LLM groups are summarized separately and never combined into a
    single overall number. If a private map is provided, per-method/dataset
    unblinded summaries are produced.
    """
    report: dict[str, Any] = {"groups": {}, "correlations": {}}

    human_df = None
    if human_csv is not None:
        human_df = load_human_csv(Path(human_csv))
        report["groups"]["human"] = summarize_group(human_df, "human", seed=seed)
        report["groups"]["human"]["inter_rater"] = inter_rater_agreement(human_df)

    llm_frames: dict[str, pd.DataFrame] = {}
    for label, path in llm_result_paths.items():
        df = load_llm_results(Path(path))
        llm_frames[label] = df
        report["groups"][label] = summarize_group(df, label, seed=seed)

    if human_df is not None:
        for label, df in llm_frames.items():
            report["correlations"][f"human_vs_{label}"] = llm_human_correlation(human_df, df)

    # unblinded per-method summary
    if private_map_path is not None:
        mapping = json.loads(Path(private_map_path).read_text(encoding="utf-8")).get("mapping", {})
        report["unblinded"] = _unblind_summary(human_df, llm_frames, mapping)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _unblind_summary(
    human_df: pd.DataFrame | None,
    llm_frames: dict[str, pd.DataFrame],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    def by_method(df: pd.DataFrame) -> dict[str, Any]:
        tmp = df.copy()
        tmp["_overall"] = tmp[list(RUBRIC_DIMENSIONS)].mean(axis=1)
        tmp["method"] = tmp["item_id"].map(lambda x: (mapping.get(x, {}) or {}).get("method"))
        tmp["dataset"] = tmp["item_id"].map(lambda x: (mapping.get(x, {}) or {}).get("dataset"))
        out: dict[str, Any] = {}
        for method, grp in tmp.dropna(subset=["method"]).groupby("method"):
            out[str(method)] = {
                "n": int(len(grp)),
                "overall_mean": float(grp["_overall"].mean()),
            }
        return out

    result: dict[str, Any] = {}
    if human_df is not None:
        result["human_by_method"] = by_method(human_df)
    for label, df in llm_frames.items():
        result[f"{label}_by_method"] = by_method(df)
    return result
