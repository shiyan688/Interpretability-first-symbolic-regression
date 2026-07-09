from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from .config import WorkflowConfig
from .expr import eval_expr
from .features import finite_frame, safe_read_csv
from .reports import read_feature_dir

EPS = 1.0e-12


@dataclass
class Candidate:
    name: str
    expression: str
    values: np.ndarray
    order: int
    score_abs_corr: float
    signed_corr: float
    mean: float
    std: float
    source: str


def _corr_score(values: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(yy)
    if int(ok.sum()) < 3:
        return 0.0, 0.0
    if float(np.std(x[ok])) <= EPS:
        return 0.0, 0.0
    corr = np.corrcoef(x[ok], yy[ok])[0, 1]
    corr = float(corr) if np.isfinite(corr) else 0.0
    return abs(corr), corr


def _zscore(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if not math.isfinite(std) or std <= EPS:
        std = 1.0
    return (arr - mean) / std, mean, std


def _signature(values: np.ndarray) -> str:
    z, _, _ = _zscore(values)
    rounded = np.round(z, 8)
    return hashlib.blake2b(rounded.tobytes(), digest_size=12).hexdigest()


def _valid_values(values: np.ndarray, max_abs_value: float) -> bool:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        return False
    if not np.isfinite(arr).all():
        return False
    if float(np.std(arr)) <= EPS:
        return False
    if float(np.max(np.abs(arr))) > max_abs_value:
        return False
    return True


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return a / np.where(np.abs(b) > EPS, b, np.nan)


def _unary_ops(enabled: list[str]) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    ops: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    if "abs" in enabled:
        ops["abs"] = np.abs
    if "square" in enabled:
        ops["square"] = lambda x: x * x
    if "cube" in enabled:
        ops["cube"] = lambda x: x * x * x
    if "inv" in enabled:
        ops["inv"] = lambda x: _safe_div(np.ones_like(x), x)
    if "sqrt_abs" in enabled:
        ops["sqrt_abs"] = lambda x: np.sqrt(np.abs(x))
    if "log_abs" in enabled:
        ops["log_abs"] = lambda x: np.log(np.abs(x) + EPS)
    return ops


def _binary_candidates(
    a: Candidate,
    b: Candidate,
    enabled: list[str],
) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    av = a.values
    bv = b.values
    ae = a.expression
    be = b.expression
    if "+" in enabled:
        out.append((f"({ae} + {be})", av + bv))
    if "-" in enabled:
        out.append((f"({ae} - {be})", av - bv))
        out.append((f"({be} - {ae})", bv - av))
    if "*" in enabled:
        out.append((f"({ae} * {be})", av * bv))
    if "/" in enabled:
        out.append((f"({ae} / {be})", _safe_div(av, bv)))
        out.append((f"({be} / {ae})", _safe_div(bv, av)))
    return out


def _make_candidate(
    expression: str,
    values: np.ndarray,
    order: int,
    y: np.ndarray,
    source: str,
    index: int,
) -> Candidate:
    score, corr = _corr_score(values, y)
    _, mean, std = _zscore(values)
    return Candidate(
        name=f"factor_{index:06d}",
        expression=expression,
        values=np.asarray(values, dtype=float),
        order=order,
        score_abs_corr=score,
        signed_corr=corr,
        mean=mean,
        std=std,
        source=source,
    )


def _candidate_row(c: Candidate) -> dict[str, Any]:
    return {
        "factor_name": c.name,
        "expression": c.expression,
        "order": int(c.order),
        "score_abs_corr": float(c.score_abs_corr),
        "signed_corr": float(c.signed_corr),
        "mean": float(c.mean),
        "std": float(c.std),
        "source": c.source,
    }


def mine_factors_from_frame(
    X: pd.DataFrame,
    y: np.ndarray,
    options: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    opts = dict(options or {})
    base_top_k = int(opts.get("base_top_k", min(40, X.shape[1])))
    pair_top_k = int(opts.get("pair_top_k", min(60, base_top_k)))
    beam_width = int(opts.get("beam_width", 300))
    final_top_k = int(opts.get("final_top_k", 1000))
    max_order = int(opts.get("max_order", 2))
    max_abs_value = float(opts.get("max_abs_value", 1.0e12))
    unary_enabled = list(opts.get("unary_ops", ["abs", "square", "inv", "sqrt_abs", "log_abs"]))
    binary_enabled = list(opts.get("binary_ops", ["+", "-", "*", "/"]))

    X = finite_frame(X)
    yy = np.asarray(y, dtype=float)
    raw_scores = []
    for col in X.columns:
        values = X[col].to_numpy(dtype=float)
        score, corr = _corr_score(values, yy)
        if score > 0.0:
            raw_scores.append((score, corr, str(col), values))
    raw_scores.sort(reverse=True, key=lambda item: item[0])
    raw_scores = raw_scores[: max(1, min(base_top_k, len(raw_scores)))]

    all_candidates: list[Candidate] = []
    beam: list[Candidate] = []
    seen_expr: set[str] = set()
    seen_sig: set[str] = set()
    next_index = 1
    for score, corr, col, values in raw_scores:
        if not _valid_values(values, max_abs_value):
            continue
        sig = _signature(values)
        if sig in seen_sig:
            continue
        cand = _make_candidate(col, values, 0, yy, "raw_base", next_index)
        next_index += 1
        cand.score_abs_corr = score
        cand.signed_corr = corr
        all_candidates.append(cand)
        beam.append(cand)
        seen_expr.add(col)
        seen_sig.add(sig)

    unary_ops = _unary_ops(unary_enabled)
    per_order_counts: dict[str, int] = {"0": len(beam)}
    for order in range(1, max_order + 1):
        seed = sorted(all_candidates, key=lambda c: c.score_abs_corr, reverse=True)[:pair_top_k]
        generated: list[Candidate] = []

        for cand in seed:
            for op_name, op_func in unary_ops.items():
                expr = f"{op_name}({cand.expression})"
                if expr in seen_expr:
                    continue
                with np.errstate(all="ignore"):
                    values = op_func(cand.values)
                if not _valid_values(values, max_abs_value):
                    continue
                sig = _signature(values)
                if sig in seen_sig:
                    continue
                new = _make_candidate(expr, values, order, yy, f"unary:{op_name}", next_index)
                next_index += 1
                generated.append(new)
                seen_expr.add(expr)
                seen_sig.add(sig)

        for i, left in enumerate(seed):
            for right in seed[i + 1 :]:
                for expr, values in _binary_candidates(left, right, binary_enabled):
                    if expr in seen_expr:
                        continue
                    if not _valid_values(values, max_abs_value):
                        continue
                    sig = _signature(values)
                    if sig in seen_sig:
                        continue
                    new = _make_candidate(expr, values, order, yy, "binary", next_index)
                    next_index += 1
                    generated.append(new)
                    seen_expr.add(expr)
                    seen_sig.add(sig)

        generated.sort(key=lambda c: c.score_abs_corr, reverse=True)
        keep = generated[:beam_width]
        per_order_counts[str(order)] = len(keep)
        all_candidates.extend(keep)
        beam = keep
        if not beam:
            break

    all_candidates.sort(key=lambda c: c.score_abs_corr, reverse=True)
    final = all_candidates[:final_top_k]
    rows = pd.DataFrame([_candidate_row(c) for c in final])
    values = pd.DataFrame({c.name: _zscore(c.values)[0] for c in final})
    manifest = {
        "n_raw_features": int(X.shape[1]),
        "n_base_features": int(len(raw_scores)),
        "n_candidates_kept": int(len(final)),
        "options": opts,
        "per_order_counts": per_order_counts,
        "value_policy": "mined factor values are z-scored before output",
    }
    return rows, values, manifest


def _safe_proposed_name(name: str, index: int) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")
    if not text:
        text = f"proposed_{index:03d}"
    if text[0].isdigit():
        text = f"p_{text}"
    return f"llmprop_{text[:80]}"


def _read_llm_proposals(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.get("proposed_factors", data.get("factors", []))
    else:
        items = data
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            out.append({"name": "", "expression": item})
        elif isinstance(item, dict) and item.get("expression"):
            out.append(item)
    return out


def proposed_factor_frame(
    X: pd.DataFrame,
    y: np.ndarray,
    proposals_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not proposals_path:
        return pd.DataFrame(index=X.index), pd.DataFrame()
    proposals_path = proposals_path.expanduser()
    if not proposals_path.exists():
        raise FileNotFoundError(f"LLM proposal file not found: {proposals_path}")
    rows = []
    cols: dict[str, np.ndarray] = {}
    used: set[str] = set()
    for i, item in enumerate(_read_llm_proposals(proposals_path), 1):
        expr = str(item.get("expression", "")).strip()
        if not expr:
            continue
        name = _safe_proposed_name(item.get("name") or f"proposed_{i:03d}", i)
        base = name
        j = 2
        while name in used or name in X.columns:
            name = f"{base}_{j}"
            j += 1
        try:
            values = eval_expr(expr, X)
        except Exception as exc:
            rows.append(
                {
                    "proposed_name": name,
                    "expression": expr,
                    "status": "error",
                    "error": repr(exc),
                    "meaning": item.get("meaning", ""),
                    "dimension_note": item.get("dimension_note", ""),
                    "priority": item.get("priority", ""),
                }
            )
            continue
        if not _valid_values(values, 1.0e12):
            rows.append(
                {
                    "proposed_name": name,
                    "expression": expr,
                    "status": "dropped",
                    "error": "nonfinite_or_constant",
                    "meaning": item.get("meaning", ""),
                    "dimension_note": item.get("dimension_note", ""),
                    "priority": item.get("priority", ""),
                }
            )
            continue
        z, _, _ = _zscore(values)
        score, corr = _corr_score(values, y)
        cols[name] = z
        used.add(name)
        rows.append(
            {
                "proposed_name": name,
                "expression": expr,
                "status": "kept",
                "score_abs_corr": score,
                "signed_corr": corr,
                "meaning": item.get("meaning", ""),
                "dimension_note": item.get("dimension_note", ""),
                "priority": item.get("priority", ""),
            }
        )
    return pd.DataFrame(cols), pd.DataFrame(rows)


def mine_factors(
    cfg: WorkflowConfig,
    target: str,
    feature_dir: Path | None = None,
    llm_proposals_path: Path | None = None,
) -> dict[str, Any]:
    feature_dir = feature_dir or (cfg.output_root / "feature_tables" / target)
    X, y = read_feature_dir(feature_dir)
    proposed_X, proposal_report = proposed_factor_frame(X, y, llm_proposals_path)
    if not proposed_X.empty:
        X = pd.concat([X.reset_index(drop=True), proposed_X.reset_index(drop=True)], axis=1)
    factor_cfg = dict(cfg.data.get("factor_mining") or {})
    rows, values, manifest = mine_factors_from_frame(X, y, factor_cfg)
    out_dir = cfg.output_root / "factor_pools" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "mined_factors.csv", index=False)
    values.to_csv(out_dir / "mined_factor_values.csv", index=False)
    if not proposal_report.empty:
        proposal_report.to_csv(out_dir / "llm_proposed_factors_report.csv", index=False)
    manifest.update({"target": target, "feature_dir": str(feature_dir), "factor_pool_dir": str(out_dir)})
    if llm_proposals_path:
        manifest["llm_proposals_path"] = str(llm_proposals_path)
        manifest["n_llm_proposed_kept"] = int(proposed_X.shape[1])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "target": target,
        "factor_pool_dir": str(out_dir),
        "n_factors": int(len(rows)),
        "best_abs_corr": float(rows["score_abs_corr"].max()) if not rows.empty else 0.0,
    }


def _read_llm_factor_selection(path: Path) -> set[str]:
    if not path.exists():
        return set()
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            items = data.get("selected_factors", data.get("factors", []))
        else:
            items = data
        selected: set[str] = set()
        for item in items:
            if isinstance(item, str):
                selected.add(item)
            elif isinstance(item, dict):
                for key in ("factor_name", "name", "expression"):
                    if item.get(key):
                        selected.add(str(item[key]))
        return selected
    df = safe_read_csv(path)
    selected = set()
    for col in ("factor_name", "name", "expression"):
        if col in df.columns:
            selected.update(str(x) for x in df[col].dropna().tolist())
    return selected


def build_pysr_pool(
    cfg: WorkflowConfig,
    target: str,
    raw_feature_dir: Path | None = None,
    factor_pool_dir: Path | None = None,
    llm_selection_path: Path | None = None,
    output_tag: str = "pysr_pool",
) -> dict[str, Any]:
    raw_feature_dir = raw_feature_dir or (cfg.output_root / "feature_tables" / target)
    factor_pool_dir = factor_pool_dir or (cfg.output_root / "factor_pools" / target)
    X_raw, y = read_feature_dir(raw_feature_dir)
    factors = safe_read_csv(factor_pool_dir / "mined_factors.csv")
    factor_values = finite_frame(safe_read_csv(factor_pool_dir / "mined_factor_values.csv"))
    select_cfg = dict(cfg.data.get("factor_selection") or {})
    raw_top_k = select_cfg.get("raw_top_k")
    raw_top_fraction = select_cfg.get("raw_top_fraction")
    factor_top_k = int(select_cfg.get("factor_top_k", 200))
    llm_keep_extra_top_k = int(select_cfg.get("llm_keep_extra_top_k", factor_top_k))

    raw_scores = []
    for col in X_raw.columns:
        score, _ = _corr_score(X_raw[col].to_numpy(dtype=float), y)
        raw_scores.append((score, col))
    raw_scores.sort(reverse=True)
    if raw_top_k is None and raw_top_fraction is not None:
        raw_top_k = max(1, int(math.ceil(float(raw_top_fraction) * X_raw.shape[1])))
    if raw_top_k is None:
        raw_keep = list(X_raw.columns)
    else:
        raw_keep = [col for _, col in raw_scores[: int(raw_top_k)]]

    factor_ranked = factors.sort_values("score_abs_corr", ascending=False)
    selected_names = set(factor_ranked.head(factor_top_k)["factor_name"].astype(str))
    selected_source = {name: "factor_ranked_corr" for name in selected_names}
    if llm_selection_path:
        llm_selected = _read_llm_factor_selection(llm_selection_path)
        if llm_selected:
            by_name = factors[factors["factor_name"].astype(str).isin(llm_selected)]["factor_name"].astype(str)
            by_expr = factors[factors["expression"].astype(str).isin(llm_selected)]["factor_name"].astype(str)
            matched = list(dict.fromkeys([*by_name.tolist(), *by_expr.tolist()]))
            for name in matched[:llm_keep_extra_top_k]:
                selected_names.add(name)
                selected_source[name] = "llm_selected"

    factor_keep = [name for name in factor_ranked["factor_name"].astype(str).tolist() if name in selected_names]
    X_factor = factor_values[factor_keep].copy() if factor_keep else pd.DataFrame(index=X_raw.index)
    X_factor = X_factor.rename(columns={name: f"mine_{name}" for name in X_factor.columns})
    X = pd.concat([X_raw[raw_keep].reset_index(drop=True), X_factor.reset_index(drop=True)], axis=1)
    X = finite_frame(X)

    lin = LinearRegression().fit(X.to_numpy(dtype=float), y)
    pred = lin.predict(X.to_numpy(dtype=float))
    linear_r2 = float(r2_score(y, pred))
    linear_rmse = float(math.sqrt(mean_squared_error(y, pred)))

    out_dir = cfg.output_root / "feature_tables" / f"{target}__{output_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_dir / "features.csv", index=False)
    X.to_csv(out_dir / "hybrid_features.csv", index=False)
    pd.DataFrame({"target": y}).to_csv(out_dir / "y.csv", index=False)

    selected_factor_rows = factors[factors["factor_name"].astype(str).isin(factor_keep)].copy()
    selected_factor_rows["pysr_column"] = selected_factor_rows["factor_name"].map(lambda x: f"mine_{x}")
    selected_factor_rows["selection_source"] = selected_factor_rows["factor_name"].astype(str).map(selected_source).fillna("")
    selected_factor_rows.to_csv(out_dir / "selected_mined_factors.csv", index=False)
    manifest = {
        "target": target,
        "builder": "raw_plus_mined_factor_pool",
        "raw_feature_dir": str(raw_feature_dir),
        "factor_pool_dir": str(factor_pool_dir),
        "llm_selection_path": str(llm_selection_path) if llm_selection_path else None,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_raw_features": int(len(raw_keep)),
        "n_mined_factors": int(len(factor_keep)),
        "linear_r2": linear_r2,
        "linear_rmse": linear_rmse,
        "feature_source": {col: ("raw_dataset" if col in raw_keep else "mined_factor") for col in X.columns},
        "selection_config": select_cfg,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "target": target,
        "feature_dir": str(out_dir),
        "n_features": int(X.shape[1]),
        "linear_r2": linear_r2,
        "linear_rmse": linear_rmse,
    }
