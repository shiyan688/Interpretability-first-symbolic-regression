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
    score_mask: np.ndarray | None = None,
) -> Candidate:
    if score_mask is None:
        score, corr = _corr_score(values, y)
    else:
        m = np.asarray(score_mask, dtype=bool)
        score, corr = _corr_score(np.asarray(values, dtype=float)[m], np.asarray(y, dtype=float)[m])
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
    train_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Mine candidate factors.

    If ``train_mask`` is given, ALL correlation scores (base ranking and beam
    search selection) are computed on train rows only, so test labels never
    influence which factors are mined or kept. The output factor values still
    span all rows (needed downstream for validation/test evaluation).
    """
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
    if train_mask is None:
        score_mask = np.ones(len(yy), dtype=bool)
    else:
        score_mask = np.asarray(train_mask, dtype=bool)
    y_score = yy[score_mask]

    def score_values(values: np.ndarray) -> tuple[float, float]:
        return _corr_score(np.asarray(values, dtype=float)[score_mask], y_score)

    raw_scores = []
    for col in X.columns:
        values = X[col].to_numpy(dtype=float)
        score, corr = score_values(values)
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
        cand = _make_candidate(col, values, 0, yy, "raw_base", next_index, score_mask)
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
                new = _make_candidate(expr, values, order, yy, f"unary:{op_name}", next_index, score_mask)
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
                    new = _make_candidate(expr, values, order, yy, "binary", next_index, score_mask)
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
        "scored_on": "train_rows_only" if train_mask is not None else "all_rows",
        "n_scoring_rows": int(score_mask.sum()),
    }
    return rows, values, manifest


@dataclass
class FeatureSpec:
    """A selected model input, expressed so it can be re-evaluated on any
    raw feature frame via expr.eval_expr.

    - ``source == "raw"``: ``expression`` is a bare raw column name.
    - ``source == "mined"``: ``expression`` is a mined factor expression over
      raw columns (uses only operators available in expr.namespace_from_frame).
    """

    name: str
    expression: str
    source: str
    abs_corr: float


def select_factor_expressions(
    X: pd.DataFrame,
    y: np.ndarray,
    mining_opts: dict[str, Any] | None = None,
    selection_opts: dict[str, Any] | None = None,
) -> list[FeatureSpec]:
    """Run mine + correlation-based selection purely in memory and return the
    chosen feature set as re-evaluable expressions.

    This is the leakage-safe core of the pipeline: every decision here uses only
    the (X, y) passed in, so callers can invoke it on a single CV training fold
    without touching held-out labels. It intentionally does NOT read/write files
    and does NOT apply LLM-in-the-loop selection (an LLM selection file is a
    full-data artifact and would reintroduce leakage inside a fold).
    """

    X = finite_frame(X)
    yy = np.asarray(y, dtype=float)
    sel = dict(selection_opts or {})

    # 1) mine factors on this (X, y) only.
    rows, _, _ = mine_factors_from_frame(X, yy, mining_opts)

    # 2) raw feature keep-set, mirroring build_pysr_pool raw_top_k / raw_top_fraction.
    raw_scores: list[tuple[float, str]] = []
    for col in X.columns:
        score, _ = _corr_score(X[col].to_numpy(dtype=float), yy)
        raw_scores.append((score, str(col)))
    raw_scores.sort(reverse=True, key=lambda item: item[0])
    raw_top_k = sel.get("raw_top_k")
    raw_top_fraction = sel.get("raw_top_fraction")
    if raw_top_k is None and raw_top_fraction is not None:
        raw_top_k = max(1, int(math.ceil(float(raw_top_fraction) * X.shape[1])))
    if raw_top_k is None:
        raw_keep = [(score, col) for score, col in raw_scores]
    else:
        raw_keep = raw_scores[: int(raw_top_k)]

    specs: list[FeatureSpec] = [
        FeatureSpec(name=col, expression=col, source="raw", abs_corr=float(score))
        for score, col in raw_keep
    ]

    # 3) mined factor keep-set, mirroring build_pysr_pool factor_top_k.
    factor_top_k = int(sel.get("factor_top_k", 200))
    if not rows.empty:
        ranked = rows.sort_values("score_abs_corr", ascending=False).head(factor_top_k)
        for _, row in ranked.iterrows():
            expr = str(row.get("expression", "")).strip()
            if not expr:
                continue
            specs.append(
                FeatureSpec(
                    name=f"mine_{row.get('factor_name', expr)}",
                    expression=expr,
                    source="mined",
                    abs_corr=float(row.get("score_abs_corr", 0.0)),
                )
            )
    return specs


def build_design_matrix(X_raw: pd.DataFrame, specs: list[FeatureSpec]) -> pd.DataFrame:
    """Evaluate each FeatureSpec expression on a raw feature frame.

    Used to materialise the same selected feature set on a held-out fold. Columns
    that fail to evaluate or are non-finite are filled with zeros (finite_frame),
    matching the rest of the pipeline's numeric hygiene.
    """

    cols: dict[str, np.ndarray] = {}
    n = len(X_raw)
    for i, spec in enumerate(specs):
        try:
            values = eval_expr(spec.expression, X_raw)
        except Exception:
            values = np.zeros(n, dtype=float)
        if np.asarray(values).shape != (n,):
            values = np.full(n, float(np.asarray(values).ravel()[0]) if np.asarray(values).size else 0.0)
        name = spec.name if spec.name not in cols else f"{spec.name}__{i}"
        cols[name] = np.asarray(values, dtype=float)
    return finite_frame(pd.DataFrame(cols, index=X_raw.index))


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
    train_mask: np.ndarray | None = None,
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
        if train_mask is None:
            score, corr = _corr_score(values, y)
        else:
            m = np.asarray(train_mask, dtype=bool)
            score, corr = _corr_score(np.asarray(values, dtype=float)[m], np.asarray(y, dtype=float)[m])
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


def _train_mask_from_feature_dir(feature_dir: Path, n_rows: int) -> np.ndarray | None:
    """Read row_roles.csv (written by the no-leakage builder) into a train mask."""
    roles_path = feature_dir / "row_roles.csv"
    if not roles_path.exists():
        return None
    roles = safe_read_csv(roles_path)
    col = "role" if "role" in roles.columns else roles.columns[0]
    values = roles[col].astype(str).tolist()
    if len(values) != n_rows:
        raise ValueError(f"row_roles.csv length {len(values)} != feature rows {n_rows}")
    return np.array([v == "train" for v in values], dtype=bool)


def mine_factors(
    cfg: WorkflowConfig,
    target: str,
    feature_dir: Path | None = None,
    llm_proposals_path: Path | None = None,
) -> dict[str, Any]:
    feature_dir = feature_dir or (cfg.output_root / "feature_tables" / target)
    X, y = read_feature_dir(feature_dir)
    train_mask = _train_mask_from_feature_dir(feature_dir, len(X))
    proposed_X, proposal_report = proposed_factor_frame(X, y, llm_proposals_path, train_mask)
    if not proposed_X.empty:
        X = pd.concat([X.reset_index(drop=True), proposed_X.reset_index(drop=True)], axis=1)
    factor_cfg = dict(cfg.data.get("factor_mining") or {})
    rows, values, manifest = mine_factors_from_frame(X, y, factor_cfg, train_mask=train_mask)
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


def _read_llm_factor_selection(path: Path) -> tuple[set[str], dict[str, bool]]:
    """Return (selected identifiers, final_formula_allowed flags by identifier)."""
    if not path.exists():
        return set(), {}
    selected: set[str] = set()
    final_allowed: dict[str, bool] = {}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            items = data.get("selected_factors", data.get("factors", []))
        else:
            items = data
        for item in items:
            if isinstance(item, str):
                selected.add(item)
            elif isinstance(item, dict):
                keys = [item[k] for k in ("factor_name", "name", "expression") if item.get(k)]
                for key in keys:
                    selected.add(str(key))
                    if "final_formula_allowed" in item:
                        final_allowed[str(key)] = bool(item["final_formula_allowed"])
        return selected, final_allowed
    df = safe_read_csv(path)
    for col in ("factor_name", "name", "expression"):
        if col in df.columns:
            for _, row in df.iterrows():
                val = row.get(col)
                if pd.isna(val):
                    continue
                selected.add(str(val))
                if "final_formula_allowed" in df.columns and not pd.isna(row.get("final_formula_allowed")):
                    final_allowed[str(val)] = bool(row.get("final_formula_allowed"))
    return selected, final_allowed


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
    train_mask = _train_mask_from_feature_dir(raw_feature_dir, len(X_raw))
    score_mask = np.ones(len(y), dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    factors = safe_read_csv(factor_pool_dir / "mined_factors.csv")
    factor_values = finite_frame(safe_read_csv(factor_pool_dir / "mined_factor_values.csv"))
    select_cfg = dict(cfg.data.get("factor_selection") or {})
    raw_top_k = select_cfg.get("raw_top_k")
    raw_top_fraction = select_cfg.get("raw_top_fraction")
    factor_top_k = int(select_cfg.get("factor_top_k", 200))
    # When True, an LLM selection DRIVES the candidate pool instead of being
    # unioned into the full corr top-k (which would make selection a no-op).
    llm_authoritative = bool(select_cfg.get("llm_authoritative", True))
    # Small corr fallback kept alongside the LLM picks (search aids).
    llm_fallback_top_k = int(select_cfg.get("llm_fallback_top_k", 0))

    raw_scores = []
    for col in X_raw.columns:
        score, _ = _corr_score(X_raw[col].to_numpy(dtype=float)[score_mask], y[score_mask])
        raw_scores.append((score, col))
    raw_scores.sort(reverse=True)
    if raw_top_k is None and raw_top_fraction is not None:
        raw_top_k = max(1, int(math.ceil(float(raw_top_fraction) * X_raw.shape[1])))
    if raw_top_k is None:
        raw_keep = list(X_raw.columns)
    else:
        raw_keep = [col for _, col in raw_scores[: int(raw_top_k)]]

    factor_ranked = factors.sort_values("score_abs_corr", ascending=False)
    final_formula_allowed: dict[str, bool] = {}
    llm_selected: set[str] = set()
    llm_final_flags: dict[str, bool] = {}
    if llm_selection_path:
        llm_selected, llm_final_flags = _read_llm_factor_selection(llm_selection_path)

    if llm_selection_path and llm_selected and llm_authoritative:
        # Candidate pool is driven by the LLM selection. Map ids/expressions to
        # factor_name and honor final_formula_allowed.
        matched_names: list[str] = []
        for _, row in factors.iterrows():
            fname = str(row["factor_name"])
            fexpr = str(row["expression"])
            hit_key = None
            if fname in llm_selected:
                hit_key = fname
            elif fexpr in llm_selected:
                hit_key = fexpr
            if hit_key is not None:
                matched_names.append(fname)
                final_formula_allowed[fname] = bool(llm_final_flags.get(hit_key, False))
        selected_names = set(matched_names)
        selected_source = {name: "llm_selected" for name in selected_names}
        # optional small corr fallback to aid PySR search
        for name in factor_ranked.head(llm_fallback_top_k)["factor_name"].astype(str):
            if name not in selected_names:
                selected_names.add(name)
                selected_source[name] = "corr_fallback"
                final_formula_allowed.setdefault(name, False)
    else:
        # Legacy / no-LLM path: corr top-k pool.
        selected_names = set(factor_ranked.head(factor_top_k)["factor_name"].astype(str))
        selected_source = {name: "factor_ranked_corr" for name in selected_names}
        for name in selected_names:
            final_formula_allowed.setdefault(name, False)
        if llm_selection_path and llm_selected and not llm_authoritative:
            llm_keep_extra_top_k = int(select_cfg.get("llm_keep_extra_top_k", factor_top_k))
            by_name = factors[factors["factor_name"].astype(str).isin(llm_selected)]["factor_name"].astype(str)
            by_expr = factors[factors["expression"].astype(str).isin(llm_selected)]["factor_name"].astype(str)
            matched = list(dict.fromkeys([*by_name.tolist(), *by_expr.tolist()]))
            for name in matched[:llm_keep_extra_top_k]:
                selected_names.add(name)
                selected_source[name] = "llm_selected"
                final_formula_allowed[name] = bool(llm_final_flags.get(name, False))

    factor_keep = [name for name in factor_ranked["factor_name"].astype(str).tolist() if name in selected_names]
    X_factor = factor_values[factor_keep].copy() if factor_keep else pd.DataFrame(index=X_raw.index)
    X_factor = X_factor.rename(columns={name: f"mine_{name}" for name in X_factor.columns})
    X = pd.concat([X_raw[raw_keep].reset_index(drop=True), X_factor.reset_index(drop=True)], axis=1)
    X = finite_frame(X)

    Xtr = X.to_numpy(dtype=float)[score_mask]
    ytr = y[score_mask]
    lin = LinearRegression().fit(Xtr, ytr)
    pred = lin.predict(Xtr)
    linear_r2 = float(r2_score(ytr, pred))
    linear_rmse = float(math.sqrt(mean_squared_error(ytr, pred)))

    out_dir = cfg.output_root / "feature_tables" / f"{target}__{output_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_dir / "features.csv", index=False)
    X.to_csv(out_dir / "hybrid_features.csv", index=False)
    pd.DataFrame({"target": y}).to_csv(out_dir / "y.csv", index=False)
    if train_mask is not None:
        roles = np.array(["unassigned"] * len(X), dtype=object)
        roles[score_mask] = "train"
        # copy full role vector from raw feature dir if present
        raw_roles_path = raw_feature_dir / "row_roles.csv"
        if raw_roles_path.exists():
            (out_dir / "row_roles.csv").write_text(raw_roles_path.read_text(encoding="utf-8"), encoding="utf-8")

    selected_factor_rows = factors[factors["factor_name"].astype(str).isin(factor_keep)].copy()
    selected_factor_rows["pysr_column"] = selected_factor_rows["factor_name"].map(lambda x: f"mine_{x}")
    selected_factor_rows["selection_source"] = selected_factor_rows["factor_name"].astype(str).map(selected_source).fillna("")
    selected_factor_rows["final_formula_allowed"] = selected_factor_rows["factor_name"].astype(str).map(
        lambda x: bool(final_formula_allowed.get(x, False))
    )
    selected_factor_rows.to_csv(out_dir / "selected_mined_factors.csv", index=False)
    manifest = {
        "target": target,
        "builder": "raw_plus_mined_factor_pool",
        "raw_feature_dir": str(raw_feature_dir),
        "factor_pool_dir": str(factor_pool_dir),
        "llm_selection_path": str(llm_selection_path) if llm_selection_path else None,
        "llm_authoritative": llm_authoritative,
        "n_llm_selected": int(len(llm_selected)),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_raw_features": int(len(raw_keep)),
        "n_mined_factors": int(len(factor_keep)),
        "n_final_formula_allowed": int(sum(1 for v in final_formula_allowed.values() if v)),
        "train_linear_r2" if train_mask is not None else "linear_r2": linear_r2,
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
