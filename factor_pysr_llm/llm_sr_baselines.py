from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

# LLM-based symbolic regression baselines.
#
# These are faithful re-implementations of the two dominant LLM-SR paradigms so
# they can be compared against IF-SR/PySR on the SAME data, with the SAME
# no-leakage protocol (fit on train, select on validation, test read once) and
# the SAME evaluation (ExprSim + blind LLM interpretability judge).
#
#   * DirectLLM   : zero/few-shot -- ask the LLM for a closed-form expression
#                   given variable descriptions + train data summary.
#   * LLMSR       : Shojaee et al. (ICLR 2025). The LLM proposes an equation
#                   SKELETON as a python function body with free parameters
#                   params[0..k]; constants are fit on TRAIN by least squares;
#                   the candidate is scored on VALIDATION; a buffer of the best
#                   skeletons is kept and sampled back into the prompt
#                   (evolutionary in-context search).
#
# call_fn(prompt) -> raw model text. Injecting it keeps everything testable and
# provider-agnostic (DeepSeek here).

_ALLOWED_FUNCS = {
    "np", "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "arcsin", "arccos",
    "arctan", "tanh", "sinh", "cosh", "pi", "e", "power",
}


def _safe_namespace() -> dict[str, Any]:
    return {
        "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": lambda x: np.log(np.abs(x) + 1e-12),
        "sqrt": lambda x: np.sqrt(np.abs(x)), "abs": np.abs,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh,
        "power": np.power, "pi": np.pi, "e": np.e,
        "__builtins__": {},
    }


def _extract_code(text: str) -> str:
    """Pull a python function body / expression from an LLM response."""
    t = text.strip()
    m = re.search(r"```(?:python)?\s*(.+?)```", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


@dataclass
class SkeletonCandidate:
    skeleton: str            # expression in terms of variables and params[i]
    n_params: int
    params: list[float] = field(default_factory=list)
    r2_val: float = float("-inf")
    r2_train: float = float("-inf")
    error: str | None = None


def _make_eval(skeleton: str, var_names: list[str]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return f(X_cols_dict, params) evaluating the skeleton."""
    def f(cols: dict[str, np.ndarray], params: np.ndarray) -> np.ndarray:
        ns = _safe_namespace()
        ns.update(cols)
        ns["params"] = params
        with np.errstate(all="ignore"):
            return np.asarray(eval(skeleton, ns), dtype=float)  # noqa: S307 (restricted ns)
    return f


def fit_skeleton(
    skeleton: str,
    n_params: int,
    cols: dict[str, np.ndarray],
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    seed: int = 0,
) -> SkeletonCandidate:
    """Fit params on TRAIN via least squares; score on train and validation."""
    cand = SkeletonCandidate(skeleton=skeleton, n_params=n_params)
    f = _make_eval(skeleton, list(cols))
    tr_cols = {k: v[train_mask] for k, v in cols.items()}
    ytr = y[train_mask]

    def resid(p: np.ndarray) -> np.ndarray:
        try:
            pred = f(tr_cols, p)
        except Exception:
            return np.full(ytr.shape, 1e6)
        pred = np.asarray(pred, dtype=float)
        if pred.shape != ytr.shape:
            pred = np.full(ytr.shape, 1e6)
        r = pred - ytr
        return np.where(np.isfinite(r), r, 1e6)

    rng = np.random.default_rng(seed)
    p0 = rng.normal(0.0, 1.0, max(1, n_params)) if n_params > 0 else np.array([])
    try:
        if n_params > 0:
            sol = least_squares(resid, p0, max_nfev=2000, method="lm")
            cand.params = [float(x) for x in sol.x]
        else:
            cand.params = []
        # score
        def r2_on(mask):
            m = mask
            pred = f({k: v[m] for k, v in cols.items()}, np.asarray(cand.params))
            pred = np.asarray(pred, dtype=float)
            ok = np.isfinite(pred) & np.isfinite(y[m])
            if int(ok.sum()) < 3:
                return float("-inf")
            return float(r2_score(y[m][ok], pred[ok]))
        cand.r2_train = r2_on(train_mask)
        cand.r2_val = r2_on(val_mask)
    except Exception as exc:
        cand.error = repr(exc)
    return cand


def substitute_params(skeleton: str, params: list[float]) -> str:
    """Replace params[i] with fitted numeric values -> a concrete expression."""
    expr = skeleton
    # replace longest indices first to avoid params[10] vs params[1]
    for i in sorted(range(len(params)), reverse=True):
        expr = re.sub(rf"params\[\s*{i}\s*\]", f"({params[i]:.6g})", expr)
    return expr


def _data_summary(cols: dict[str, np.ndarray], y: np.ndarray, mask: np.ndarray, k: int = 5) -> str:
    lines = ["Variable ranges and sample rows (train only):"]
    for name, v in cols.items():
        vv = v[mask]
        lines.append(f"  {name}: min={vv.min():.4g} max={vv.max():.4g} mean={vv.mean():.4g}")
    yy = y[mask]
    lines.append(f"  target y: min={yy.min():.4g} max={yy.max():.4g} mean={yy.mean():.4g}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Direct LLM baseline
# ---------------------------------------------------------------------------
def direct_llm_expression(
    var_names: list[str],
    var_meanings: dict[str, str],
    cols: dict[str, np.ndarray],
    y: np.ndarray,
    train_mask: np.ndarray,
    call_fn: Callable[[str], str],
) -> str:
    meanings = "\n".join(f"  {n}: {var_meanings.get(n,'')}" for n in var_names)
    prompt = f"""You are a scientific symbolic regression assistant.

Given the variables and a summary of noisy training data, propose ONE closed-form
expression for y as a function of the variables. Use only these variables
{var_names} and operators + - * / ** and functions sin cos exp log sqrt abs.

Variable meanings:
{meanings}

{_data_summary(cols, y, train_mask)}

Return strict JSON: {{"expression": "<python expression in the variables>"}}.
No text outside JSON."""
    raw = call_fn(prompt)
    txt = _extract_code(raw)
    try:
        data = json.loads(txt)
        return str(data["expression"])
    except Exception:
        # fall back: try to find an expression-looking line
        m = re.search(r'"expression"\s*:\s*"([^"]+)"', raw)
        if m:
            return m.group(1)
        raise ValueError(f"could not parse direct-LLM response: {raw[:200]}")


# ---------------------------------------------------------------------------
# LLM-SR (skeleton + fit + evolutionary buffer)
# ---------------------------------------------------------------------------
LLMSR_SYSTEM = """You perform scientific equation discovery. You propose an equation
SKELETON as a single Python expression for y in terms of the given variables and a
parameter vector `params` (a numpy array). Unknown numeric constants MUST be written
as params[0], params[1], ... (they will be fit to data, do not guess their values).
Use only + - * / ** and functions: sin, cos, tan, exp, log, sqrt, abs, arcsin, tanh.
Return strict JSON: {"skeleton": "<expr using variables and params[i]>", "n_params": <int>}."""


def _llmsr_prompt(
    var_names: list[str],
    var_meanings: dict[str, str],
    cols: dict[str, np.ndarray],
    y: np.ndarray,
    train_mask: np.ndarray,
    buffer: list[SkeletonCandidate],
) -> str:
    meanings = "\n".join(f"  {n}: {var_meanings.get(n,'')}" for n in var_names)
    examples = ""
    if buffer:
        top = sorted(buffer, key=lambda c: c.r2_val, reverse=True)[:3]
        examples = "\nBest skeletons so far (higher val R^2 is better). Propose a DIFFERENT, improved skeleton:\n"
        for c in top:
            examples += f"  skeleton: {c.skeleton}  (n_params={c.n_params}, val_R2={c.r2_val:.4f})\n"
    return f"""{LLMSR_SYSTEM}

Variables: {var_names}
Variable meanings:
{meanings}

{_data_summary(cols, y, train_mask)}
{examples}
Return only JSON."""


def _parse_skeleton(raw: str) -> tuple[str, int]:
    txt = _extract_code(raw)
    data = json.loads(txt)
    skeleton = str(data["skeleton"])
    n_params = int(data.get("n_params", len(re.findall(r"params\[", skeleton))))
    # basic safety: reject imports / dunder
    if "__" in skeleton or "import" in skeleton or ";" in skeleton:
        raise ValueError("unsafe skeleton")
    return skeleton, n_params


def run_llmsr(
    var_names: list[str],
    var_meanings: dict[str, str],
    cols: dict[str, np.ndarray],
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    call_fn: Callable[[str], str],
    n_iterations: int = 12,
    seed: int = 0,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Iterative LLM-SR search. Returns best-by-validation concrete expression."""
    buffer: list[SkeletonCandidate] = []
    history: list[dict[str, Any]] = []
    for it in range(n_iterations):
        prompt = _llmsr_prompt(var_names, var_meanings, cols, y, train_mask, buffer)
        try:
            raw = call_fn(prompt)
            skeleton, n_params = _parse_skeleton(raw)
        except Exception as exc:
            history.append({"iter": it, "status": "parse_error", "error": repr(exc)[:200]})
            continue
        cand = fit_skeleton(skeleton, n_params, cols, y, train_mask, val_mask, seed=seed + it)
        buffer.append(cand)
        history.append({
            "iter": it, "status": "ok" if cand.error is None else "fit_error",
            "skeleton": skeleton, "n_params": n_params,
            "r2_val": cand.r2_val, "r2_train": cand.r2_train, "error": cand.error,
        })
        if log_fn:
            log_fn(f"    llmsr iter {it}: val_R2={cand.r2_val:.4f} skeleton={skeleton[:60]}")
    valid = [c for c in buffer if np.isfinite(c.r2_val)]
    if not valid:
        return {"status": "no_valid_candidate", "history": history}
    best = max(valid, key=lambda c: c.r2_val)
    concrete = substitute_params(best.skeleton, best.params)
    return {
        "status": "success",
        "best_skeleton": best.skeleton,
        "best_expression": concrete,
        "n_params": best.n_params,
        "r2_val": best.r2_val,
        "r2_train": best.r2_train,
        "n_candidates": len(buffer),
        "history": history,
    }
