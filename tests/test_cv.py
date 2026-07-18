from __future__ import annotations

import numpy as np
import pandas as pd

from factor_pysr_llm.cv import evaluate_cv
from factor_pysr_llm.expr import eval_expr


def test_mining_operators_round_trip() -> None:
    # sqrt_abs / log_abs are used inside mined factor expression strings; they
    # must be evaluable by eval_expr so factors can be applied to held-out folds.
    frame = pd.DataFrame({"raw_a": np.array([-4.0, -1.0, 0.0, 1.0, 4.0])})
    got_sqrt = eval_expr("sqrt_abs(raw_a)", frame)
    got_log = eval_expr("log_abs(raw_a)", frame)
    np.testing.assert_allclose(got_sqrt, np.sqrt(np.abs(frame["raw_a"].to_numpy())))
    assert np.isfinite(got_log).all()  # log_abs(0) is finite thanks to the +EPS guard
    # nested mined-style expression also round-trips
    got_nested = eval_expr("(square(raw_a) / sqrt_abs(raw_a))", frame)
    assert got_nested.shape == (5,)


def _make_frame(n: int, n_noise: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    cols = {f"raw_n{i}": rng.normal(size=n) for i in range(n_noise)}
    X = pd.DataFrame(cols)
    return X, rng.normal(size=n)


def test_leaky_cv_is_optimistic_on_pure_noise() -> None:
    # y is independent of every feature. Honest CV should report ~0 (or negative)
    # R2. Leaky CV -- which selects the top factors on the full data before
    # splitting -- should report a clearly higher R2 purely from winner's curse.
    n = 120
    X, y = _make_frame(n, n_noise=25, seed=7)
    mining = {
        "base_top_k": 20,
        "pair_top_k": 20,
        "beam_width": 200,
        "final_top_k": 400,
        "max_order": 2,
    }
    selection = {"raw_top_k": 8, "factor_top_k": 60}
    report = evaluate_cv(X, y, mining, selection, k=5, mode="both", seed=1)

    honest_r2 = report["honest"]["mean_r2"]
    leaky_r2 = report["leaky"]["mean_r2"]

    # The leak makes the noise-only dataset look predictive.
    assert leaky_r2 > honest_r2
    assert report["leakage_gap_r2"] > 0.05
    # Honest estimate correctly refuses to find signal that is not there.
    assert honest_r2 < 0.2


def test_honest_cv_recovers_real_signal() -> None:
    # A genuine (noisy) signal should survive honest, fold-internal selection.
    rng = np.random.default_rng(3)
    n = 200
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    noise = rng.normal(scale=0.1, size=n)
    y = 2.0 * a * b + noise
    X = pd.DataFrame({"raw_a": a, "raw_b": b, **{f"raw_j{i}": rng.normal(size=n) for i in range(6)}})
    mining = {
        "base_top_k": 8,
        "pair_top_k": 8,
        "beam_width": 100,
        "final_top_k": 100,
        "max_order": 1,
        "binary_ops": ["*", "+", "-"],
        "unary_ops": ["square"],
    }
    selection = {"raw_top_k": 8, "factor_top_k": 30}
    report = evaluate_cv(X, y, mining, selection, k=5, mode="honest", seed=2)
    assert report["honest"]["mean_r2"] > 0.8
