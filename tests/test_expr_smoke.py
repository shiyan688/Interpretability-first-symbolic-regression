from __future__ import annotations

import numpy as np
import pandas as pd

from factor_pysr_llm.expr import eval_expr, metric_dict


def test_eval_expr_sisso_and_pysr_style() -> None:
    X = pd.DataFrame({"x": [1.0, -2.0, 3.0], "y": [2.0, 4.0, 8.0]})
    pred = eval_expr("|x| + inv(y) + square(x)", X)
    expected = np.abs(X["x"].to_numpy()) + 1.0 / X["y"].to_numpy() + X["x"].to_numpy() ** 2
    assert np.allclose(pred, expected)


def test_metric_dict() -> None:
    y = np.array([1.0, 2.0, 3.0])
    out = metric_dict(y, y)
    assert out["r2_verified"] == 1.0
    assert out["n_finite"] == 3

