from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_pysr_llm.lineage import (
    FactorCard,
    build_card_index,
    canonicalize,
    cards_from_mined_factors,
    check_numeric_consistency,
    complexity_stats,
    expand_expression,
    inverse_standardize,
    substitute_standardized_variables,
)


def test_expand_nested_factors():
    cards = [
        FactorCard("f1", "ratio", "(a / b)", ["a", "b"]),
        FactorCard("f2", "scaled", "(ratio * c)", ["ratio", "c"]),
    ]
    idx = build_card_index(cards)
    expanded = expand_expression("scaled + 1", idx)
    stats = complexity_stats("scaled + 1", idx)
    assert set(stats["variables"]) == {"a", "b", "c"}
    # short alias must not hide complexity: node count of expanded > alias
    assert stats["expanded_node_count"] > 3


def test_cycle_detection():
    cards = [FactorCard("x", "x", "(y + 1)"), FactorCard("y", "y", "(x + 1)")]
    idx = build_card_index(cards)
    with pytest.raises(ValueError):
        expand_expression("x", idx)


def test_numeric_consistency_after_expansion():
    cards = [FactorCard("f1", "prod", "(a * b)", ["a", "b"])]
    idx = build_card_index(cards)
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "prod": [4.0, 10.0, 18.0]})
    result = check_numeric_consistency("prod + a", idx, frame)
    assert result["consistent"] is True
    assert result["max_abs_diff"] < 1e-9


def test_inverse_standardize():
    z = np.array([0.0, 1.0, -1.0])
    raw = inverse_standardize(z, y_mean=10.0, y_scale=2.0)
    assert np.allclose(raw, [10.0, 12.0, 8.0])


def test_substitute_standardized_variables():
    out = substitute_standardized_variables("f_a + f_b", {"f_a": 1.0, "f_b": 2.0}, {"f_a": 2.0, "f_b": 4.0})
    assert "f_a" not in out.replace("(f_a", "")  # replaced
    # numeric check
    import numpy as np

    f_a = 5.0
    f_b = 9.0
    expr = out
    val = eval(expr.replace("^", "**"), {"__builtins__": {}}, {"f_a": f_a, "f_b": f_b})
    assert abs(val - (((f_a - 1.0) / 2.0) + ((f_b - 2.0) / 4.0))) < 1e-9


def test_cards_from_mined_factors(tmp_path):
    csv = tmp_path / "mined_factors.csv"
    pd.DataFrame({"factor_name": ["factor_000001"], "expression": ["(a * b)"]}).to_csv(csv, index=False)
    cards = cards_from_mined_factors(csv)
    assert cards[0].factor_id == "factor_000001"
    assert cards[0].unit_status == "screening_only"
    assert cards[0].approved_for_final_formula is False


def test_complexity_counts_expanded_nodes():
    # a deep nesting hidden behind alias must count fully
    cards = [FactorCard("big", "big", "(((a + b) * (c + d)) / (e + f))", ["a", "b", "c", "d", "e", "f"])]
    idx = build_card_index(cards)
    alias_stats = complexity_stats("big", idx)
    assert alias_stats["n_variables"] == 6
    assert alias_stats["expanded_node_count"] >= 11
