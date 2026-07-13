from __future__ import annotations

from factor_pysr_llm.expression_similarity import (
    expression_similarity_report,
    algebraic_equivalence,
    numeric_equivalence,
    support_f1,
)


def test_identical_formula_scores_high():
    rep = expression_similarity_report("v1 * v2 + v3", "v1 * v2 + v3", ["v1", "v2", "v3"])
    assert rep["expr_sim"] > 0.98
    assert rep["separate_metrics"]["algebraic_equivalence"] is True
    assert rep["separate_metrics"]["numeric_equivalence"] is True
    assert rep["separate_metrics"]["support_f1"] == 1.0


def test_variable_renaming_detected_via_algebraic():
    # different variable names => not algebraically equal, low variable F1
    rep = expression_similarity_report("a * b", "v1 * v2", ["a", "b", "v1", "v2"])
    assert rep["subscores"]["variable_set_f1"] == 0.0
    assert rep["separate_metrics"]["algebraic_equivalence"] is False


def test_commutativity_is_equivalent():
    assert algebraic_equivalence("v1 + v2", "v2 + v1") is True
    assert numeric_equivalence("v1 * v2", "v2 * v1", ["v1", "v2"]) is True


def test_approximate_constant_close():
    rep = expression_similarity_report("v1 * 2.0", "v1 * 2.0001", ["v1"])
    assert rep["subscores"]["numeric_similarity"] > 0.9
    # not exactly equivalent
    assert rep["separate_metrics"]["algebraic_equivalence"] is False


def test_wrong_formula_scores_low():
    rep = expression_similarity_report("v1 + v2", "v1 * v2 * v3 / v4", ["v1", "v2", "v3", "v4"])
    assert rep["expr_sim"] < 0.8
    assert rep["separate_metrics"]["algebraic_equivalence"] is False


def test_missing_variable_lowers_support_f1():
    # predicted misses v3
    f1 = support_f1("v1 + v2", "v1 + v2 + v3")
    assert f1 < 1.0


def test_all_subscores_in_unit_interval():
    rep = expression_similarity_report("sin(v1) + v2^2", "cos(v1) * v2", ["v1", "v2"])
    for k, v in rep["subscores"].items():
        assert 0.0 <= v <= 1.0
    assert 0.0 <= rep["expr_sim"] <= 1.0


def test_undefined_points_handled():
    # log of negative handled by safe_log; division by potential zero
    rep = expression_similarity_report("inv(v1)", "1.0 / v1", ["v1"])
    assert rep["separate_metrics"]["numeric_equivalence"] is True


def test_weights_are_frozen():
    rep = expression_similarity_report("v1", "v1", ["v1"])
    assert rep["weights"]["numeric_similarity"] == 0.50
    assert rep["weights"]["variable_set_f1"] == 0.20
    assert rep["weights"]["operator_set_f1"] == 0.20
    assert rep["weights"]["tree_structure_similarity"] == 0.10
