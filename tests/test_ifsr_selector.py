from __future__ import annotations

import pytest

from factor_pysr_llm.ifsr_selector import Candidate, select_if_sr
from factor_pysr_llm.lineage import FactorCard


def test_delta_boundary():
    cands = [
        Candidate("best", "a + b", r2_val=0.90),
        Candidate("edge", "c", r2_val=0.88),        # exactly at threshold 0.90-0.02
        Candidate("out", "d", r2_val=0.8799),        # just below
    ]
    d = select_if_sr(cands, delta=0.02)
    assert d["n_acceptable"] == 2
    # "edge" is simplest (1 node) so it should win
    assert d["selected"]["candidate_id"] == "edge"


def test_test_r2_does_not_affect_selection():
    cands_a = [
        Candidate("c1", "a + b", r2_val=0.90, r2_test=0.1),
        Candidate("c2", "a", r2_val=0.89, r2_test=0.99),
    ]
    cands_b = [
        Candidate("c1", "a + b", r2_val=0.90, r2_test=0.99),
        Candidate("c2", "a", r2_val=0.89, r2_test=0.1),
    ]
    da = select_if_sr(cands_a, delta=0.02)
    db = select_if_sr(cands_b, delta=0.02)
    assert da["selected"]["candidate_id"] == db["selected"]["candidate_id"]
    # simplest acceptable = c2
    assert da["selected"]["candidate_id"] == "c2"


def test_short_alias_does_not_escape_complexity():
    # 'big' alias expands to a large expression; a genuinely simple candidate wins
    cards = [FactorCard("big", "big", "(((a + b) * (c + d)) / (e + f))", ["a", "b", "c", "d", "e", "f"])]
    cands = [
        Candidate("alias", "big", r2_val=0.90, uses_domain_factor=True),
        Candidate("simple", "a + b", r2_val=0.89),
    ]
    d = select_if_sr(cands, delta=0.02, cards=cards)
    assert d["selected"]["candidate_id"] == "simple"


def test_hard_violation_beats_complexity():
    cands = [
        Candidate("leaky", "leakvar", r2_val=0.95),   # simplest but leaks
        Candidate("clean", "a + b + c", r2_val=0.94),
    ]
    d = select_if_sr(cands, delta=0.02, leakage_variables={"leakvar"})
    assert d["selected"]["candidate_id"] == "clean"


def test_domain_factor_tiebreak():
    cands = [
        Candidate("plain", "a + b", r2_val=0.90),
        Candidate("domain", "a + b", r2_val=0.90, uses_domain_factor=True),
    ]
    d = select_if_sr(cands, delta=0.02)
    assert d["selected"]["candidate_id"] == "domain"


def test_stability_and_id_tiebreak_reproducible():
    cands = [
        Candidate("b_id", "a + b", r2_val=0.90, uses_domain_factor=True, stability=0.5),
        Candidate("a_id", "a + b", r2_val=0.90, uses_domain_factor=True, stability=0.5),
    ]
    d1 = select_if_sr(cands, delta=0.02)
    d2 = select_if_sr(list(reversed(cands)), delta=0.02)
    # same tie-break regardless of input order -> smallest id
    assert d1["selected"]["candidate_id"] == "a_id"
    assert d2["selected"]["candidate_id"] == "a_id"


def test_all_violations_returns_none():
    cands = [Candidate("only", "leak", r2_val=0.9)]
    d = select_if_sr(cands, delta=0.02, leakage_variables={"leak"})
    assert d["selected"] is None
