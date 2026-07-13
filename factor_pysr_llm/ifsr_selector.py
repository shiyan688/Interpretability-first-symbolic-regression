from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .lineage import FactorCard, build_card_index, complexity_stats

# IF-SR selector
# --------------
# Accuracy-constrained, interpretability-first selection. The selection uses
# ONLY validation metrics. Test metrics must not be read until a formula is
# locked. The rule is a fixed, reproducible lexicographic order:
#
#   1. acceptable set: R2_val >= best_R2_val - delta
#   2. drop hard violations (leakage vars, invalid expr, undefined, domain
#      coverage too low, definite dimensional add/sub errors)
#   3. minimize expanded complexity
#   4. tie -> prefer candidate using an approved domain factor
#   5. tie -> higher cross-seed stability
#   6. tie -> lexicographically smallest candidate id (fully reproducible)


@dataclass
class Candidate:
    candidate_id: str
    expression: str
    r2_val: float
    # optional, only read AFTER selection
    r2_test: float | None = None
    uses_domain_factor: bool = False
    stability: float = 0.0
    hard_violation: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "expression": self.expression,
            "r2_val": self.r2_val,
            "r2_test": self.r2_test,
            "uses_domain_factor": self.uses_domain_factor,
            "stability": self.stability,
            "hard_violation": self.hard_violation,
            "meta": self.meta,
        }


DEFAULT_DELTA = 0.02


def default_hard_violation_checker(
    leakage_variables: set[str] | None = None,
    card_index: dict[str, FactorCard] | None = None,
) -> Callable[[Candidate], str | None]:
    """Return a checker returning a violation reason or None."""
    leakage_variables = leakage_variables or set()
    card_index = card_index or {}

    def check(cand: Candidate) -> str | None:
        if cand.hard_violation:
            return cand.hard_violation
        expr = cand.expression
        if not str(expr).strip():
            return "empty_expression"
        # try expansion + complexity; failure => invalid/undefined
        try:
            stats = complexity_stats(expr, card_index)
        except Exception as exc:  # cyclic reference, unparseable, etc.
            return f"invalid_expression:{type(exc).__name__}"
        used_vars = set(stats.get("variables", []))
        leaked = used_vars & leakage_variables
        if leaked:
            return f"leakage_variable:{sorted(leaked)}"
        return None

    return check


def _annotate_complexity(
    candidates: list[Candidate], card_index: dict[str, FactorCard]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        try:
            out[cand.candidate_id] = complexity_stats(cand.expression, card_index)
        except Exception:
            out[cand.candidate_id] = {"expanded_node_count": 10**9, "variables": []}
    return out


def select_if_sr(
    candidates: list[Candidate],
    delta: float = DEFAULT_DELTA,
    cards: list[FactorCard] | None = None,
    leakage_variables: set[str] | None = None,
    hard_violation_checker: Callable[[Candidate], str | None] | None = None,
) -> dict[str, Any]:
    """Run the IF-SR lexicographic selection. Returns a decision record.

    Never reads r2_test to make the decision. r2_test is only echoed in the
    trace if the caller already populated it (which should happen only after
    lock in a proper pipeline).
    """
    if not candidates:
        raise ValueError("no candidates to select from")
    card_index = build_card_index(cards or [])
    checker = hard_violation_checker or default_hard_violation_checker(leakage_variables, card_index)

    # 1. best validation R2 and acceptable set
    best_r2_val = max(c.r2_val for c in candidates)
    threshold = best_r2_val - float(delta)
    trace: list[dict[str, Any]] = []
    acceptable: list[Candidate] = []
    for c in candidates:
        record = {"candidate_id": c.candidate_id, "r2_val": c.r2_val}
        if c.r2_val < threshold:
            record["status"] = "rejected"
            record["reason"] = f"below_tolerance (r2_val<{threshold:.6f})"
            trace.append(record)
            continue
        acceptable.append(c)

    # 2. hard violations
    complexity = _annotate_complexity(candidates, card_index)
    survivors: list[Candidate] = []
    for c in acceptable:
        violation = checker(c)
        record = {"candidate_id": c.candidate_id, "r2_val": c.r2_val}
        if violation:
            record["status"] = "rejected"
            record["reason"] = f"hard_violation:{violation}"
            trace.append(record)
            continue
        record["status"] = "candidate"
        record["expanded_node_count"] = complexity[c.candidate_id]["expanded_node_count"]
        record["uses_domain_factor"] = c.uses_domain_factor
        record["stability"] = c.stability
        trace.append(record)
        survivors.append(c)

    if not survivors:
        return {
            "selected": None,
            "best_r2_val": best_r2_val,
            "threshold": threshold,
            "delta": float(delta),
            "n_candidates": len(candidates),
            "n_acceptable": len(acceptable),
            "n_survivors": 0,
            "trace": trace,
            "reason": "all candidates violated hard constraints",
        }

    # 3-6. lexicographic sort key
    def sort_key(c: Candidate):
        return (
            complexity[c.candidate_id]["expanded_node_count"],  # 3 minimize complexity
            0 if c.uses_domain_factor else 1,                    # 4 prefer domain factor
            -float(c.stability),                                 # 5 higher stability first
            str(c.candidate_id),                                 # 6 reproducible tie-break
        )

    ranked = sorted(survivors, key=sort_key)
    selected = ranked[0]

    return {
        "selected": selected.to_dict(),
        "selected_complexity": complexity[selected.candidate_id],
        "best_r2_val": best_r2_val,
        "threshold": threshold,
        "delta": float(delta),
        "n_candidates": len(candidates),
        "n_acceptable": len(acceptable),
        "n_survivors": len(survivors),
        "ranking": [c.candidate_id for c in ranked],
        "trace": trace,
        "metric_space": "validation_only",
        "note": "test metrics are not used for selection",
    }


def candidates_from_records(records: list[dict[str, Any]]) -> list[Candidate]:
    out: list[Candidate] = []
    for i, r in enumerate(records):
        out.append(
            Candidate(
                candidate_id=str(r.get("candidate_id", f"cand_{i:04d}")),
                expression=str(r.get("expression", "")),
                r2_val=float(r.get("r2_val", r.get("r2_validation", 0.0))),
                r2_test=(float(r["r2_test"]) if r.get("r2_test") is not None else None),
                uses_domain_factor=bool(r.get("uses_domain_factor", False)),
                stability=float(r.get("stability", 0.0)),
                hard_violation=r.get("hard_violation"),
                meta=dict(r.get("meta", {})),
            )
        )
    return out


def save_decision(decision: dict[str, Any], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
