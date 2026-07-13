from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .expr import eval_expr, to_python_expr

# Factor lineage
# --------------
# Every mined/domain factor is recorded as a "factor card" so that a short
# factor name can never hide a complex expression from complexity scoring or
# interpretability review. Cards can reference other factors by name; expansion
# recursively substitutes those references down to raw variables, with cycle
# detection. Complexity is always measured on the FULLY EXPANDED expression.

FACTOR_CARD_FIELDS = (
    "factor_id",
    "name",
    "expression",
    "variables",
    "meaning",
    "unit_status",
    "source",
    "approved_for_final_formula",
)

VALID_UNIT_STATUS = {"valid", "unknown", "screening_only"}
VALID_SOURCE = {"expert", "llm", "literature", "mined"}


@dataclass
class FactorCard:
    factor_id: str
    name: str
    expression: str
    variables: list[str] = field(default_factory=list)
    meaning: str = ""
    unit_status: str = "unknown"
    source: str = "mined"
    approved_for_final_formula: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "name": self.name,
            "expression": self.expression,
            "variables": list(self.variables),
            "meaning": self.meaning,
            "unit_status": self.unit_status,
            "source": self.source,
            "approved_for_final_formula": bool(self.approved_for_final_formula),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FactorCard":
        unit_status = str(data.get("unit_status", "unknown"))
        if unit_status not in VALID_UNIT_STATUS:
            raise ValueError(f"invalid unit_status {unit_status!r}; allowed {sorted(VALID_UNIT_STATUS)}")
        source = str(data.get("source", "mined"))
        if source not in VALID_SOURCE:
            raise ValueError(f"invalid source {source!r}; allowed {sorted(VALID_SOURCE)}")
        if not str(data.get("factor_id", "")).strip():
            raise ValueError("factor card requires a non-empty factor_id")
        if not str(data.get("expression", "")).strip():
            raise ValueError("factor card requires a non-empty expression")
        return cls(
            factor_id=str(data["factor_id"]),
            name=str(data.get("name", data["factor_id"])),
            expression=str(data["expression"]),
            variables=[str(v) for v in data.get("variables", [])],
            meaning=str(data.get("meaning", "")),
            unit_status=unit_status,
            source=source,
            approved_for_final_formula=bool(data.get("approved_for_final_formula", False)),
        )


def _identifiers(expr: str) -> set[str]:
    """Identifiers appearing in an expression (excludes known function names)."""
    funcs = {
        "abs", "sin", "cos", "tan", "exp", "log", "sqrt", "inv",
        "square", "cube", "cbrt", "pow", "pi", "asin", "acos", "atan", "tanh",
    }
    text = to_python_expr(expr)
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    # drop names that are immediately followed by '(' -> function calls
    call_names = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text))
    return {n for n in names if n not in funcs and n not in call_names}


def build_card_index(cards: list[FactorCard]) -> dict[str, FactorCard]:
    index: dict[str, FactorCard] = {}
    for card in cards:
        for key in (card.factor_id, card.name):
            if key:
                index[str(key)] = card
    return index


def expand_expression(
    expression: str,
    card_index: dict[str, FactorCard],
    _stack: tuple[str, ...] = (),
    _depth: int = 0,
    _max_depth: int = 200,
) -> str:
    """Recursively expand factor references down to raw variables.

    Raises ValueError on cyclic references.
    """
    if _depth > _max_depth:
        raise ValueError("factor expansion exceeded max depth (possible cycle)")
    tokens = _identifiers(expression)
    result = expression
    for tok in tokens:
        if tok in card_index:
            if tok in _stack:
                raise ValueError(f"cyclic factor reference detected: {' -> '.join([*_stack, tok])}")
            card = card_index[tok]
            # do not expand a card into itself if its expression is just a raw var
            if card.expression.strip() == tok:
                continue
            sub = expand_expression(
                card.expression, card_index, _stack=(*_stack, tok), _depth=_depth + 1, _max_depth=_max_depth
            )
            # replace whole-word token with parenthesized expansion
            result = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(tok)}(?![A-Za-z0-9_])", f"({sub})", result)
    return result


def _sympy_expr(expression: str):
    import sympy as sp

    local = {
        "inv": lambda x: 1 / x,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
        "sqrt_abs": lambda x: sp.sqrt(sp.Abs(x)),
        "log_abs": lambda x: sp.log(sp.Abs(x)),
        "Abs": sp.Abs,
        "abs": sp.Abs,
    }
    return sp.sympify(to_python_expr(expression), locals=local)


def canonicalize(expression: str) -> str:
    """Basic canonicalization via sympy. Falls back to the raw string."""
    try:
        import sympy as sp

        expr = _sympy_expr(expression)
        return str(sp.simplify(expr))
    except Exception:
        return expression


def complexity_stats(expression: str, card_index: dict[str, FactorCard] | None = None) -> dict[str, Any]:
    """Complexity measured on the fully expanded expression."""
    expanded = expand_expression(expression, card_index or {})
    try:
        import sympy as sp

        expr = _sympy_expr(expanded)
        node_count = _count_nodes(expr)
        depth = _tree_depth(expr)
        variables = sorted(str(s) for s in expr.free_symbols)
        n_constants = len([a for a in sp.preorder_traversal(expr) if a.is_number])
    except Exception:
        variables = sorted(_identifiers(expanded))
        node_count = len(re.findall(r"[A-Za-z0-9_.]+|[+\-*/()]", expanded))
        depth = expanded.count("(")
        n_constants = len(re.findall(r"(?<![A-Za-z0-9_.])\d+\.?\d*", expanded))
    return {
        "expanded_expression": expanded,
        "expanded_node_count": int(node_count),
        "expanded_depth": int(depth),
        "n_variables": int(len(variables)),
        "variables": variables,
        "n_constants": int(n_constants),
    }


def _count_nodes(expr) -> int:
    import sympy as sp

    return sum(1 for _ in sp.preorder_traversal(expr))


def _tree_depth(expr) -> int:
    if not getattr(expr, "args", None):
        return 1
    return 1 + max(_tree_depth(a) for a in expr.args)


def inverse_standardize(
    z_expression_value: np.ndarray,
    y_mean: float,
    y_scale: float,
) -> np.ndarray:
    """Map a z-scored prediction back to raw target units."""
    return float(y_mean) + float(y_scale) * np.asarray(z_expression_value, dtype=float)


def substitute_standardized_variables(
    expression: str,
    x_mean: dict[str, float],
    x_scale: dict[str, float],
) -> str:
    """Rewrite an expression over z-scored features into raw-variable terms.

    Each standardized feature ``f`` is replaced by ``((f - mean_f)/scale_f)``
    using the fitted preprocessing statistics, so the displayed formula matches
    raw variables. Names not in the maps are left untouched.
    """
    result = expression
    for name in sorted(set(x_mean) | set(x_scale), key=len, reverse=True):
        mean = float(x_mean.get(name, 0.0))
        scale = float(x_scale.get(name, 1.0)) or 1.0
        replacement = f"(({name} - {mean!r}) / {scale!r})"
        result = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", replacement, result)
    return result


def check_numeric_consistency(
    expression: str,
    card_index: dict[str, FactorCard],
    frame: pd.DataFrame,
    atol: float = 1.0e-8,
    rtol: float = 1.0e-6,
) -> dict[str, Any]:
    """Verify expanded expression reproduces the original expression numerically.

    frame must provide values for all raw variables AND all referenced factor
    columns so that both the original and expanded forms can be evaluated.
    """
    original = eval_expr(expression, frame)
    expanded = expand_expression(expression, card_index)
    expanded_val = eval_expr(expanded, frame)
    ok = np.isfinite(original) & np.isfinite(expanded_val)
    if int(ok.sum()) == 0:
        consistent = False
        max_abs_diff = float("inf")
    else:
        max_abs_diff = float(np.max(np.abs(original[ok] - expanded_val[ok])))
        consistent = bool(np.allclose(original[ok], expanded_val[ok], atol=atol, rtol=rtol))
    return {
        "expression": expression,
        "expanded_expression": expanded,
        "consistent": consistent,
        "max_abs_diff": max_abs_diff,
        "n_compared": int(ok.sum()),
    }


def load_factor_cards(path: Path) -> list[FactorCard]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.get("factors", data.get("cards", []))
    else:
        items = data
    return [FactorCard.from_dict(item) for item in items]


def save_factor_cards(cards: list[FactorCard], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"factors": [c.to_dict() for c in cards]}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cards_from_mined_factors(mined_csv: Path, source: str = "mined") -> list[FactorCard]:
    """Build factor cards from a mined_factors.csv produced by factor mining."""
    df = pd.read_csv(mined_csv)
    cards: list[FactorCard] = []
    for _, row in df.iterrows():
        expr = str(row.get("expression", "")).strip()
        if not expr:
            continue
        fid = str(row.get("factor_name", "")).strip() or f"mined_{len(cards):06d}"
        cards.append(
            FactorCard(
                factor_id=fid,
                name=fid,
                expression=expr,
                variables=sorted(_identifiers(expr)),
                meaning="",
                unit_status="screening_only",
                source=source,
                approved_for_final_formula=False,
            )
        )
    return cards
