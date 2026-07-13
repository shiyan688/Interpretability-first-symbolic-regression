from __future__ import annotations

import json

import numpy as np
import pandas as pd

from factor_pysr_llm.config import WorkflowConfig
from factor_pysr_llm.dataset import build_raw_feature_table
from factor_pysr_llm.factor_miner import build_pysr_pool, mine_factors
from factor_pysr_llm.llm_stages import write_factor_selection_prompt


def _setup(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    x1 = np.linspace(-2, 2, n)
    x2 = 1.5 + np.sin(np.linspace(0, 3, n))
    x3 = rng.normal(0, 1, n)
    y = x1 * x2 + 0.3 * x3 + rng.normal(0, 0.02, n)
    csv = tmp_path / "data.csv"
    pd.DataFrame({"id": range(n), "x1": x1, "x2": x2, "x3": x3, "y": y}).to_csv(csv, index=False)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "input_csv": str(csv),
                "output_root": str(tmp_path / "out"),
                "targets": ["y"],
                "dataset": {"id_columns": ["id"], "target_columns": ["y"], "scaling": "zscore"},
                "factor_mining": {
                    "base_top_k": 3,
                    "pair_top_k": 3,
                    "beam_width": 30,
                    "final_top_k": 40,
                    "max_order": 2,
                    "unary_ops": ["square"],
                    "binary_ops": ["*", "+", "-"],
                },
                "factor_selection": {"raw_top_k": 3, "llm_authoritative": True, "llm_fallback_top_k": 0},
            }
        ),
        encoding="utf-8",
    )
    return WorkflowConfig.from_json(cfg_path)


def test_selection_prompt_embeds_factor_content(tmp_path):
    cfg = _setup(tmp_path)
    build_raw_feature_table(cfg, "y")
    mine_factors(cfg, "y")
    prompt_path = write_factor_selection_prompt(cfg, "y", top_k=10)
    text = prompt_path.read_text(encoding="utf-8")
    factors = pd.read_csv(cfg.output_root / "factor_pools" / "y" / "mined_factors.csv")
    # at least one real factor expression must be embedded in the prompt body
    top_expr = factors.sort_values("score_abs_corr", ascending=False)["expression"].iloc[0]
    assert top_expr in text
    assert "output schema" in text.lower() or "schema" in text.lower()


def test_different_llm_selection_yields_different_pool(tmp_path):
    cfg = _setup(tmp_path)
    build_raw_feature_table(cfg, "y")
    mine_factors(cfg, "y")
    factors = pd.read_csv(cfg.output_root / "factor_pools" / "y" / "mined_factors.csv")
    names = factors["factor_name"].astype(str).tolist()
    assert len(names) >= 2

    sel1 = tmp_path / "sel1.json"
    sel1.write_text(json.dumps({"selected_factors": [{"factor_name": names[0], "final_formula_allowed": True}]}), encoding="utf-8")
    sel2 = tmp_path / "sel2.json"
    sel2.write_text(json.dumps({"selected_factors": [{"factor_name": names[1], "final_formula_allowed": False}]}), encoding="utf-8")

    build_pysr_pool(cfg, "y", llm_selection_path=sel1, output_tag="pool1")
    build_pysr_pool(cfg, "y", llm_selection_path=sel2, output_tag="pool2")
    p1 = pd.read_csv(cfg.output_root / "feature_tables" / "y__pool1" / "selected_mined_factors.csv")["factor_name"].astype(str).tolist()
    p2 = pd.read_csv(cfg.output_root / "feature_tables" / "y__pool2" / "selected_mined_factors.csv")["factor_name"].astype(str).tolist()
    assert set(p1) != set(p2)
    assert names[0] in p1 and names[1] in p2


def test_final_formula_allowed_recorded(tmp_path):
    cfg = _setup(tmp_path)
    build_raw_feature_table(cfg, "y")
    mine_factors(cfg, "y")
    factors = pd.read_csv(cfg.output_root / "factor_pools" / "y" / "mined_factors.csv")
    name = factors["factor_name"].astype(str).iloc[0]
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"selected_factors": [{"factor_name": name, "final_formula_allowed": True}]}), encoding="utf-8")
    build_pysr_pool(cfg, "y", llm_selection_path=sel, output_tag="poolx")
    rows = pd.read_csv(cfg.output_root / "feature_tables" / "y__poolx" / "selected_mined_factors.csv")
    row = rows[rows["factor_name"].astype(str) == name].iloc[0]
    assert bool(row["final_formula_allowed"]) is True
