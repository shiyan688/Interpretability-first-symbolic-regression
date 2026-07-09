from __future__ import annotations

import json

import numpy as np
import pandas as pd

from factor_pysr_llm.config import WorkflowConfig
from factor_pysr_llm.dataset import build_raw_feature_table
from factor_pysr_llm.factor_miner import build_pysr_pool, mine_factors
from factor_pysr_llm.llm_stages import write_factor_proposal_prompt, write_factor_selection_prompt


def test_factor_mining_and_pool(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x1 = np.linspace(-2, 2, 30)
    x2 = 1.5 + np.sin(np.linspace(0, 3.0, 30))
    noise = rng.normal(0.0, 0.01, size=len(x1))
    y = x1 * x2 + noise
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"id": range(len(x1)), "x1": x1, "x2": x2, "y": y}).to_csv(csv_path, index=False)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "input_csv": str(csv_path),
                "output_root": str(tmp_path / "out"),
                "targets": ["y"],
                "dataset": {
                    "id_columns": ["id"],
                    "target_columns": ["y"],
                    "missing": {"fill": "median", "add_indicators": True},
                    "scaling": "zscore",
                },
                "factor_mining": {
                    "base_top_k": 2,
                    "pair_top_k": 2,
                    "beam_width": 20,
                    "final_top_k": 20,
                    "max_order": 1,
                    "unary_ops": ["square"],
                    "binary_ops": ["*", "+", "-"],
                },
                "factor_selection": {"raw_top_k": 2, "factor_top_k": 5},
            }
        ),
        encoding="utf-8",
    )
    cfg = WorkflowConfig.from_json(cfg_path)
    build_raw_feature_table(cfg, "y")
    proposal = write_factor_proposal_prompt(cfg, "y", top_k=2)
    assert proposal.exists()
    proposal_json = tmp_path / "proposal.json"
    proposal_json.write_text(
        json.dumps(
            {
                "target": "y",
                "proposed_factors": [
                    {
                        "name": "interaction",
                        "expression": "(raw_x1 * raw_x2)",
                        "meaning": "synthetic interaction",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    mined = mine_factors(cfg, "y", llm_proposals_path=proposal_json)
    assert mined["n_factors"] > 0
    factors = pd.read_csv(tmp_path / "out" / "factor_pools" / "y" / "mined_factors.csv")
    assert factors["expression"].astype(str).str.contains(r"\*").any()
    proposal_report = pd.read_csv(tmp_path / "out" / "factor_pools" / "y" / "llm_proposed_factors_report.csv")
    assert proposal_report["status"].eq("kept").any()
    selection = write_factor_selection_prompt(cfg, "y", top_k=5)
    assert selection.exists()
    pool = build_pysr_pool(cfg, "y")
    assert pool["n_features"] >= 3
