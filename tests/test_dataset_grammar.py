from __future__ import annotations

import json

import pandas as pd

from factor_pysr_llm.config import WorkflowConfig
from factor_pysr_llm.dataset import build_raw_feature_table, inspect_dataset


def test_generic_dataset_build_raw(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    out_root = tmp_path / "out"
    pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "x one": [1.0, 2.0, None, 4.0],
            "x-two": [2.0, 4.0, 6.0, 8.0],
            "drop_me": [5.0, 5.0, 5.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(csv_path, index=False)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "input_csv": str(csv_path),
                "output_root": str(out_root),
                "dataset": {
                    "id_columns": ["sample_id"],
                    "target_columns": ["y"],
                    "feature_rules": {"exclude": ["drop_me"], "numeric_only": True},
                    "missing": {"fill": "median", "add_indicators": True},
                    "scaling": "zscore",
                    "naming": {"safe_prefix": "raw_"},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = WorkflowConfig.from_json(cfg_path)
    info = inspect_dataset(cfg)
    assert info["targets"] == ["y"]
    row = build_raw_feature_table(cfg, "y")
    assert row["n_rows"] == 4
    assert row["n_features"] == 3
    manifest = json.loads((out_root / "feature_tables" / "y" / "manifest.json").read_text())
    assert manifest["builder"] == "raw_dataset_grammar"
    assert "raw_x_one" in manifest["feature_name_map"]
    assert "raw_x_one__is_missing" in manifest["feature_name_map"]

