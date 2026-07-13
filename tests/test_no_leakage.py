from __future__ import annotations

import json

import numpy as np
import pandas as pd

from factor_pysr_llm.config import WorkflowConfig
from factor_pysr_llm.dataset import build_raw_feature_table
from factor_pysr_llm.factor_miner import build_pysr_pool, mine_factors
from factor_pysr_llm.splits import build_split_manifest, save_split_manifest


def _make_dataset(tmp_path, seed=0, n=60):
    rng = np.random.default_rng(seed)
    x1 = np.linspace(-3, 3, n)
    x2 = rng.normal(0, 1, n)
    x3 = np.sin(np.linspace(0, 5, n))
    y = 2.0 * x1 + 0.5 * x2 * x3 + rng.normal(0, 0.05, n)
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {"sample_id": [f"s{i:03d}" for i in range(n)], "x1": x1, "x2": x2, "x3": x3, "y": y}
    ).to_csv(csv_path, index=False)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "input_csv": str(csv_path),
                "output_root": str(tmp_path / "out"),
                "targets": ["y"],
                "dataset": {"id_columns": ["sample_id"], "target_columns": ["y"], "scaling": "zscore"},
                "raw_feature_selection": {"mode": "top_k_abs_corr", "top_k": 3},
                "factor_mining": {
                    "base_top_k": 5,
                    "pair_top_k": 5,
                    "beam_width": 30,
                    "final_top_k": 40,
                    "max_order": 1,
                    "unary_ops": ["square"],
                    "binary_ops": ["*", "+", "-"],
                },
                "factor_selection": {"raw_top_k": 3, "factor_top_k": 10},
            }
        ),
        encoding="utf-8",
    )
    return csv_path, cfg_path


def _corrupt_test_labels(csv_path, split_manifest, factor=1000.0):
    """Return a new csv path where test-row labels are multiplied by factor."""
    df = pd.read_csv(csv_path)
    test_ids = set(split_manifest.test_ids)
    mask = df["sample_id"].astype(str).isin(test_ids)
    df.loc[mask, "y"] = df.loc[mask, "y"] * factor + 12345.0
    new_path = csv_path.parent / "data_corrupt_test.csv"
    df.to_csv(new_path, index=False)
    return new_path


def test_train_fit_preprocess_ignores_test_extremes(tmp_path):
    csv_path, cfg_path = _make_dataset(tmp_path)
    cfg = WorkflowConfig.from_json(cfg_path)
    split = build_split_manifest(cfg, "y", mode="random", seed=1)
    split_path = tmp_path / "split.json"
    save_split_manifest(split, split_path)

    build_raw_feature_table(cfg, "y", split_manifest_path=split_path)
    manifest_a = json.loads((cfg.output_root / "feature_tables" / "y" / "manifest.json").read_text())

    # Corrupt a FEATURE extreme value only in test rows.
    df = pd.read_csv(csv_path)
    test_ids = set(split.test_ids)
    mask = df["sample_id"].astype(str).isin(test_ids)
    df.loc[mask, "x1"] = df.loc[mask, "x1"] * 1e6
    corrupt_csv = tmp_path / "data_corrupt_feat.csv"
    df.to_csv(corrupt_csv, index=False)

    cfg2_path = tmp_path / "config2.json"
    cfg2 = json.loads(cfg_path.read_text())
    cfg2["input_csv"] = str(corrupt_csv)
    cfg2["output_root"] = str(tmp_path / "out2")
    cfg2_path.write_text(json.dumps(cfg2), encoding="utf-8")
    cfg_b = WorkflowConfig.from_json(cfg2_path)
    # rebuild split on same seed/ids (ids identical since rows unchanged)
    split_b = build_split_manifest(cfg_b, "y", mode="random", seed=1)
    split_b_path = tmp_path / "split_b.json"
    save_split_manifest(split_b, split_b_path)
    build_raw_feature_table(cfg_b, "y", split_manifest_path=split_b_path)
    manifest_b = json.loads((cfg_b.output_root / "feature_tables" / "y" / "manifest.json").read_text())

    # train-fitted means/scales must be identical because test features changed only
    assert manifest_a["x_mean"] == manifest_b["x_mean"]
    assert manifest_a["x_scale"] == manifest_b["x_scale"]
    assert manifest_a["feature_selection"]["kept_features"] == manifest_b["feature_selection"]["kept_features"]


def test_changing_test_labels_does_not_change_mined_factors(tmp_path):
    csv_path, cfg_path = _make_dataset(tmp_path)
    cfg = WorkflowConfig.from_json(cfg_path)
    split = build_split_manifest(cfg, "y", mode="random", seed=2)
    split_path = tmp_path / "split.json"
    save_split_manifest(split, split_path)

    build_raw_feature_table(cfg, "y", split_manifest_path=split_path)
    mine_factors(cfg, "y")
    factors_a = pd.read_csv(cfg.output_root / "factor_pools" / "y" / "mined_factors.csv")
    pool_a = build_pysr_pool(cfg, "y")
    selected_a = pd.read_csv(
        cfg.output_root / "feature_tables" / "y__pysr_pool" / "selected_mined_factors.csv"
    )["factor_name"].astype(str).tolist()

    # Corrupt TEST labels.
    corrupt_csv = _corrupt_test_labels(csv_path, split)
    cfg2 = json.loads(cfg_path.read_text())
    cfg2["input_csv"] = str(corrupt_csv)
    cfg2["output_root"] = str(tmp_path / "out2")
    cfg2_path = tmp_path / "config2.json"
    cfg2_path.write_text(json.dumps(cfg2), encoding="utf-8")
    cfg_b = WorkflowConfig.from_json(cfg2_path)
    split_b = build_split_manifest(cfg_b, "y", mode="random", seed=2)
    save_split_manifest(split_b, tmp_path / "split_b.json")
    build_raw_feature_table(cfg_b, "y", split_manifest_path=tmp_path / "split_b.json")
    mine_factors(cfg_b, "y")
    factors_b = pd.read_csv(cfg_b.output_root / "factor_pools" / "y" / "mined_factors.csv")
    build_pysr_pool(cfg_b, "y")
    selected_b = pd.read_csv(
        cfg_b.output_root / "feature_tables" / "y__pysr_pool" / "selected_mined_factors.csv"
    )["factor_name"].astype(str).tolist()

    # Mined factor expressions and their train-scored ranking must be identical.
    assert factors_a["expression"].tolist() == factors_b["expression"].tolist()
    assert np.allclose(
        factors_a["score_abs_corr"].to_numpy(), factors_b["score_abs_corr"].to_numpy(), atol=1e-12
    )
    assert selected_a == selected_b


def test_split_seed_reproducible_ids(tmp_path):
    csv_path, cfg_path = _make_dataset(tmp_path)
    cfg = WorkflowConfig.from_json(cfg_path)
    s1 = build_split_manifest(cfg, "y", mode="random", seed=42)
    s2 = build_split_manifest(cfg, "y", mode="random", seed=42)
    assert s1.content_sha256() == s2.content_sha256()
    assert sorted(s1.test_ids) == sorted(s2.test_ids)
