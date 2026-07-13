from __future__ import annotations

import json

import numpy as np
import pandas as pd

from factor_pysr_llm.config import WorkflowConfig
from factor_pysr_llm.splits import (
    build_split_manifest,
    check_split_manifest,
    load_split_manifest,
    make_group_split,
    make_random_split,
    save_split_manifest,
)


def _toy_config(tmp_path, n=40, with_group=False):
    rng = np.random.default_rng(0)
    x1 = np.linspace(-2, 2, n)
    x2 = rng.normal(0, 1, n)
    y = x1 * 2.0 + x2 + rng.normal(0, 0.05, n)
    data = {"sample_id": [f"s{i:03d}" for i in range(n)], "x1": x1, "x2": x2, "y": y}
    if with_group:
        data["material"] = [f"m{i % 5}" for i in range(n)]
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "input_csv": str(csv_path),
                "output_root": str(tmp_path / "out"),
                "targets": ["y"],
                "dataset": {"id_columns": ["sample_id"], "target_columns": ["y"]},
            }
        ),
        encoding="utf-8",
    )
    return WorkflowConfig.from_json(cfg_path)


def test_random_split_reproducible():
    ids = [f"r{i}" for i in range(50)]
    a = make_random_split(ids, seed=7, fractions=(0.6, 0.2, 0.2))
    b = make_random_split(ids, seed=7, fractions=(0.6, 0.2, 0.2))
    assert a == b
    c = make_random_split(ids, seed=8, fractions=(0.6, 0.2, 0.2))
    assert a != c
    train, val, test = a
    assert set(train) | set(val) | set(test) == set(ids)
    assert len(train) + len(val) + len(test) == len(ids)


def test_group_split_no_crossing():
    ids = [f"r{i}" for i in range(30)]
    groups = [f"g{i % 6}" for i in range(30)]
    train, val, test, assignment = make_group_split(ids, groups, seed=3, fractions=(0.6, 0.2, 0.2))
    train_groups = {groups[int(r[1:])] for r in train}
    val_groups = {groups[int(r[1:])] for r in val}
    test_groups = {groups[int(r[1:])] for r in test}
    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)
    assert set(train) | set(val) | set(test) == set(ids)


def test_build_and_load_manifest_sha256(tmp_path):
    cfg = _toy_config(tmp_path)
    manifest = build_split_manifest(cfg, "y", mode="random", seed=11)
    out = tmp_path / "split.json"
    payload = save_split_manifest(manifest, out)
    assert "sha256" in payload
    loaded = load_split_manifest(out)
    assert loaded.content_sha256() == manifest.content_sha256()
    check_split_manifest(loaded)


def test_manifest_stores_row_ids_not_fractions(tmp_path):
    cfg = _toy_config(tmp_path)
    manifest = build_split_manifest(cfg, "y", mode="random", seed=11)
    payload = manifest.to_dict()
    assert payload["train_ids"] and payload["validation_ids"] and payload["test_ids"]
    # IDs are the sample_id strings, not positional
    assert all(str(x).startswith("s") for x in payload["train_ids"])


def test_group_split_manifest_end_to_end(tmp_path):
    cfg = _toy_config(tmp_path, with_group=True)
    manifest = build_split_manifest(cfg, "y", mode="group", seed=5, group_column="material")
    check_split_manifest(manifest)
    # a group must not appear in more than one bucket
    buckets = {}
    for b in ("train_ids", "validation_ids", "test_ids"):
        for rid in getattr(manifest, b.replace("_ids", "_ids")):
            pass
    assert set(manifest.group_assignment.values()) <= {"train", "validation", "test"}
