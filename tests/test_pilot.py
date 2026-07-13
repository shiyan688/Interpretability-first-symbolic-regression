from __future__ import annotations

import numpy as np
import pandas as pd

from factor_pysr_llm.known_data import generate_known_formula_dataset, write_known_formula_task
from factor_pysr_llm.known_formulas import sample_known_formula_tasks
from factor_pysr_llm.pilot import (
    ablation_is_nontrivial,
    interpretability_first_selection,
    pilot_conditions,
    standard_selection,
    surrogate_search,
)


def test_known_formula_sampler_selects_20():
    info = sample_known_formula_tasks("configs/known_formula_tasks.yaml")
    assert info["n_selected"] == 20
    # deterministic
    info2 = sample_known_formula_tasks("configs/known_formula_tasks.yaml")
    assert [t["task_id"] for t in info["selected_tasks"]] == [t["task_id"] for t in info2["selected_tasks"]]


def test_generate_and_write_known_formula(tmp_path):
    ds = generate_known_formula_dataset("t", "v1 * v2", 2, n_train=50, n_validation=20, n_test=30, n_exprsim=40, seed=1)
    meta = write_known_formula_task(tmp_path / "t", ds)
    assert meta["n_train"] == 50 and meta["n_test"] == 30
    combined = pd.read_csv(tmp_path / "t" / "data.csv")
    assert len(combined) == 100
    roles = pd.read_csv(tmp_path / "t" / "predefined_roles.csv")
    assert set(roles["role"]) == {"train", "validation", "test"}
    # exprsim points are noise-free and independent
    exprsim = pd.read_csv(tmp_path / "t" / "exprsim_points.csv")
    assert len(exprsim) == 40


def test_pilot_pool_ablation_nontrivial(tmp_path):
    ds = generate_known_formula_dataset("t", "v1 * (v2 + v3)", 3, n_train=200, n_validation=100, n_test=100, n_exprsim=50, seed=20260709)
    comb = pd.concat([ds["train"], ds["validation"], ds["test"]], ignore_index=True)
    roles = ["train"] * 200 + ["validation"] * 100 + ["test"] * 100
    y = comb["y"].to_numpy(dtype=float)
    tr = np.array([r == "train" for r in roles])
    va = np.array([r == "validation" for r in roles])
    ff = comb[ds["variables"] + ds["irrelevant_variables"]].copy()
    ff["mine_sum"] = comb["v2"] + comb["v3"]
    ff["domain_prod"] = comb["v1"] * (comb["v2"] + comb["v3"])
    arc = pilot_conditions(
        ds["variables"] + ds["irrelevant_variables"], ["mine_sum"], ["domain_prod"], ff, y, tr, va
    )
    ab = ablation_is_nontrivial(arc)
    sels = ab["selections"]
    # raw / mine / if_sr must give at least 3 distinct selections
    assert len({sels["raw_pysr"], sels["mine_pysr"], sels["if_sr"]}) == 3


def test_selection_rule_can_diverge():
    # a task where a simpler candidate is within delta of the best -> IF diverges
    ds = generate_known_formula_dataset("t", "v1 + 0.05*v2*v3", 3, n_train=400, n_validation=200, n_test=100, n_exprsim=50, seed=20260709, var_low=1.0, var_high=2.0)
    comb = pd.concat([ds["train"], ds["validation"], ds["test"]], ignore_index=True)
    roles = ["train"] * 400 + ["validation"] * 200 + ["test"] * 100
    y = comb["y"].to_numpy(dtype=float)
    tr = np.array([r == "train" for r in roles])
    va = np.array([r == "validation" for r in roles])
    ff = comb[ds["variables"] + ds["irrelevant_variables"]].copy()
    ff["mine_prod"] = comb["v2"] * comb["v3"]
    cols = ds["variables"] + ds["irrelevant_variables"] + ["mine_prod"]
    cands = surrogate_search(ff, y, tr, va, cols)
    std = standard_selection(cands)
    iff = interpretability_first_selection(cands)
    assert std != iff  # accuracy-first vs interpretability-first genuinely differ
