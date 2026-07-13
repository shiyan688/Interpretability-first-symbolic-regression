from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .expr import eval_expr

# Synthetic data generation for known-formula tasks (experiment 2 pilot).
# Frozen generation settings live in configs/known_formula_tasks.yaml. This
# module renders one task into train/validation/test CSVs plus an independent
# point set used only for ExprSim (never reused for training).


def generate_known_formula_dataset(
    task_id: str,
    expression: str,
    n_variables: int,
    n_train: int = 400,
    n_validation: int = 200,
    n_test: int = 400,
    n_exprsim: int = 400,
    target_noise: float = 0.05,
    n_irrelevant: int = 5,
    seed: int = 20260709,
    var_low: float = 1.0,
    var_high: float = 3.0,
) -> dict[str, Any]:
    """Render a known formula into split CSVs + independent ExprSim points.

    Variables are named v1..vk (matching the frozen pool). Irrelevant variables
    are named z1..zm and do not enter the true expression. Target noise is
    relative Gaussian noise on the clean signal.
    """
    rng = np.random.default_rng(int(seed))
    variables = [f"v{i}" for i in range(1, n_variables + 1)]
    irrelevant = [f"z{i}" for i in range(1, n_irrelevant + 1)]

    def make_frame(n: int, sub_seed: int) -> pd.DataFrame:
        r = np.random.default_rng(sub_seed)
        cols = {v: r.uniform(var_low, var_high, n) for v in variables}
        for z in irrelevant:
            cols[z] = r.uniform(var_low, var_high, n)
        return pd.DataFrame(cols)

    def add_target(frame: pd.DataFrame, sub_seed: int, noisy: bool) -> pd.DataFrame:
        with np.errstate(all="ignore"):
            signal = eval_expr(expression, frame)
        signal = np.where(np.isfinite(signal), signal, 0.0)
        out = frame.copy()
        if noisy:
            r = np.random.default_rng(sub_seed)
            scale = float(np.std(signal)) or 1.0
            out["y"] = signal + r.normal(0.0, target_noise * scale, len(signal))
        else:
            out["y"] = signal
        out.insert(0, "sample_id", [f"{task_id}_{sub_seed}_{i:04d}" for i in range(len(out))])
        return out

    base = int(seed)
    train = add_target(make_frame(n_train, base + 1), base + 101, noisy=True)
    val = add_target(make_frame(n_validation, base + 2), base + 102, noisy=True)
    test = add_target(make_frame(n_test, base + 3), base + 103, noisy=True)
    exprsim = add_target(make_frame(n_exprsim, base + 4), base + 104, noisy=False)

    return {
        "task_id": task_id,
        "expression": expression,
        "variables": variables,
        "irrelevant_variables": irrelevant,
        "train": train,
        "validation": val,
        "test": test,
        "exprsim_points": exprsim,
    }


def write_known_formula_task(
    out_dir: Path,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # combined CSV (train+val+test) for the dataset-grammar pipeline
    combined = pd.concat(
        [dataset["train"], dataset["validation"], dataset["test"]], axis=0, ignore_index=True
    )
    combined_path = out_dir / "data.csv"
    combined.to_csv(combined_path, index=False)
    dataset["exprsim_points"].to_csv(out_dir / "exprsim_points.csv", index=False)
    # explicit role vector so a split manifest can be built to match
    n_tr = len(dataset["train"])
    n_va = len(dataset["validation"])
    n_te = len(dataset["test"])
    roles = ["train"] * n_tr + ["validation"] * n_va + ["test"] * n_te
    role_df = pd.DataFrame({"sample_id": combined["sample_id"], "role": roles})
    role_df.to_csv(out_dir / "predefined_roles.csv", index=False)
    meta = {
        "task_id": dataset["task_id"],
        "expression": dataset["expression"],
        "variables": dataset["variables"],
        "irrelevant_variables": dataset["irrelevant_variables"],
        "n_train": n_tr,
        "n_validation": n_va,
        "n_test": n_te,
        "n_exprsim_points": int(len(dataset["exprsim_points"])),
        "data_csv": str(combined_path),
    }
    (out_dir / "task_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
