from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml

# Deterministic stratified sampler for known-formula tasks (experiment 2).
# No human may inspect method results and hand-pick tasks. Selection is a pure
# function of the frozen candidate pool + seed + strata proportions.


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _variable_bin(n_vars: int, bins: list[list[int]]) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= n_vars <= hi:
            return i
    return len(bins) - 1


def _node_bin(nodes: int, bins: list[list[int]]) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= nodes <= hi:
            return i
    return len(bins) - 1


def sample_known_formula_tasks(
    config_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Deterministically select tasks by stratified sampling.

    Returns the selected task list and writes it back into a copy of the config
    (or a standalone JSON) if output_path is given.
    """
    cfg = _load_config(config_path)
    sampler = cfg["sampler"]
    pool = cfg["candidate_pool"]
    seed = int(sampler.get("seed", 20260709))
    n_select = int(sampler.get("n_select", 20))
    strata = sampler.get("strata", {})

    var_bins = strata.get("variable_count", {}).get("bins", [[1, 99]])
    var_props = strata.get("variable_count", {}).get("proportions", [1.0])
    node_bins = strata.get("node_count", {}).get("bins", [[1, 999]])
    node_props = strata.get("node_count", {}).get("proportions", [1.0])
    min_nodes = int(strata.get("node_count", {}).get("min_nodes", 0))
    require_ops = set(strata.get("operator_type", {}).get("require_at_least_one_of", []))

    # filter by min complexity
    eligible = [t for t in pool if int(t.get("node_count", 0)) >= min_nodes]

    rng = random.Random(seed)

    # bucket by (var_bin, node_bin)
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for t in eligible:
        vb = _variable_bin(int(t["n_variables"]), var_bins)
        nb = _node_bin(int(t["node_count"]), node_bins)
        buckets.setdefault((vb, nb), []).append(t)

    # target count per (var_bin) and per (node_bin) using proportions; combine
    # into a joint target by multiplying marginal proportions.
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    joint_targets: dict[tuple[int, int], int] = {}
    for vb in range(len(var_bins)):
        for nb in range(len(node_bins)):
            p = float(var_props[vb] if vb < len(var_props) else 0) * float(
                node_props[nb] if nb < len(node_props) else 0
            )
            joint_targets[(vb, nb)] = int(round(p * n_select))

    # deterministic draw from each bucket
    for key in sorted(joint_targets.keys()):
        want = joint_targets[key]
        candidates = sorted(buckets.get(key, []), key=lambda t: str(t["task_id"]))
        rng.shuffle(candidates)
        for t in candidates[:want]:
            if t["task_id"] not in selected_ids:
                selected.append(t)
                selected_ids.add(t["task_id"])

    # ensure operator coverage: if a required operator class is missing, pull
    # one eligible task providing it.
    def covered_ops() -> set[str]:
        ops: set[str] = set()
        for t in selected:
            ops.update(t.get("operators", []))
        return ops

    for req in sorted(require_ops):
        if req not in covered_ops():
            extras = sorted(
                [t for t in eligible if req in t.get("operators", []) and t["task_id"] not in selected_ids],
                key=lambda t: str(t["task_id"]),
            )
            if extras:
                selected.append(extras[0])
                selected_ids.add(extras[0]["task_id"])

    # top up / trim to exactly n_select deterministically
    if len(selected) < n_select:
        remaining = sorted(
            [t for t in eligible if t["task_id"] not in selected_ids], key=lambda t: str(t["task_id"])
        )
        rng.shuffle(remaining)
        for t in remaining:
            if len(selected) >= n_select:
                break
            selected.append(t)
            selected_ids.add(t["task_id"])
    selected = sorted(selected, key=lambda t: str(t["task_id"]))[:n_select]

    result = {
        "seed": seed,
        "n_requested": n_select,
        "n_selected": len(selected),
        "selected_tasks": [
            {
                "task_id": t["task_id"],
                "expression": t["expression"],
                "n_variables": t["n_variables"],
                "node_count": t["node_count"],
                "operators": t.get("operators", []),
            }
            for t in selected
        ],
        "operator_coverage": sorted(covered_ops()),
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
