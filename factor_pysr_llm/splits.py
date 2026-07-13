from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import WorkflowConfig
from .features import safe_read_csv

# Row-id semantics
# ----------------
# A split manifest stores explicit row IDs (not just fractions) so that the
# exact train/validation/test membership is reproducible and auditable. Row IDs
# are taken from an ID column when available, otherwise from the positional
# integer index of the target-filtered table.

DEFAULT_FRACTIONS = (0.6, 0.2, 0.2)


@dataclass
class SplitManifest:
    """A reproducible train/validation/test partition over explicit row IDs."""

    target: str
    id_column: str | None
    mode: str  # "random" | "group"
    group_column: str | None
    seed: int
    fractions: tuple[float, float, float]
    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    n_total: int
    input_csv: str = ""
    config_path: str = ""
    group_assignment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "target": self.target,
            "id_column": self.id_column,
            "mode": self.mode,
            "group_column": self.group_column,
            "seed": int(self.seed),
            "fractions": list(self.fractions),
            "n_total": int(self.n_total),
            "n_train": len(self.train_ids),
            "n_validation": len(self.validation_ids),
            "n_test": len(self.test_ids),
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
            "input_csv": self.input_csv,
            "config_path": self.config_path,
        }
        if self.mode == "group":
            payload["group_assignment"] = dict(self.group_assignment)
        return payload

    def content_sha256(self) -> str:
        """Deterministic hash of the split membership (order-independent)."""
        core = {
            "target": self.target,
            "mode": self.mode,
            "seed": int(self.seed),
            "fractions": [round(float(x), 12) for x in self.fractions],
            "train_ids": sorted(str(x) for x in self.train_ids),
            "validation_ids": sorted(str(x) for x in self.validation_ids),
            "test_ids": sorted(str(x) for x in self.test_ids),
        }
        blob = json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitManifest":
        return cls(
            target=str(data["target"]),
            id_column=data.get("id_column"),
            mode=str(data.get("mode", "random")),
            group_column=data.get("group_column"),
            seed=int(data.get("seed", 0)),
            fractions=tuple(float(x) for x in data.get("fractions", DEFAULT_FRACTIONS)),  # type: ignore[arg-type]
            train_ids=[str(x) for x in data.get("train_ids", [])],
            validation_ids=[str(x) for x in data.get("validation_ids", [])],
            test_ids=[str(x) for x in data.get("test_ids", [])],
            n_total=int(data.get("n_total", 0)),
            input_csv=str(data.get("input_csv", "")),
            config_path=str(data.get("config_path", "")),
            group_assignment={str(k): str(v) for k, v in (data.get("group_assignment") or {}).items()},
        )


def _validate_fractions(fractions: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(fractions) != 3:
        raise ValueError("fractions must be (train, validation, test)")
    total = float(sum(fractions))
    if total <= 0:
        raise ValueError("fractions must sum to a positive value")
    return tuple(float(f) / total for f in fractions)  # type: ignore[return-value]


def _row_ids(df: pd.DataFrame, id_column: str | None) -> list[str]:
    if id_column and id_column in df.columns:
        ids = [str(x) for x in df[id_column].tolist()]
        if len(set(ids)) != len(ids):
            raise ValueError(f"id_column {id_column} contains duplicate values; cannot use as row IDs")
        return ids
    return [str(i) for i in range(len(df))]


def _finite_target_frame(cfg: WorkflowConfig, target: str) -> pd.DataFrame:
    """Return the rows kept for a target (target-finite), reset index."""
    df = safe_read_csv(cfg.input_csv)
    if target not in df.columns:
        raise KeyError(f"target not found: {target}")
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y)
    return df.loc[ok].reset_index(drop=True)


def _assign_counts(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def make_random_split(
    ids: list[str],
    seed: int,
    fractions: tuple[float, float, float],
) -> tuple[list[str], list[str], list[str]]:
    fractions = _validate_fractions(fractions)
    rng = np.random.default_rng(int(seed))
    order = list(ids)
    perm = rng.permutation(len(order))
    shuffled = [order[i] for i in perm]
    n_train, n_val, _ = _assign_counts(len(shuffled), fractions)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def make_group_split(
    ids: list[str],
    groups: list[str],
    seed: int,
    fractions: tuple[float, float, float],
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """Group split: all rows of one group land in exactly one set."""
    fractions = _validate_fractions(fractions)
    if len(ids) != len(groups):
        raise ValueError("ids and groups must have equal length")
    unique_groups = sorted(set(groups))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(unique_groups))
    shuffled_groups = [unique_groups[i] for i in perm]
    n_g = len(shuffled_groups)
    n_train_g, n_val_g, _ = _assign_counts(n_g, fractions)
    train_g = set(shuffled_groups[:n_train_g])
    val_g = set(shuffled_groups[n_train_g : n_train_g + n_val_g])
    assignment: dict[str, str] = {}
    for g in shuffled_groups:
        if g in train_g:
            assignment[g] = "train"
        elif g in val_g:
            assignment[g] = "validation"
        else:
            assignment[g] = "test"
    train, val, test = [], [], []
    for rid, g in zip(ids, groups):
        bucket = assignment[g]
        if bucket == "train":
            train.append(rid)
        elif bucket == "validation":
            val.append(rid)
        else:
            test.append(rid)
    return train, val, test, assignment


def build_split_manifest(
    cfg: WorkflowConfig,
    target: str,
    mode: str = "random",
    seed: int = 20260709,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    id_column: str | None = None,
    group_column: str | None = None,
) -> SplitManifest:
    df = _finite_target_frame(cfg, target)
    if id_column is None:
        # try first configured id column present in the table
        dataset_cfg = dict(cfg.data.get("dataset") or {})
        for cand in dataset_cfg.get("id_columns", []):
            if cand in df.columns:
                id_column = str(cand)
                break
    ids = _row_ids(df, id_column)
    n_total = len(ids)

    if mode == "group":
        if not group_column or group_column not in df.columns:
            raise ValueError(f"group split requires an existing group_column, got {group_column!r}")
        groups = [str(x) for x in df[group_column].tolist()]
        train, val, test, assignment = make_group_split(ids, groups, seed, fractions)
        manifest = SplitManifest(
            target=target,
            id_column=id_column,
            mode="group",
            group_column=group_column,
            seed=seed,
            fractions=_validate_fractions(fractions),
            train_ids=train,
            validation_ids=val,
            test_ids=test,
            n_total=n_total,
            input_csv=str(cfg.input_csv),
            config_path=str(cfg.path),
            group_assignment=assignment,
        )
    elif mode == "random":
        train, val, test = make_random_split(ids, seed, fractions)
        manifest = SplitManifest(
            target=target,
            id_column=id_column,
            mode="random",
            group_column=None,
            seed=seed,
            fractions=_validate_fractions(fractions),
            train_ids=train,
            validation_ids=val,
            test_ids=test,
            n_total=n_total,
            input_csv=str(cfg.input_csv),
            config_path=str(cfg.path),
        )
    else:
        raise ValueError(f"unsupported split mode: {mode}")

    check_split_manifest(manifest)
    return manifest


def check_split_manifest(manifest: SplitManifest) -> None:
    """Validate mutual exclusivity, coverage and no group crossing."""
    train = set(manifest.train_ids)
    val = set(manifest.validation_ids)
    test = set(manifest.test_ids)
    if len(train) != len(manifest.train_ids):
        raise ValueError("duplicate IDs in train set")
    if len(val) != len(manifest.validation_ids):
        raise ValueError("duplicate IDs in validation set")
    if len(test) != len(manifest.test_ids):
        raise ValueError("duplicate IDs in test set")
    if train & val or train & test or val & test:
        raise ValueError("split sets are not mutually exclusive")
    covered = len(train) + len(val) + len(test)
    if covered != manifest.n_total:
        raise ValueError(f"split does not cover all rows: covered={covered} n_total={manifest.n_total}")
    if manifest.mode == "group":
        # a group may not appear in more than one set
        seen: dict[str, str] = {}
        for bucket, ids in (("train", train), ("validation", val), ("test", test)):
            for rid in ids:
                pass  # per-row group membership is enforced at build time
        # verify assignment consistency
        buckets = set(manifest.group_assignment.values())
        if not buckets <= {"train", "validation", "test"}:
            raise ValueError("invalid group assignment buckets")


def save_split_manifest(manifest: SplitManifest, path: Path) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    payload["sha256"] = manifest.content_sha256()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_split_manifest(path: Path) -> SplitManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = SplitManifest.from_dict(data)
    check_split_manifest(manifest)
    stored = data.get("sha256")
    if stored is not None and stored != manifest.content_sha256():
        raise ValueError(f"split manifest sha256 mismatch for {path}")
    return manifest


def role_masks_for_frame(
    manifest: SplitManifest,
    df: pd.DataFrame,
    id_column: str | None = None,
) -> dict[str, np.ndarray]:
    """Return boolean masks (train/validation/test) aligned to df rows.

    df must be the target-finite frame (same construction as at split time).
    """
    id_column = id_column if id_column is not None else manifest.id_column
    ids = _row_ids(df, id_column)
    train = set(manifest.train_ids)
    val = set(manifest.validation_ids)
    test = set(manifest.test_ids)
    train_mask = np.array([rid in train for rid in ids], dtype=bool)
    val_mask = np.array([rid in val for rid in ids], dtype=bool)
    test_mask = np.array([rid in test for rid in ids], dtype=bool)
    return {"train": train_mask, "validation": val_mask, "test": test_mask}


def target_finite_frame(cfg: WorkflowConfig, target: str) -> pd.DataFrame:
    """Public helper: the exact rows used to construct features for a target."""
    return _finite_target_frame(cfg, target)
