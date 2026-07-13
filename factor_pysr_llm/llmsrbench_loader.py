from __future__ import annotations

import io
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# LLM-SRBench loader via the ModelScope mirror.
#
# The official HuggingFace dataset (nnheui/llm-srbench) is gated and unreachable
# here. ModelScope hosts a normalized copy at
#   scientific-intelligent-modelling/sim-datasets-bak/llm-srbench/<category>/<task>/
# with per-task {formula.py, metadata.yaml, train.csv, valid.csv, id_test.csv,
# ood_test.csv}. This module lists, downloads and adapts those tasks into the
# same structure our runners use (feature frame + roles + ground-truth expr).

MS_BASE = "https://www.modelscope.cn/api/v1/datasets/scientific-intelligent-modelling/sim-datasets-bak"
REPO_TREE = MS_BASE + "/repo/tree?Revision=master&Root={root}"
REPO_FILE = MS_BASE + "/repo?Revision=master&FilePath={path}"
CATEGORIES = ["lsrtransform", "bio_pop_growth", "chem_react", "matsci", "phys_osc"]


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def list_tasks(category: str) -> list[str]:
    data = json.loads(_get(REPO_TREE.format(root=f"llm-srbench/{category}")))
    return [f["Path"] for f in data["Data"]["Files"] if f["Type"] == "tree"]


def download_task(task_path: str, dest: Path) -> dict[str, Any]:
    """Download the files of one task into dest/. Returns local file map."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = ["formula.py", "metadata.yaml", "train.csv", "valid.csv", "id_test.csv", "ood_test.csv"]
    got = {}
    for f in files:
        url = REPO_FILE.format(path=f"{task_path}/{f}")
        try:
            content = _get(url)
        except Exception:
            continue
        out = dest / f
        out.write_bytes(content)
        got[f] = str(out)
    return got


def _parse_formula_py(text: str) -> tuple[str, list[str], str]:
    """Extract (expression, arg_names, target_name) from a formula.py body.

    Handles two shapes:
        def v(m, m_0, c):
            return -c * np.sqrt(1 - m_0**2 / m**2)
    and (LSR-Synth) local-constant style:
        def dv_dt(x, t, v):
            alpha = 0.325...
            beta = 0.333...
            return -alpha*v**3 - beta*np.abs(v)**0.333 - x**3
    Local constant assignments are inlined into the returned expression so the
    result depends only on the function arguments.
    """
    # ignore comment lines
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    body = "\n".join(lines)
    m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)\s*:", body)
    if not m:
        raise ValueError("no def found in formula.py")
    target = m.group(1)
    args = [a.strip() for a in m.group(2).split(",") if a.strip()]
    # collect local constant assignments before the return
    consts: dict[str, str] = {}
    for am in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", body, re.MULTILINE):
        name, val = am.group(1), am.group(2)
        if name in args or name == target:
            continue
        consts[name] = val
    ret = re.search(r"return\s+(.+)", body, re.DOTALL)
    if not ret:
        raise ValueError("no return in formula.py")
    expr = ret.group(1).strip()
    expr = re.sub(r"\bnp\.", "", expr)
    # inline constants (longest names first to avoid partial overlap)
    for name in sorted(consts, key=len, reverse=True):
        val = re.sub(r"\bnp\.", "", consts[name])
        expr = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", f"({val})", expr)
    return expr, args, target


def load_task(dest: Path) -> dict[str, Any]:
    """Load a downloaded task into feature frame + roles + metadata.

    Uses the benchmark's own train/valid/id_test splits (id_test as the held-out
    test set). Row roles are built so downstream no-leakage code applies.
    """
    dest = Path(dest)
    meta = yaml.safe_load((dest / "metadata.yaml").read_text(encoding="utf-8"))
    formula_txt = (dest / "formula.py").read_text(encoding="utf-8")
    expr, args, target = _parse_formula_py(formula_txt)

    def read_split(name: str) -> pd.DataFrame:
        return pd.read_csv(dest / name)

    train = read_split("train.csv")
    valid = read_split("valid.csv")
    test = read_split("id_test.csv")
    feats = [c for c in train.columns if c != target]

    # subsample huge splits to keep SR tractable but representative
    def cap(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
        if len(df) <= n:
            return df.reset_index(drop=True)
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    train = cap(train, 400, 1)
    valid = cap(valid, 200, 2)
    test = cap(test, 400, 3)
    frame = pd.concat([train, valid, test], axis=0, ignore_index=True)
    roles = np.array(["train"] * len(train) + ["validation"] * len(valid) + ["test"] * len(test), dtype=object)
    return {
        "task_id": meta["dataset"]["name"],
        "expression": expr,
        "variables": feats,
        "target": target,
        "frame": frame,
        "roles": roles,
        "metadata": meta,
    }


def download_subset(
    out_root: Path,
    per_category: dict[str, int],
    seed: int = 20260709,
    log_fn=None,
) -> list[str]:
    """Download a reproducible sampled subset. Returns list of local task dirs."""
    out_root = Path(out_root)
    rng = np.random.default_rng(seed)
    dirs = []
    for cat, k in per_category.items():
        tasks = sorted(list_tasks(cat))
        if k < len(tasks):
            idx = sorted(rng.choice(len(tasks), size=k, replace=False).tolist())
            tasks = [tasks[i] for i in idx]
        for tp in tasks:
            name = tp.split("/")[-1]
            dest = out_root / cat / name
            if (dest / "metadata.yaml").exists():
                dirs.append(str(dest))
                if log_fn:
                    log_fn(f"  cached {cat}/{name}")
                continue
            got = download_task(tp, dest)
            if "metadata.yaml" in got and "train.csv" in got:
                dirs.append(str(dest))
                if log_fn:
                    log_fn(f"  downloaded {cat}/{name} ({len(got)} files)")
            time.sleep(0.2)
    return dirs
