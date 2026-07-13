#!/usr/bin/env python3
"""Run LLM-SR-style baselines on the LSR-Transform reconstruction tasks
(memorization-resistant, non-standard target variable). Same API/eval/protocol.
"""
from __future__ import annotations

import time
from pathlib import Path

from factor_pysr_llm.llm_sr_runner import run_baselines
from factor_pysr_llm.model_api import call_openai_compatible

ROOT = Path("/public/home/wangyg/Interpretability-first-symbolic-regression")
OUT = ROOT / "outputs" / "lsr_transform_baselines"
OUT.mkdir(parents=True, exist_ok=True)
PROG = OUT / "PROGRESS.log"
PROVIDER = ROOT / "configs/llm_provider.judge_a.json"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with PROG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def call_fn(prompt: str) -> str:
    return str(call_openai_compatible(PROVIDER, prompt)["content"])


def main() -> None:
    seeds = [20260709, 20260710, 20260711]
    log(f"=== LSR-TRANSFORM LLM BASELINES START (seeds={seeds}) ===")
    out = run_baselines(
        ROOT / "configs/lsr_transform_formulas.json",
        OUT, call_fn,
        task_ids=None, seeds=seeds,
        methods=["direct_llm", "llm_sr"], llmsr_iterations=10, log_fn=log,
    )
    (OUT / "ALL_DONE.txt").write_text("done\n", encoding="utf-8")
    log(f"=== LSR-TRANSFORM LLM BASELINES COMPLETE: {out['n_runs']} runs ===")


if __name__ == "__main__":
    main()
