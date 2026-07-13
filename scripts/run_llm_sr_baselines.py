#!/usr/bin/env python3
"""Run LLM-SR-style baselines (DirectLLM + LLM-SR) on the experiment-2 tasks,
using the same DeepSeek API, no-leakage protocol and ExprSim evaluation as
IF-SR/PySR. API-bound, so it can run alongside the PySR confirmatory job.
"""
from __future__ import annotations

import time
from pathlib import Path

from factor_pysr_llm.llm_sr_runner import run_baselines
from factor_pysr_llm.model_api import call_openai_compatible

ROOT = Path("/public/home/wangyg/Interpretability-first-symbolic-regression")
OUT = ROOT / "outputs" / "llm_sr_baselines"
OUT.mkdir(parents=True, exist_ok=True)
PROG = OUT / "PROGRESS.log"

# use deepseek-chat as the SR engine LLM (a DIFFERENT role from the judges;
# judges remain deepseek-chat + reasoner but never rate their own generations
# because generation identity is blinded and the judge only sees formulas).
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
    log(f"=== LLM-SR BASELINES START (seeds={seeds}) ===")
    out = run_baselines(
        ROOT / "configs/experiment2_formulas.json",
        OUT,
        call_fn,
        task_ids=None,        # all regressible tasks
        seeds=seeds,
        methods=["direct_llm", "llm_sr"],
        llmsr_iterations=10,
        log_fn=log,
    )
    (OUT / "ALL_DONE.txt").write_text("done\n", encoding="utf-8")
    log(f"=== LLM-SR BASELINES COMPLETE: {out['n_runs']} runs ===")


if __name__ == "__main__":
    main()
