#!/usr/bin/env python3
"""Run PySR-based conditions (raw_pysr, mine_pysr, if_sr) on the LSR-Transform
reconstruction tasks, so PySR/IF-SR can be compared to LLM-SR baselines on
memorization-resistant problems. Same harness as experiment 2.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from factor_pysr_llm.experiment2 import run_task

ROOT = Path("/public/home/wangyg/Interpretability-first-symbolic-regression")
OUT = ROOT / "outputs" / "lsr_transform_pysr"
OUT.mkdir(parents=True, exist_ok=True)
PROG = OUT / "PROGRESS.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with PROG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def main() -> None:
    catalog = json.loads((ROOT / "configs/lsr_transform_formulas.json").read_text(encoding="utf-8"))
    gen = catalog["meta"]["data_generation"]
    budget = {"niterations": 60, "timeout_seconds": 120, "delta": 0.02}
    seeds = [20260709, 20260710, 20260711]
    formulas = [f for f in catalog["formulas"] if f.get("regressible", True)]
    results_path = OUT / "experiment2_results.json"
    done = {}
    if results_path.exists():
        for r in json.loads(results_path.read_text(encoding="utf-8")):
            done[(r["task_id"], r["seed"])] = r
    results = list(done.values())
    total = len(formulas) * len(seeds)
    log(f"=== LSR-TRANSFORM PySR START: {total} runs, {len(done)} done ===")
    for f in formulas:
        for seed in seeds:
            if (f["task_id"], seed) in done:
                continue
            t0 = time.time()
            try:
                r = run_task(f, gen, budget, seed); r["status"] = "success"
            except Exception as exc:
                import traceback
                r = {"task_id": f["task_id"], "seed": seed, "status": "error",
                     "error": repr(exc), "traceback": traceback.format_exc()}
            r["wall_time_seconds"] = round(time.time() - t0, 1)
            results.append(r)
            results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"  {f['task_id']} seed={seed} -> {r['status']} ({r['wall_time_seconds']}s) [{len(results)}/{total}]")
    (OUT / "ALL_DONE.txt").write_text("done\n", encoding="utf-8")
    log("=== LSR-TRANSFORM PySR COMPLETE ===")


if __name__ == "__main__":
    main()
