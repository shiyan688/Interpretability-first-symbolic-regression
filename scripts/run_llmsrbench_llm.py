#!/usr/bin/env python3
"""LLM methods (direct_llm + llm_sr) on REAL LLM-SRBench tasks (ModelScope mirror)."""
from __future__ import annotations
import json, time
from pathlib import Path
from factor_pysr_llm.llmsrbench_runner import run_llmsrbench_task
from factor_pysr_llm.model_api import call_openai_compatible

ROOT = Path("/public/home/wangyg/Interpretability-first-symbolic-regression")
OUT = ROOT / "outputs" / "llmsrbench_llm"
OUT.mkdir(parents=True, exist_ok=True)
PROG = OUT / "PROGRESS.log"
PROVIDER = ROOT / "configs/llm_provider.judge_a.json"

def log(m):
    line=f"[{time.strftime('%H:%M:%S')}] {m}"
    PROG.open("a").write(line+"\n"); print(line, flush=True)

def call_fn(p): return str(call_openai_compatible(PROVIDER, p)["content"])

def main():
    dirs = json.loads((ROOT/"outputs/llmsrbench_data/task_dirs.json").read_text())
    seeds = [20260709, 20260710, 20260711]
    budget = {"delta": 0.02}
    rp = OUT / "llmsrbench_llm_results.json"
    done = {}
    if rp.exists():
        for r in json.loads(rp.read_text()): done[(r["task_id"], r["seed"])]=r
    results = list(done.values())
    log(f"=== LLMSRBench LLM start: {len(dirs)} tasks x {len(seeds)} seeds ===")
    for d in dirs:
        for seed in seeds:
            t0=time.time()
            try:
                res = run_llmsrbench_task(d, call_fn, seed, ["direct_llm","llm_sr"], budget, llmsr_iterations=10, log_fn=None)
                if (res["task_id"], seed) in done: continue
                res["status"]="success"
            except Exception as e:
                import traceback; res={"task_id":d.split("/")[-1],"seed":seed,"status":"error","error":repr(e),"traceback":traceback.format_exc()}
            res["wall_time_seconds"]=round(time.time()-t0,1)
            results.append(res); rp.write_text(json.dumps(results,indent=2,ensure_ascii=False))
            log(f"  {res.get('task_id','?')[:40]} seed={seed} -> {res.get('status')} ({res['wall_time_seconds']}s)")
    (OUT/"ALL_DONE.txt").write_text("done"); log("=== DONE ===")

if __name__=="__main__": main()
