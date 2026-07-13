#!/usr/bin/env python3
"""PySR conditions (raw_pysr, mine_pysr, if_sr) on REAL LLM-SRBench tasks."""
from __future__ import annotations
import json, time
from pathlib import Path
from factor_pysr_llm.llmsrbench_runner import run_llmsrbench_task
ROOT=Path("/public/home/wangyg/Interpretability-first-symbolic-regression")
OUT=ROOT/"outputs/llmsrbench_pysr"; OUT.mkdir(parents=True, exist_ok=True)
PROG=OUT/"PROGRESS.log"
def log(m):
    line=f"[{time.strftime('%H:%M:%S')}] {m}"; PROG.open("a").write(line+"\n"); print(line,flush=True)
def main():
    dirs=json.loads((ROOT/"outputs/llmsrbench_data/task_dirs.json").read_text())
    seeds=[20260709,20260710,20260711]
    budget={"niterations":60,"timeout_seconds":120,"delta":0.02}
    rp=OUT/"llmsrbench_pysr_results.json"
    done={}
    if rp.exists():
        for r in json.loads(rp.read_text()): done[(r.get("task_id"),r.get("seed"))]=r
    results=list(done.values())
    total=len(dirs)*len(seeds)
    log(f"=== LLMSRBench PySR START: {total} runs, {len(done)} done ===")
    for d in dirs:
        for seed in seeds:
            # skip if done (by task dir name proxy via load); run then check
            t0=time.time()
            try:
                res=run_llmsrbench_task(d, None, seed, ["raw_pysr","mine_pysr","if_sr"], budget, log_fn=None)
                if (res["task_id"],seed) in done: 
                    continue
                res["status"]="success"
            except Exception as e:
                import traceback; res={"task_id":d.split("/")[-1],"seed":seed,"status":"error","error":repr(e),"traceback":traceback.format_exc()}
            res["wall_time_seconds"]=round(time.time()-t0,1)
            results.append(res); rp.write_text(json.dumps(results,indent=2,ensure_ascii=False))
            log(f"  {res.get('task_id','?')[:45]} seed={seed} -> {res.get('status')} ({res['wall_time_seconds']}s) [{len(results)}/{total}]")
    (OUT/"ALL_DONE.txt").write_text("done"); log("=== DONE ===")
if __name__=="__main__": main()
