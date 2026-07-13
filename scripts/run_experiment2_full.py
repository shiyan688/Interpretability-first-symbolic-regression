#!/usr/bin/env python3
"""Confirmatory experiment 2 driver (run inside tmux).

Phases (each resumable / incremental):
  1. SR over all regressible tasks x seeds (train-only fit, val select, test once)
  2. Build blind rating candidates from selected formulas
  3. Blind export (anonymized manifest + isolated private map)
  4. Two independent DeepSeek judges (chat + reasoner), cached/resumable
  5. Aggregate ratings (separate human/LLM groups, unblinded by method)

Writes progress markers to outputs/experiment2_full/PROGRESS.log.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path("/public/home/wangyg/Interpretability-first-symbolic-regression")
OUT = ROOT / "outputs" / "experiment2_full"
OUT.mkdir(parents=True, exist_ok=True)
PROG = OUT / "PROGRESS.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with PROG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def phase1_sr() -> None:
    from factor_pysr_llm.experiment2 import run_task

    catalog = json.loads((ROOT / "configs/experiment2_formulas.json").read_text(encoding="utf-8"))
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
    log(f"PHASE1 start: {len(formulas)} tasks x {len(seeds)} seeds = {total} runs; {len(done)} already done")
    for f in formulas:
        for seed in seeds:
            if (f["task_id"], seed) in done:
                continue
            t0 = time.time()
            try:
                r = run_task(f, gen, budget, seed)
                r["status"] = "success"
            except Exception as exc:
                import traceback
                r = {"task_id": f["task_id"], "seed": seed, "status": "error",
                     "error": repr(exc), "traceback": traceback.format_exc()}
            r["wall_time_seconds"] = round(time.time() - t0, 1)
            results.append(r)
            results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            ok = sum(1 for x in results if x.get("status") == "success")
            log(f"  {f['task_id']} seed={seed} -> {r['status']} ({r['wall_time_seconds']}s) [{len(results)}/{total}, ok={ok}]")
    log(f"PHASE1 done: {len(results)} runs")


def phase2_build_ratings() -> None:
    from factor_pysr_llm.experiment2 import _rename_variables

    catalog = json.loads((ROOT / "configs/experiment2_formulas.json").read_text(encoding="utf-8"))
    byid = {f["task_id"]: f for f in catalog["formulas"]}
    res = json.loads((OUT / "experiment2_results.json").read_text(encoding="utf-8"))
    # one rating item per (task, condition) using the SEED-0 (first) run to avoid
    # over-weighting; use best (median) — here we take the first successful seed.
    seen = set()
    cands = []
    for x in sorted(res, key=lambda r: (r.get("task_id", ""), r.get("seed", 0))):
        if x.get("status") != "success":
            continue
        if x["task_id"] in seen:
            continue
        seen.add(x["task_id"])
        f, _ = _rename_variables(byid[x["task_id"]])
        vardefs = [{"name": v["name"], "definition": v.get("meaning", ""), "unit": "-",
                    "allowed_range": f"[{v['low']},{v['high']}]"} for v in f["variables"]]
        for name, c in x["conditions"].items():
            if not c.get("expression"):
                continue
            formula = c.get("expanded_expression") or c["expression"]
            cands.append({
                "formula": formula,
                "variables": vardefs,
                "task_context": f"Recover a scientific relationship for target y. Domain: {byid[x['task_id']].get('name','')}.",
                "dataset_label": f"task_{x['task_id'][:3]}",
                "method": name, "seed": x["seed"], "r2_test": c.get("r2_test"),
                "dataset": x["task_id"], "target": "y",
            })
    (OUT / "rating_candidates.json").write_text(json.dumps(cands, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"PHASE2 done: {len(cands)} rating candidates ({len(seen)} tasks x 3 conditions)")


def phase3_export() -> None:
    from factor_pysr_llm.interpretability_eval import export_blind_ratings

    cands = json.loads((OUT / "rating_candidates.json").read_text(encoding="utf-8"))
    info = export_blind_ratings(cands, OUT / "rating_manifest.jsonl", OUT / "rating_private_map.json", seed=20260709)
    log(f"PHASE3 done: exported {info['n_items']} blind items")


def phase4_judges() -> None:
    from factor_pysr_llm.interpretability_eval import run_llm_judge
    from factor_pysr_llm.model_api import call_openai_compatible

    def make_call(cfg_path):
        def call(prompt):
            return str(call_openai_compatible(Path(cfg_path), prompt)["content"])
        return call

    man = OUT / "rating_manifest.jsonl"
    rub = ROOT / "configs/interpretability_rubric.json"
    jdir = OUT / "judge"
    for cfg, model_id, seed in [
        (ROOT / "configs/llm_provider.judge_a.json", "deepseek_chat", 1),
        (ROOT / "configs/llm_provider.judge_b.json", "deepseek_reasoner", 999),
    ]:
        log(f"PHASE4 judge {model_id} start")
        out = run_llm_judge(man, rub, jdir, make_call(cfg), model_id, temperature=0.0, seed=seed, max_retries=3, resume=True)
        log(f"PHASE4 judge {model_id}: rated={out['summary']['n_rated']} errors={out['summary']['n_errors']}")


def phase5_aggregate() -> None:
    from factor_pysr_llm.rating_aggregate import aggregate_ratings

    jdir = OUT / "judge"
    report = aggregate_ratings(
        human_csv=None,
        llm_result_paths={
            "deepseek_chat": jdir / "judge_deepseek_chat_seed1.jsonl",
            "deepseek_reasoner": jdir / "judge_deepseek_reasoner_seed999.jsonl",
        },
        private_map_path=OUT / "rating_private_map.json",
        out_path=OUT / "rating_aggregate.json",
    )
    for g, gd in report["groups"].items():
        log(f"PHASE5 {g}: overall={gd['overall']['mean']:.3f} CI={[round(x,3) for x in gd['overall']['ci95']]}")
    for k, v in report.get("unblinded", {}).items():
        for method, md in v.items():
            log(f"PHASE5 {k}.{method}: n={md['n']} mean={md['overall_mean']:.3f}")


def main() -> None:
    log("=== CONFIRMATORY RUN START ===")
    phase1_sr()
    phase2_build_ratings()
    phase3_export()
    phase4_judges()
    phase5_aggregate()
    (OUT / "ALL_DONE.txt").write_text("done\n", encoding="utf-8")
    log("=== CONFIRMATORY RUN COMPLETE ===")


if __name__ == "__main__":
    main()
