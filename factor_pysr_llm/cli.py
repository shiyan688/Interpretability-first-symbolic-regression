from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import WorkflowConfig
from .dataset import build_raw_feature_table, inspect_dataset
from .factor_miner import build_pysr_pool, mine_factors
from .features import build_union
from .llm import write_llm_brief
from .llm_stages import (
    write_factor_proposal_prompt,
    write_factor_selection_prompt,
    write_interpretability_prompt,
)
from .mining import mine_expression_list
from .model_api import call_prompt_file
from .pysr_runner import run_pysr
from .reports import verify_expression, verify_hof, verify_result, write_verify_json
from .splits import build_split_manifest, save_split_manifest


def _add_common_config(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="Workflow JSON config.")


def _targets_from_args(cfg: WorkflowConfig, args: argparse.Namespace) -> list[str]:
    values = []
    if getattr(args, "target", None):
        values.append(args.target)
    if getattr(args, "targets", None):
        values.extend(args.targets)
    if not values:
        values.extend(cfg.targets)
    if not values:
        dataset_cfg = dict(cfg.data.get("dataset") or {})
        values.extend(dataset_cfg.get("targets", dataset_cfg.get("target_columns", [])))
    if not values:
        raise SystemExit("No target specified and config has no targets.")
    return list(dict.fromkeys(str(x) for x in values))


def cmd_build_union(args: argparse.Namespace) -> None:
    cfg = WorkflowConfig.from_json(args.config)
    rows = []
    for target in _targets_from_args(cfg, args):
        row = build_union(cfg, target)
        rows.append(row)
        print(
            f"{target}: feature_dir={row['feature_dir']} "
            f"n_features={row['n_features']} linear_r2={row['linear_r2']:.6f}"
        )
    summary_path = cfg.output_root / "feature_union_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {summary_path}")


def cmd_inspect_dataset(args: argparse.Namespace) -> None:
    cfg = WorkflowConfig.from_json(args.config)
    info = inspect_dataset(cfg)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_build_raw(args: argparse.Namespace) -> None:
    cfg = WorkflowConfig.from_json(args.config)
    split_manifest = getattr(args, "split_manifest", None)
    split_path = Path(split_manifest).expanduser() if split_manifest else None
    rows = []
    for target in _targets_from_args(cfg, args):
        row = build_raw_feature_table(cfg, target, split_manifest_path=split_path)
        rows.append(row)
        print(
            f"{target}: feature_dir={row['feature_dir']} rows={row['n_rows']} "
            f"n_features={row['n_features']} linear_r2={row['linear_r2']:.6f}"
        )
    summary_path = cfg.output_root / "raw_feature_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {summary_path}")


def _override_pysr_options(base: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    opts = dict(base)
    for key in (
        "maxsize",
        "procs",
        "population_size",
        "populations",
        "timeout_seconds",
        "niterations",
        "parsimony",
        "seed",
        "progress",
    ):
        val = getattr(args, key, None)
        if val is not None:
            opts[key] = val
    if args.tmp_root:
        opts["pysr_output_dir"] = str(Path(args.tmp_root).expanduser() / args.run_name / args.target)
    return opts


def cmd_run_pysr(args: argparse.Namespace) -> None:
    cfg = WorkflowConfig.from_json(args.config)
    opts = _override_pysr_options(cfg.pysr_options(), args)
    if args.feature_dir:
        feature_dir = Path(args.feature_dir).expanduser()
    else:
        pool_dir = cfg.output_root / "feature_tables" / f"{args.target}__pysr_pool"
        raw_dir = cfg.output_root / "feature_tables" / args.target
        feature_dir = pool_dir if pool_dir.exists() else raw_dir
    run_dir = cfg.output_root / "runs" / args.run_name / args.target
    result = run_pysr(feature_dir, run_dir, opts, args.target)
    print(
        f"{args.target}: status={result.get('status')} "
        f"best_r2={result.get('best_r2')} run_dir={run_dir}"
    )


def cmd_verify(args: argparse.Namespace) -> None:
    feature_dir = Path(args.feature_dir).expanduser()
    outputs = []
    if args.expression:
        outputs.append(verify_expression(feature_dir, args.expression))
    if args.result:
        outputs.append(verify_result(feature_dir, Path(args.result).expanduser()))
    if args.hof:
        outputs.append(verify_hof(feature_dir, Path(args.hof).expanduser()))
    if not outputs:
        raise SystemExit("verify requires --expression, --result, or --hof")
    data: Any = outputs[0] if len(outputs) == 1 else outputs
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if args.output:
        write_verify_json(Path(args.output).expanduser(), {"results": outputs} if len(outputs) > 1 else outputs[0])


def cmd_llm_brief(args: argparse.Namespace) -> None:
    cfg = WorkflowConfig.from_json(args.config)
    feature_dir = Path(args.feature_dir).expanduser() if args.feature_dir else None
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else None
    out = write_llm_brief(cfg, args.target, feature_dir=feature_dir, run_dir=run_dir)
    print(f"wrote {out}")


def cmd_mine_exprs(args: argparse.Namespace) -> None:
    roots = [Path(x).expanduser() for x in args.roots]
    df = mine_expression_list(
        roots=roots,
        output_path=Path(args.output).expanduser(),
        targets=args.targets,
        top_k_per_target=args.top_k_per_target,
    )
    print(f"wrote {args.output} rows={len(df)} targets={df['target'].nunique() if not df.empty else 0}")


def cmd_llm_propose_factors(args: argparse.Namespace) -> None:
    cfg = WorkflowConfig.from_json(args.config)
    feature_dir = Path(args.feature_dir).expanduser() if args.feature_dir else None
    out = write_factor_proposal_prompt(cfg, args.target, feature_dir=feature_dir, top_k=args.top_k)
    print(f"wrote {out}")


def cmd_mine_factors(args: argparse.Namespace) -> None:
    cfg = WorkflowConfig.from_json(args.config)
    rows = []
    for target in _targets_from_args(cfg, args):
        feature_dir = Path(args.feature_dir).expanduser() if args.feature_dir else None
        llm_proposals = Path(args.llm_proposals).expanduser() if args.llm_proposals else None
        row = mine_factors(cfg, target, feature_dir=feature_dir, llm_proposals_path=llm_proposals)
        rows.append(row)
        print(
            f"{target}: factor_pool_dir={row['factor_pool_dir']} "
            f"n_factors={row['n_factors']} best_abs_corr={row['best_abs_corr']:.6f}"
        )
    out = cfg.output_root / "factor_mining_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


def cmd_llm_select_factors(args: argparse.Namespace) -> None:
    cfg = WorkflowConfig.from_json(args.config)
    factor_pool_dir = Path(args.factor_pool_dir).expanduser() if args.factor_pool_dir else None
    out = write_factor_selection_prompt(cfg, args.target, factor_pool_dir=factor_pool_dir, top_k=args.top_k)
    print(f"wrote {out}")


def cmd_build_pysr_pool(args: argparse.Namespace) -> None:
    cfg = WorkflowConfig.from_json(args.config)
    raw_feature_dir = Path(args.raw_feature_dir).expanduser() if args.raw_feature_dir else None
    factor_pool_dir = Path(args.factor_pool_dir).expanduser() if args.factor_pool_dir else None
    llm_selection = Path(args.llm_selection).expanduser() if args.llm_selection else None
    row = build_pysr_pool(
        cfg,
        args.target,
        raw_feature_dir=raw_feature_dir,
        factor_pool_dir=factor_pool_dir,
        llm_selection_path=llm_selection,
        output_tag=args.output_tag,
    )
    print(
        f"{args.target}: feature_dir={row['feature_dir']} "
        f"n_features={row['n_features']} linear_r2={row['linear_r2']:.6f}"
    )


def cmd_llm_interpret(args: argparse.Namespace) -> None:
    cfg = WorkflowConfig.from_json(args.config)
    result_path = Path(args.result).expanduser() if args.result else None
    expr_list = Path(args.expr_list).expanduser() if args.expr_list else None
    out = write_interpretability_prompt(
        cfg,
        args.target,
        result_path=result_path,
        expr_list_path=expr_list,
        top_k=args.top_k,
    )
    print(f"wrote {out}")


def cmd_export_blind_ratings(args: argparse.Namespace) -> None:
    from .interpretability_eval import export_blind_ratings

    candidates = json.loads(Path(args.candidates).expanduser().read_text(encoding="utf-8"))
    if isinstance(candidates, dict):
        candidates = candidates.get("candidates", candidates.get("items", []))
    info = export_blind_ratings(
        candidates,
        out_manifest=Path(args.out_manifest).expanduser(),
        out_private_map=Path(args.out_private_map).expanduser(),
        seed=args.seed,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))


def cmd_score_interpretability_prompt(args: argparse.Namespace) -> None:
    from .interpretability_eval import score_interpretability_prompt

    info = score_interpretability_prompt(
        manifest_path=Path(args.manifest).expanduser(),
        rubric_path=Path(args.rubric).expanduser(),
        out_path=Path(args.output).expanduser(),
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))


def cmd_run_llm_judge(args: argparse.Namespace) -> None:
    from .interpretability_eval import build_judge_prompt, load_rubric, run_llm_judge
    from .model_api import call_openai_compatible

    provider_path = Path(args.provider_config).expanduser()

    def call_fn(prompt: str) -> str:
        result = call_openai_compatible(provider_path, prompt)
        return str(result["content"])

    out = run_llm_judge(
        manifest_path=Path(args.manifest).expanduser(),
        rubric_path=Path(args.rubric).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        call_fn=call_fn,
        model_id=args.model_id,
        temperature=args.temperature,
        seed=args.seed,
        max_retries=args.max_retries,
        resume=not args.no_resume,
    )
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))


def cmd_aggregate_ratings(args: argparse.Namespace) -> None:
    from .rating_aggregate import aggregate_ratings

    llm_paths = {}
    for spec in args.llm_results or []:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--llm-results expects label=path, got {spec!r}")
        llm_paths[label] = Path(path).expanduser()
    report = aggregate_ratings(
        human_csv=Path(args.human_csv).expanduser() if args.human_csv else None,
        llm_result_paths=llm_paths,
        private_map_path=Path(args.private_map).expanduser() if args.private_map else None,
        out_path=Path(args.output).expanduser() if args.output else None,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_expression_similarity(args: argparse.Namespace) -> None:
    from .expression_similarity import expression_similarity_report

    report = expression_similarity_report(
        predicted=args.predicted,
        truth=args.truth,
        variables=args.variables or None,
        seed=args.seed,
        n_points=args.n_points,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_sample_known_formulas(args: argparse.Namespace) -> None:
    from .known_formulas import sample_known_formula_tasks

    info = sample_known_formula_tasks(
        config_path=Path(args.config).expanduser(),
        output_path=Path(args.output).expanduser() if args.output else None,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))


def cmd_generate_split(args: argparse.Namespace) -> None:
    cfg = WorkflowConfig.from_json(args.config)
    fractions = (args.train_fraction, args.validation_fraction, args.test_fraction)
    manifest = build_split_manifest(
        cfg,
        args.target,
        mode=args.mode,
        seed=args.seed,
        fractions=fractions,
        id_column=args.id_column,
        group_column=args.group_column,
    )
    if args.output:
        out_path = Path(args.output).expanduser()
    else:
        out_path = cfg.output_root / "splits" / f"{args.target}__{args.mode}_seed{args.seed}.json"
    payload = save_split_manifest(manifest, out_path)
    print(
        f"{args.target}: mode={manifest.mode} n_total={manifest.n_total} "
        f"train={payload['n_train']} val={payload['n_validation']} test={payload['n_test']} "
        f"sha256={payload['sha256'][:12]} -> {out_path}"
    )


def cmd_llm_call(args: argparse.Namespace) -> None:
    result = call_prompt_file(
        provider_config_path=Path(args.provider_config).expanduser(),
        prompt_file=Path(args.prompt_file).expanduser(),
        output_path=Path(args.output).expanduser(),
        content_only=bool(args.content_only),
        extract_json=bool(args.extract_json),
        system_prompt=args.system_prompt,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="factor-pysr-llm")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("inspect-dataset", help="Inspect dataset grammar resolution for a config.")
    _add_common_config(i)
    i.add_argument("--output")
    i.set_defaults(func=cmd_inspect_dataset)

    raw = sub.add_parser("build-raw", help="Build raw feature tables from a generic CSV dataset grammar.")
    _add_common_config(raw)
    raw.add_argument("--target")
    raw.add_argument("--targets", nargs="*")
    raw.add_argument("--split-manifest", help="Split manifest JSON for no-leakage train-fit preprocessing.")
    raw.set_defaults(func=cmd_build_raw)

    gs = sub.add_parser("generate-split", help="Generate a reproducible train/validation/test split manifest.")
    _add_common_config(gs)
    gs.add_argument("--target", required=True)
    gs.add_argument("--mode", choices=["random", "group"], default="random")
    gs.add_argument("--seed", type=int, default=20260709)
    gs.add_argument("--train-fraction", type=float, default=0.6, dest="train_fraction")
    gs.add_argument("--validation-fraction", type=float, default=0.2, dest="validation_fraction")
    gs.add_argument("--test-fraction", type=float, default=0.2, dest="test_fraction")
    gs.add_argument("--id-column", dest="id_column")
    gs.add_argument("--group-column", dest="group_column")
    gs.add_argument("--output")
    gs.set_defaults(func=cmd_generate_split)

    lp = sub.add_parser("llm-propose-factors", help="Write prompt/template for LLM feature-factor proposal.")
    _add_common_config(lp)
    lp.add_argument("--target", required=True)
    lp.add_argument("--feature-dir")
    lp.add_argument("--top-k", type=int, default=80)
    lp.set_defaults(func=cmd_llm_propose_factors)

    mf = sub.add_parser("mine-factors", help="Python SISSO-like factor enumeration over a prepared feature table.")
    _add_common_config(mf)
    mf.add_argument("--target")
    mf.add_argument("--targets", nargs="*")
    mf.add_argument("--feature-dir")
    mf.add_argument("--llm-proposals")
    mf.set_defaults(func=cmd_mine_factors)

    ls = sub.add_parser("llm-select-factors", help="Write prompt/template for LLM selection from mined factor pool.")
    _add_common_config(ls)
    ls.add_argument("--target", required=True)
    ls.add_argument("--factor-pool-dir")
    ls.add_argument("--top-k", type=int, default=200)
    ls.set_defaults(func=cmd_llm_select_factors)

    bp = sub.add_parser("build-pysr-pool", help="Combine raw variables and selected/mined factors for PySR.")
    _add_common_config(bp)
    bp.add_argument("--target", required=True)
    bp.add_argument("--raw-feature-dir")
    bp.add_argument("--factor-pool-dir")
    bp.add_argument("--llm-selection")
    bp.add_argument("--output-tag", default="pysr_pool")
    bp.set_defaults(func=cmd_build_pysr_pool)

    b = sub.add_parser("build-union", help="Build effective-union feature table from historical factors.")
    _add_common_config(b)
    b.add_argument("--target")
    b.add_argument("--targets", nargs="*")
    b.set_defaults(func=cmd_build_union)

    r = sub.add_parser("run-pysr", help="Run PySR on a prepared feature table.")
    _add_common_config(r)
    r.add_argument("--target", required=True)
    r.add_argument("--run-name", required=True)
    r.add_argument("--feature-dir")
    r.add_argument("--maxsize", type=int)
    r.add_argument("--procs", type=int)
    r.add_argument("--population-size", type=int, dest="population_size")
    r.add_argument("--populations", type=int)
    r.add_argument("--timeout-seconds", type=int, dest="timeout_seconds")
    r.add_argument("--niterations", type=int)
    r.add_argument("--parsimony", type=float)
    r.add_argument("--seed", type=int)
    r.add_argument("--progress", action="store_true")
    r.add_argument("--tmp-root", default="")
    r.set_defaults(func=cmd_run_pysr)

    v = sub.add_parser("verify", help="Recompute R2/RMSE for a result, HOF, or expression.")
    v.add_argument("--feature-dir", required=True)
    v.add_argument("--result")
    v.add_argument("--hof")
    v.add_argument("--expression")
    v.add_argument("--output")
    v.set_defaults(func=cmd_verify)

    l = sub.add_parser("llm-brief", help="Write a structured brief for LLM-assisted next-factor design.")
    _add_common_config(l)
    l.add_argument("--target", required=True)
    l.add_argument("--feature-dir")
    l.add_argument("--run-dir")
    l.set_defaults(func=cmd_llm_brief)

    interp = sub.add_parser("llm-interpret", help="Write prompt for formula interpretability enhancement.")
    _add_common_config(interp)
    interp.add_argument("--target", required=True)
    interp.add_argument("--result")
    interp.add_argument("--expr-list")
    interp.add_argument("--top-k", type=int, default=20)
    interp.set_defaults(func=cmd_llm_interpret)

    lc = sub.add_parser("llm-call", help="Call an OpenAI-compatible chat API with a prompt file.")
    lc.add_argument("--provider-config", default="configs/llm_provider.local.json")
    lc.add_argument("--prompt-file", required=True)
    lc.add_argument("--output", required=True)
    lc.add_argument("--content-only", action="store_true")
    lc.add_argument("--extract-json", action="store_true")
    lc.add_argument("--system-prompt")
    lc.set_defaults(func=cmd_llm_call)

    m = sub.add_parser("mine-exprs", help="Collect best/HOF expressions across result folders.")
    m.add_argument("--roots", nargs="+", required=True)
    m.add_argument("--output", required=True)
    m.add_argument("--targets", nargs="*")
    m.add_argument("--top-k-per-target", type=int, default=None)
    m.set_defaults(func=cmd_mine_exprs)

    ebr = sub.add_parser("export-blind-ratings", help="Export anonymized blind rating manifest + private map.")
    ebr.add_argument("--candidates", required=True, help="JSON list of candidates with formula/variables.")
    ebr.add_argument("--out-manifest", required=True)
    ebr.add_argument("--out-private-map", required=True)
    ebr.add_argument("--seed", type=int, default=20260709)
    ebr.set_defaults(func=cmd_export_blind_ratings)

    sip = sub.add_parser("score-interpretability-prompt", help="Write embedded judge prompts for a rating manifest.")
    sip.add_argument("--manifest", required=True)
    sip.add_argument("--rubric", default="configs/interpretability_rubric.json")
    sip.add_argument("--output", required=True)
    sip.set_defaults(func=cmd_score_interpretability_prompt)

    rlj = sub.add_parser("run-llm-judge", help="Run an LLM interpretability judge over a rating manifest.")
    rlj.add_argument("--manifest", required=True)
    rlj.add_argument("--rubric", default="configs/interpretability_rubric.json")
    rlj.add_argument("--out-dir", required=True)
    rlj.add_argument("--provider-config", default="configs/llm_provider.local.json")
    rlj.add_argument("--model-id", required=True)
    rlj.add_argument("--temperature", type=float, default=0.0)
    rlj.add_argument("--seed", type=int, default=20260709)
    rlj.add_argument("--max-retries", type=int, default=3)
    rlj.add_argument("--no-resume", action="store_true")
    rlj.set_defaults(func=cmd_run_llm_judge)

    agg = sub.add_parser("aggregate-ratings", help="Aggregate human and LLM interpretability ratings.")
    agg.add_argument("--human-csv")
    agg.add_argument("--llm-results", nargs="*", help="label=path.jsonl entries (e.g. llm_a=judge.jsonl).")
    agg.add_argument("--private-map")
    agg.add_argument("--output")
    agg.add_argument("--seed", type=int, default=20260709)
    agg.set_defaults(func=cmd_aggregate_ratings)

    es = sub.add_parser("expression-similarity", help="Compute frozen ExprSim between predicted and true formula.")
    es.add_argument("--predicted", required=True)
    es.add_argument("--truth", required=True)
    es.add_argument("--variables", nargs="*", help="Variable names to sample over.")
    es.add_argument("--n-points", type=int, default=400)
    es.add_argument("--seed", type=int, default=20260709)
    es.add_argument("--output")
    es.set_defaults(func=cmd_expression_similarity)

    skf = sub.add_parser("sample-known-formulas", help="Deterministically sample known-formula tasks (experiment 2).")
    skf.add_argument("--config", default="configs/known_formula_tasks.yaml")
    skf.add_argument("--output")
    skf.set_defaults(func=cmd_sample_known_formulas)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
