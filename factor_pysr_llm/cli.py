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
from .pysr_runner import run_pysr
from .reports import verify_expression, verify_hof, verify_result, write_verify_json


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
    rows = []
    for target in _targets_from_args(cfg, args):
        row = build_raw_feature_table(cfg, target)
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
    feature_dir = Path(args.feature_dir).expanduser() if args.feature_dir else cfg.output_root / "feature_tables" / args.target
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
    raw.set_defaults(func=cmd_build_raw)

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

    m = sub.add_parser("mine-exprs", help="Collect best/HOF expressions across result folders.")
    m.add_argument("--roots", nargs="+", required=True)
    m.add_argument("--output", required=True)
    m.add_argument("--targets", nargs="*")
    m.add_argument("--top-k-per-target", type=int, default=None)
    m.set_defaults(func=cmd_mine_exprs)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
