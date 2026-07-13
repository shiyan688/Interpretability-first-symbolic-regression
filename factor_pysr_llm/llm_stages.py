from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import WorkflowConfig
from .factor_miner import _corr_score
from .features import safe_read_csv
from .reports import read_feature_dir


def _top_correlations(feature_dir: Path, top_k: int) -> pd.DataFrame:
    X, y = read_feature_dir(feature_dir)
    rows = []
    for col in X.columns:
        score, corr = _corr_score(X[col].to_numpy(dtype=float), y)
        rows.append({"feature": col, "abs_corr": score, "signed_corr": corr})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).head(top_k)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                values.append(f"{val:.6g}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_factor_proposal_prompt(
    cfg: WorkflowConfig,
    target: str,
    feature_dir: Path | None = None,
    top_k: int = 80,
) -> Path:
    feature_dir = feature_dir or (cfg.output_root / "feature_tables" / target)
    out_dir = cfg.output_root / "llm_prompts" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = feature_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    corr = _top_correlations(feature_dir, top_k)
    corr.to_csv(out_dir / "top_raw_feature_correlations.csv", index=False)
    template = {
        "target": target,
        "proposed_factors": [
            {
                "name": "short_factor_name",
                "expression": "(raw_feature_a / raw_feature_b)",
                "meaning": "domain meaning",
                "dimension_note": "dimensionally consistent or screening only",
                "priority": 1,
            }
        ],
        "preferred_raw_features": ["raw_feature_a"],
        "avoid": ["leakage variables or dimensionally invalid final formula terms"],
    }
    (out_dir / "factor_proposals_template.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    prompt = f"""# Factor Proposal Prompt

目标：`{target}`

你需要基于数据集性质提出候选特性因子，供后续 Python SISSO-like 枚举和 PySR 使用。

## 数据集概况

- feature_dir: `{feature_dir}`
- n_rows: `{manifest.get("n_rows", "NA")}`
- n_features: `{manifest.get("n_features", "NA")}`
- raw linear_r2: `{manifest.get("linear_r2", "NA")}`

## 相关性最高的原始变量

{_markdown_table(corr)}

## 允许输出

请返回 JSON，格式见 `factor_proposals_template.json`。要求：

1. 只使用上表或 manifest 中存在的安全特征名。
2. 表达式只用 `+ - * / abs square sqrt_abs log_abs inv` 的低阶组合。
3. 区分两类因子：可解释最终式因子、仅用于筛选/搜索的中间因子。
4. 对每个因子说明物理/化学/结构含义、量纲是否合理、优先级。
5. 不要引入 target 或 target 的变形，避免泄漏。
"""
    out_path = out_dir / "factor_proposal_prompt.md"
    out_path.write_text(prompt, encoding="utf-8")
    return out_path


def _factor_variable_map(expression: str) -> str:
    from .expr import identifier_names

    return ", ".join(sorted(identifier_names(str(expression)))) or "(none)"


def write_factor_selection_prompt(
    cfg: WorkflowConfig,
    target: str,
    factor_pool_dir: Path | None = None,
    top_k: int = 200,
    batch_size: int | None = None,
) -> Path:
    factor_pool_dir = factor_pool_dir or (cfg.output_root / "factor_pools" / target)
    factors = safe_read_csv(factor_pool_dir / "mined_factors.csv")
    ranked = factors.sort_values("score_abs_corr", ascending=False).reset_index(drop=True)
    top = ranked.head(top_k).copy()
    out_dir = cfg.output_root / "llm_prompts" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_dir / "top_mined_factors_for_llm.csv", index=False)

    # Build the candidate table that is actually EMBEDDED in the prompt so a
    # remote LLM (which cannot read local files) sees the real content.
    display = top.copy()
    display["variables"] = display["expression"].map(_factor_variable_map)
    table_cols = [c for c in ("factor_name", "expression", "variables", "score_abs_corr", "order", "source") if c in display.columns]
    embedded_table = _markdown_table(display[table_cols])

    # Explicit batching rule when the pool is large.
    n = len(top)
    batch_note = ""
    if batch_size and n > batch_size:
        n_batches = (n + batch_size - 1) // batch_size
        batch_note = (
            f"\n> 因子较多（{n} 个）。请按每批 {batch_size} 个分 {n_batches} 批评估，"
            f"最终合并 selected_factors；不要因为超长而丢弃后面批次。\n"
        )

    template = {
        "target": target,
        "selected_factors": [
            {
                "factor_name": "factor_000001",
                "reason": "why this factor is meaningful",
                "final_formula_allowed": True,
            }
        ],
        "notes": "Prefer meaningful and dimensionally defensible factors, but keep a few strong screening factors.",
    }
    (out_dir / "factor_selection_template.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    prompt = f"""# Factor Selection Prompt

目标：`{target}`

下面是 Python SISSO-like 枚举后按 `abs(corr(factor, y))`（仅 train 行计算）排序的
因子池 top-{n}。请从中选择进入 PySR 的因子，兼顾 R2 潜力和解释性。
{batch_note}
## 候选因子（已内嵌，请只从下表中选择 factor_name）

{embedded_table}

## 选择规则

1. 优先选择表达式短、变量含义明确、量纲关系可解释的因子。
2. 保留少量相关性很高但解释性一般的因子，作为 PySR 搜索辅助。
3. 用 `final_formula_allowed` 明确标记哪些因子允许进入最终解释公式；
   只有 `final_formula_allowed: true` 的因子才会被视为可进入最终公式的领域因子。
4. `factor_name` 必须精确匹配上表中的值；未知名称会被判为匹配失败。

## 输出 schema（严格 JSON）

```json
{{
  "target": "{target}",
  "selected_factors": [
    {{"factor_name": "<上表中的 factor_name>", "reason": "<简短理由>", "final_formula_allowed": true}}
  ]
}}
```

只返回 JSON，不要额外文字。
"""
    out_path = out_dir / "factor_selection_prompt.md"
    out_path.write_text(prompt, encoding="utf-8")
    return out_path


def write_interpretability_prompt(
    cfg: WorkflowConfig,
    target: str,
    result_path: Path | None = None,
    expr_list_path: Path | None = None,
    top_k: int = 20,
) -> Path:
    out_dir = cfg.output_root / "llm_prompts" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    formulas: list[dict[str, Any]] = []
    if result_path and result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        formulas.append(
            {
                "source": str(result_path),
                "r2": data.get("best_r2", data.get("r2_verified")),
                "rmse": data.get("best_rmse", data.get("rmse_verified")),
                "expression": data.get("best_equation", data.get("expression", "")),
            }
        )
    if expr_list_path and expr_list_path.exists():
        df = safe_read_csv(expr_list_path)
        if "target" in df.columns:
            df = df[df["target"].astype(str).eq(target)]
        if "r2" in df.columns:
            df["_r2"] = pd.to_numeric(df["r2"], errors="coerce")
            df = df.sort_values("_r2", ascending=False)
        for _, row in df.head(top_k).iterrows():
            formulas.append(
                {
                    "source": row.get("source_file", str(expr_list_path)),
                    "r2": row.get("r2"),
                    "rmse": row.get("rmse"),
                    "expression": row.get("expression"),
                }
            )
    formulas_path = out_dir / "formulas_for_interpretability.json"
    formulas_path.write_text(json.dumps(formulas, indent=2, ensure_ascii=False), encoding="utf-8")
    prompt = f"""# Interpretability Enhancement Prompt

目标：`{target}`

你需要对 R2 表现较好的符号回归公式做解释性增强。

## 输入公式

- formulas json: `{formulas_path}`

## 任务

1. 挑出 R2 不错且结构较短的公式。
2. 将 `mine_factor_*` 或 meta-factor 展开回原始变量表达式。
3. 对表达式做结构性经验润色：合并同类结构、量纲对齐、去除明显无意义的嵌套。
4. 给出重新拟合常数后的候选表达式。
5. 每个润色候选都必须回到数值验证：train R2、K-fold R2、RMSE。

输出 JSON：`candidate_name / original_expression / refined_expression / interpretation / expected_tradeoff / verification_needed`。
"""
    out_path = out_dir / "interpretability_prompt.md"
    out_path.write_text(prompt, encoding="utf-8")
    return out_path
