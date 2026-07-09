from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import WorkflowConfig


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _source_counts(manifest: dict[str, Any]) -> pd.Series:
    source_map = manifest.get("feature_source", {})
    if not isinstance(source_map, dict) or not source_map:
        return pd.Series(dtype=int)
    return pd.Series(source_map).value_counts()


def write_llm_brief(
    cfg: WorkflowConfig,
    target: str,
    feature_dir: Path | None = None,
    run_dir: Path | None = None,
) -> Path:
    feature_dir = feature_dir or (cfg.output_root / "feature_tables" / target)
    manifest = _load_json(feature_dir / "manifest.json")
    best = _load_json(run_dir / "best_result.json") if run_dir else {}
    out_dir = cfg.output_root / "llm_briefs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target}_brief.md"

    counts = _source_counts(manifest)
    source_lines = "\n".join(f"- {idx}: {int(val)}" for idx, val in counts.items()) or "- 暂无 source map"
    best_expr = str(best.get("best_equation", "")).strip()
    best_r2 = best.get("best_r2", best.get("r2_verified", "NA"))

    text = f"""# LLM Strategy Brief: {target}

## 当前输入空间

- input_csv: `{cfg.input_csv}`
- feature_dir: `{feature_dir}`
- n_rows: `{manifest.get("n_rows", "NA")}`
- n_features: `{manifest.get("n_features", "NA")}`
- linear_r2: `{manifest.get("linear_r2", "NA")}`
- linear_rmse: `{manifest.get("linear_rmse", "NA")}`

## 特征来源计数

{source_lines}

## 当前 PySR 最佳

- run_dir: `{run_dir if run_dir else "NA"}`
- best_r2: `{best_r2}`
- best_equation: `{best_expr if best_expr else "NA"}`

## 给 LLM 的任务

请基于上述输入空间和已有最佳式子，提出下一轮可解释性因子与搜索约束：

1. 从已有高分式子里提取可解释的结构因子，优先考虑吸附能、键长、配位/几何、电子结构变量的低阶组合。
2. 标记哪些 meta-factor 必须展开回原始输入，哪些可以作为筛选用中间变量但不能进入最终论文表达式。
3. 给出 10-30 个候选经验因子，说明化学含义、量纲关系、可能适用目标。
4. 给出 PySR 搜索建议：maxsize、算符、是否需要除法/abs/inv、是否增加原始特征比例。
5. 对最佳式子提出结构性润色方案：量纲对齐、合并同类项、常数重新拟合、K 折验证。

## 硬性约束

- 所有建议必须回到数值验证：训练 R2、K 折 R2、RMSE。
- 最终解释公式不允许残留不可展开的 `*_best_pred` 或 equation snapshot meta-factor。
- 如果高 R2 依赖 meta-factor，必须先展开，再重新拟合常数。
"""
    out_path.write_text(text, encoding="utf-8")
    return out_path

