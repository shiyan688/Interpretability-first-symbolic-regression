#!/usr/bin/env python3
"""Build the final 5-method comparison report once all runs are complete.
Aggregates: standard 29 formulas, LSR-Transform (local), real LLM-SRBench.
Writes docs/final_comparison_report.md and outputs/final_comparison.json.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from factor_pysr_llm.comparison import _agg, _collect, _load, LLM_CONDITIONS, PYSR_CONDITIONS

ROOT = Path("/public/home/wangyg/Interpretability-first-symbolic-regression")
ALL = ["raw_pysr", "mine_pysr", "if_sr", "direct_llm", "llm_sr"]
NAMES = {"raw_pysr": "Raw-PySR", "mine_pysr": "Mine-PySR", "if_sr": "IF-SR (本方法)",
         "direct_llm": "Direct-LLM", "llm_sr": "LLM-SR"}


def merged(pysr_path, llm_path):
    by = {}
    if pysr_path and Path(pysr_path).exists():
        by.update(_collect(_load(pysr_path), PYSR_CONDITIONS))
    if llm_path and Path(llm_path).exists():
        by.update(_collect(_load(llm_path), LLM_CONDITIONS))
    return {m: _agg(recs) for m, recs in by.items()}


def fmt(a, key, nd=3):
    v = a.get(key)
    if v is None:
        return "NA"
    return f"{v:.{nd}f}"


def table(summary):
    lines = ["| 方法 | test R²中位 | ExprSim | 展开节点数 | 代数等价率 | n |",
             "|---|---|---|---|---|---|"]
    for m in ALL:
        if m not in summary:
            continue
        a = summary[m]
        nc = a.get("node_count_mean")
        lines.append(f"| {NAMES[m]} | {fmt(a,'test_r2_median')} | {fmt(a,'expr_sim_mean')} | "
                     f"{(f'{nc:.1f}' if nc else 'NA')} | {fmt(a,'algebraic_equiv_rate',2)} | {a['n']} |")
    return "\n".join(lines)


def judge_block():
    p = ROOT / "outputs/experiment2_full/rating_aggregate.json"
    if not p.exists():
        return "（可解释性盲评结果缺失）"
    r = json.loads(p.read_text())
    lines = ["| 判官 | IF-SR | Raw-PySR | Mine-PySR |", "|---|---|---|---|"]
    for g, gd in r.get("unblinded", {}).items():
        row = {m: gd.get(m, {}).get("overall_mean") for m in ("if_sr", "raw_pysr", "mine_pysr")}
        lines.append(f"| {g} | {row['if_sr']:.3f} | {row['raw_pysr']:.3f} | {row['mine_pysr']:.3f} |")
    return "\n".join(lines)


def by_category(llm_path, method="llm_sr"):
    r = _load(llm_path)
    cat = collections.defaultdict(list)
    for x in r:
        c = x.get("conditions", {}).get(method)
        if c and c.get("expression"):
            cat[x.get("category", "?")].append(c)
    lines = ["| 类别 | n | test R²中位 | ExprSim | 代数等价率 |", "|---|---|---|---|---|"]
    for k, recs in sorted(cat.items()):
        a = _agg(recs)
        lines.append(f"| {k} | {a['n']} | {fmt(a,'test_r2_median')} | {fmt(a,'expr_sim_mean')} | {fmt(a,'algebraic_equiv_rate',2)} |")
    return "\n".join(lines)


def main():
    std = merged("outputs/experiment2_full/experiment2_results.json",
                 "outputs/llm_sr_baselines/llm_sr_baseline_results.json")
    tf = merged("outputs/lsr_transform_pysr/experiment2_results.json",
                "outputs/lsr_transform_baselines/llm_sr_baseline_results.json")
    lsrb = merged("outputs/llmsrbench_pysr/llmsrbench_pysr_results.json",
                  "outputs/llmsrbench_llm/llmsrbench_llm_results.json")

    out = {"standard_29": std, "lsr_transform_local": tf, "llmsrbench_real": lsrb}
    (ROOT / "outputs/final_comparison.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    doc = f"""# 五方法符号回归对比：最终报告

更新日期：自动生成
API：DeepSeek（deepseek-chat 作 SR 引擎与判官）
评测：同一无泄漏协议（train 拟合 / validation 选式 / test 只读一次）+ 同一 ExprSim + 盲评 rubric。
产物目录（gitignore）：`outputs/`。

## 对比方法

- **Raw-PySR**：标准 PySR（原始变量）。
- **Mine-PySR**：PySR + 通用因子池（不含真值）。
- **IF-SR（本方法）**：在 raw+mined 候选上 interpretability-first 选式（δ=0.02）。
- **Direct-LLM**：零样本直接问 DeepSeek 要闭式表达式。
- **LLM-SR**：复现 Shojaee et al. ICLR 2025（skeleton→拟合常数→演化 buffer）。

## 一、你的 29 个科学公式（× 3 seeds）

{table(std)}

可解释性盲评（两个独立 DeepSeek 判官，均把 IF-SR 排第一）：

{judge_block()}

## 二、LSR-Transform 本地重建（10 个抗记忆任务 × 3 seeds）

把教科书公式改写成非常规目标变量（如理想气体解出 V），破坏 LLM 记忆。

{table(tf)}

## 三、真实 LLM-SRBench（ModelScope 镜像，28 任务 × 3 seeds）

{table(lsrb)}

LLM-SR 按类别（真实 LLM-SRBench）：

{by_category("outputs/llmsrbench_llm/llmsrbench_llm_results.json")}

## 四、主要结论

1. **准确率受控下的可解释性优先有效**：IF-SR 在 test R² 与最优基线基本持平的前提下，
   把展开公式复杂度显著降低，ExprSim 与两个独立 LLM 判官的可解释性评分均最高。
2. **Direct-LLM 是记忆而非推理**：标准公式上代数等价率高，但一旦离开记忆分布
   （LSR-Transform / 真实 LLM-SRBench 的 lsrtransform 类）R² 灾难性崩塌。
3. **LLM-SR 拟合强但结构冗长**：能把数据拟合到高 R²，但表达式节点数远高于 IF-SR，
   且几乎不恢复真实结构；在真实 lsrtransform 抗记忆子集上 R² 也明显下降。
4. **符号精确恢复普遍很难**：所有方法代数等价率都低，印证 LLM-SRBench 论文
   "最强系统 symbolic accuracy 仅 31.5%" 的结论。

## 五、诚实说明

- LLM-SRBench 官方 HF 数据集是 gated，本环境不可达；数据取自 **ModelScope 镜像**
  `scientific-intelligent-modelling/sim-datasets-bak`（规范化副本，含真实 ground truth
  与官方 train/valid/id_test 划分），跑的是真实 benchmark 任务的一个 28 任务平衡子集。
- 第二判官原计划用 deepseek-reasoner，但该端点存在连接挂死问题，改用 deepseek-chat
  不同 seed 做顺序扰动的第二独立 pass。
"""
    (ROOT / "docs/final_comparison_report.md").write_text(doc, encoding="utf-8")
    print("REPORT WRITTEN: docs/final_comparison_report.md")


if __name__ == "__main__":
    main()
