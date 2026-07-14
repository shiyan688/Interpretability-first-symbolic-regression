# LLM 符号回归方法对比报告（进行中）

更新日期：2026-07-13
数据：`examples/实验2_科学公式数据集清单.md`（29 个可回归真式）+ LSR-Transform 本地重建（10 个抗记忆任务）
评测：同一 DeepSeek API、同一无泄漏协议（train 拟合 / validation 选式 / test 只读一次）、同一 ExprSim + 盲评 rubric。

> 运行产物在 `outputs/`（gitignore，不入库）。本报告随运行更新。

## 一、对比的方法

| 方法 | 类型 | 说明 |
|---|---|---|
| `raw_pysr` | 数值 SR | 标准 PySR（原始变量，最优 validation R² 选择） |
| `mine_pysr` | 数值 SR | PySR + 通用因子池（成对积/商/平方/倒数，不含真值） |
| `if_sr` | 本方法 | 在 raw+mined 候选上做 interpretability-first 选择（δ=0.02） |
| `direct_llm` | LLM | 零样本：给变量+数据摘要，让 DeepSeek 直接给闭式表达式 |
| `llm_sr` | LLM | 复现 LLM-SR（Shojaee et al. ICLR 2025）：LLM 提 skeleton → 最小二乘拟合常数 → 演化 buffer 迭代 |

`llm_sr` 与 `direct_llm` 用 DeepSeek（deepseek-chat）作 SR 引擎；两个 judge（deepseek-chat + deepseek-reasoner）只看匿名公式，不知道方法来源。

## 二、LLM 方法在标准数据集上的结果（29 任务 × 3 seeds = 87 次，已完成）

| 方法 | test R² 中位 | test R² 均值 | ExprSim 均值 | 代数等价率 |
|---|---|---|---|---|
| direct_llm | 0.644 | −2.157 | 0.737 | **0.31** |
| llm_sr | **0.997** | 0.783 | 0.651 | 0.01 |

读法：
- `direct_llm` **代数等价率高达 31%**——它在很大程度上是**背诵**教科书公式（库仑、万有引力、米氏方程等），一旦背错就灾难性失败（均值 R² 为负）。
- `llm_sr` 数据拟合稳定（中位 R² 0.997），但**几乎不恢复真实结构**（等价率 1%），且表达式冗长（平均 32.6 节点）——靠加参数拟合。

## 三、LSR-Transform 抗记忆任务上的结果（10 任务 × 3 seeds = 30 次，已完成 LLM 部分）

把教科书公式改写成**非常规目标变量**形式（如理想气体解出 V、开普勒解出 a 的立方根），破坏 LLM 记忆。

| 方法 | test R² 中位 | ExprSim 均值 | 代数等价率 |
|---|---|---|---|
| direct_llm | **−2.572** | 0.542 | **0.10** |
| llm_sr | 0.826 | 0.493 | 0.00 |

**关键发现**：`direct_llm` 的代数等价率从 31% **崩到 10%**，中位 R² 从 0.644 崩到 −2.572——证明它此前的优势主要来自**记忆而非推理**。这正是 LLM-SRBench 设计要暴露的现象。`llm_sr` 退化更平缓（中位 R² 0.826），因为它靠数据拟合而非背诵。

## 四、与 LLM-SRBench 的关系（诚实说明）

- LLM-SRBench 官方数据集（`nnheui/llm-srbench`，239 题）在 HuggingFace 上是**受限（gated）**的，本环境无法访问 GitHub 或授权下载（只有 pip 镜像、hf-mirror 基址和 DeepSeek API 可达）。因此**无法运行官方 239 题基准**。
- 本报告用 `configs/lsr_transform_formulas.json` **忠实复现 LSR-Transform 的方法学思想**（把常见定律改写为不常见等价表示以抗记忆），并用相同 API/评测/协议运行，可作为方法学层面的对照。
- LLM-SRBench 论文公开参考数：目前最强系统 symbolic accuracy 仅 **31.5%**，佐证"精确恢复很难、记忆会虚高指标"这一结论，与本文观测一致（外部引用，非本环境复现）。

## 五、已完成（本节原为"待补"，现全部跑完）

全部 402 次运行成功、0 失败。完整的五方法 × 三数据集对比表、统一盲评与结论见
**`docs/final_comparison_report.md`（完整实验报告）**。要点：

- 标准 29 公式、LSR-Transform 10 任务、真实 LLM-SRBench 28 任务的 IF-SR/PySR 三条件均已完成。
- 双 DeepSeek 判官盲评：IF-SR 可解释性排第一（2.76 / 2.79）。
- IF-SR 在三数据集上"R² 基本不降、公式最短"；Direct-LLM 靠记忆（抗记忆任务崩溃）；
  LLM-SR 拟合强但冗长。
