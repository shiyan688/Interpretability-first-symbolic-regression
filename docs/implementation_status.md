# 仓库审计与实现状态

更新日期：2026-07-13
审计范围：`factor_pysr_llm` 包、`configs/`、`examples/`、`tests/`，对照
`docs/first_three_weeks_execution_prompt_zh.md` 的验收项。

本文件是活文档。每完成一个阶段就在对应小节更新“当前状态”。

## 0. 审计方法

- 逐文件通读 `cli.py`、`dataset.py`、`features.py`、`factor_miner.py`、
  `llm_stages.py`、`model_api.py`、`pysr_runner.py`、`reports.py`、`expr.py`、
  `config.py`。
- 运行现有测试基线：`pytest -q` → 6 passed（审计时）。
- 依据可复现证据回答审计问题，不依赖文档口径。

## 1. 逐项审计结论（审计基线：阶段一实现之前）

### 1.1 数据何时划分 train/validation/test

**审计基线：不划分。** `dataset.build_raw_feature_table` 在完整表上删除
target 缺失行后直接构造特征、拟合缩放、计算 `linear_r2`
（`dataset.py:212-308`）。没有任何 train/validation/test 概念，也没有
split manifest。`features.build_union`、`factor_miner.mine_factors`、
`build_pysr_pool` 全部在整表上运行。

**阶段一后：** 新增 `factor_pysr_llm/splits.py` 与 CLI `generate-split`，
产生带 row-id 和 SHA256 的 JSON manifest；支持普通与 group split，并有
互斥/覆盖/组不跨集检查。预处理与因子挖掘可接收 `role_mask` 只在 train 上 fit。

### 1.2 imputation/scaling/相关筛选/因子挖掘是否只在 train 上 fit

**审计基线：否。** `fill_numeric`、`scale_values`
（`dataset.py:142-167`）在全表上计算 median/mean/std；
`apply_feature_selection`（`dataset.py:184-209`）用全表 `feature_scores`
选 top-k；`mine_factors_from_frame`（`factor_miner.py:158-264`）用全部
label `y` 计算相关性并做 beam 搜索。全部存在泄漏。

**阶段一后：** 新增 `factor_pysr_llm/preprocess.py`，实现
`fit`（仅 train 行）+ `transform`（应用到全部行），保存 fitted state。
`build_raw_feature_table` 增加可选 `split_manifest` 参数走 train-fit 路径。
因子挖掘的相关性排序通过 train mask 计算。

### 1.3 PySR 候选是否在 validation 上选择、test 是否只评价一次

**审计基线：无此机制。** `pysr_runner.run_pysr`（`pysr_runner.py:42-145`）
在传入的整表上 fit 并用同一表算 `best_r2`。没有 acceptable set、没有
validation 选择、没有 test 一次性评价。

**阶段二后：** 新增 `factor_pysr_llm/ifsr_selector.py` 实现
`δ=0.02` 容忍带 + 词典序选择，只读 validation 指标，选定后才允许读 test。

### 1.4 mined/domain factor 是否保存完整 expression lineage

**审计基线：部分。** `mined_factors.csv` 保存 `expression`（基于 safe
feature name 的字符串，如 `(raw_x1 * raw_x2)`）、`order`、`source`
（`factor_miner.py:145-155`）。但没有 `factor_id`、`meaning`、
`unit_status`、`approved_for_final_formula`，也没有递归展开到原始变量、
循环检测或逆标准化。

**阶段二后：** 新增 `factor_pysr_llm/lineage.py`：factor card schema、
递归展开、循环检测、展开到原始列名、逆标准化、复杂度统计与数值一致性检查。

### 1.5 最终公式能否展开到原始变量并逆标准化

**审计基线：否。** `expr.eval_expr`（`expr.py:67-73`）能在给定特征表上求值，
但特征已 z-score 化，公式停留在 `raw_*` / `mine_*` 层，没有回到原始物理
变量，也没有乘回 `x_scale` 加回 `x_mean`。`llm-interpret` 只生成让模型
自行展开的 prompt 文本，无程序化保证。

**阶段二后：** `lineage.py` 提供 `expand_to_raw` 与 `inverse_standardize`，
并有单测验证展开前后数值预测一致。

### 1.6 LLM factor-selection prompt 是否实际嵌入候选内容

**审计基线：否（关键缺陷）。** `write_factor_selection_prompt`
（`llm_stages.py:103-151`）把 top 因子写到旁边的 CSV，但 prompt 正文只
给出**文件路径**（`top_mined_factors_for_llm.csv`），没有把因子表、表达式、
变量元数据嵌入正文。远端 LLM 看不到本地文件，无法真正选择。
`write_factor_proposal_prompt` 好一些（内嵌相关性表格）。

**阶段二后：** 重写为把候选因子表（factor_id/expression/变量/相关性）直接
嵌入 prompt 正文，并加入明确 top-k/分批规则和 JSON schema 说明。

### 1.7 LLM selection 是否真正改变候选池，还是与 top-k union 后接近 no-op

**审计基线：接近 no-op。** `build_pysr_pool`（`factor_miner.py:419-506`）
先无条件把相关性 top-k（`factor_top_k`，默认 200）全部加入
`selected_names`，再把 LLM 选择的因子 union 进去
（`factor_miner.py:450-461`）。当因子池 ≤200 时，corr top-k 已包含几乎所有
因子，LLM 选择实际不改变候选池。`final_formula_allowed` 字段在
`_read_llm_factor_selection`（`factor_miner.py:393-416`）中被完全忽略。

**阶段二后：** 新增 LLM-authoritative 选择模式：当提供 selection 时，候选池
由 LLM 选择驱动（可配 fallback 数量），`final_formula_allowed` 写入
selected factor 元数据并影响最终资格；加测试证明不同 selection → 不同候选池。

### 1.8 是否存在可解释性 judge、盲评导出和 rating aggregation

**审计基线：完全没有。** 仓库只有 `llm-interpret`（生成公式解释/改写
prompt，`llm_stages.py:154-214`）。没有盲评导出、匿名映射、rating manifest、
judge prompt、1–5 分 schema 校验、多模型批量调用、聚合统计。
`llm-interpret` 不是可解释性评测器。

**阶段二后：** 新增 `interpretability_eval.py`（盲评导出 + judge prompt +
schema 校验 + 假响应可跑通）、`rating_aggregate.py`（人类/LLM 分开聚合、
bootstrap CI、一致性、Spearman）、`configs/interpretability_rubric.json`、
`examples/rating_manifest.example.jsonl` 及 CLI。

### 1.9 当前测试覆盖哪些功能、哪些论文关键功能没有测试

**审计基线覆盖：**
- `test_dataset_grammar.py`：raw feature table 构建与命名。
- `test_factor_miner.py`：因子挖掘、proposal、selection prompt 生成、pool 构建。
- `test_model_api.py`：JSON 提取、provider config 读取。
- `test_expr_smoke.py`：表达式求值 smoke。

**审计基线未覆盖（论文关键）：** split 可复现与 group 不交叉、无泄漏
（改 test label 不变候选）、公式展开/逆标准化数值一致、IF-SR selector 边界、
LLM selection 非 no-op、盲评导出匿名性、judge schema 校验、rating 聚合统计、
表达式相似度。这些在阶段一/二/三逐步补齐。

## 2. 已知事实确认

`llm-interpret` 是公式解释/改写 prompt 生成器，**不是**论文需要的可解释性
评测 judge。本项目按此前提实现独立评测设施，不复用 `llm-interpret`。

## 3. 阶段进度总览

| 阶段 | 目标 | 状态 |
|---|---|---|
| 审计 | implementation_status.md | 完成 |
| 阶段一 | 范围冻结 + 无泄漏管线 + 泄漏测试 | 见 docs/phase1_report.md |
| 阶段二 | IF-SR + 可解释性评测设施 | 见 docs/phase2_report.md |
| 阶段三 | 端到端 pilot + 协议冻结 | 见 docs/pilot_report.md |

## 4. 环境说明

- 依赖 numpy/pandas/scikit-learn/pyyaml/sympy/scipy。
- 解释器：`/public/home/wangyg/workspace/llm_pysr_project/.venv/bin/python`（含全部依赖）。
- 测试命令：`<venv>/bin/python -m pytest -q`。
- 本机无 PySR、无真实 LLM API；所有测试不依赖 PySR 或真实 API，使用固定假响应与
  toy/合成数据。

## 5. 三阶段完成状态（2026-07-13）

三阶段代码与文档已交付，`pytest -q` 全绿（63 passed）。

新增模块：
- `factor_pysr_llm/splits.py`、`preprocess.py`（阶段一：无泄漏 split + train-fit）。
- `factor_pysr_llm/lineage.py`、`ifsr_selector.py`（阶段二：展开/逆标准化 + IF-SR selector）。
- `factor_pysr_llm/interpretability_eval.py`、`rating_aggregate.py`（阶段二：盲评 + judge + 聚合）。
- `factor_pysr_llm/expression_similarity.py`、`known_formulas.py`、`known_data.py`、`pilot.py`
  （阶段三：ExprSim + 分层抽取 + 数据生成 + pilot）。

新增配置/示例：
- `configs/paper_scope.yaml`、`dataset_inventory.yaml`、`known_formula_tasks.yaml`、
  `interpretability_rubric.json`、`frozen_protocol.json`、`examples/rating_manifest.example.jsonl`。

新增 CLI：`generate-split`、`export-blind-ratings`、`score-interpretability-prompt`、
`run-llm-judge`、`aggregate-ratings`、`expression-similarity`、`sample-known-formulas`。

新增测试：`test_splits.py`、`test_no_leakage.py`、`test_lineage.py`、`test_ifsr_selector.py`、
`test_llm_selection.py`、`test_interpretability_eval.py`、`test_rating_aggregate.py`、
`test_expression_similarity.py`、`test_pilot.py`。

阶段报告：`docs/phase1_report.md`、`phase2_report.md`、`pilot_report.md`。

剩余前置项（需真实资源，不改变管线结构）：真实催化数据接入、PySR 引擎替换 surrogate、
真实 LLM judge 与 rubric 校准。详见 `docs/pilot_report.md` 风险与下一步。
