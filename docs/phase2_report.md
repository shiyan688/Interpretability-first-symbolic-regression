# 阶段二报告：IF-SR 与可解释性评测基础设施

更新日期：2026-07-13
对应任务来源：`docs/first_three_weeks_execution_prompt_zh.md` 第五节。

## 本阶段结论：通过

阶段二实现了公式 lineage/展开/逆标准化、IF-SR selector、修复了 LLM factor prompt 与
selection no-op，并交付了完整的可解释性盲评导出 + judge + 聚合设施（用固定假响应即可
跑通 prompt → parse → aggregate 全链路，不依赖真实 API）。

## 已完成

### 任务 1：公式 lineage、展开与逆标准化（factor_pysr_llm/lineage.py）

- `FactorCard`：schema 含 `factor_id/name/expression/variables/meaning/unit_status/
  source/approved_for_final_formula`，`unit_status` 与 `source` 取值受校验。
- `expand_expression`：递归展开 nested factors，`_stack` 检测循环引用并报错。
- `complexity_stats`：展开后计算 node count、depth、变量数、常数数；短别名无法逃避
  展开后复杂度（有测试证明）。
- `substitute_standardized_variables`：把 z-score 特征改写为 `((f-mean)/scale)` 原始变量式。
- `inverse_standardize`：z 预测乘回 `y_scale` 加回 `y_mean`。
- `check_numeric_consistency`：验证展开前后数值预测一致（`max_abs_diff`）。
- `canonicalize`：基于 sympy 的基础规范化。

### 任务 2：IF-SR selector（factor_pysr_llm/ifsr_selector.py）

固定词典序规则：validation 容忍带（`R²_val ≥ best - 0.02`）→ 剔除硬违规（泄漏变量、
非法/无定义表达式）→ 最小化展开复杂度 → 优先 approved domain factor → 跨 seed 稳定性 →
固定 ID tie-break。选择过程**不读取 test 指标**；保存完整 trace（每候选保留/剔除原因、
ranking、threshold）。

### 任务 3：修复 LLM factor prompt 与 selection no-op（llm_stages.py, factor_miner.py）

- `write_factor_selection_prompt` 现在把候选因子表（factor_name/expression/变量/相关性/
  source）**直接嵌入 prompt 正文**，含明确 top-k、可选分批规则、JSON schema 与
  `final_formula_allowed` 说明；不再只给本地文件路径。
- `build_pysr_pool` 新增 `llm_authoritative`（默认 True）模式：提供 selection 时候选池
  由 LLM 选择驱动，可配 `llm_fallback_top_k` 少量 corr 兜底，不再和全量 corr top-k union。
- `final_formula_allowed` 写入 `selected_mined_factors.csv` 并统计入 manifest。
- 相关性排序在有 split 时只用 train 行（读取 `row_roles.csv`）。

### 任务 4 & 5：可解释性盲评、judge 与聚合

- `factor_pysr_llm/interpretability_eval.py`：
  - `export_blind_ratings`：随机化匿名 `item_id`、导出 rating manifest（jsonl）、
    隔离的 private mapping（method/seed/r2）、人类评分 CSV 模板；防御性检查禁止私有字段泄漏。
  - `build_judge_prompt`：嵌入公式、变量字典、rubric 四维锚点、输出 schema。
  - `parse_rating_response`：严格校验（缺字段、越界、非整数、item_id 不匹配、code fence
    均拒绝，不静默补默认分）。
  - `run_llm_judge`：注入 `call_fn`（可用假响应测试）、缓存/断点续跑、有上限重试、
    seed 驱动顺序扰动、保存 prompt hash/model ID/原始记录/错误日志与 summary。
  - `score_interpretability_prompt`：导出每题内嵌 prompt。
- `configs/interpretability_rubric.json`：四维 1–5 分，每分含文字锚点，含 output_schema。
- `examples/rating_manifest.example.jsonl`：3 条匿名样例。
- `factor_pysr_llm/rating_aggregate.py`：人类 CSV 导入；人类/LLM-A/LLM-B **分开**汇总；
  每维均值 + bootstrap 95% CI；评分者一致性；LLM–human Spearman 与成对偏好一致率；
  private map 解盲后 by-method 汇总；**不**把人类分与 LLM 分平均成一个总分。
- CLI：`export-blind-ratings`、`score-interpretability-prompt`、`run-llm-judge`、
  `aggregate-ratings`。

### 附加（阶段三前置）

- `factor_pysr_llm/expression_similarity.py` + `expression-similarity` CLI（见阶段三）。
- `factor_pysr_llm/known_formulas.py` + `sample-known-formulas` CLI：确定性分层抽取，
  已产出 20 个任务 ID。

## 验证证据

```bash
VP=/public/home/wangyg/workspace/llm_pysr_project/.venv/bin/python
$VP -m pytest -q      # 59 passed
```

关键测试：
- `tests/test_lineage.py`（7）：展开、循环检测、数值一致、逆标准化、别名不逃避复杂度。
- `tests/test_ifsr_selector.py`（7）：δ 边界、test R² 不影响选择、短别名、硬违规优先、
  domain 因子 tie-break、ID tie-break 可复现、全违规返回 None。
- `tests/test_llm_selection.py`（3）：prompt 嵌入因子内容、不同 selection → 不同候选池、
  `final_formula_allowed` 被记录。
- `tests/test_interpretability_eval.py`（13）：盲评隐藏私有字段、确定性、prompt 嵌入、
  schema 校验各异常、假响应跑通、错误记录、断点续跑、顺序扰动。
- `tests/test_rating_aggregate.py`（5）：均值/CI、一致性、Spearman、人类与 LLM 分开、解盲。

最小端到端产物：

```bash
$VP -m factor_pysr_llm.cli score-interpretability-prompt \
    --manifest examples/rating_manifest.example.jsonl \
    --rubric configs/interpretability_rubric.json --output /tmp/prompts.json   # n_prompts=3
```

## 仍存在的风险

- **真实 judge 尚未跑**：`run_llm_judge` 用假响应全链路验证，但真实 provider 的 token/
  成本/稳定性需在阶段三 pilot 用小规模真实调用测算。
- **canonicalization 深度有限**：`canonicalize`/tree 相似度依赖 sympy `simplify`，对极复杂
  表达式可能超时或不收敛；已在 expression_similarity 中加异常回退。
- **domain factor 冻结**：`approved_for_final_formula` 机制已就位，但真实领域因子卡片需
  至少 1 位领域人员在阶段三审查冻结。

## 下一步（阶段三）

1. 用 2 真实 + 4 真式开发任务准备 pilot 数据、split、domain card 与运行配置。
2. 跑通内部四条件（Raw-PySR / Mine-PySR / IF-SR w/o domain / IF-SR），确认消融非 no-op。
3. 跑通 candidate archive → blind export → 假/真 judge → aggregate 全链路 pilot。
4. 成本测算并生成 `configs/frozen_protocol.json` 与 `docs/pilot_report.md`。
