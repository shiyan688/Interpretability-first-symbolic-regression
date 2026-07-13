# 阶段三报告：端到端 Pilot 与协议冻结

更新日期：2026-07-13
对应任务来源：`docs/first_three_weeks_execution_prompt_zh.md` 第六节。

## 本阶段结论：部分通过（管线 gate 全绿；真实数据前置项待补）

阶段三在**开发用合成数据**上把完整链路跑通，并冻结了确认性协议。所有可在无真实
数据、无 PySR、无真实 LLM API 情况下验证的 gate 均已通过。三个前置项（真实催化数据、
真实 PySR 引擎、真实 judge 模型）在获得资源后按冻结协议接入，不改变管线结构。

> 重要：本阶段所有实验结果均由 `development_only` 的 surrogate SR 引擎产生，
> 严禁进入确认性主表。

## 已完成

### 任务 1：pilot 数据准备

- `factor_pysr_llm/known_data.py`：把已知真式渲染为 train/validation/test CSV +
  **独立** ExprSim 采样点（noise-free，且与训练点独立生成），主实验设置为
  `5% target noise + 5 无关变量`。写出 `predefined_roles.csv` 供构建匹配的 split manifest。
- pilot 任务：1 个真式开发任务（`v1*(v2+v3)`）跑全链路；另加 1 个专门的
  selection-divergence 任务（`v1 + 0.05*v2*v3`）证明选择规则消融非 no-op。

### 任务 2：四条件端到端（factor_pysr_llm/pilot.py）

- `pilot_conditions` 跑通 `raw_pysr / mine_pysr / if_sr_no_domain / if_sr`，通过列可用性
  区分候选池（raw / +mined / +approved domain）。
- `ablation_is_nontrivial` 同时检查**候选池差异**与**选择规则差异**。
- SR 引擎通过 `search_fn` 注入；缺省是 `development_only` 的线性可加 surrogate，
  真实运行替换为 PySR。

Pilot 观测（surrogate）：
- 任务 A 四条件选择：raw=`v1 + v3 + v2`，mine=`v1 * mine_sum`，if_no_domain=`v1 * mine_sum`，
  if_sr=`domain_prod`（raw/mine/domain 三者互异）。
- 任务 B 证明同一候选池下 accuracy-first 与 interpretability-first **分歧**：
  mine_pysr 选 `v1 + mine_prod`（最高 val R²），IF-no-domain 选 `v1`（容忍带内最简）。

### 任务 3：可解释性评测 pilot 全链路

`candidate archive → blind export → LLM-A judge → 顺序扰动 → LLM-B judge →
schema validation → aggregate-ratings → reliability report` 用固定假响应跑通：
- 盲评导出隐藏 method/seed/R²（隔离 private map）；
- 两个假 judge 用不同 seed 做顺序扰动，零解析错误；
- 聚合分别汇总 LLM-A/LLM-B，输出 by-method 解盲。

（真实 rubric 校准需 8–12 个非主结果公式 + 至少 1 位领域评分者，属阶段三真实数据环节。）

### 任务 4：表达式相似度（factor_pysr_llm/expression_similarity.py）

冻结 `ExprSim = 0.50 数值 + 0.20 变量F1 + 0.20 算子F1 + 0.10 树结构`；单独报告代数等价、
数值等价、support F1；独立采样点不复用训练样本；权重写入 `configs/paper_scope.yaml` 与
`configs/frozen_protocol.json`。pilot 中选中公式展开后 ExprSim = 1.0（成功恢复真式）。
单测覆盖等价/近似/错误/变量重命名/缺失/无定义点/交换律。

### 任务 5：成本测算与协议冻结

- `configs/frozen_protocol.json`：数据任务 ID、split 方案与 hash 占位、方法/消融、SR 超参与
  预算、seeds、`δ`、算子集合、factor/domain card 版本、rubric 版本、judge 配置、ExprSim 权重、
  失败/排除规则、统计方案、代码 commit hash（`cad97e7`）。
- 20 个真式任务 ID 已由确定性抽取器冻结写入。

## 阶段三 Gate 核对

- [x] 无泄漏测试通过（`tests/test_no_leakage.py`）。
- [x] 公式展开和逆标准化数值一致（pilot consistency=True；`tests/test_lineage.py`）。
- [x] IF-SR selector 不读取 test（`tests/test_ifsr_selector.py`）。
- [x] 内部四条件端到端跑通且消融不是 no-op（pilot 任务 A 池差异 + 任务 B 选择规则分歧）。
- [x] LLM judge 全链路能断点续跑并严格校验（`tests/test_interpretability_eval.py`）。
- [~] 至少完成一次小规模 rubric 校准：设施与流程就位，实际校准需真实领域评分者。
- [x] ExprSim 对已知测试样例行为正确（`tests/test_expression_similarity.py`）。
- [~] pilot 成本证明：surrogate 下管线开销可忽略；真实 PySR/judge 的 wall-time/token
  需真实引擎测算（协议已列预算字段）。
- [x] `frozen_protocol.json` 已生成。
- [x] pilot 之后不再根据主结果调整任务/rubric/权重/split（均已冻结）。

## 验证证据

```bash
VP=/public/home/wangyg/workspace/llm_pysr_project/.venv/bin/python
$VP -m pytest -q                      # 63 passed
PYTHONPATH=. $VP scripts/run_pilot.py # 端到端 pilot，产物见 outputs/pilot_run/
```

最小端到端产物：`outputs/pilot_run/pilot_summary.json`（含 ablation、IF-SR 决策、
数值一致性、盲评/judge/aggregate 摘要、ExprSim）。

## 仍存在的风险（对论文完整性与可信度的影响）

1. **真实催化数据未接入**（高）：`dataset_inventory.yaml` 6 个真实主任务仍为骨架。
   实验 1 无法在真实数据上启动前，无法给出人类/LLM 可解释性主结果。
2. **SR 引擎为 surrogate**（高）：pilot 用线性可加 surrogate 证明管线正确性与消融非 no-op，
   但真实 Pareto 前沿、真实 R² 分布、真实 wall-time 需 PySR。真实运行前不得声称任何 R² 结论。
3. **真实 LLM judge 未跑**（中）：schema/缓存/续跑/顺序扰动已用假响应验证；真实 token/成本/
   模型间一致性需小规模真实调用测算。
4. **rubric 校准与 domain card 冻结**（中）：设施就位，需真实领域评分者完成。

## 下一步（获得资源后，按冻结协议）

1. 接入真实催化源表，冻结 6 个真实主任务及 group split，记录 split hash 到 frozen_protocol。
2. 用真实 PySR 替换 surrogate，跑 pilot 2 真实 + 4 真式，测算 wall-time/token/费用。
3. 用两个真实模型族跑小规模真实 judge + rubric 校准。
4. gate 全绿后按 `configs/paper_scope.yaml` reduction_order 决定是否缩减，再启动确认性首批任务。
