# 实验二结果报告：已知真式的符号回归恢复（初步真实运行）

更新日期：2026-07-13
数据来源：`examples/实验2_科学公式数据集清单.md`（30 条科学公式）
运行产物：`outputs/experiment2_run/`（gitignore，不入库）

> 状态：**初步真实运行**。SR 引擎为真实 PySR，LLM judge 为真实 DeepSeek
> （deepseek-chat + deepseek-reasoner）。本轮跑通 10 个代表任务 × 3 条件 × 1 seed，
> 用于验证方法与全链路。确认性主表需扩到全部可回归任务 × 3 seeds（见"下一步"）。

## 一、实验设置

- **数据生成**：每个真式渲染 train=300 / validation=150 / test=300，外加 300 个
  **独立** ExprSim 采样点（无噪声）。`5% target noise + 5 个无关变量 z1..z5`。物理常数
  折算为代表数值，SR 需恢复变量间的**结构**。
- **无泄漏协议**：PySR 只在 train 行上拟合；候选 Pareto archive 在 validation 上评分；
  test R² 在每条件锁定公式后只读一次。
- **三个条件**（均来自同一批 PySR 运行）：
  - `raw_pysr`：仅原始变量，标准（最优 validation R²）选择；
  - `mine_pysr`：原始变量 + 通用因子（成对积/商/平方/倒数，**不含真值**），标准选择；
  - `if_sr`：在 raw+mined 候选上做 interpretability-first 选择（`δ=0.02` validation
    容忍带内最小展开复杂度）。
- **公式展开**：mined 列公式通过 factor lineage 展开回原始变量后再算 ExprSim 与复杂度，
  保证与真值可比。
- **可解释性评分**：两个独立 DeepSeek 模型族盲评（deepseek-chat / deepseek-reasoner），
  四维 1–5 rubric，公式匿名、隐藏方法名与 R²、顺序扰动、严格 JSON schema 校验。

10 个代表任务（覆盖简单/中等/复杂）：F01 自由落体、F04 库仑力、F05 万有引力、
F06 理想气体、F09 一维高斯、F15 米氏方程、F17 单摆周期、F19 逻辑斯蒂增长、
F22 两体引力势、F25 康普顿散射。全部运行成功，0 失败。

## 二、主结果

### 2.1 预测性能、表达式相似度、复杂度（10 任务均值）

| 指标 | raw_pysr | mine_pysr | if_sr |
|---|---|---|---|
| test R² 均值 | 0.996 | 0.997 | 0.996 |
| ExprSim 均值 | 0.857 | 0.930 | 0.917 |
| 展开节点数均值 | 12.3 | 8.0 | **5.5** |
| 代数等价 | 3/10 | 2/10 | 2/10 |
| 数值等价 | 3/10 | 2/10 | 2/10 |

**读法**：三条件 test R² 基本相同（0.996–0.997），符合"准确率受控"前提。在几乎不损失
R² 的情况下，IF-SR 把公式复杂度从 12.3 节点降到 5.5 节点（≈-55%），mined 因子把
ExprSim 从 0.857 提升到 0.930。IF-SR 在复杂任务上把冗长公式压到最短（F19：24→7 节点且
ExprSim 0.63→0.98；F22：22→10 节点且 0.81→0.99）。

### 2.2 LLM 可解释性盲评（两个独立模型族，方向一致）

| 判官 | raw_pysr | mine_pysr | if_sr |
|---|---|---|---|
| deepseek-chat | 3.425 | 3.475 | **3.800** |
| deepseek-reasoner | 3.800 | 4.125 | **4.275** |

两个模型族**独立地**给出相同排序：`IF-SR > Mine-PySR > Raw-PySR`。
判官间一致性：Spearman = 0.732，成对偏好一致率 = 0.892（n=30）。

## 三、与论文主线的关系

本轮初步结果支持论文核心主张：**在 test R² 基本不降的前提下，interpretability-first
选择产生更短、结构更接近真式、且被独立 LLM 判定更可解释的表达式**。这正是
accuracy-constrained、interpretability-first 的设计目标。

需要诚实指出：
- 代数/数值**精确恢复率**三条件相近且不高（2–3/10）。这与 PySR 常引入小常数近似项、
  以及本轮预算较小有关；ExprSim 与复杂度反映的是"更接近、更短"，不是"完全恢复"。
- IF-SR 在个别任务上以极小 R² 代价换取大幅简化（如 F25：R² 0.997→0.983，节点 13→5），
  属于预期的 accuracy–interpretability trade-off，应在论文中如实呈现，不宣称全面占优。

## 四、复现命令

```bash
VP=/public/home/wangyg/workspace/llm_pysr_project/.venv/bin/python
# 1) 运行 SR（train-only 拟合，validation 选择，test 只读一次）
PYTHONPATH=. $VP -c "from factor_pysr_llm.experiment2 import run_experiment2; \
  run_experiment2('configs/experiment2_formulas.json','outputs/experiment2_run', \
  task_ids=[...], seeds=[20260709], budget={'niterations':40,'timeout_seconds':90,'delta':0.02})"
# 2) 盲评导出
PYTHONPATH=. $VP -m factor_pysr_llm.cli export-blind-ratings \
  --candidates outputs/experiment2_run/rating_candidates.json \
  --out-manifest outputs/experiment2_run/rating_manifest.jsonl \
  --out-private-map outputs/experiment2_run/rating_private_map.json
# 3) 两个 DeepSeek judge（配置见 configs/llm_provider.judge_{a,b}.json，已 gitignore）
PYTHONPATH=. $VP -m factor_pysr_llm.cli run-llm-judge --manifest ... --model-id deepseek_chat ...
# 4) 聚合
PYTHONPATH=. $VP -m factor_pysr_llm.cli aggregate-ratings \
  --llm-results deepseek_chat=... deepseek_reasoner=... \
  --private-map outputs/experiment2_run/rating_private_map.json --output ...
```

## 五、风险与下一步

风险：
- **样本量小**：10 任务 × 1 seed。方向一致但未做多 seed 稳定性与配对显著性检验。
- **PySR 预算小**（niterations=40, timeout=90s）：更大预算可能提高精确恢复率。
- **单一 API 家族**：两个 judge 都是 DeepSeek 系列。论文要求两个**不同**模型家族，
  正式主评需再接入一个非 DeepSeek 家族（如 OpenAI/Qwen）。

下一步：
1. 扩到全部 29 个可回归任务 × 3 seeds，提高 PySR 预算，做配对效应量 + bootstrap CI。
2. 接入第二个真正不同的模型家族做 LLM judge，保留 DeepSeek 之一。
3. 加入至少 2 位领域人员对 8–12 个非主结果公式做 rubric 校准，再上人类主评。
4. 把 F14 范德瓦尔斯（已显式化）与 F30 洛伦兹系统（单分量）纳入敏感性分析或明确排除。
