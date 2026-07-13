# 给下一个模型的三个阶段执行提示词

下面整段可以直接复制给下一个负责代码与实验的模型。

---

你正在接手项目：

```text
/public/home/wangyg/Interpretability-first-symbolic-regression
```

你的任务不是重新讨论论文方向，而是依次把论文所需的无泄漏实验管线、IF-SR 最小方法、LLM 可解释性评测设施和端到端 pilot 真正实现出来。

## 一、论文主线不可偏离

论文主线是：

> 真实科学数据通常有噪声、样本有限且关系复杂，唯一真实公式往往不可辨识。符号回归在这种场景中的价值，是在预测性能可接受的前提下，产生人类能够理解、审查并用于提出假设的表达式。现有方法主要依赖预测误差和语法复杂度，但低节点数并不等于领域可解释。因此，我们提出一个 accuracy-constrained、interpretability-first 的符号回归流程，在接近最优验证性能的候选中，根据硬语义约束、完全展开后的复杂度、领域因子和稳定性选择公式。

论文只保留两个核心实验：

1. 无明确真式的真实黑盒科学数据：比较 test R²、人类专家可解释性评分和独立 LLM 可解释性评分；
2. 有复杂真式的数据：除上述指标外，增加表达式等价恢复率和表达式相似度。

不要把工作扩张为通用解释性理论、新的遗传编程引擎或大型 benchmark。

## 二、开始前必须阅读和核对

先完整阅读：

```text
docs/fast_publication_checklist_zh.md
docs/next_gpt_paper_handoff_zh.md
README.md
factor_pysr_llm/cli.py
factor_pysr_llm/dataset.py
factor_pysr_llm/features.py
factor_pysr_llm/factor_miner.py
factor_pysr_llm/llm_stages.py
factor_pysr_llm/model_api.py
factor_pysr_llm/pysr_runner.py
factor_pysr_llm/reports.py
tests/
```

然后先做一次仓库审计并记录到：

```text
docs/implementation_status.md
```

审计必须逐项回答：

- 数据何时划分 train/validation/test；
- imputation、scaling、相关筛选和因子挖掘是否只在 train 上 fit；
- PySR 候选是否在 validation 上选择、test 是否只评价一次；
- mined/domain factor 是否保存完整 expression lineage；
- 最终公式能否展开到原始变量并逆标准化；
- LLM factor-selection prompt 是否实际嵌入候选内容；
- LLM selection 是否真正改变候选池，还是与 top-k 做 union 后接近 no-op；
- 是否存在可解释性 judge、盲评导出和 rating aggregation；
- 当前测试覆盖哪些功能、哪些论文关键功能没有测试。

已知事实：当前仓库只有通用 `llm-call`、因子提议/筛选 prompt 和用于解释/改写公式的 `llm-interpret`。`llm-interpret` **不是论文需要的可解释性评测器**。不要把它当作已经实现的 LLM judge。

## 三、总执行原则

1. 直接修改代码、配置、测试和文档，不要只给建议。
2. 保留用户已有修改；不要使用破坏性 git 操作。
3. 每完成一个可独立验收的模块就运行对应测试。
4. 所有随机过程显式保存 seed；所有实验保存配置和版本信息。
5. 历史运行只能标记为 `development_only`，不得直接进入确认性主表。
6. test labels 不得参与缺失值处理、缩放、筛选、因子挖掘、prompt 构造、常数拟合或最终公式选择。
7. LLM 输出必须经过结构化校验；非法输出不得静默接受或自动填默认分。
8. 给远端 LLM 的 prompt 必须嵌入它需要看到的实际内容，不能只提供本地文件路径。
9. 论文评价用的 LLM judge 不得参与 IF-SR 最终公式选择。
10. 如果资源不足，优先砍额外数据、额外 baseline 和额外 seed，不能砍无泄漏、独立 test、盲评和失败日志。

## 四、阶段一：范围、数据与无泄漏管线

### 阶段目标

完成范围冻结、数据盘点和无泄漏 split 基础设施。阶段结束时，至少应能在 toy 数据上证明：改变 test labels 不会改变训练得到的特征、候选或最终选择。

### 任务 1：冻结论文与实验范围

创建：

```text
configs/paper_scope.yaml
configs/dataset_inventory.yaml
configs/known_formula_tasks.yaml
```

`paper_scope.yaml` 至少写死：

- 两个研究问题；
- 真实数据主任务数量为 6；
- 已知真式任务数量为 20；
- 比较条件：PySR、一个外部 SR baseline、Mine-PySR、IF-SR、IF-SR w/o domain factors；
- 主指标：test R²、人类评分、两个独立 LLM 评分、ExprSim；
- `δ=0.02` accuracy tolerance；
- 3 个确认性 seeds；
- 不允许根据最终结果更换任务、split、rubric 或 ExprSim 权重。

`dataset_inventory.yaml` 对每个真实候选任务记录：

- 数据来源和文件；
- source dataset ID；
- target；
- 样本数；
- 原始变量及含义；
- 单位状态：known/unknown/inferred；
- group split 候选字段；
- 是否和其他 target 共用源表；
- 是否可公开；
- 是否选入 6 个主任务及预先定义的原因。

不得把同一张源表的多个 target 写成多个独立数据集。

`known_formula_tasks.yaml` 先给出 20 个任务的冻结规则和任务 ID；如果最终列表尚不能当天确定，先实现可复现的分层抽取器，按变量数、节点数和算子类型用固定 seed 抽取，禁止人工看结果后挑题。

### 任务 2：实现固定 split manifest

新增适当模块，例如：

```text
factor_pysr_llm/splits.py
```

至少支持：

- 普通 train/validation/test split；
- 按材料、active phase、support 或实验批次的 group split；
- 固定 seed；
- 保存 row IDs，而不是只保存比例；
- 输出 JSON manifest 和 SHA256；
- 检查三个集合互斥且覆盖预期样本；
- 同一 group 不得跨集合。

CLI 至少提供可复现的 split 生成或读取入口。所有后续数据命令必须显式接收 split manifest，不能在各模块内部各自随机切分。

### 任务 3：把预处理改成 train-fit

检查并修改数据处理，使以下操作只在 train 上拟合：

- 缺失值统计与填补；
- 标准化/归一化；
- 原始变量相关性筛选；
- 常数列和异常列判断；
- 因子相关性排序和 top-k 选择。

保存 fitted preprocessing state，并将同一 state 确定性应用到 validation/test。不要在完整数据表上先预处理再切分。

### 任务 4：建立泄漏测试

至少新增：

```text
tests/test_splits.py
tests/test_no_leakage.py
```

必须覆盖：

- split 可复现；
- group 不交叉；
- 修改 validation/test 特征中的极端值不会改变 train-fitted 参数；
- 修改 test labels 不会改变选中特征、挖掘因子、候选 archive 和最终公式 ID；
- validation 可以选式，test 不可用于选式。

### 阶段一验收标准

- `paper_scope.yaml`、dataset inventory 和真式任务规则存在且可读取；
- 固定 split manifest 能在 toy 数据和至少 1 个真实开发任务上生成；
- 预处理 state 明确区分 fit/transform；
- 无泄漏测试通过；
- 现有测试没有被破坏；
- 输出 `docs/phase1_report.md`，列出完成项、未完成项、测试命令和结果。

阶段一未通过验收，不得开始确认性实验。

## 五、阶段二：IF-SR 与可解释性评测基础设施

### 阶段目标

完成 IF-SR 最小方法和 LLM 可解释性评测基础设施。阶段结束时必须能从一组候选公式导出匿名盲评包，生成包含实际内容的 judge prompt，并用固定假响应完成解析与统计。

### 任务 1：实现公式 lineage、展开和逆标准化

每个 mined/domain factor 至少保存：

```json
{
  "factor_id": "...",
  "name": "...",
  "expression": "...",
  "variables": ["..."],
  "meaning": "...",
  "unit_status": "valid|unknown|screening_only",
  "source": "expert|llm|literature|mined",
  "approved_for_final_formula": true
}
```

实现并测试：

- 递归展开 nested factors；
- 循环引用检测；
- 展开到原始变量；
- 标准化变量的逆变换；
- 基础 canonicalization；
- expanded node count、depth、变量数和常数数；
- 展开前后数值预测一致性。

隐藏在一个短因子名后的复杂表达式必须按展开后复杂度计分。

### 任务 2：实现 IF-SR selector

新增独立、可测试的 selector。规则固定为：

```text
1. 只使用 validation 指标；
2. acceptable set: R²_val >= best_R²_val - 0.02；
3. 剔除硬违规：泄漏变量、非法表达式、无定义、有效域覆盖不足、确定的量纲加减错误；
4. 最小化 expanded complexity；
5. 同复杂度时优先 approved domain factor；
6. 再按跨 seed 稳定性和固定 ID tie-break；
7. 选择完成后才读取 test 指标。
```

保存选择前完整 candidate archive、每个候选被保留/剔除的原因和最终决策轨迹。

新增测试验证：

- `δ` 边界正确；
- test R² 改变不影响选择；
- 短别名不会逃避 expanded complexity；
- 硬违规优先于复杂度；
- tie-break 可复现。

### 任务 3：修复 LLM 因子 prompt 与 selection no-op

修改 `write_factor_selection_prompt` 等相关代码：

- prompt 内实际包含候选因子表、表达式和变量元数据；
- 限制 prompt 大小时采用明确的 top-k/分批规则；
- 对 LLM selection 使用 JSON schema 校验；
- `final_formula_allowed` 真正影响最终候选资格；
- LLM selection 必须真实改变候选池，而不是和原 top-k 全量 union；
- 保存原始响应、prompt hash、模型配置和匹配失败项。

为上述行为添加测试，证明提供不同 selection 会得到不同候选池。

### 任务 4：实现可解释性盲评导出

新增：

```text
factor_pysr_llm/interpretability_eval.py
configs/interpretability_rubric.json
examples/rating_manifest.example.jsonl
```

CLI 新增：

```text
export-blind-ratings
score-interpretability-prompt
```

`rating_manifest.jsonl` 每条至少包含：

- 匿名 `item_id`；
- dataset/target 匿名或中性描述；
- 完全展开、逆标准化并统一排版的公式；
- 变量名称、定义、单位和允许范围；
- 必要任务背景；
- 不包含方法名、R²、生成理由或已有 LLM 解释。

方法名、seed、真实 R² 等保存在隔离的 private mapping 中。

rubric 使用四个 1–5 分维度：

1. 变量及组合的领域意义；
2. 结构合理性；
3. 易读、易概括性；
4. 能否支持可讨论、可检验的假设。

每个 1–5 分必须提供文字锚点，不能只写“1 差、5 好”。prompt 必须嵌入公式、变量说明、rubric 和输出 schema，不能引用模型看不到的本地路径。

### 任务 5：实现 LLM judge 和评分聚合

CLI 新增：

```text
run-llm-judge
aggregate-ratings
```

`run-llm-judge` 至少支持：

- OpenAI-compatible provider config；
- 多批次；
- 两个独立模型配置；
- 固定 temperature；
- 缓存和断点续跑；
- 有上限的重试；
- 公式顺序随机扰动；
- 保存 prompt hash、model ID、原始响应、解析结果、token/成本和错误；
- JSON schema 严格校验；
- 分数必须是 1–5；
- 缺失、重复和未知 item ID 判为失败，不得静默填补。

`aggregate-ratings` 至少支持：

- 人类 CSV 导入；
- 人类、LLM-A、LLM-B 分开汇总；
- 每维均值和总体均值；
- bootstrap 95% CI；
- 人类评分者一致性；
- LLM–human Spearman；
- 成对偏好一致率；
- 按 dataset/target/method 解盲后的汇总；
- 不把人类分与 LLM 分平均成一个总分。

测试不得依赖真实 API。使用固定假响应覆盖正常输出、越界分数、缺字段、markdown code fence、重复 ID、部分批次失败和断点续跑。

建议新增：

```text
factor_pysr_llm/rating_aggregate.py
tests/test_interpretability_eval.py
tests/test_rating_aggregate.py
```

### 阶段二验收标准

- 任意候选公式能展开、逆标准化并重算，数值一致；
- selector 只用 validation 且决策可审计；
- LLM factor prompt 实际含候选内容，selection 不再是 no-op；
- 能导出匿名专家 CSV 和 LLM rating manifest；
- 能用假响应跑通 prompt → parse → aggregate；
- 关键异常都有测试；
- 输出 `docs/phase2_report.md`，记录 CLI 示例、测试结果和剩余风险。

阶段二未完成 LLM 评测设施，不得把“LLM 可解释性评分”写成已完成实验。

## 六、阶段三：端到端 Pilot 与协议冻结

### 阶段目标

完成端到端 pilot、rubric 小规模校准、成本测算和确认性协议冻结。只有所有 gate 通过后，才能启动正式主实验。

### 任务 1：准备 pilot 数据

使用：

- 2 个真实开发任务；
- 4 个已知真式开发任务；
- 不得使用最终 6 个真实主任务和 20 个真式任务中的全部结果反复调参；如有重合，明确标记并限制用途。

为每个 pilot 任务生成固定 split、metadata/domain card 和运行配置。

### 任务 2：跑通方法条件

至少跑通：

```text
Raw-PySR
Mine-PySR
IF-SR w/o domain factors
IF-SR
```

外部 baseline 在内部四条件全链路通过后接入，不得阻塞内部验证。

检查：

- 四种条件是否真的产生不同候选池或选择结果；
- 候选 archive 是否完整；
- 公式是否都能展开和逆标准化；
- test 是否只在公式锁定后读取；
- 失败和超时是否自动记录；
- 每次运行是否记录 seed、split hash、代码版本、CPU/wall time 和 LLM 调用成本。

### 任务 3：跑通可解释性评测 pilot

从 pilot candidate archive 中，仅根据 validation 结果构造预测性能相近的公式集合。

完成以下闭环：

```text
candidate archive
→ blind rating export
→ 专家评分 CSV
→ LLM-A judge
→ 顺序扰动
→ LLM-B judge
→ schema validation
→ aggregate-ratings
→ reliability report
```

先选 8–12 个非主结果公式，由至少 1 位领域人员做 rubric 校准；条件允许时由第 2 位评分者复核。校准只允许修改含糊的评分说明，不允许根据“哪个方法得分高”修改 rubric。

必须检查：

- 评分者能否仅凭所给变量定义理解任务；
- 公式展开后是否长到无法评价；
- LLM 是否因为公式顺序产生明显偏差；
- 两个 LLM 是否输出有效 JSON；
- 人类与 LLM 是否至少有合理方向的一致性；
- 若一致性很低，记录为风险，不得偷偷更换模型直到结果好看。

### 任务 4：实现表达式相似度最小版本

为实验 2 实现并测试：

```text
ExprSim = 0.50 × 独立采样点数值相似度
        + 0.20 × 变量集合 F1
        + 0.20 × 运算符集合 F1
        + 0.10 × 规范化树结构相似度
```

另外单独报告：

- 代数等价成功；
- 独立采样点数值等价成功；
- support F1。

要求：

- 所有子分数在 `[0,1]`；
- 对变量重命名、交换律、常数近似和无定义点有明确处理；
- 独立采样点不能复用训练样本；
- 权重写入冻结配置；
- 用手工构造的等价、近似、错误和变量缺失公式写单元测试。

### 任务 5：成本测算与协议冻结

根据 pilot 记录估算：

- 每个 SR method × task × seed 的 wall time；
- 6 个真实任务和 20 个真式任务总运行量；
- 两个 LLM judge 的请求数、token、费用和预计耗时；
- 专家需要评价的公式数量和预计工时；
- 失败重跑余量。

如果根据 pilot 测算发现资源不足，按以下顺序缩减：

1. 删除补充 clean 条件；
2. 不扩展剩余真实 targets；
3. 不增加 seed；
4. 不增加第二个外部 baseline；
5. 将已知真式从 20 降至最低 15，并保持分层抽取。

不得删除两个核心实验、两个独立 LLM judge、至少两位专家的最终主评、独立 test 或无泄漏检查。

冻结并生成：

```text
configs/frozen_protocol.json
docs/pilot_report.md
```

`frozen_protocol.json` 至少包含：

- 数据任务 ID；
- split hashes；
- 方法和消融；
- SR 超参数和预算；
- seeds；
- `δ`；
- 算子集合；
- factor/domain cards 版本；
- rubric 版本；
- LLM judge model IDs、temperature、prompt hash 和重复次数；
- ExprSim 权重；
- 失败/排除规则；
- 统计方案；
- 代码 commit hash。

### 任务 6：启动确认性实验的首批任务

只有满足阶段三 gate 后，才启动首批确认性运行：

- 先运行 1 个真实主任务和 3 个真式主任务；
- 自动检查产物完整性；
- 确认汇总脚本不需要人工修表；
- 通过后再排队其余任务。

不要在 gate 未通过时用大规模计算掩盖管线问题。

### 阶段三 Gate

以下全部满足才算阶段三完成：

- [ ] 无泄漏测试通过；
- [ ] 公式展开和逆标准化数值一致；
- [ ] IF-SR selector 不读取 test；
- [ ] 内部四条件端到端跑通且消融不是 no-op；
- [ ] LLM judge 全链路能够断点续跑并严格校验；
- [ ] 至少完成一次小规模 rubric 校准；
- [ ] ExprSim 对已知测试样例行为正确；
- [ ] pilot 成本证明完整实验在现有计算与 API 预算内可完成；
- [ ] `frozen_protocol.json` 已生成；
- [ ] pilot 之后不再根据主结果调整任务、rubric、权重或 split。

## 七、三个阶段完成后必须交付

代码侧至少包括：

```text
factor_pysr_llm/splits.py                         # 或等价模块
factor_pysr_llm/interpretability_eval.py
factor_pysr_llm/rating_aggregate.py
factor_pysr_llm/expression_similarity.py          # 或等价模块
configs/paper_scope.yaml
configs/dataset_inventory.yaml
configs/known_formula_tasks.yaml
configs/interpretability_rubric.json
configs/frozen_protocol.json
examples/rating_manifest.example.jsonl
tests/test_splits.py
tests/test_no_leakage.py
tests/test_interpretability_eval.py
tests/test_rating_aggregate.py
tests/test_expression_similarity.py
```

文档侧至少包括：

```text
docs/implementation_status.md
docs/phase1_report.md
docs/phase2_report.md
docs/pilot_report.md
```

CLI 或等价可执行入口至少包括：

```text
generate-split
export-blind-ratings
score-interpretability-prompt
run-llm-judge
aggregate-ratings
expression-similarity
```

如果实际模块名不同，可以调整，但功能、测试和可复现产物不能减少。

## 八、每次向用户汇报的格式

每个里程碑结束后，用下面格式汇报，不要只说“完成了”：

```text
本阶段结论：通过 / 未通过

已完成：
- 具体文件和功能

验证证据：
- 执行的测试命令
- 通过数量
- 一个最小端到端产物路径

仍存在的风险：
- 风险及其对论文完整性和可信度的影响

下一步：
- 接下来 1–3 个最高优先级动作
```

如果发现现有设计与代码事实冲突，以可复现证据为准；更新状态文档和工作清单，但不要擅自改变论文的两个核心实验与“可解释性优先、准确率受控”主线。

现在开始执行。先完成仓库审计和阶段一任务，不要仅返回新的计划。

---
