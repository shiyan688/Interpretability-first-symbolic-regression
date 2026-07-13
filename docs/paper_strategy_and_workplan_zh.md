# 可解释性优先符号回归：论文定位、方法重构与工作计划

更新日期：2026-07-12

> 若目标是尽快投 CCF B/C，而不是冲击高水平通用 venue，请直接执行精简版清单：[`fast_publication_checklist_zh.md`](fast_publication_checklist_zh.md)。

## 0. 结论先行

这个方向有明确价值，但当前仓库更接近一个可运行的 workflow scaffold，还不是一篇足以支撑“interpretability-first symbolic regression”的方法论文。

真正值得写的研究问题不是：

> LLM、因子枚举和 PySR 串起来后，能不能找到一些看起来更合理的公式？

而应该是：

> 当噪声、未观测变量和多重共线性使“唯一真实公式”不可辨识时，能否在预测性能统计上不劣于强基线的候选集中，稳定选出领域语义违规更少、专家更偏好、跨重采样更稳定的表达式？

建议将论文核心从泛化的“可解释性优先”进一步收紧为：

> **Accuracy-constrained, domain-aware symbolic regression**：用可审计的领域知识约束符号搜索，在预先定义的精度容差内优化科学可解释性。

论文不能声称从有噪声的观察数据中恢复了“真实机制”或因果规律。可支撑的表述是：输出更可信、更容易讨论、更适合作为后续实验假设的预测性表达式。

## 1. 为什么这个问题值得做

### 1.1 科学价值

真实科学数据中经常存在：

- 测量噪声、缺失值和批次效应；
- 隐变量与未测机制；
- 强共线性和有限采样范围；
- 多个表达式在观测域内近似等价；
- 真实过程过于复杂，不存在适合人类阅读的唯一闭式表达式。

此时，把“精确复原真公式”作为唯一成功标准并不现实。符号回归仍有价值，因为它可以把数据压缩成可审计的候选关系，供专家检查变量作用、极限行为、尺度律和后续实验。

### 1.2 方法价值

传统的 `loss + expression size` 只能鼓励短公式，不能保证公式在领域上合理。例如，一个节点数很少的公式仍可能量纲错误、包含无意义的非线性嵌套，或依赖明显不合理的变量组合。

可解释性是面向使用者和任务的属性，不能由节点数单独代表。因此有必要把领域元数据、物理约束和专家偏好变成可执行、可验证的搜索条件。

### 1.3 实用价值

如果 LLM 的作用被限制为“把自然语言领域知识编译成候选因子和结构化约束”，而所有公式都由确定性代码检查并由独立数据验证，那么 LLM 可以降低知识注入门槛，同时保留审计性。其价值是知识接口，不是替代数值验证或领域专家。

## 2. 必须修正的 related-work 判断

“现有方法没有解释性约束”这个表述不成立，不能写进论文。已有工作已经覆盖了多个相邻方向：

- [Learning a Formula of Interpretability to Learn Interpretable Formulas](https://arxiv.org/abs/2004.11170) 已从人类反馈学习解释性代理，并将其用于双目标符号回归。
- [SRBench++](https://doi.org/10.1109/TEVC.2024.3423681) 已在真实任务中引入领域专家解释性评价，并包含噪声、特征选择、错误捷径、外推和真式恢复任务。
- [LLM-SR（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/28df8e730c054c5331855fd4d5403ba9-Abstract-Conference.html) 已用 LLM 科学先验迭代生成公式骨架并结合数值优化。
- [LaSR（NeurIPS 2024）](https://papers.nips.cc/paper_files/paper/2024/hash/4ec3ddc465c6d650c9c419fb91f1c00a-Abstract-Conference.html) 已用 LLM 发现和演化概念库来指导符号搜索。
- [Knowledge integration for physics-informed symbolic regression using pre-trained LLMs](https://www.nature.com/articles/s41598-026-35327-6) 已把 LLM 对量纲、简单性和物理合理性的评分直接加入 SR 损失。
- [Beyond Accuracy and Complexity / EIC（ICML 2026）](https://fi.ee.tsinghua.edu.cn/~dingjingtao/papers/2509.21780v2.pdf) 已提出结构稳定性指标，并通过 108 位专家评价验证它与解释性偏好的一致性。
- [Pareto-Optimal Fronts for Benchmarking SR Algorithms（ICML 2025）](https://proceedings.mlr.press/v267/fong25b.html) 已系统研究准确率—表达式长度 Pareto 前沿。
- [LLM-SRBench（ICML 2025）](https://proceedings.mlr.press/v267/shojaee25a.html) 已用 239 个跨领域问题明确处理著名公式被 LLM 记忆导致的 benchmark 污染。
- 2026 年 7 月的 [LLM-PySR](https://arxiv.org/abs/2607.04156) 已让 LLM 控制变量、算子、变换和搜索深度，而由确定性指标决定候选保留；这与当前工作流高度接近。
- 2026 年 6 月的 [FunctionEvolve](https://arxiv.org/abs/2606.07704) 已用显式 AST、局部结构变异和结构感知常数拟合在 LLM-SRBench 上取得很强的精确恢复结果。

因此，更可辩护、但仍需继续核实的 gap 是：

> 现有方案多依赖通用复杂度、人工编码的特定物理约束，或不可审计的 LLM 直接评分；仍缺少一个将自然语言领域元数据一次性编译为有来源记录、可执行、可消融的语义约束，并在严格无泄漏的精度非劣效条件下处理无唯一真式真实数据的通用工作流。

## 3. 建议的论文主张

### 3.1 可证伪的主假设

论文的联合主结论应为：

1. **预测非劣效**：相对预注册强基线，本方法的外层测试性能下降不超过领域允许的容差 `δ`。
2. **解释性优越**：在验证性能相近的表达式之间，本方法得到更高的盲评专家成对偏好率和更低的领域语义违规率。
3. **前沿改进**：本方法改善 accuracy–interpretability 前沿，而不只是挑出一个漂亮案例。
4. **结构恢复**：在有真公式任务中，本方法提高代数/数值语义恢复率，并在结构恢复指标上不劣于强基线。

H1 和 H2 必须同时成立。只提高解释性但明显牺牲精度，或只提高 R²，都不足以支持主线。

### 3.2 形式化定义

设：

- `D_train`、`D_val`、`D_test` 为严格分离的数据；
- `K` 为在查看确认性结果前冻结的 domain card；
- `F` 为只使用训练数据产生的候选表达式集合；
- `L_val(f)` 为训练内层验证损失；
- `V_K(f)` 为表达式对领域约束 `K` 的违规向量；
- `C_expand(f)` 为完全展开到原始变量、逆标准化后的认知/描述长度；
- `S_stability(f)` 为跨训练重采样的结构稳定性。

可把问题定义为：

```text
F_delta = {f in F : L_val(f) <= L_best + delta}

f* = argmin_{f in F_delta}
     [V_K(f), C_expand(f), -S_stability(f)]
```

也可保留完整多目标 Pareto 集，但论文必须预先定义最终公式选择规则，不能在测试集上看图后再挑。

这里的关键不是某个任意加权总分，而是：先满足预测精度约束，再优先减少硬语义违规和展开后复杂度。这才真正体现“interpretability-first, accuracy-controlled”。

## 4. 方法应如何重构

### 4.1 Domain card

每个数据集必须有机器可读的 domain card，至少包含：

```text
target:
  name, definition, unit, valid_range
variables:
  name, definition, unit, role, valid_range
allowed_transformations:
  operator, input/output unit rules, domain restrictions
candidate_relations:
  ratio/difference/product/coupling, rationale, evidence
hard_constraints:
  dimensional consistency, forbidden variables, invariances
soft_constraints:
  monotonicity, limiting behavior, plausible interactions
split_constraints:
  group/material-family/patient/time/batch identifiers
provenance:
  expert, textbook/paper, LLM proposal, approval status
```

domain card 必须在确认性实验前冻结。LLM 可以提出内容，但领域专家或明确规则必须确认硬约束。

### 4.2 LLM 的正确角色

主方法中，LLM 应是一次性的知识编译器，而不是每个候选式的最终裁判：

```text
domain metadata
  -> LLM proposes typed factors and constraints
  -> parser/schema validation
  -> unit/type checker
  -> expert approval or rule-based validation
  -> frozen factor grammar / constraint set
```

建议把两种信号分开，方便消融：

- 语义通道：LLM 只看变量定义、单位和任务背景，不看标签数值；
- 数据通道：确定性因子挖掘只看当前训练折的数据，不看领域文本。

两者最后合并。这样才能区分收益来自领域知识，还是仅仅来自扩大了特征搜索空间。

LLM 直接给候选公式打分可以保留为 baseline 或探索性实验，但不应作为唯一解释性约束。

### 4.3 可执行的解释性约束

至少实现四类确定性指标：

1. **单位与类型有效性**：加减项同量纲；指数、对数和三角函数输入满足类型规则；输出单位匹配 target。
2. **领域语义违规**：违反冻结的变量角色、允许交互、单调性、对称性或极限行为。
3. **展开后认知复杂度**：节点数、变量数、参数数、嵌套深度、非线性复合、重复/抵消子树和 MDL。
4. **结构稳定性**：跨 split/seed 的变量集合、规范化子表达式和预测行为一致性。

### 4.4 统一的候选表达式档案

所有方法输出都应进入同一 candidate archive：

```text
method / seed / split / budget
raw_expression
expanded_expression
original_unit_expression
constant_refit_status
train/validation metrics
constraint violations
expanded complexity
runtime / CPU / memory / LLM tokens
provenance and failure logs
```

任何 mined factor 或 meta-factor 都必须展开后再计算复杂度和交给专家。否则一个很复杂的子树在 PySR 中只算一个叶节点，会造成不公平的“隐藏复杂度”。

### 4.5 LLM 后处理的边界

“解释性增强”不能是论文最后手工挑一个式子让 LLM 美化。若保留该阶段，LLM 改写后的表达式必须作为新候选重新进入：

```text
parse -> expand -> unit check -> refit constants on train
      -> inner validation -> frozen selection -> outer test once
```

否则建议把该阶段从主方法中删除，只让 LLM生成自然语言解释，不修改公式。

## 5. 当前代码与论文主张的差距

| 问题 | 当前证据 | 审稿风险 | 必须的修复 |
|---|---|---|---|
| 整表选择泄漏 | 填补、标准化、相关筛选、因子挖掘和 PySR 都在整表进行 | test R² 和因子优势无效 | split 先行，整条 pipeline 在每个训练折内 fit/transform |
| 只有训练指标 | PySR 在同一 `X,y` 上 fit 和 predict | 无法证明泛化 | 外层 test + 训练内层 CV；test 只评一次 |
| 搜索不是解释性优先 | `model_selection="accuracy"`，极小 parsimony，因子按相关性排序 | 标题与算法不一致 | 语义约束进入 beam、候选筛选或最终预注册重排 |
| LLM selection 近似 no-op | 先保留相关 top-k，再把 LLM 结果做 union | 无法证明 LLM 选择贡献 | 让 LLM/语义规则真正改变候选集，并保留严格消融 |
| 远端 LLM 看不到本地文件 | prompt 只写 CSV/JSON 路径，API 只发送 prompt 文本 | selection/interpretability 阶段实际缺输入 | 将所需内容安全嵌入 prompt，做 schema 校验和缓存 |
| 没有领域语义 schema | prompt 主要包含安全变量名和相关性 | 不能可靠讨论量纲和领域含义 | 新增变量定义、单位、范围、约束和 provenance |
| z-score 后直接解释 | 最终公式依赖标准化变量/因子 | 量纲与真式相似度失真 | 自动逆标准化，恢复原始单位和常数单位 |
| 隐藏因子复杂度 | mined factor 在 PySR 中是一个列 | 对基线不公平 | 自动展开 lineage，并按展开 AST 计费/评分 |
| “SISSO-like”贡献偏弱 | 只有相关性 beam enumeration，没有 sparsifying operator | 容易被判为普通 feature engineering | 改称 typed/correlation-guided factor expansion，或补齐真正稀疏描述符选择 |
| 历史预测 meta-factor | 可能复用同一标签选择出的历史最佳式 | 叠加式泄漏 | 每个训练折内重跑并只使用 OOF prediction，主实验最好先移除 |

当前 6 个单元测试可以通过，但它们只覆盖 toy 功能，不覆盖数据隔离、公式展开、跨折变换和研究协议。

## 6. 实验 1：无唯一真式的真实数据

### 6.1 数据集策略

分为两层：

- **核心专家评价层**：8–12 个真正独立的科学/工程任务，集中在 1–2 个能获得可靠专家的领域；必须有变量定义、单位和合理的数据切分依据。
- **广覆盖层**：15–25 个公共回归任务，主要用于数值泛化、稳定性和经专家校准后的自动评分；变量匿名或无领域含义的数据不承担“领域可解释性”主结论。

现有 19 个 target 和 4 个约 2000 样本数据集可以作为主案例，但要先回答：

- 19 个 target 是否来自同一源数据；若是，不能当作 19 个独立数据集；
- 样本是否存在材料族、结构族、病人、设备、时间或批次相关性；
- 数据能否公开；若不能，能否公开数据字典、split hash、评测接口和脱敏公式；
- 是否有真正能够评审表达式的领域专家。

切分必须根据数据生成机制选择：材料/分子按族或结构分组，病人按患者分组，时间序列按时间外推，重复测量按实体分组。随机逐行切分通常不足以支撑科学泛化。

### 6.2 Baseline

最低配置：

1. raw-feature PySR；
2. 强 complexity-aware SR：Operon 或 GP-GOMEA 至少一个；
3. SISSO/SISSO++，适用于描述符问题时加入；
4. 一个可稳定复现的现代 LLM/神经 SR；LLM-SR/LaSR 为已发表强基线，LLM-PySR 和 FunctionEvolve 至少应纳入最新结果对照，代码可用时再做同预算复现；
5. 本方法的完整消融。

gplearn 可以作为弱参考，但不能充当唯一 GP baseline。AI Feynman 对数据和问题结构要求特殊，也不宜作为所有真实表格任务的主基线。

最重要的受控比较是同一个 PySR 后端下：raw、纯因子挖掘、LLM seed、语义约束和 full workflow。外部方法用于说明系统级位置，不能替代内部因果消融。

### 6.3 预算公平

主表采用相同端到端资源包络：

- 相同硬件、CPU 核数、墙钟时间和内存上限；
- 因子枚举、LLM 调用和后处理时间全部计入；
- 报告 LLM token/API 成本和失败率；
- 至少 5 个端到端 seed；
- 失败和超时保留，不能删掉；
- 提供 15 min / 1 h / 4 h 等 anytime 曲线。

由于不同 SR 实现每次“候选评估”的成本不同，建议同时补充算法原生预算结果，但不要把它当作唯一公平标准。

### 6.4 主指标

预测指标：

- outer-test R²；
- RMSE、MAE、标准化 RMSE；
- OOD/group-test 表现；
- 失败率和非法输出率。

可解释性与可靠性指标：

- 专家成对偏好概率；
- 五维专家评分；
- 自动语义约束满足率；
- 展开后节点数、变量数、参数数、深度和 MDL；
- 跨 split/seed 的变量 Jaccard、规范子表达式 overlap 和预测稳定性；
- accuracy–interpretability dominance probability；
- 预注册精度容忍带内的最佳解释性。

不要把所有指标随意平均成一个总冠军分数。论文主结果应是：预测非劣效检验 + 同精度成对解释性偏好。

## 7. 实验 2：复杂且有真公式

### 7.1 两部分 benchmark

1. **公共可比部分**：约 30–50 个按节点数、深度、变量数和运算符分层的 Feynman、SRBench++ 或 LLM-SRBench 任务。
2. **污染受控部分**：约 40–60 个在方法、prompt 和超参数冻结后，用程序化量纲一致 grammar 生成并封存的新公式。

公共著名公式不能承担 LLM 方法的唯一结论。对公共任务至少做变量随机改名、隐藏公式编号和物理背景的匿名条件；另设 context-rich 条件，量化领域文本带来的收益。

### 7.2 压力条件

对公式复杂度分层后，使用平衡的部分因子设计覆盖：

- clean；
- 1% target noise + 5 个无关变量；
- 5% target noise + 10 个无关变量；
- 相关干扰变量；
- 小样本；
- 输入域外推。

每个任务都保留独立 IID test 和 OOD test。公式生成 seed 在方法冻结后封存，实验完成后公开。

### 7.3 表达式恢复指标的优先级

1. **精确代数恢复**：canonicalization 后验证符号等价；
2. **数值语义恢复**：在预注册域内用大量 Sobol/随机点比较无噪真函数，分别报告 IID 与 OOD 误差；
3. **公式骨架恢复**：把可重拟合数值常数替换为占位符后比较；
4. **结构恢复**：变量 support F1、运算符 multiset F1、非平凡子表达式 F1、归一化 tree-edit distance；
5. **复杂度差和域有效性**：相对真式的展开 MDL、奇点/溢出/无定义比例；
6. **约束行为恢复**：单位、单调性、对称性和极限行为匹配。

tree-edit distance 只能作为次指标。代数等价的表达式可能有完全不同的树，字符串相似度更不能作为主要证据。

## 8. 专家与 LLM 评分协议

### 8.1 人类专家是主证据

人工主指标优先使用“相近验证性能下的成对盲选”，Likert 五维分数用于解释差异。

建议五维 rubric：

| 维度 | 1 分 | 3 分 | 5 分 |
|---|---|---|---|
| 变量语义合理性 | 明显无关或疑似泄漏 | 大体相关但冗余/角色不清 | 变量精炼且领域角色明确 |
| 结构关系合理性 | 任意组合或违反基本知识 | 可作经验关系但机制模糊 | 比值、耦合、尺度律等有清楚含义 |
| 量纲/类型有效性 | 存在确定违规 | 信息不足或依赖明确归一化 | 运算与输出严格一致 |
| 认知可压缩性 | 难以自然语言概括 | 可拆成若干效应但较复杂 | 可概括为一两个清楚关系 |
| 可检验性/科学用途 | 无法导出后续验证 | 有趋势但方案模糊 | 能导出清楚边界、方向或实验 |

量纲信息未知时应记为 `NA`，不能默认中间分。

人工流程：

- 每个表达式至少 3 位对应领域专家；
- 方法名、R²、生成理由和 LLM 解释全部隐藏；
- mined/meta-factor 全展开、逆标准化、统一化简和排版；
- 候选按验证集表现匹配，不按测试集结果匹配；
- 随机展示顺序，10% 重复样本测同一评分者稳定性；
- 用平衡不完全区组控制工作量；
- 多评分者有序一致性报告 ordinal Krippendorff’s alpha 或 ICC 及 bootstrap CI；
- 成对偏好用 Bradley–Terry 或带 dataset/rater 随机效应的 logistic model。

若一致性 `alpha < 0.67`，不要强行汇总成一个“专家总分”，应退回分维度和成对分析。

### 8.2 LLM 只是经校准的代理

- 至少 3 个不同模型家族；
- 生成模型不能同时担任唯一主裁判；
- 固定模型快照、prompt、temperature 和输出 schema；
- 每个成对判断交换左右位置，测顺序偏差；
- 人类评分分成 calibration 与完全留出的 validation；
- 报告 LLM–human Spearman、pairwise agreement、MAE 和分数据集 CI；
- 可预设门槛，例如留出集 `rho >= 0.6` 且成对一致率 `>= 70%`；未达门槛则 LLM-only 结果只能是探索性结果；
- 不得把 LLM 分和专家分简单平均成“综合专家分”。

## 9. 消融与负对照

确认性消融：

1. raw PySR；
2. deterministic factor mining + PySR；
3. LLM proposal + PySR，无 data mining；
4. 完整候选池，但只按 accuracy + expanded complexity 选择；
5. 完整语义约束 workflow；
6. full workflow 去掉 LLM formula enhancement。

诊断性负对照：

- 数量、复杂度和相关性匹配的随机因子池；
- 打乱变量描述或错配 domain card；
- 匿名变量条件；
- 同一候选 archive 上分别用节点数、LLM judge、确定性语义约束重排；
- 量纲约束开/关；
- 不同 factor、token 和时间预算；
- 同模型评分自身输出与异模型评分的偏差。

这组实验必须回答：提升究竟来自更多计算、更多派生特征、领域语义、重排策略，还是事后公式润色。

## 10. 统计与预注册

### 10.1 预注册主检验

- H1：测试预测非劣效。无领域依据时可在 pilot 中暂用 `delta_R2 = 0.02`，但最终容差必须在确认性运行前冻结。
- H2：同精度候选的专家成对偏好概率显著大于 0.5。
- H3：自动语义违规率和展开复杂度降低。
- H4：污染受控真式任务的代数/数值语义恢复率提高。

### 10.2 推断单位

独立统计单位是数据集或生成公式，不是 fold、seed、target 或候选表达式。若 19 个 target 同源，应在层次模型中按源数据集聚类，避免伪重复。

建议使用：

- 数据集级配对 bootstrap 或 hierarchical model，报告效应量和 95% CI；
- 有序专家评分用 hierarchical ordinal model；
- 成对偏好用 Bradley–Terry/mixed logistic model；
- 精确恢复用配对二元或 mixed logistic model；
- 主比较只冻结 full vs 2 个强基线，其余比较做 Holm 校正；
- Pareto hypervolume 作为次指标，reference point 与归一化方式在开发集上冻结。

### 10.3 结果措辞边界

可以说：

- 更受领域专家偏好；
- 更少违反预定义领域约束；
- 在相似预测性能下更短、更稳定或更可检验；
- 更接近已知真式的语义/结构。

不能仅凭上述结果说：

- 恢复了真实因果机制；
- 发现了自然定律；
- LLM 理解了物理；
- 专家打分高就证明公式为真。

## 11. 12 周关键路径

这是最短可行版本；若专家招募或多方法复现延迟，现实排期应预留到 14–16 周。

| 周 | 工作 | 可交付物 | 决策门 |
|---|---|---|---|
| 1 | 冻结论文主张、H1–H4、成功标准；盘点现有数据 | preregistration v0；dataset inventory；风险表 | 若没有单位/变量定义/专家，收窄为应用论文或先补数据 |
| 1–2 | 建 dataset/domain-card schema；生成固定 outer/group splits | dataset manifests；split hashes；domain cards v0 | split 必须先于任何因子选择 |
| 2–3 | 把预处理、相关筛选、因子挖掘、常数拟合改成 fold-aware | fit/transform pipeline；无泄漏单测 | 泄漏测试不通过，不启动 benchmark |
| 3–4 | 实现表达式 lineage、完全展开、逆标准化、canonicalization、单位检查 | expression IR；expanded AST；equivalence tests | 所有方法必须进入统一表达式格式 |
| 4–5 | 实现语义约束编译、候选 archive、精度约束选择和稳定性指标 | scorer/selector；domain compiler；审计日志 | 主目标必须实际改变选择结果 |
| 4–5 | 接入 raw PySR、Operon/GP-GOMEA、SISSO++ 和一个现代 LLM/神经基线 | 统一 runner；预算与失败日志 | 至少两个强基线稳定运行 |
| 5 | 在 3–4 个开发数据集和 10–15 个合成公式上 pilot；专家 rubric 校准 | pilot report；功效与预算估计；评分界面 | 冻结数据、prompt、模型、算子、预算和统计脚本 |
| 6–8 | 跑全部真实/真式确认性实验和核心消融 | candidate archives；raw metrics；failure audit | 主运行有效率应 >=95% |
| 7–9 | 并行开展盲化专家成对评价和多 LLM 校准 | anonymized ratings；reliability report | 专家 alpha 不足则不用总分；LLM 未过门槛不用作主证据 |
| 9–10 | 执行冻结统计分析 | 非劣效图；偏好模型；Pareto/恢复/稳定性图 | H1+H2 必须联合成立才能作强主张 |
| 11 | 做预注册鲁棒性分析与 2–3 个案例研究 | ablation tables；case-study hypotheses | 禁止结果后调 prompt 或手工删公式 |
| 12 | 写论文并打包复现材料 | paper v1；container；code/data cards；rating release | 按证据强度选择投稿档位 |

## 12. 代码任务优先级

P0，任何大实验前必须完成：

1. `split-manifest`：固化 outer/group split 和 hash；
2. fold-aware `fit/transform`：填补、标准化、相关筛选、因子生成只 fit 训练折；
3. `candidate-archive`：统一记录所有候选和预算；
4. `expand-expression`：展开 mined/meta-factor lineage；
5. `inverse-scale-expression`：恢复原始变量与单位；
6. `refit-expression`：只在训练折重拟合常数；
7. `evaluate-expression`：val/test/OOD 分开，test 只读一次；
8. 修复 LLM prompt：真正嵌入候选内容并做 JSON schema 校验；
9. 移除或 fold 内重做历史预测 meta-factor；
10. 添加泄漏、展开、逆标准化和跨折一致性测试。

P1，构成论文方法：

1. `domain-card` schema；
2. `compile-domain-constraints`；
3. dimensional/type checker；
4. expanded complexity/MDL；
5. semantic violation evaluator；
6. accuracy-constrained/Pareto selector；
7. bootstrap structure stability；
8. ablation runner 和统一预算计量。

P2，构成评测与论文证据：

1. blind rating exporter/UI；
2. rating aggregation 与可靠性；
3. LLM judge calibration；
4. algebraic/numeric equivalence；
5. skeleton/subtree/operator/support similarity；
6. confirmatory statistics 和自动制图。

## 13. 投稿定位

| 项目完成度 | 论文性质 | 现实投稿定位 |
|---|---|---|
| 当前代码 + toy demo + 少量案例 | 工程串联/workflow demo | workshop、demo、软件说明；不建议直接投通用方法主会 |
| 修完泄漏，有一个领域的盲评、完整消融和严格预算，但方法创新有限 | 应用型或专项 SR 论文 | EuroGP、GECCO main/companion 视贡献强度；ACM TELO、GPEM/相关应用期刊；强领域结果可投对应领域期刊 |
| 可审计 domain compiler + 显式 accuracy-constrained semantic objective + 2 个领域 + 可靠专家研究 | 完整方法论文 | TMLR、IEEE TEVC、ACM TELO；GECCO/PPSN 等强专项 venue |
| 上述基础上再有跨领域泛化、公开 benchmark/评分数据、大规模强基线和明显 SOTA | 通用 ML/AI 方法与 benchmark | AAAI/IJCAI 有竞争力；ICLR/ICML/NeurIPS 属冲刺档，不应作为当前默认预期 |
| 公式产生了新科学发现，并有外部实验或独立数据验证 | 科学发现/领域论文 | 高水平领域期刊；Nature 系列等需要“新科学结果”，仅 workflow 优势远远不够 |

TMLR 明确接受“新算法及可靠实证”“新任务形式化与评测方法”以及揭示已有方法优缺点的应用研究，因此若方法和协议做扎实，是比盲冲 top conference 更现实的通用 ML 目标。[TMLR scope](https://www.jmlr.org/tmlr/editorial-policies.html)

如果最终最强的资产是新数据集、冻结候选式和多领域专家评分，而算法提升有限，也可以将其独立整理为 `SciInt-SRBench`，优先考虑 [DMLR](https://data.mlr.press/submissions.html) 或 TMLR；这比把 benchmark 和尚未成熟的方法强绑成一篇更稳。

GECCO 的 Genetic Programming track 明确覆盖 regression、feature engineering 和 feature selection，是主题高度匹配的专项社区。[GECCO GP track](https://gecco-2026.sigevo.org/Track?itemId=58)

最现实的建议是：

- **主投目标**：先按 TMLR / IEEE TEVC 级别把方法和实验做完整；
- **专项目标**：若算法创新集中在 GP/PySR 搜索与多目标选择，优先 GECCO/EuroGP/ACM TELO；
- **冲刺目标**：只有在多领域、公开 benchmark、强人类研究和跨任务泛化都成立时，再考虑 ICLR/ICML/NeurIPS；
- **快速路径**：若现有 19-target 数据能产生真正的新领域认识，就收窄成高质量领域应用论文，不必强行包装成通用 SR 新范式。

截至 2026-07-12，距离 AAAI-27 正文截止时间已很近；以当前代码状态仓促投稿只会牺牲实验可信度，不建议把它作为本轮排期目标。

## 14. Go / no-go 标准

在写摘要和选择 venue 前，先检查：

- 主运行有效率是否 `>=95%`；
- 是否有至少两个稳定强基线；
- 所有候选是否按展开后复杂度比较；
- outer test 是否从未参与因子选择和公式润色；
- H1 预测非劣效是否成立；
- H2 专家解释性优势是否成立；
- 专家一致性是否足以支持汇总分；
- LLM judge 是否通过独立人类留出验证；
- 真式任务是否排除了简单记忆解释；
- 优势是否在多个独立数据集而非同源 targets 上成立；
- 是否有至少一个可复现、可检验的领域案例，而不是只展示高分公式。

如果 H1 与 H2 没有同时成立，应如实降级结论：可能是更好的因子工程、某类任务上的精度方法，或一套解释性评测资源，但不能声称实现了 accuracy–interpretability Pareto improvement。

## 15. 建议题目

首选：

> **Beyond Parsimony: Accuracy-Constrained, Domain-Aware Symbolic Regression for Noisy Scientific Data**

备选：

> **Interpretability First, Accuracy Controlled: Auditable Domain Priors for Symbolic Regression**

若最终只是 workflow，不要在标题中使用容易被理解为新优化算法的 “constrained symbolic regression”。
