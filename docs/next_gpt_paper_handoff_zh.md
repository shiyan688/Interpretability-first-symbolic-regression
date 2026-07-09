# 给下一个 GPT 的论文交接提示词

你正在接手一个论文与代码项目：`Interpretability-first-symbolic-regression`。

项目核心不是“再造一个更强的黑箱符号回归器”，而是提出一个 **可解释性优先的符号回归工作流**：

> 真实科学数据通常有噪声、变量耦合强、复杂度高，很多情况下并不存在一个可以被符号回归精确复原的“真公式”。但符号回归仍然有价值，因为它能给出短小、结构化、可解释、可被专家讨论和修正的表达式。现有方法通常主要优化拟合精度或表达式复杂度，对“领域可解释性”的约束不足。本文主张：符号回归应该从“只追求 R2 的搜索”转向“可解释性优先、精度受控”的搜索与评估。

## 论文暂定主线

题目方向：

- Interpretability-first Symbolic Regression
- 可解释性优先的符号回归
- LLM-guided interpretable factor mining for symbolic regression

核心观点：

1. 真实数据集往往噪声高、变量复杂、机制不完全可观测，因此“复原唯一真实表达式”不是现实目标。
2. 对这类数据，符号回归的主要价值是提出可解释、可检验、可被专家修正的候选规律。
3. 现有符号回归方法的优化目标通常是 `loss + complexity penalty`，但表达式复杂度不等价于科学可解释性。
4. 我们的流程显式引入“可解释性优先”的结构：
   - LLM 根据数据集背景和变量含义提出特性因子。
   - Python 实现 SISSO-like 因子枚举，不直接依赖 SISSO。
   - 先从因子池中选出高相关且有意义的因子，再和原始变量一起交给 PySR。
   - 对 R2 较好的表达式进行可解释性增强：展开中间因子、量纲对齐、结构润色、重新拟合常数、重新验证。
5. 文章要证明：在真实复杂任务上，我们的方法能在相近 R2 下得到更可解释的表达式；在有真公式的任务上，能更接近真实表达式结构。

## 当前代码项目对应的工作流

代码仓库已经实现一个初版工作流：

```text
dataset grammar
  -> raw feature table
  -> LLM factor proposal prompt
  -> Python SISSO-like factor mining
  -> LLM factor selection prompt
  -> raw + selected/mined factors as PySR input
  -> PySR search
  -> verification
  -> LLM interpretability enhancement prompt
```

关键命令：

```bash
python -m factor_pysr_llm.cli inspect-dataset --config examples/toy_config.json
python -m factor_pysr_llm.cli build-raw --config examples/toy_config.json --target y
python -m factor_pysr_llm.cli llm-propose-factors --config examples/toy_config.json --target y
python -m factor_pysr_llm.cli mine-factors --config examples/toy_config.json --target y --llm-proposals examples/factor_proposals.example.json
python -m factor_pysr_llm.cli llm-select-factors --config examples/toy_config.json --target y
python -m factor_pysr_llm.cli build-pysr-pool --config examples/toy_config.json --target y --llm-selection examples/factor_selection.example.json
python -m factor_pysr_llm.cli run-pysr --config examples/toy_config.json --target y --run-name pysr_toy_factor_pool
```

LLM API 配置：

```bash
cp configs/llm_provider.example.json configs/llm_provider.local.json
export OPENAI_API_KEY="..."
```

调用 prompt：

```bash
python -m factor_pysr_llm.cli llm-call \
  --provider-config configs/llm_provider.local.json \
  --prompt-file outputs/toy_regression_run/llm_prompts/y/factor_proposal_prompt.md \
  --output outputs/toy_regression_run/llm_prompts/y/factor_proposals.llm.json \
  --extract-json
```

## 论文实验设计

### 实验 1：真实/黑盒数据集，无明确真公式

目的：

证明在真实噪声和高复杂度数据上，可解释性优先流程能得到更可解释的表达式，同时保持有竞争力的 R2。

数据集选择：

- 没有明确封闭表达式答案的科学/工程数据集。
- 应有变量名和基本领域含义，方便专家和 LLM 判断解释性。
- 数据不能只看 R2，必须能讨论变量含义和表达式结构。

候选数据类型：

- 材料性质预测数据。
- 催化/吸附能/反应能数据。
- 物理化学模拟数据。
- 工程测量数据。
- 也可以使用当前用户已有的 19 target 数据集和 4 个 2000 数据集作为内部主案例。

比较方法：

- PySR raw features。
- DSO / deep-symbolic-optimization。
- gplearn 或 Genetic Programming baseline。
- AI Feynman 或类似物理符号回归方法。
- SISSO / SISSO++ 如果许可和环境允许。
- 我们的方法：LLM factor proposal + Python SISSO-like factor mining + LLM factor selection + PySR。

评价指标：

1. 拟合性能：
   - train R2
   - test R2 或 K-fold R2
   - RMSE / MAE
2. 表达式复杂度：
   - 节点数
   - 运算符数量
   - 使用变量数量
   - 嵌套深度
3. 可解释性评分：
   - 人类专家评分。
   - LLM 评分。
   - 最好做 blind evaluation，不展示方法名。
   - 最好报告专家间一致性，例如 Spearman/Kendall 或 Cohen's kappa。

可解释性评分建议拆成 5 个维度，每个 1-5 分：

1. 变量相关性：表达式是否使用了领域上合理的变量。
2. 结构合理性：变量组合是否符合领域经验，例如比值、差值、耦合项是否有意义。
3. 量纲/尺度合理性：表达式是否明显量纲混乱，是否需要作为 screening factor 而非最终物理公式。
4. 简洁性：表达式是否短、可读、没有无意义嵌套。
5. 可讨论性/可修正性：专家能否基于表达式提出机制解释或后续实验验证。

最终可以报告：

```text
accuracy_score = test_R2 或 K-fold R2
interpretability_score = 专家/LLM 综合评分
frontier = R2 与 interpretability 的 Pareto frontier
```

不要只说“我们 R2 最高”。主张应是：在相近 R2 下，我们的表达式解释性更好；或在相近解释性下，我们的 R2 更高。

### 实验 2：复杂但有明确真公式的数据集

目的：

证明我们不仅能在真实黑盒数据上给出可解释表达式，也能在有真实表达式的任务上更接近真实结构。

数据集选择：

- Feynman-like equations。
- 科学公式数据集。
- 自己构造的复杂表达式数据集。
- 可以加入噪声、冗余变量、变量重命名、无关变量、缺失值，模拟真实科学数据。

评价指标：

1. 拟合性能：
   - train/test/K-fold R2
   - RMSE
2. 可解释性评分：
   - 同实验 1。
3. 表达式相似度：
   - 与 ground-truth expression 的 symbolic tree edit distance。
   - 使用变量集合 overlap。
   - 运算符集合 overlap。
   - 子表达式 overlap。
   - 归一化复杂度差异。
   - 数值等价测试：在新采样点上比较 ground truth 与 recovered expression。

表达式相似度不要只用字符串相似度。应该优先考虑：

- 解析成表达式树。
- 做简单 canonicalization，例如交换律排序、常数折叠、无意义括号去除。
- 对候选表达式和真公式在独立采样点上做数值等价/近似等价检查。

## 我对当前论文思路的判断

这个方向是有价值的，但必须避免两个风险。

风险 1：可解释性太主观。

解决方式：

- 设计明确评分 rubric。
- 做 blind evaluation。
- 使用多个评分者：至少 LLM 多模型 + 人类专家。
- 报告评分一致性。
- 把 LLM 评分定位为 scalable proxy，把专家评分作为关键验证。

风险 2：LLM 参与会被质疑为“人为提示带来的偏置”。

解决方式：

- 让所有方法使用相同原始变量。
- 明确我们的方法使用 LLM 的位置：只用于因子假设和解释性筛选，不直接决定最终 R2。
- 所有候选表达式必须经过同样的数据验证。
- 做 ablation：
  - raw PySR
  - Python factor mining only
  - LLM proposal only
  - LLM proposal + factor mining
  - full workflow

## 建议的论文贡献点

可以写成 3 个 contribution：

1. 提出 interpretability-first symbolic regression framework，将领域因子假设、数据驱动因子挖掘、符号回归和解释性增强连接成闭环。
2. 实现一个不依赖 SISSO 的 Python SISSO-like factor mining 模块，用 beam search 枚举高相关候选因子，并允许 LLM 因子作为 seed。
3. 提出面向符号回归的可解释性评估协议，在真实黑盒数据集和有 ground truth 的复杂公式数据集上验证 R2、表达式相似度和可解释性评分。

## 下一步具体任务

优先级从高到低：

1. 完善 paper experiment plan：
   - 明确实验 1 数据集列表。
   - 明确实验 2 ground-truth equation 数据集列表。
   - 明确 baseline 方法和运行预算。
2. 完善 interpretability rubric：
   - 写成可给专家和 LLM 使用的评分表。
   - 每项 1-5 分，给示例。
   - 支持 blind evaluation。
3. 代码侧新增：
   - `score-interpretability-prompt`：给候选表达式生成评分 prompt。
   - `aggregate-ratings`：汇总 LLM/专家评分。
   - `expression-similarity`：计算表达式结构相似度。
   - `cross-validate-expression`：对固定表达式重新拟合常数并 K-fold 验证。
4. 做 ablation：
   - no LLM
   - no factor mining
   - raw features only
   - mined factors only
   - full workflow
5. 写论文草稿：
   - Introduction：真实数据难复原真公式，但可解释候选规律仍然有价值。
   - Related Work：SR、SISSO、PySR、LLM for science、interpretability evaluation。
   - Method：四段式 workflow。
   - Experiments：黑盒真实数据 + 有真公式数据。
   - Discussion：解释性、偏置、限制。

## 给下一个 GPT 的直接提示词

请接手这个项目并继续推进论文与代码。论文主线是“可解释性优先的符号回归”：真实科学数据通常噪声高、机制复杂，符号回归不一定能复原唯一真公式，但可以给出高可解释、可验证、可被专家修正的候选表达式。现有方法主要优化 R2 和复杂度，对领域可解释性约束不足。我们的方法用 LLM 提出领域因子假设，用 Python 实现 SISSO-like 因子枚举，用 LLM 从因子池中筛选有意义因子，再与原始变量一起交给 PySR，最后对高 R2 表达式做解释性增强、展开和重新验证。

你需要优先做三件事：

1. 把论文实验设计具体化：
   - 实验 1：真实/黑盒数据集，无明确真公式。比较不同符号回归方法，评价 R2、K-fold R2、复杂度、人类专家和 LLM 的可解释性评分。
   - 实验 2：复杂但有明确真公式的数据集。除了上述指标，再加表达式相似度，包括 tree edit distance、变量/运算符 overlap、子表达式 overlap、独立采样点数值等价。
2. 设计可解释性评分 rubric：
   - 变量相关性、结构合理性、量纲/尺度合理性、简洁性、可讨论性。
   - 每项 1-5 分。
   - 支持 blind evaluation 和多评分者一致性统计。
3. 检查并扩展代码仓库：
   - 仓库名：`Interpretability-first-symbolic-regression`。
   - 目前已有 `factor_pysr_llm` 包、`examples/toy_config.json`、LLM API provider config、Python SISSO-like factor mining、PySR runner、verification。
   - 下一步应新增 interpretability scoring、rating aggregation、expression similarity、fixed-expression refit/K-fold verification、ablation runner。

请始终保持两个原则：

- 不要只追 R2。论文核心是 R2 与可解释性的 Pareto improvement。
- LLM 只能提出假设、筛选因子、辅助解释；所有最终结论必须由数值验证、专家评分和可复现实验支撑。

