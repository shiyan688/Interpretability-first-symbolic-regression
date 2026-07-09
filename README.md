# Interpretability-first Symbolic Regression

一个“LLM 因子假设 + Python SISSO-like 因子枚举 + PySR 回归 + 可解释性增强”的符号回归工作流。

核心原则：

- 不只重跑 PySR，而是先做“LLM 因子假设 + Python SISSO-like 因子枚举 + 因子池筛选”。
- 对旧高分公式，既保留其原始因子，也加入公式预测值作为可展开的 meta-factor。
- 每轮 PySR 都必须保存 `best_result.json`、`hall_of_fame.csv` 快照、验证 R2、输入 manifest。
- LLM 只负责策略建议和因子解释，不替代数值验证。

## 完整工作流

项目按 4 段设计：

1. `LLM 提出特性因子`
   - 输入：数据集文法、原始变量、target、单变量相关性、缺失情况。
   - 输出：`llm_prompts/<target>/factor_proposal_prompt.md` 和 JSON 模板。
   - 作用：让大模型根据领域知识提出候选因子和变量组合方向。

2. `Python SISSO-like 挖因子`
   - 输入：raw feature table。
   - 方法：在 Python 中做 beam 枚举，不直接调用 SISSO。
   - 默认算符：`+ - * / abs square inv sqrt_abs log_abs`。
   - 控制参数：`base_top_k / pair_top_k / beam_width / final_top_k / max_order`。
   - 输出：`factor_pools/<target>/mined_factors.csv` 和 `mined_factor_values.csv`。

3. `LLM 选因子 + PySR 回归`
   - 输入：高相关性因子池、LLM 选择结果、原始变量。
   - 输出：`feature_tables/<target>__pysr_pool/`。
   - 作用：把高相关、可解释、有领域意义的 mined factors 与原始变量一起放进 PySR。

4. `LLM 解释性增强`
   - 输入：PySR best/HOF/expr list。
   - 输出：解释性增强 prompt，要求展开 mined factor/meta-factor，做量纲对齐、结构润色、重新拟合常数、重新验证 R2/K-fold R2。

## 安装方式

在已有环境中直接用源码运行：

```bash
cd Interpretability-first-symbolic-regression
python -m factor_pysr_llm.cli --help
```

如果在新环境中安装：

```bash
python -m pip install -e .
```

## 典型流程

### 通用 CSV 数据集

本项目不假设固定数据集。任意表格数据先用 `dataset` 文法声明列角色：

```json
{
  "input_csv": "/path/to/data.csv",
  "output_root": "/path/to/outputs/run1",
  "targets": ["target_y"],
  "dataset": {
    "format": "tabular_csv_v1",
    "id_columns": ["structure_id", "sample_id"],
    "target_columns": ["target_y"],
    "feature_rules": {
      "include": [],
      "regex": [],
      "exclude": [],
      "exclude_regex": [],
      "numeric_only": true
    },
    "missing": {
      "fill": "median",
      "add_indicators": true
    },
    "scaling": "zscore",
    "naming": {
      "safe_prefix": "raw_",
      "max_length": 96
    }
  }
}
```

字段规则：

- `id_columns`：不参与建模的编号/结构名列。
- `target_columns`：明确的 Y 列。也可以用 `target_rules.prefixes` 或 `target_rules.regex` 自动识别。
- `feature_rules.include`：为空表示候选所有非 ID、非 target 列；非空表示白名单。
- `feature_rules.regex`：只保留匹配正则的特征列。
- `feature_rules.exclude` / `exclude_regex`：排除列。
- `missing.fill`：`median`、`mean` 或 `zero`。
- `missing.add_indicators`：有缺失的列额外生成 `__is_missing` 指示变量。
- `scaling`：`zscore` 或 `none`。
- `raw_feature_selection.mode`：`all` 或 `top_k_abs_corr`。

检查文法解析：

```bash
python -m factor_pysr_llm.cli inspect-dataset \
  --config configs/generic_tabular_template.json
```

构建 raw feature table：

```bash
python -m factor_pysr_llm.cli build-raw \
  --config configs/generic_tabular_template.json \
  --target target_y
```

生成 LLM 因子提议 prompt：

```bash
python -m factor_pysr_llm.cli llm-propose-factors \
  --config configs/generic_tabular_template.json \
  --target target_y
```

Python SISSO-like 挖因子：

```bash
python -m factor_pysr_llm.cli mine-factors \
  --config configs/generic_tabular_template.json \
  --target target_y \
  --llm-proposals /path/to/factor_proposals.json
```

`--llm-proposals` 是可选项；如果提供，LLM 提出的表达式会先按当前
raw feature table 计算，作为额外 seed 进入 Python 枚举和后续因子池。

生成 LLM 因子池筛选 prompt：

```bash
python -m factor_pysr_llm.cli llm-select-factors \
  --config configs/generic_tabular_template.json \
  --target target_y
```

合并原始变量和 mined/LLM 选择因子，形成 PySR 输入：

```bash
python -m factor_pysr_llm.cli build-pysr-pool \
  --config configs/generic_tabular_template.json \
  --target target_y \
  --llm-selection /path/to/factor_selection.json
```

然后直接接 PySR：

```bash
python -m factor_pysr_llm.cli run-pysr \
  --config configs/generic_tabular_template.json \
  --target target_y \
  --feature-dir /path/to/feature_tables/target_y__pysr_pool \
  --run-name pysr_factor_pool_target_y
```

生成解释性增强 prompt：

```bash
python -m factor_pysr_llm.cli llm-interpret \
  --config configs/generic_tabular_template.json \
  --target target_y \
  --result /path/to/best_result.json
```

### 复用历史高分因子

1. 构建 effective-union 输入表：

```bash
python -m factor_pysr_llm.cli build-union \
  --config configs/history_union_template.json \
  --target target_y
```

2. 维护跨目录表达式清单：

```bash
python -m factor_pysr_llm.cli mine-exprs \
  --roots /path/to/previous_run_a \
          /path/to/previous_run_b \
          /path/to/previous_run_c \
  --output /path/to/outputs/expr_list.csv \
  --top-k-per-target 1000
```

3. 跑 PySR：

```bash
python -m factor_pysr_llm.cli run-pysr \
  --config configs/history_union_template.json \
  --target target_y \
  --run-name pysr_history_union_target_y \
  --maxsize 50 \
  --procs 32 \
  --population-size 1000 \
  --populations 96 \
  --timeout-seconds 86400
```

4. 验证当前 best 或 HOF：

```bash
python -m factor_pysr_llm.cli verify \
  --feature-dir /path/to/feature_tables/target_y \
  --result /path/to/best_result.json
```

5. 生成 LLM brief：

```bash
python -m factor_pysr_llm.cli llm-brief \
  --config configs/history_union_template.json \
  --target target_y
```

## 输出结构

```text
outputs/
  feature_tables/<target>/
    features.csv
    hybrid_features.csv        # 兼容旧脚本
    y.csv
    manifest.json
  factor_pools/<target>/
    mined_factors.csv
    mined_factor_values.csv
    manifest.json
  llm_prompts/<target>/
    factor_proposal_prompt.md
    factor_proposals_template.json
    factor_selection_prompt.md
    factor_selection_template.json
    interpretability_prompt.md
  runs/<run_name>/<target>/
    best_result.json
    model_equations_snapshot.csv
  llm_briefs/<target>_brief.md
```

## 说明

`configs/history_union_template.json` 是历史结果合并模板：它可以合并旧 PySR raw 输入、
旧因子表、旧最佳公式 prediction、旧 HOF/equation snapshot prediction。实际使用时将
`/path/to/...` 替换成自己的结果目录。
