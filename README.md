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

Clone 后直接安装：

```bash
git clone https://github.com/shiyan688/Interpretability-first-symbolic-regression.git
cd Interpretability-first-symbolic-regression
python -m pip install -e .
python -m factor_pysr_llm.cli --help
```

## 典型流程

### LLM API 配置

LLM API 不写在数据集 config 里，单独放在 provider config 里，避免和实验配置混在一起。

仓库提供两个示例：

- `configs/llm_provider.example.json`：OpenAI-compatible 默认示例。
- `configs/llm_provider.deepseek.example.json`：DeepSeek/OpenAI-compatible 示例。

使用方式：

```bash
cp configs/llm_provider.example.json configs/llm_provider.local.json
```

然后二选一：

1. 推荐：只填环境变量名，不把 key 写进文件。

```json
{
  "base_url": "https://api.openai.com/v1",
  "api_key_env": "OPENAI_API_KEY",
  "api_key": "",
  "model": "gpt-4o-mini"
}
```

运行前设置：

```bash
export OPENAI_API_KEY="your_api_key_here"
```

2. 本地临时使用：把 key 填到 `configs/llm_provider.local.json` 的 `api_key` 字段。

`configs/llm_provider.local.json` 已经写进 `.gitignore`，不会被提交到仓库。

调用一个 prompt 文件：

```bash
python -m factor_pysr_llm.cli llm-call \
  --provider-config configs/llm_provider.local.json \
  --prompt-file outputs/toy_regression_run/llm_prompts/y/factor_proposal_prompt.md \
  --output outputs/toy_regression_run/llm_prompts/y/factor_proposals.llm.json \
  --extract-json
```

`--extract-json` 会从模型回复里提取 JSON，适合直接接到：

```bash
python -m factor_pysr_llm.cli mine-factors \
  --config configs/generic_tabular_template.json \
  --target y \
  --llm-proposals outputs/toy_regression_run/llm_prompts/y/factor_proposals.llm.json
```

### 30 秒 Demo

仓库自带一个小型 toy CSV：`examples/toy_regression.csv`。路径全部是相对路径，
用户 clone 后可以直接跑数据解析、因子挖掘、LLM prompt 生成和 PySR 输入表构建：

```bash
python -m factor_pysr_llm.cli inspect-dataset \
  --config configs/generic_tabular_template.json

python -m factor_pysr_llm.cli build-raw \
  --config configs/generic_tabular_template.json \
  --target y

python -m factor_pysr_llm.cli llm-propose-factors \
  --config configs/generic_tabular_template.json \
  --target y

python -m factor_pysr_llm.cli mine-factors \
  --config configs/generic_tabular_template.json \
  --target y \
  --llm-proposals examples/factor_proposals.example.json

python -m factor_pysr_llm.cli llm-select-factors \
  --config configs/generic_tabular_template.json \
  --target y

python -m factor_pysr_llm.cli build-pysr-pool \
  --config configs/generic_tabular_template.json \
  --target y \
  --llm-selection examples/factor_selection.example.json
```

输出会写到 `outputs/toy_regression_run/`。PySR 是可选依赖，装好 PySR 后再跑：

```bash
python -m factor_pysr_llm.cli run-pysr \
  --config configs/generic_tabular_template.json \
  --target y \
  --run-name pysr_toy_factor_pool
```

也可以用脚本跑完整 demo 的非 PySR 部分：

```bash
bash scripts/run_generic_pipeline.sh
```

如果已经安装 PySR，并且希望脚本最后启动 PySR：

```bash
RUN_PYSR=1 bash scripts/run_generic_pipeline.sh
```

### 通用 CSV 数据集

本项目不假设固定数据集。任意表格数据先用 `dataset` 文法声明列角色：

```json
{
  "input_csv": "../data/my_data.csv",
  "output_root": "../outputs/my_run",
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

配置文件中的相对路径按“配置文件所在目录”解析。比如配置在 `configs/` 下，
`"../data/my_data.csv"` 指向仓库根目录的 `data/my_data.csv`。

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
  --llm-proposals examples/factor_proposals.example.json
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
  --llm-selection examples/factor_selection.example.json
```

然后直接接 PySR：

```bash
python -m factor_pysr_llm.cli run-pysr \
  --config configs/generic_tabular_template.json \
  --target target_y \
  --run-name pysr_factor_pool_target_y
```

生成解释性增强 prompt：

```bash
python -m factor_pysr_llm.cli llm-interpret \
  --config configs/generic_tabular_template.json \
  --target target_y \
  --result outputs/my_run/runs/pysr_factor_pool_target_y/target_y/best_result.json
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
  --roots previous_runs/run_a \
          previous_runs/run_b \
          previous_runs/run_c \
  --output outputs/history_union_run/expr_list.csv \
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
  --feature-dir outputs/history_union_run/feature_tables/target_y \
  --result outputs/history_union_run/runs/pysr_history_union_target_y/target_y/best_result.json
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
`data/`、`previous_runs/`、`outputs/` 换成自己的相对目录即可。
