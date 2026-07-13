# 阶段一报告：范围、数据与无泄漏管线

更新日期：2026-07-13
对应任务来源：`docs/first_three_weeks_execution_prompt_zh.md` 第四节。

## 本阶段结论：通过

阶段一在 toy 数据上已证明：改变 test labels（以及仅改动 test 行的特征极端值）
不会改变 train-fit 得到的预处理参数、被选特征、挖掘因子及最终因子池选择。

## 已完成

### 范围冻结（configs/）

- `configs/paper_scope.yaml`：冻结两个研究问题、两个实验、6 个真实主任务、
  20 个真式任务、五种比较条件、主指标、`δ=0.02`、3 个确认性 seed、rubric 版本、
  ExprSim 权重、不可协商项与资源缩减顺序。
- `configs/dataset_inventory.yaml`：每个源表记录来源、source dataset ID、target、
  样本数、变量含义、单位状态、group split 候选字段、是否共用源表、是否可公开、
  是否入选主任务及原因。toy 数据显式标记 `development_only: true`。真实候选为待填
  骨架，未看结果前不得提升为主任务。
- `configs/known_formula_tasks.yaml`：冻结分层抽取规则（按变量数、节点数、算子类型，
  固定 seed），并给出候选公式池。`selected_tasks` 在运行确定性抽取器前为空。

### 无泄漏 split 基础设施

- `factor_pysr_llm/splits.py`：
  - `SplitManifest`：保存 row IDs（非仅比例）、seed、fractions、mode、
    group assignment，并计算顺序无关的 `content_sha256`。
  - `make_random_split` / `make_group_split`：固定 seed 可复现；group split 保证同组
    不跨集合。
  - `build_split_manifest`：从 target-finite 表构建，自动选取 ID 列。
  - `check_split_manifest`：互斥、覆盖、无重复、group 合法性检查。
  - `save/load_split_manifest`：JSON + SHA256 校验；加载时验证 hash 一致。
  - `role_masks_for_frame`：把 manifest 映射为 train/validation/test 布尔掩码。
- CLI：`generate-split`（mode/seed/fractions/id-column/group-column/output）。

### train-fit 预处理

- `factor_pysr_llm/preprocess.py`：
  - `fit_preprocess`：缺失填补值、z-score 均值/方差、常数列判断全部**只用 train 行**拟合。
  - `transform_preprocess`：把 fitted state 确定性应用到全部行。
  - `feature_scores_train` / `fit_feature_selection`：相关筛选只读 train labels。
- `dataset.build_raw_feature_table` 新增可选 `split_manifest_path`，走
  `_build_raw_feature_table_split` 无泄漏路径；输出 `row_roles.csv` 和
  `preprocess_state`，headline `train_linear_r2` 仅在 train 上计算。
- `factor_miner.mine_factors_from_frame` / `mine_factors` / `proposed_factor_frame`
  新增 train mask：所有相关性排序与 beam 选择只在 train 行上计算。mine_factors 自动读取
  `row_roles.csv`。

### 测试

- `tests/test_splits.py`（5）：随机 split 可复现且不同 seed 不同、group 不交叉、
  manifest SHA256 往返、保存 row IDs 而非比例、group split 端到端。
- `tests/test_no_leakage.py`（4）：
  - 改动 test 行特征极端值不改变 train-fit 均值/方差/被选特征；
  - 改动 test labels 不改变挖掘因子表达式、train 相关性得分和因子池选择；
  - 同 seed split 产生一致 row IDs。

## 验证证据

```bash
VP=/public/home/wangyg/workspace/llm_pysr_project/.venv/bin/python
$VP -m pytest -q            # 14 passed
$VP -m pytest -q tests/test_splits.py tests/test_no_leakage.py   # 8 passed
```

最小端到端产物：

```bash
$VP -m factor_pysr_llm.cli generate-split --config examples/toy_config.json --target y
# -> outputs/toy_regression_run/splits/y__random_seed20260709.json  (含 sha256)
$VP -m factor_pysr_llm.cli build-raw --config examples/toy_config.json --target y \
    --split-manifest outputs/toy_regression_run/splits/y__random_seed20260709.json
# -> feature_tables/y/{features.csv, row_roles.csv, manifest.json(builder=..._train_fit)}
```

现有 6 个基线测试未被破坏（14 = 原 6 + 新 8）。

## 仍存在的风险

- **真实数据尚未接入**：`dataset_inventory.yaml` 的 6 个真实主任务仍是骨架。无泄漏
  管线已在 toy 与合成数据上验证，但 group split 字段（material/support/batch）需要真实
  源表才能落定。影响：阶段三 pilot 前必须补齐，否则实验 1 无法启动。
- **known_formula 抽取器**：规则与候选池已冻结，但确定性抽取器（`known_formulas.py`）
  与 `sample-known-formulas` CLI 在阶段三实现，当前 `selected_tasks` 为空。
- **非 split 旧路径仍存在**：`build_raw_feature_table` 不传 split 时仍是全表拟合（供
  toy demo 与向后兼容）。确认性实验必须显式传 split manifest。

## 下一步（阶段二，最高优先级）

1. 实现 `lineage.py`：factor card schema、递归展开、循环检测、展开到原始变量、
   逆标准化、展开前后数值一致性检查。
2. 实现 `ifsr_selector.py`：`δ=0.02` 容忍带 + 词典序选择，只读 validation，选定后才读 test。
3. 修复 LLM factor prompt（嵌入候选内容）与 selection no-op，并加测试证明不同 selection → 不同候选池。
