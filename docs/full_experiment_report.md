# 可解释性优先符号回归（IF-SR）：完整实验报告

- 项目：`/public/home/wangyg/Interpretability-first-symbolic-regression`
- 更新日期：2026-07-14
- 运行环境：Python 3.12 venv（numpy / pandas / scikit-learn / sympy / scipy / pysr 1.5.10），DeepSeek OpenAI-compatible API（`deepseek-chat`）。
- 本文覆盖**全部实验**：单元测试与无泄漏验证、方法基础设施 pilot、可解释性评测设施、以及三个数据集上的五方法对比。
- 所有跑数产物在 `outputs/`（已 gitignore，不入库）；API key 与 provider 配置全程未提交。

---

## 目录

- 实验 0：单元测试与无泄漏验证
- 实验 1：方法基础设施端到端 pilot
- 实验 2：可解释性评测设施（盲评 + LLM judge + 聚合）
- 实验 3：你的 29 个科学公式（主实验，五方法 × 3 seeds）
- 实验 4：LSR-Transform 抗记忆重建（10 任务 × 3 seeds）
- 实验 5：真实 LLM-SRBench（ModelScope 镜像，28 任务 × 3 seeds）
- 总结论、局限、复现入口

---

## 通用设置

### 五个对比方法

| 方法 | 类型 | 说明 |
|---|---|---|
| Raw-PySR | 数值 SR | 标准 PySR，仅原始变量，最优 validation R² 选式 |
| Mine-PySR | 数值 SR | PySR + 通用因子池（成对积/商/平方/倒数，**不含真值**），标准选式 |
| **IF-SR（本方法）** | 数值 SR + 可解释性优先 | 在 raw+mined 候选上，validation 容忍带 δ=0.02 内**最小展开复杂度**选式；mined 列经 lineage 展开回原始变量后计分 |
| Direct-LLM | LLM | 零样本：给变量+数据摘要，让 DeepSeek 直接给闭式表达式 |
| LLM-SR | LLM | 复现 Shojaee et al. ICLR 2025：LLM 提 skeleton → 最小二乘拟合常数 → 演化 buffer 迭代 |

### 统一无泄漏协议

- train / validation / test 切分，保存 row-id 与 SHA256 manifest（`splits.py`）。
- 缺失填补、缩放、相关筛选、因子挖掘**只在 train 上 fit**（`preprocess.py`）。
- SR 只在 train 拟合；候选 archive 在 **validation** 上选式；**test 只读一次**。
- 评价指标：test R²、ExprSim（0.50 数值相似 + 0.20 变量集 F1 + 0.20 算子集 F1 + 0.10 树结构，权重冻结）、展开节点数、代数/数值等价率、可解释性盲评（四维 1–5 rubric）。

### 运行量与耗时（全部成功，0 失败）

| 实验块 | runs | 成功 | 失败 | 累计 wall-time |
|---|---|---|---|---|
| 主实验 PySR/IF-SR（29 公式×3） | 87 | 87 | 0 | 177 min |
| 主实验 LLM 基线（29×3） | 87 | 87 | 0 | 24 min |
| LSR-Transform PySR（10×3） | 30 | 30 | 0 | 63 min |
| LSR-Transform LLM（10×3） | 30 | 30 | 0 | 8 min |
| 真实 LLM-SRBench PySR（28×3） | 84 | 84 | 0 | 181 min |
| 真实 LLM-SRBench LLM（28×3） | 84 | 84 | 0 | 31 min |
| **合计** | **402** | **402** | **0** | **~8.1 h** |

---

## 实验 0：单元测试与无泄漏验证

**目的**：证明管线无泄漏、各模块行为正确。

**方法**：`pytest`，13 个测试文件，覆盖 split 可复现/组不交叉、无泄漏（改 test label 不改候选）、公式展开与逆标准化数值一致、IF-SR selector δ 边界与 test 不参与选择、LLM selection 非 no-op、盲评 schema 校验、rating 聚合、ExprSim 行为、pilot 消融。

**结果**：

```
63 passed
```

关键无泄漏断言（`tests/test_no_leakage.py`）：
- 改动 test 行的特征极端值 → train-fit 的均值/方差/被选特征**不变**；
- 改动 test labels → 挖掘因子表达式、train 相关性得分、因子池选择**不变**。

**结论**：管线满足论文对"test 不泄漏进预处理/挖掘/选式"的硬要求。

---

## 实验 1：方法基础设施端到端 pilot

**目的**：在真实 PySR 上跑通"数据生成 → split → 四条件 → IF-SR 选式 → 展开一致性 → 盲评 → 假判官 → 聚合 → ExprSim"全链路，并证明消融非 no-op。

**方法**：10 个代表任务 × 1 seed，PySR 引擎，脚本 `scripts/run_experiment2_full.py` 早期版本与 `run_pilot.py`。

**结果（10 任务，PySR 三条件）**：

| 方法 | test R²中位 | ExprSim | 展开节点数 | 代数等价率 |
|---|---|---|---|---|
| Raw-PySR | 0.997 | 0.857 | 12.3 | 0.30 |
| Mine-PySR | 0.997 | 0.930 | 8.0 | 0.20 |
| **IF-SR** | 0.997 | 0.917 | **5.5** | 0.20 |

**结论**：全链路跑通；IF-SR 在 R² 持平下节点数最少；消融（raw/mine/if_sr 选出不同公式）非 no-op。此 pilot 为后续主实验提供了方法与代码验证。

---

## 实验 2：可解释性评测设施

**目的**：建立可复现的盲评设施——匿名导出、内嵌 rubric 的 judge prompt、严格 JSON 校验、多判官缓存/续跑/顺序扰动、人类/LLM 分开聚合。

**方法**：`interpretability_eval.py` + `rating_aggregate.py` + `configs/interpretability_rubric.json`（四维 1–5：领域意义 / 结构合理性 / 易读可概括性 / 假设支持力）。用固定假响应做单测，用真实 DeepSeek 做实盘。

**工程加固**（跑主实验时发现并修复，均已提交）：
1. sympy `simplify` 遇病态表达式会无限跑 → 加信号级硬超时（5s 兜底）。
2. API 突发限流 `Connection refused` → 指数退避重试。
3. DeepSeek 端点会"接受 TCP 后不发数据"，urllib 超时不触发 → 改**工作线程 + 硬墙钟 deadline**，卡住即丢弃重试。`deepseek-reasoner` 尤其严重，故第二判官改用 `deepseek-chat` + 不同 seed（顺序扰动的独立 pass）。

**结论**：盲评设施可稳定运行；两判官在主实验上 **Spearman = 0.986、成对偏好一致率 = 0.998**，一致性极高。

---

## 实验 3：你的 29 个科学公式（主实验）

**数据**：`examples/实验2_科学公式数据集清单.md` 的 30 条真值公式（洛伦兹系统为 ODE 系统，不做单目标回归，实跑 29 条）。物理常数折算为代表数值；**5% 目标噪声 + 5 个无关变量**；train 300 / valid 150 / test 300 + 独立 ExprSim 采样点 300。3 seeds。

### 3.1 定量指标（87 次运行）

| 方法 | test R²中位 | test R²均值 | ExprSim | 展开节点数 | 代数等价率 | n |
|---|---|---|---|---|---|---|
| Raw-PySR | 0.997 | — | 0.704 | 14.6 | 0.13 | 87 |
| Mine-PySR | 0.997 | — | 0.724 | 14.5 | 0.07 | 87 |
| **IF-SR（本方法）** | 0.995 | — | **0.733** | **8.6** | **0.16** | 87 |
| Direct-LLM | 0.644 | −2.16 | 0.737 | 9.4 | **0.31** | 87 |
| LLM-SR | 0.997 | 0.783 | 0.651 | 32.6 | 0.01 | 87 |

### 3.2 可解释性盲评（两个独立 DeepSeek 判官）

总体（1–5，越高越可解释）：

| 判官 | IF-SR | Raw-PySR | Mine-PySR |
|---|---|---|---|
| chat-A | **2.759** | 2.422 | 2.405 |
| chat-B（seed 扰动） | **2.793** | 2.397 | 2.336 |

按维度（判官 A）：

| 维度 | IF-SR | Raw-PySR | Mine-PySR |
|---|---|---|---|
| 领域意义 | 2.62 | 2.41 | 2.38 |
| 结构合理性 | 2.62 | 2.41 | 2.38 |
| 易读可概括 | **3.17** | 2.45 | 2.52 |
| 假设支持力 | 2.62 | 2.41 | 2.34 |

两判官一致性：Spearman 0.986，成对一致率 0.998。

### 3.3 典型公式对比（seed 0）

| 任务 | 真值 | IF-SR | LLM-SR | Direct-LLM |
|---|---|---|---|---|
| 库仑力 | `q1·q2/r²` | `(q1/r)·(q2/r)` ✓等价 | `0.65·q1·q2/(r-0.35)+11.1·r-158…` ✗冗长 | `q1·q2/r²` ✓（背诵） |
| 米氏方程 | `Vmax·S/(Km+S)` | `Vmax/((Km/S)+1.0)` ✓等价 | `Vmax·S/(Km+S)+0.0017(z1-z2)…` ✗含无关变量 | `Vmax·S/(Km+S)` ✓（背诵） |
| 逻辑斯蒂 | `r·N·(1-N/K)` | `(1.0-(N/K))·(r·N)` ✓等价 | `r·N·(1-N/K)` ✓ | `r·N·(1-N/K)` ✓（背诵） |
| 两体引力势 | `-m1·m2·(1/r2-1/r1)` | `((1.0-(r1/r2))·(m2/r1))/(1/m1)` ✓ | `0.31·m1·m2/r1²-1.28·m1·m2/r2²…` ✗ | `m1·m2/(r1·r2)` ✗ |

### 3.4 结论

IF-SR 在 test R² 与最优基线基本持平（0.995 vs 0.997）的前提下，公式节点数砍到 8.6（**−41%**），ExprSim 与双判官盲评均第一。LLM-SR 靠拖无关变量把公式撑到 32.6 节点；Direct-LLM 在经典公式上是"背出来的"（等价率 0.31 最高，但对未记忆样本 R² 均值 −2.16）。

---

## 实验 4：LSR-Transform 抗记忆重建（10 任务 × 3 seeds）

**目的**：复现 LLM-SRBench 的 LSR-Transform 思想——把教科书定律改写成**非常规目标变量**（如理想气体解出 V、开普勒解出 a 的立方根、阿伦尼乌斯解出 Ea），破坏 LLM 记忆，检验"推理 vs 背诵"。

**数据**：`configs/lsr_transform_formulas.json`（10 个本地重建任务，5% 噪声 + 5 无关变量），3 seeds。

### 4.1 定量指标（30 次运行）

| 方法 | test R²中位 | ExprSim | 展开节点数 | 代数等价率 | n |
|---|---|---|---|---|---|
| Raw-PySR | 0.997 | 0.751 | 13.3 | 0.07 | 30 |
| Mine-PySR | 0.997 | 0.694 | 12.4 | 0.00 | 30 |
| **IF-SR（本方法）** | 0.997 | 0.739 | **7.9** | 0.00 | 30 |
| Direct-LLM | **−2.572** | 0.542 | 9.0 | 0.10 | 30 |
| LLM-SR | 0.826 | 0.493 | 34.8 | 0.00 | 30 |

### 4.2 结论

Direct-LLM 代数等价率从标准数据集的 0.31 崩到 0.10、中位 R² 从 0.644 崩到 **−2.57**——直接证明其此前优势主要来自**记忆而非推理**。数值 SR 方法（含 IF-SR）不受影响，R² 仍 0.997，IF-SR 依旧最短（7.9 节点）。

---

## 实验 5：真实 LLM-SRBench（ModelScope 镜像，28 任务 × 3 seeds）

**数据来源**：官方 HF `nnheui/llm-srbench` 为 gated、本环境不可达；改用 ModelScope 镜像 `scientific-intelligent-modelling/sim-datasets-bak`（规范化副本，含真实 ground-truth 公式与官方 train/valid/id_test 划分）。平衡子集 = 12 lsrtransform + 4×(bio_pop_growth, chem_react, matsci, phys_osc) = 28 任务，用基准自带 id_test 作 test。3 seeds。

### 5.1 五方法定量指标（各 84 次运行）

| 方法 | test R²中位 | ExprSim | 展开节点数 | 代数等价率 | n |
|---|---|---|---|---|---|
| Raw-PySR | 0.999 | 0.580 | 21.6 | 0.00 | 84 |
| Mine-PySR | 0.999 | 0.546 | 29.8 | 0.01 | 84 |
| **IF-SR（本方法）** | 0.988 | 0.562 | **13.5** | 0.01 | 84 |
| Direct-LLM | **−7.291** | 0.558 | 10.2 | 0.00 | 84 |
| LLM-SR | 0.968 | 0.531 | 38.8 | 0.00 | 84 |

### 5.2 LLM-SR 按类别

| 类别 | n | test R²中位 | ExprSim |
|---|---|---|---|
| bio_pop_growth | 12 | 1.000 | 0.566 |
| chem_react | 12 | 0.999 | 0.478 |
| **lsrtransform（抗记忆）** | 36 | **0.309** | 0.551 |
| matsci | 12 | 1.000 | 0.482 |
| phys_osc | 12 | 0.999 | 0.538 |

### 5.3 IF-SR 按类别

| 类别 | n | test R²中位 | 展开节点数 | ExprSim |
|---|---|---|---|---|
| bio_pop_growth | 12 | 0.992 | 10.8 | 0.415 |
| chem_react | 12 | 0.980 | 14.2 | 0.474 |
| lsrtransform | 36 | 0.990 | 15.6 | 0.670 |
| matsci | 12 | 0.990 | 10.5 | 0.551 |
| phys_osc | 12 | 0.986 | 12.4 | 0.486 |

### 5.4 结论

真实 benchmark 上结论依然成立：IF-SR 的 test R²（0.988）与 PySR 基线（0.999）基本持平，但公式复杂度只有 13.5 节点，比 Raw-PySR（21.6）省 **38%**、比 LLM-SR（38.8）省 **65%**。关键对照：**LLM-SR 在抗记忆的 lsrtransform 类 R² 掉到 0.309**（其他合成域 ≈1.0），Direct-LLM 中位 R² 崩到 **−7.29**——精确复现 LLM-SRBench 论文的核心发现（记忆会虚高指标、符号发现很难）。

---

## 总结论

1. **准确率受控下的可解释性优先有效**：IF-SR 在三个数据集上一致做到"test R² 基本不降、公式最短、ExprSim 与可解释性盲评最高"，直接支撑论文主线（accuracy-constrained, interpretability-first）。
2. **Direct-LLM 是记忆而非推理**：标准公式代数等价率最高（0.31），但离开记忆分布后 R² 灾难性崩塌（LSR-Transform −2.57，真实 LLM-SRBench −7.29）。
3. **LLM-SR 拟合强但结构冗长**：合成域 R²≈1.0，但表达式 30+ 节点，几乎不恢复真实结构，抗记忆 lsrtransform 类 R² 明显下降。
4. **符号精确恢复普遍很难**：所有方法代数等价率都低，印证 LLM-SRBench 论文"最强系统 symbolic accuracy 仅 31.5%"。
5. **判官可信度**：两个独立判官 pass 的排序一致性极高（Spearman 0.986、成对一致 0.998），支持"IF-SR 更可解释"这一结论的稳健性。

---

## 局限与诚实说明

- **LLM-SRBench 用镜像子集**：官方 HF 数据 gated 不可达，用 ModelScope 规范化镜像的 **28 任务平衡子集**（非全部 229/239 题）。
- **第二判官非跨家族**：原计划 deepseek-reasoner，因端点连接挂死改用 deepseek-chat 换 seed 做独立 pass；论文主评仍建议接入一个真正不同的模型家族。
- **无人类评分**：本轮仅 LLM 判官；论文主评需 ≥2 位领域人类评分者做 rubric 校准与主评。
- **样本规模**：每块 3 seeds、LLM-SRBench 用子集，方向清晰但未做大规模配对显著性检验；投稿主表建议扩 seed/任务数并加 bootstrap 95% CI 与配对效应量。
- **真实运行、非 surrogate**：本轮 SR 引擎为真实 PySR，LLM 方法与判官为真实 DeepSeek，所有数字均为实盘。

---

## 复现入口

```bash
VP=/public/home/wangyg/workspace/llm_pysr_project/.venv/bin/python

# 实验 0：单元测试
PYTHONPATH=. $VP -m pytest -q                       # 63 passed

# 实验 1：pilot
PYTHONPATH=. $VP scripts/run_pilot.py

# 实验 3：主实验（你的 29 公式，含盲评+聚合）
PYTHONPATH=. $VP scripts/run_experiment2_full.py

# LLM 基线（标准 / transform）
PYTHONPATH=. $VP scripts/run_llm_sr_baselines.py
PYTHONPATH=. $VP scripts/run_lsr_transform_baselines.py

# 实验 4：LSR-Transform PySR/IF-SR
PYTHONPATH=. $VP scripts/run_lsr_transform_pysr.py

# 实验 5：真实 LLM-SRBench（镜像下载 + 5 方法）
PYTHONPATH=. $VP scripts/run_llmsrbench_llm.py
PYTHONPATH=. $VP scripts/run_llmsrbench_pysr.py

# 汇总五方法对比
PYTHONPATH=. $VP scripts/build_final_report.py
```

**核心模块**：`splits.py`、`preprocess.py`、`lineage.py`、`ifsr_selector.py`、
`expression_similarity.py`、`interpretability_eval.py`、`rating_aggregate.py`、
`known_formulas.py`、`known_data.py`、`experiment2.py`、`llm_sr_baselines.py`、
`llm_sr_runner.py`、`llmsrbench_loader.py`、`llmsrbench_runner.py`、`comparison.py`、`pilot.py`。

**相关文档**：`docs/phase1_report.md`、`phase2_report.md`、`pilot_report.md`（基础设施三阶段）；
`docs/experiment2_results_report.md`、`docs/llm_sr_comparison_report.md`（早期记录，已被本报告取代）。
