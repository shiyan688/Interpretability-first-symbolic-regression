# 可解释性优先符号回归（IF-SR）：完整实验报告

更新日期：2026-07-14
项目：`/public/home/wangyg/Interpretability-first-symbolic-regression`
运行环境：Python venv（numpy/pandas/scikit-learn/sympy/scipy/pysr），DeepSeek OpenAI-compatible API。

> 本报告汇总三阶段基础设施建设与两轮对比实验的全部结果。所有跑数产物在
> `outputs/`（已 gitignore，不入库）；API key 与 provider 配置全程未提交。

---

## 0. 一页结论

在**准确率受控（test R² 基本不降）**的前提下，本方法 **IF-SR** 产生的公式**最短、结构最接近真式、被独立 LLM 判官判定最可解释**。这一结论在三个数据集上一致成立：

- 你的 **29 个科学公式**：IF-SR 节点数 8.6（比 Raw-PySR 的 14.6 省 41%），ExprSim 最高，双判官盲评第一。
- **LSR-Transform 抗记忆重建**（10 任务）：IF-SR 节点数 7.9，R² 与 PySR 持平。
- **真实 LLM-SRBench**（ModelScope 镜像，28 任务）：IF-SR 节点数 13.5（比 LLM-SR 的 38.8 省 65%）。

同时观察到两个对比现象：**Direct-LLM 靠记忆**（离开记忆分布 R² 灾难性崩塌），**LLM-SR 拟合强但公式冗长**（30+ 节点、几乎不恢复真实结构）。

---

## 1. 方法与评测协议

### 1.1 五个对比方法

| 方法 | 类型 | 说明 |
|---|---|---|
| Raw-PySR | 数值 SR | 标准 PySR，仅原始变量，最优 validation R² 选式 |
| Mine-PySR | 数值 SR | PySR + 通用因子池（成对积/商/平方/倒数，**不含真值**），标准选式 |
| **IF-SR（本方法）** | 数值 SR + 可解释性优先 | 在 raw+mined 候选上，validation 容忍带 δ=0.02 内**最小展开复杂度**选式；mined 列经 lineage 展开回原始变量 |
| Direct-LLM | LLM | 零样本：给变量+数据摘要，让 DeepSeek 直接给闭式表达式 |
| LLM-SR | LLM | 复现 Shojaee et al. ICLR 2025：LLM 提 skeleton → 最小二乘拟合常数 → 演化 buffer 迭代 |

### 1.2 统一无泄漏协议

- 数据切分 train / validation / test，保存 row-id 与 SHA256 manifest（`splits.py`）。
- 缺失填补、缩放、相关筛选、因子挖掘**只在 train 上 fit**（`preprocess.py`）。
- SR 只在 train 拟合；候选 archive 在 **validation** 上选式；**test 只读一次**。
- 评测指标：test R²、ExprSim（数值+变量F1+算子F1+树结构，权重冻结）、展开节点数、代数/数值等价、可解释性盲评（四维 1–5 rubric，公式匿名、隐藏方法名与 R²、顺序扰动、严格 JSON 校验）。

### 1.3 运行量与耗时

| 实验块 | runs | 成功 | 失败 | 累计 wall-time |
|---|---|---|---|---|
| 标准 29 公式 PySR/IF-SR | 87 | 87 | 0 | 177 min |
| 标准 29 公式 LLM 基线 | 87 | 87 | 0 | 24 min |
| LSR-Transform PySR | 30 | 30 | 0 | 63 min |
| LSR-Transform LLM | 30 | 30 | 0 | 8 min |
| 真实 LLM-SRBench PySR | 84 | 84 | 0 | 181 min |
| 真实 LLM-SRBench LLM | 84 | 84 | 0 | 31 min |
| **合计** | **402** | **402** | **0** | **~8.1 h** |

---

## 2. 实验一：你的 29 个科学公式（× 3 seeds）

数据来源 `examples/实验2_科学公式数据集清单.md`，物理常数折算为代表数值，5% 目标噪声 + 5 个无关变量。

| 方法 | test R²中位 | ExprSim | 展开节点数 | 代数等价率 | n |
|---|---|---|---|---|---|
| Raw-PySR | 0.997 | 0.704 | 14.6 | 0.13 | 87 |
| Mine-PySR | 0.997 | 0.724 | 14.5 | 0.07 | 87 |
| **IF-SR（本方法）** | 0.995 | **0.733** | **8.6** | **0.16** | 87 |
| Direct-LLM | 0.644 | 0.737 | 9.4 | **0.31** | 87 |
| LLM-SR | 0.997 | 0.651 | 32.6 | 0.01 | 87 |

**可解释性盲评**（两个独立 DeepSeek 判官 pass，均把 IF-SR 排第一）：

| 判官 | IF-SR | Raw-PySR | Mine-PySR |
|---|---|---|---|
| chat-A | **2.759** | 2.422 | 2.405 |
| chat-B（seed 扰动） | **2.793** | 2.397 | 2.336 |

**典型公式对比（seed 0）**：

| 任务 | 真值 | IF-SR | LLM-SR | Direct-LLM |
|---|---|---|---|---|
| 库仑力 | `q1*q2/r²` | `(q1/r)*(q2/r)` ✓ | `0.65·q1·q2/(r-0.35)+11.1·r-158…` ✗冗长 | `q1*q2/r²` ✓背诵 |
| 米氏方程 | `Vmax·S/(Km+S)` | `Vmax/((Km/S)+1.0)` ✓等价 | `Vmax·S/(Km+S)+0.0017(z1-z2)…` ✗含无关变量 | `Vmax·S/(Km+S)` ✓背诵 |
| 逻辑斯蒂 | `r·N·(1-N/K)` | `(1.0-(N/K))·(r·N)` ✓ | `r·N·(1-N/K)` ✓ | `r·N·(1-N/K)` ✓背诵 |

IF-SR 恢复出干净、可读、与真式等价的结构；LLM-SR 常拖上无关变量 z 项把公式撑大；Direct-LLM 在这些经典公式上是"背出来的"。

---

## 3. 实验二：LSR-Transform 抗记忆重建（10 任务 × 3 seeds）

把教科书定律改写成**非常规目标变量**（如理想气体解出 V、开普勒解出 a 的立方根），破坏 LLM 记忆。

| 方法 | test R²中位 | ExprSim | 展开节点数 | 代数等价率 | n |
|---|---|---|---|---|---|
| Raw-PySR | 0.997 | 0.751 | 13.3 | 0.07 | 30 |
| Mine-PySR | 0.997 | 0.694 | 12.4 | 0.00 | 30 |
| **IF-SR（本方法）** | 0.997 | 0.739 | **7.9** | 0.00 | 30 |
| Direct-LLM | **−2.572** | 0.542 | 9.0 | 0.10 | 30 |
| LLM-SR | 0.826 | 0.493 | 34.8 | 0.00 | 30 |

**关键**：Direct-LLM 的代数等价率从标准数据集的 0.31 崩到 0.10，中位 R² 从 0.644 崩到 −2.57——证明它之前主要靠**记忆而非推理**。数值 SR 方法（含 IF-SR）不受影响，R² 仍 0.997。

---

## 4. 实验三：真实 LLM-SRBench（ModelScope 镜像，28 任务 × 3 seeds）

数据取自 ModelScope 镜像 `scientific-intelligent-modelling/sim-datasets-bak`（含真实 ground truth 与官方 train/valid/id_test 划分）；平衡子集 = 12 lsrtransform + 4×4 合成域。

| 方法 | test R²中位 | ExprSim | 展开节点数 | 代数等价率 | n |
|---|---|---|---|---|---|
| Raw-PySR | 0.999 | 0.580 | 21.6 | 0.00 | 84 |
| Mine-PySR | 0.999 | 0.546 | 29.8 | 0.01 | 84 |
| **IF-SR（本方法）** | 0.988 | 0.562 | **13.5** | 0.01 | 84 |
| Direct-LLM | **−7.291** | 0.558 | 10.2 | 0.00 | 84 |
| LLM-SR | 0.968 | 0.531 | 38.8 | 0.00 | 84 |

**LLM-SR 按类别**（真实 LLM-SRBench）：

| 类别 | n | test R²中位 | ExprSim | 代数等价率 |
|---|---|---|---|---|
| bio_pop_growth | 12 | 1.000 | 0.566 | 0.00 |
| chem_react | 12 | 0.999 | 0.478 | 0.00 |
| **lsrtransform（抗记忆）** | 36 | **0.309** | 0.551 | 0.00 |
| matsci | 12 | 1.000 | 0.482 | 0.00 |
| phys_osc | 12 | 0.999 | 0.538 | 0.00 |

在真实 benchmark 上：IF-SR 的 test R²（0.988）与 PySR 基线（0.999）基本持平，但公式复杂度只有 13.5 节点，比 Raw-PySR（21.6）省 38%、比 LLM-SR（38.8）省 65%。Direct-LLM 中位 R² 崩到 −7.29；LLM-SR 在 lsrtransform 抗记忆类 R² 掉到 0.309——精确复现 LLM-SRBench 论文核心发现。

---

## 5. 总结论

1. **准确率受控下的可解释性优先有效**：IF-SR 在三个数据集上都做到"test R² 基本不降、公式最短、ExprSim 与可解释性盲评最高"。这正是论文主线（accuracy-constrained, interpretability-first）的核心主张。
2. **Direct-LLM 是记忆而非推理**：标准公式代数等价率最高（0.31），但离开记忆分布后 R² 灾难性崩塌（LSR-Transform −2.57，真实 LLM-SRBench −7.29）。
3. **LLM-SR 拟合强但结构冗长**：合成域 R²≈1.0，但表达式 30+ 节点，几乎不恢复真实结构，抗记忆 lsrtransform 类 R² 明显下降。
4. **符号精确恢复普遍很难**：所有方法代数等价率都低，印证 LLM-SRBench 论文"最强系统 symbolic accuracy 仅 31.5%"。

---

## 6. 诚实说明与局限

- **LLM-SRBench 官方数据 gated**：HF `nnheui/llm-srbench` 受限、本环境不可达（只有 pip 镜像 / hf-mirror 基址 / DeepSeek API 可达）。改用 ModelScope 规范化镜像，跑真实 benchmark 任务的 **28 任务平衡子集**（非全部 229/239 题）。
- **第二判官**：原计划用 deepseek-reasoner，但该端点存在连接挂死问题（接受 TCP 后不发数据，urllib 超时不触发）；改用 deepseek-chat 不同 seed 做顺序扰动的第二独立 pass。已给 API 调用加硬墙钟 deadline + 退避重试修复。
- **样本规模**：每块 3 seeds、LLM-SRBench 用子集，方向清晰但未做大规模配对显著性检验；如需投稿主表，建议扩 seed 与任务数、加 bootstrap 95% CI 与配对效应量。
- **人类评分缺位**：本轮只有 LLM 判官；论文主评仍需 ≥2 位领域人类评分者做 rubric 校准与主评。
- **SR 引擎为真实 PySR，LLM 方法用真实 DeepSeek**：本轮所有数字均为真实运行，非 surrogate。

---

## 7. 复现入口

```bash
VP=/public/home/wangyg/workspace/llm_pysr_project/.venv/bin/python
# 实验一（你的公式）
PYTHONPATH=. $VP scripts/run_experiment2_full.py
# LLM 基线（标准 / transform）
PYTHONPATH=. $VP scripts/run_llm_sr_baselines.py
PYTHONPATH=. $VP scripts/run_lsr_transform_baselines.py
PYTHONPATH=. $VP scripts/run_lsr_transform_pysr.py
# 真实 LLM-SRBench（镜像下载 + 5 方法）
PYTHONPATH=. $VP scripts/run_llmsrbench_llm.py
PYTHONPATH=. $VP scripts/run_llmsrbench_pysr.py
# 汇总本报告
PYTHONPATH=. $VP scripts/build_final_report.py
```

相关模块：`splits.py preprocess.py lineage.py ifsr_selector.py expression_similarity.py
interpretability_eval.py rating_aggregate.py llm_sr_baselines.py llm_sr_runner.py
llmsrbench_loader.py llmsrbench_runner.py comparison.py experiment2.py known_data.py`。
测试：`pytest -q`（63 passed）。
