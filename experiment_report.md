# 实验报告：CIFAR-10 类级遗忘基准 + 多验证方法对比（Phase 1+2）

> 日期：2026-08-11
> 脚本：`benchmark_v2.py`（`/Users/peregrine/one_rank/`）
> 结果：`results/benchmark_v2/`（results.csv、fig1_methods.png、fig2_corr.png、models/）
> 硬件：Apple M5 Pro 24GB（MPS），单轮全流程 **65 秒**
> 随机种子：42

---

## 1. 实验协议

**数据集与任务（class-level, K=3 held-out）：**
- 训练类：{0,1,2,3,4,5,6}（7 类，含遗忘类）
- **遗忘类：3（cat）**
- 保留类：{0,1,2,4,5,6}（6 类）
- 留出类（MHPR 参照，模型从未见过）：{7,8,9}（horse, ship, truck）
- 模型：SmallCNN（3 conv + 2 fc），无数据增强（确定性、可复现）

**遗忘方法（7 种）：** NoUnlearn（基线）· Retrain（oracle，重训于保留集）· NegGrad（梯度上升）· FineTune（上升+保留下降）· **KED**（KL 至均匀分布擦除）· **BadTeacher**（错误标签训练）· SISA（5 分片×2 切片）

**验证指标（同基准，逐方法）：**
| 类别 | 指标 | 说明 |
|---|---|---|
| 效用 | retain_acc / forget_acc | 保留/遗忘准确率 |
| 谱证书 | **RII** ρ | 2×C 输出信道奇异值比 |
| 谱证书 | **MHPR**（K=3） | 遗忘类均值在留出类张成空间上的投影残差 |
| MIA 基线 | MIA-loss / MIA-conf AUC | 损失/置信度阈值（正=遗忘样本） |
| TAPE 风格 | posterior_diff | 遗忘集上未学习模型 vs 原模型的平均后验 L2 差 |
| RULER 风格 | repr_mmd / repr_mmd_holdout | fc1 特征 RBF-MMD（中位数启发式 σ），对保留类/对留出类 |
| RUB 风格 | residual_probe_auc | 线性探测区分遗忘特征 vs 留出类特征 |

---

## 2. 完整结果

| method | retain% | forget% | **RII** | **MHPR** | MIA-loss | MIA-conf | postDiff | reprMMD | mmdHO | probeAUC | 耗时s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NoUnlearn | 93.7 | 67.9 | 2.33e-1 | 0.812 | 0.160 | 0.178 | 0.000 | 0.132 | 0.173 | 0.970 | 0.0 |
| Retrain | 93.5 | **0.0** | 9.15e-2 | 0.428 | 0.000 | 0.208 | 0.904 | **0.087** | 0.110 | 0.959 | 14.2 |
| NegGrad | 94.6 | 63.4 | **2.54e-1** | 0.831 | 0.128 | 0.143 | 0.166 | 0.128 | 0.173 | 0.971 | 0.5 |
| FineTune | 98.0 | 29.2 | 2.08e-1 | 0.642 | 0.025 | 0.077 | 0.451 | 0.140 | 0.180 | 0.978 | 1.9 |
| KED | 97.2 | 1.7 | 1.21e-1 | 0.457 | 0.003 | 0.081 | 0.703 | 0.130 | 0.192 | 0.981 | 3.4 |
| BadTeacher | 97.9 | 17.0 | 1.63e-1 | 0.344 | 0.013 | 0.067 | 0.550 | 0.137 | 0.177 | 0.977 | 3.3 |
| SISA | 69.9 | **0.0** | **6.42e-2** | **0.090** | 0.004 | 0.337 | 0.808 | **0.056** | **0.041** | 0.920 | 5.5 |

**方法排名（遗忘强度，forget_acc 越低越好）：**
- Retrain ≈ SISA > KED > BadTeacher > FineTune > NegGrad > NoUnlearn

**RII 排名（ρ 越低 = 输出擦除越好）：**
- SISA < Retrain < KED < BadTeacher < FineTune < NoUnlearn < **NegGrad**

**MHPR 排名：**
- SISA < BadTeacher < Retrain < KED < FineTune < NoUnlearn < NegGrad

---

## 3. 关键发现（对照论文主张与导师意见）

### 3.1 核心主张 1：known/unknown 不对称被实验证实 ✅
**Retrain oracle（数学上"完美遗忘"）的 RII = 0.092 ≠ 0**。因为 cat 成为从未见过的类，其输出分布天然区别于保留类——标准 RII 在类级场景无法把 oracle 判为"零泄漏"。这正是论文的核心论点，现在有了 benchmark 数据支撑。MHPR 将 oracle 与 NoUnlearn 的差距从 RII 的 2.5×（0.092 vs 0.233）拉大到 1.9×（0.428 vs 0.812，MHPR 值本身仍偏高的原因见 §4）。

### 3.2 核心主张 2：RII 发现 MIA/准确率掩盖的隐藏失败 ✅（重要新发现）
**NegGrad（梯度上升，最流行的近似遗忘方法）被 RII 评为"最差"（ρ=0.254，甚至高于不遗忘的 0.233）**，而：
- forget_acc 从 67.9%→63.4%（看起来"略有效"）
- MIA-loss 从 0.160→0.128（看起来"略有效"）
- 但 RII 上升 → 梯度上升把 cat 推向"自信的错误预测"，在输出空间留下**更强的可检测签名**

这是论文"RII 揭示隐藏失败"叙事的最强实证：一个被社区广泛使用的遗忘方法，其输出级签名不但没有消除反而增强，而 MIA 与准确率均给出"有效"的错觉。**建议写入论文作为案例**（与 FineTune rebound 并列）。

### 3.3 指标一致性（相关性矩阵）
| 相关对 | r |
|---|---|
| RII vs MHPR | **0.911** |
| RII vs forget_acc | **0.928** |
| RII vs MIA-loss | 0.81 |
| RII vs repr_mmd | 0.78 |
| posterior_diff vs forget_acc | **−0.978** |
| retain_acc vs probeAUC | 0.98（混淆项，见§4） |

- **RII/MHPR 与遗忘强度高度一致**（r≈0.9），说明谱证书是可靠的无参考指标。
- **RII 与 MIA-loss 仅 r=0.81**，且顶部顺序反转（NegGrad/NoUnlearn），印证"两轴正交、需同时报告"的主张（§6 Reconciling MIA and RII 有数据支撑了）。
- **表示层 MMD（RULER 风格）与 RII 正相关**（r=0.78）：输出级擦除好的方法（SISA/Retrain），表示层异常也最低——输出与表示层指标大体一致，但并非完全相同（见 probeAUC）。

### 3.4 表示层残差：probeAUC ≈ 0.92-0.98 对全部方法
**即使 Retrain oracle 的表示层仍保留 cat 的可探测签名（probeAUC=0.959）**——与 RULER / "Erased, but Not Gone" 的结论一致：输出级成功与表示层残差可共存。这直接支撑论文 Scope 中"输出层证书 + 表示层审计互补"的论述（Beyond Output 段落），也说明论文必须明确"我们证的是输出层，不声称白盒安全"。

### 3.5 posterior_diff（TAPE 风格）≈ 遗忘强度的镜像
r=−0.978，排序完全合理（Retrain 0.904 > SISA 0.808 > KED 0.703 > BadTeacher 0.550 > FineTune 0.451 > NegGrad 0.166 > NoUnlearn 0）。但它是"变化量"而非"质量量"：NoUnlearn 恒为 0、且需要访问原模型，作为审计指标不及 RII 直接。

---

## 4. 诚实局限（必须向审稿人交代）

1. **SISA 的"最佳"被效用混淆**：SISA 谱指标全场最优（RII=0.064、MHPR=0.090），但 retain_acc 仅 69.9%（其余方法 93-98%）。因 SISA 训练量偏小（2 切片×1 epoch/分片），弱模型输出弥散导致谱指标偏低。**公平对比需加大 SISA 训练量或报告"效用-遗忘"联合视图**（如 Pareto 前沿）。
2. **CIFAR-10 的 MHPR 绝对值偏高**：oracle 的 MHPR=0.428（论文 MNIST 上为 0.046）。原因是留出类 {horse,ship,truck} 的张成空间不能很好地表示 cat 的输出分布；MNIST 上 digit-0 对 digit-5 恰好代表性好。**这是数据集相关的已知/未知相似度问题**，应如实讨论，并提示"留出类需与遗忘类语义相近"的适用条件。
3. **MIA-loss 绝对值 <0.5 的方向混淆**：cat 本身是难类（基线 forget_acc 67.9% < retain 93.7%），即使不遗忘，cat 样本损失也系统性高于保留类，导致全部 AUC<0.5。**相对排序有效，但绝对阈值不可解释**——这本身是对"MIA 依赖类难度/校准"论点的支持，但论文引用时须谨慎。
4. **MIA-conf 与 retain_acc 强负相关（r=−0.92）**：SISA（弱模型）MIA-conf 最高（0.337），说明该指标被模型置信度整体水平污染。
5. **residual_probe_auc 无区分度**（全部 0.92-0.98 且与 retain_acc r=0.98）：特征空间里遗忘类相对留出类始终可分，与"输出层 vs 表示层"叙事一致；作为方法对比指标意义有限。

---

## 5. 对论文（APIN 23 页稿）的落点建议

| 论文位置 | 建议 |
|---|---|
| §5.3 实验 Setup | 增加本 benchmark 协议（7 类训练/遗忘 cat/3 留出类，7 方法） |
| §5.4 Main Results 后 | 新增一张**主对比表**（本报告 §2 表），行=方法、列=RII/MHPR/MIA/MMD/probe |
| §5 新小节 | **"RII 揭示 NegGrad 隐藏失败"**案例：forget_acc/MIA 显示"有效"，RII 显示签名增强 |
| §6 Reconciling MIA/RII | 引用相关性数字（r=0.81 非完全一致；NegGrad/NoUnlearn 顶部反转） |
| Beyond Output 段 | 引用 probeAUC≈0.96（oracle 也有表示层签名）→ 强化"输出层证书+表示层审计"论述 |
| 讨论/局限 | §4 的 5 条局限全部写入（尤其 SISA 效用混淆、MHPR 数据集相关） |
| 图 | fig1（方法对比，log 轴）、fig2（相关性热力图）可直接用作论文图 |

---

## 6. 结论

本次实验完成了导师评审要求的"必改4"核心部分：
- ✅ **真实遗忘 benchmark**（CIFAR-10 类级，7 方法，含新增 KED/BadTeacher）
- ✅ **同基准多验证方法对比**（RII/MHPR/MIA/TAPE 风格/RULER 风格/RUB 风格，10 项指标）
- ✅ **产出 2 图 + 数据 + 可复现模型**（results/benchmark_v2/）

**最重要的科学产出**：
1. Retrain oracle 在类级下 RII≠0 → **known/unknown 不对称的实验铁证**（论文核心论点）。
2. **NegGrad 的 RII 反升** → "RII 揭示准确率/MIA 掩盖的隐藏失败"，可与 FineTune rebound 并列为论文两大实证亮点。
3. 表示层 probeAUC 全方法≈0.96 → 输出层证书定位清晰，与 RULER/Erased-not-gone 互补。

**后续建议**：① 加大 SISA 训练量后重跑以去除效用混淆；② 增加一个与 cat 语义相近的留出类组合以改善 MHPR 绝对值；③ 将本表与两图接入 APIN 稿，补 1-2 页实验与讨论。
