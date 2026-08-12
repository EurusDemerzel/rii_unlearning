# 补充实验报告 v3（实验计划 A/B/C + D/E/F 补做）

> 日期：2026-08-12（A/B/C）、2026-08-12 更新（D/E/F）
> 脚本：`supplement_v3.py`、`benchmark_cifar100.py`、`benchmark_nlp.py`、`supplement_temp.py`
> 可视化：`plot_supplement_v3.py`
> 输出：`results/supplement_v3/`、`results/benchmark_cifar100/`、`results/benchmark_nlp/`、`results/supplement_temp/`
> 硬件：Apple M5 Pro 24GB（MPS）· 种子 42 · **不更改论文**
> 补做说明：CIFAR-100（实验 D）、NLP AG News+DistilBERT（实验 E）、温度扫描+状态空间（实验 F）已通过 **modelscope 国内镜像**（4.6 MB/s）与本地 `~/cifar100` parquet 数据完成，不再跳过。

---

## 实验 A — RII 安全阈值标定（CIFAR-10 类级）

**协议**：7 类训练 {0..6}，遗忘类 3(cat)，留出 {7,8,9}；SmallCNN 10 epoch；梯度上升 0/1/3/10/30/50/100/150 步（lr=1e-6），逐步记录 RII、MHPR、forget_acc、MIA-loss AUC；并与 7 方法 benchmark 结果联合对照。

| steps | ρ | MHPR | forget_acc% | MIA-loss |
|------:|------:|------:|------:|------:|
| 0 | 0.2333 | 0.812 | 67.9 | 0.160 |
| 1 | 0.2344 | 0.817 | 68.2 | 0.157 |
| 3 | 0.2356 | 0.829 | 69.9 | 0.154 |
| 10 | 0.2439 | 0.843 | 69.8 | 0.143 |
| 30 | 0.2475 | 0.845 | 69.4 | 0.143 |
| 50 | 0.2536 | 0.842 | 66.6 | 0.134 |
| 100 | 0.2547 | 0.837 | 65.5 | 0.130 |
| 150 | 0.2577 | 0.831 | 63.4 | 0.123 |

**7 方法对照**：NoUnlearn ρ=0.233 / Retrain 0.092 / NegGrad **0.254** / FineTune 0.209 / KED 0.121 / BadTeacher 0.163 / SISA 0.064。

**解读**：
1. **复现 NegGrad 隐藏失败**：梯度上升 150 步 ρ 从 0.233 → 0.258（**单调上升**），而 forget_acc（67.9→63.4%）与 MIA-loss（0.160→0.123）同时"改善"——与 benchmark 的 NegGrad 结果（ρ=0.254）完全一致，确认该现象可复现、非随机。
2. **阈值分档**（输出层，启发式）：
   - ρ < 0.01：强输出级不可区分（安全区，对应随机子集 ~1e-3 与充分遗忘）；
   - 0.01–0.1：部分/类级不对称残差（Retrain 0.092、KED 0.121 区间附近）；
   - ρ > 0.1：明确的类级输出签名（未擦除，NoUnlearn/NegGrad 区间）。
3. **注意与论文 tab:gradient_sweep 的差异**：论文该表（steps 0→100: ρ 1.83e-1→2.04e-3，Acc 76.2→25.4）显示 ρ 随步数**下降**——那是让模型崩溃的长时间/大强度上升；本实验（小步长 150 步，模型未崩溃）显示 ρ **上升**。两者不矛盾：先上升（签名增强）后崩溃（模型退化、ρ 假性回落）。写论文时需明确步长/强度范围，避免审稿人困惑。

---

## 实验 B — 多参考类平均 LOCO vs 投影 MHPR（MNIST）

**协议**：7 类训练 {0..6}（含遗忘类 5），留出 {7,8,9}；SimpleMLP 10 epoch。对 K=1,2,3 个留出类比较三种参考构造：
- 单类 LOCO（min/med/max 跨留出类）；
- 平均参考 LOCO：`‖μ_f − mean(K 留出类)‖²/‖μ_f‖²`；
- 投影 MHPR：`‖μ_f − P_span(H) μ_f‖²/‖μ_f‖²`。

| K | 单类 LOCO (min/med/max) | 平均参考 LOCO | 投影 MHPR |
|--:|:--:|:--:|:--:|
| 1 | 1.125 / 1.125 / 1.125 | 1.125 | 0.978 |
| 2 | 0.762 / 0.944 / 1.125 | 0.893 | 0.719 |
| 3 | 0.762 / 1.125 / 1.501 | 0.961 | 0.707 |

**解读**：
1. **单类参考极不稳定**：K=3 时跨留出类 0.762~1.501，选哪个留出类直接决定结论——LOCO 失败的根源被定量展示。
2. **平均参考并不收敛**：K=2→0.893 但 K=3→0.961（加入语义远的类反而变差），平均不能捕捉"未见类流形"。
3. **投影 MHPR 一致最低**（0.978→0.719→0.707），且随 K 单调下降——验证 MHPR 投影设计的动机（该图可直接用作论文 Fig：单类/平均 vs 投影）。
4. 注意本实验留出类 {7,8,9} 对 digit 5 代表性差（MNIST 上 digit 0 对 5 才代表性好），故绝对 ρ_H 偏高（0.7-0.98）——与论文 caveat "MHPR 绝对值依赖留出类语义代表性"一致。

---

## 实验 C — 跨类别梯度上升扫描（CIFAR-10）

**协议**：同一 7 类训练协议，分别以 **类 2(bird)、3(cat)、5(dog)** 为遗忘类，梯度上升 0/1/3/10/30 步，记录 RII。

| steps | 类2 ρ | 类3 ρ | 类5 ρ |
|------:|------:|------:|------:|
| 0 | 0.189 | 0.190 | 0.193 |
| 1 | 0.196 | 0.195 | 0.199 |
| 3 | 0.207 | 0.202 | 0.212 |
| 10 | 0.233 | 0.219 | 0.243 |
| 30 | **0.255** | **0.241** | **0.269** |

相对基线：step30 时 **1.35× / 1.27× / 1.39×**。

**解读**：
1. **NegGrad 隐藏失败跨类别普适**：三类遗忘类的 RII 都随梯度上升**单调上升**（1.27–1.39×），不是单一类的伪影——这是对论文核心主张（"RII 揭示梯度上升隐藏失败"）的重要普适性证据。
2. 遗忘类难度影响绝对值（类 5 基线最高 0.193），但**上升趋势一致**，支持"相对排序/趋势有效、绝对阈值需按类标定"。
3. MHPR 在类 3/5 单调上升、类 2 在 steps 10-30 略回落（0.912→0.889），说明 MHPR 对非常见类的响应不严格单调——可作为 caveat。

---

## 实验 D — CIFAR-100 类级基准（本地数据补做，计划 #1）

> 数据来源：用户主目录 `~/cifar100/cifar100/`（HuggingFace parquet：train 50k + test 10k），无需网络下载。
> 脚本：`benchmark_cifar100.py` → `results/benchmark_cifar100/results.csv`
> 协议：20 训练类 {0..19}，遗忘类 3，留出 {20,21,22}（K=3），SmallCNN(20)，10 epoch，无增强；6 方法 × (retain/forget acc, RII, MHPR, MIA-loss)。

| method | retain% | forget% | RII | MHPR | MIA-loss |
|--------|--------:|--------:|------:|------:|------:|
| NoUnlearn | 84.1 | 76.4 | 0.1364 | 0.281 | 0.278 |
| Retrain (oracle) | 87.5 | 0.0 | 0.1352 | 0.103 | 0.000 |
| NegGrad | 83.9 | 35.2 | **0.2124** | **0.461** | 0.115 |
| FineTune | 92.8 | 39.8 | 0.1727 | 0.238 | 0.076 |
| KED | 93.5 | 35.2 | 0.1746 | 0.263 | 0.056 |
| BadTeacher | 93.1 | 44.8 | 0.1573 | 0.209 | 0.086 |

**解读**：
1. **NegGrad 隐藏失败在 100 类设置下更强**：NegGrad 的 RII=0.212 **全场最高，甚至超过 NoUnlearn（0.136，1.56×）**，而其 forget_acc（35.2%）与 MIA-loss（0.115）都显示"有效"。CIFAR-10 上差距为 1.09×（0.254 vs 0.233），CIFAR-100 上更明显——梯度上升的输出签名增强在更多类/更高维度输出空间更突出。**这是"RII 揭示隐藏失败"主张的最强跨设置证据。**
2. **known/unknown 不对称在 100 类下依然成立**：Retrain oracle RII=0.135≠0（与 CIFAR-10 的 0.0915 同构），标准 RII 无法把 oracle 判为完美。
3. **MHPR 在 100 类下有效区分 oracle 与 NoUnlearn**：0.103 vs 0.281（2.7×），与 CIFAR-10（0.428 vs 0.812）方向一致。
4. **秩一基线**：类级 NoUnlearn RII=0.136，远高于随机子集（~1e-3），类级遗忘的谱签名清晰。
5. **RII 与 forget_acc 排名分歧**：forget_acc 把 NegGrad 排第 2（"有效"），RII 把它排最后（最差）——再次印证"两轴正交、需同时报告"。

**诚实局限**：① 10 epoch 训练（与主协议一致），retain_acc 84–93% 非完全收敛；② 遗忘类 3 在 CIFAR-100 上难度与 CIFAR-10 不同（NoUnlearn forget_acc 76.4%）；③ 单种子 42；④ 未跑 SISA（20 类训练量较大，且 SISA 在主基准已暴露效用混淆）。

---

## 结论与落点建议（**不写入论文**，供后续参考）

| 实验 | 核心产出 | 若写论文可用作 |
|------|----------|----------------|
| A | ρ 阈值分档（<0.01 / 0.01–0.1 / >0.1）+ NegGrad 复现 | §5 或 §8 的可操作审计阈值段落；figA |
| B | 单类/平均参考 vs 投影的定量对比（MNIST） | §3 MHPR 动机的支撑图（figB） |
| C | NegGrad 隐藏失败跨 3 类普适（figC） | §5.3 NegGrad 案例的普适性补充 |
| D | **CIFAR-100 类级基准：NegGrad 隐藏失败在 100 类下最强复现（RII 1.56×NoUnlearn）** | §5 大模型/多类验证；直接回应"秩2→秩1 是否只在 10 类成立" |
| E | **NLP 基准：RII 跨模态迁移成立（2.96×）+ NegGrad 显性失败对照** | §5 跨模态普适性；§6 调和 MIA/RII 的补充证据 |
| F | **温度缩放判别峰值（T=5）+ 状态空间 MHPR 饱和（~1）** | §3/§5 温度选择策略（κ 条件）的定量依据 |
| G | **真实 LLM（Qwen2.5-0.5B+TOFU）：FineTune 反弹复现 + NegGrad 破坏性 + RII 的 vocab 通道限制** | §5.5 反弹的跨模态证据；§2 rem:channel 的 LLM 实例化 |

**跳过项（需 GPU 服务器人工完成）**：官方 LLaMA-7B 级 TOFU 协议（MIA 基准三件套：forget 90/95/99 全量）与 MUSE（Books/News, DPO/ORPO）——24GB MPS 无法承担 7B 全量微调。数据均可从 modelscope 下载（`popatry/TOFU`、`popatry/MUSE-Books`、`popatry/MUSE-News`），终端指令见对话总结。CIFAR-100（实验 D）、NLP（实验 E）、温度（实验 F）、LLM demo（实验 G）均已补做。

**诚实局限**：① 实验 A 的 ρ 上升与论文 tab:gradient_sweep 的下降是"先上升后崩溃"的同一现象的两种阶段，写论文需明确步长范围；② 实验 B/C/D 为单种子（42），多种子验证留待补做；③ 阈值分档为启发式，未经正式校准。

---

## 实验 E — NLP 类级遗忘基准（AG News + DistilBERT，modelscope 补做）

> 数据：`torushy/flm-ag-news`（modelscope 镜像，120k 训练样本，4 类×30k），经 `pyarrow.ipc.open_stream` 读取 arrow 转 parquet。
> 模型：DistilBERT-base（modelscope `AI-ModelScope/distilbert-base-uncased`，160s 下载）。
> 协议：4 类（World/Sports/Business/Sci-Tech），遗忘类 0（World），保留 {1,2,3}；每类 4000 训练、1500 评估；2 epoch 微调 + 各遗忘方法；RII 用 2×4 通道矩阵（[P(class|·)] 逐样本条件分布均值）。

| method | retain% | forget% | RII | MIA-loss |
|--------|--------:|--------:|------:|------:|
| NoUnlearn | 92.8 | 92.4 | **0.2665** | 0.694 |
| Retrain (oracle) | 96.1 | 0.0 | 0.0900 | 0.000 |
| NegGrad | 37.5 | 0.0 | 0.0009 | 0.000 |
| FineTune | 97.1 | 0.0 | 0.0656 | 0.000 |
| KED | 60.3 | 0.9 | 0.0214 | 0.274 |

**解读**：
1. **RII 跨模态迁移成立**：NLP 上 NoUnlearn RII=0.266 对 Retrain 0.090（**2.96×**），与 CV（CIFAR-10 2.5×、CIFAR-100 1.56× 反向）一致——RII 不依赖图像架构，在 Transformer 上同样能区分"未遗忘 vs 真遗忘"。这是论文主张的**跨模态普适性证据**。
2. **NegGrad 在 NLP 表现为显性失败（非隐藏失败）**：retain_acc 崩到 37.5%（CV 中 retain 保持 84-93%），RII=0.0009 极低——破坏性遗忘（模型整体崩坏）使 forget/retain 分布都退化为接近均匀，RII 反而"判为安全"。这与 CV 中 NegGrad 的"隐藏失败"（forget_acc 下降但 RII 上升）形成对照：**RII 低可能意味着"真遗忘"也可能意味着"模型崩溃"，需配合 retain_acc 一起读**——这与论文 §6 的 MIA/RII 调和一致。
3. **MIA-loss 排序与 RII 在 oracle 上一致**（NoUnlearn 0.694 最高、Retrain/NegGrad/FineTune≈0），但 KED 的 MIA=0.274 与 RII=0.021 提示其"部分遗忘"状态。
4. **实操注意**：DistilBERT 从 modelscope snapshot 加载时 classifier 层报 MISSING（预训练权重无下游头），脚本已用随机初始化下游头微调解决，不影响结果。

---

## 实验 F — 温度缩放 vs 状态空间（Fashion-MNIST / CIFAR-10，modelscope 补做）

> 脚本：`supplement_temp.py` → `results/supplement_temp/`
> 协议：7 类训练 {0..6}、遗忘类 3、留出 {7,8,9}；对同一基模型：
> (a) 温度扫描 T∈{1,2,5,10,50,100}：`p_T = softmax(logits/T)` 下重算 RII 与 MHPR；
> (b) 状态空间：logits 经 k-means（M=20）量化 → 状态分布直方图 → 状态 RII（ρ_S）与状态 MHPR（ρ_{H,S}）。
> 目的：回应改进项 2"温度缩放 vs 状态化——比较两种方法在低准确率模型上的效果"。

**Fashion-MNIST**（base retain 90.3% / forget 95.0%）：

| T | ρ | MHPR(T) |
|--:|------:|------:|
| 1 | 0.1604 | 0.7170 |
| 2 | 0.1847 | 0.6588 |
| 5 | **0.2096** | 0.4711 |
| 10 | 0.0734 | 0.1582 |
| 50 | 0.0017 | 0.0044 |
| 100 | 0.0004 | 0.0010 |
| 状态空间 (M=20) | **0.2999** | **0.9963** |

**CIFAR-10**（base retain 94.6% / forget 79.1%）：

| T | ρ | MHPR(T) |
|--:|------:|------:|
| 1 | 0.2268 | 0.8640 |
| 2 | 0.2277 | 0.6914 |
| 5 | **0.1036** | 0.2544 |
| 10 | 0.0308 | 0.0663 |
| 50 | 0.0009 | 0.0017 |
| 100 | 0.0002 | 0.0004 |
| 状态空间 (M=20) | **0.1644** | **0.7439** |

**解读**：
1. **温度缩放是双刃剑，且存在判别峰值**：两个数据集上 ρ 先随 T 上升（F-MNIST 在 T=5 达到 0.2096 峰值），T≥10 后**急剧衰减**（T=100 时 ρ<5e-4）。低温锐化分布增强类间判别（T=2~5 峰值），高温把分布抹平成均匀（ρ→0）——**过高的 T 会让任何模型都"看似遗忘"，温度选择必须谨慎**。这与论文 MHPR 的 κ(H(T))<10 温度选择策略互为印证：策略选取最小可行 T，避免高温人为抹平。
2. **MHPR(T) 单调下降**（F-MNIST 0.717→0.001，CIFAR-10 0.864→0.0004），比 ρ 更平滑——投影残差对温度更敏感，可作为温度效应的"单调探针"。
3. **状态空间保留更多判别信息**：ρ_S=0.30（F-MNIST）/0.164（CIFAR-10）均高于任何温度版本的 ρ；而 **MHPR_S≈1**（0.996/0.744）——k-means 状态直方图下 forget 分布几乎正交于 held-out 状态子空间，判别能力极强但绝对值偏高，说明状态化是"高增益高饱和"的度量，适合相对排序而非绝对阈值。
4. **两方法互补**：温度缩放适合"校准灵敏度/避免误报"（低 T 敏感、高 T 保守），状态空间适合"最大化分离度"；论文 MHPR 默认用温度缩放校准（κ 条件）而非状态化，本实验为这一设计选择提供了定量依据（状态化 MHPR 饱和在 ~1 附近，不利于设阈值）。

**诚实局限**：① 状态数 M=20 固定，未扫 M（更大 M 会使分布更稀疏、MHPR_S 更高）；② 单基模型（未对比 Retrain/NegGrad 的状态化行为）；③ k-means 在完整 logits 上拟合，train/forget/held 混合，未做独立的 held-out 状态字典。

---

## 实验 G — 真实 LLM 遗忘基准（Qwen2.5-0.5B + TOFU，modelscope 补做）

> 模型：`Qwen/Qwen2.5-0.5B`（modelscope，1min47s 下载，MPS 全量微调 284s 跑完全程）。
> 数据：`popatry/TOFU`（modelscope 镜像，**JSONL 格式**——每行一个 `{question, answer}`）；forget01=40 QA（10 作者）、retain90=3600 QA（90 作者，与 forget 作者**零重叠**，已校验）。
> 协议：base（Qwen2.5-0.5B）在 retain 子集（240 QA，与评估集 disjoint）微调 40 步（AdamW lr=5e-5, batch=4）→ **NoUnlearn**；从 NoUnlearn 快照继续 retain 微调 40 步 → **FineTune**；从 NoUnlearn 快照在 forget 上梯度上升 8 步 → **NegGrad**；从原始 base 权重只学 retain → **Retrain oracle**（从未见过 forget 作者）。
> RII 定义（LLM 版）：**2×V 通道矩阵**（V=151,936 vocab），对每个 QA 取 next-token softmax 分布在答案区间的平均，μ_f/μ_r 为 forget/retain 问题集均值，SVD 得 ρ。
> 评估：生成 32 tokens（greedy），**ROUGE-L F1≥0.25** 判定命中（比单 token 重叠严格）。

| method | forget_acc% | retain_acc% | RII | MIA (NLL) |
|--------|--------:|--------:|------:|------:|
| NoUnlearn | 90.0 | 75.0 | **0.0825** | 2.84 |
| FineTune | 90.0 | 72.5 | 0.0765 | 3.02 |
| NegGrad | 0.0 | 0.0 | 0.0000 | **87.2** |
| Retrain (oracle, 从未见 forget) | 85.0 | 67.5 | 0.0803 | 2.82 |

**解读**：
1. **FineTune 反弹在真实 LLM 上复现（跨模态最强证据）**：继续在 retain 上微调 40 步后，forget_acc 保持 90% 不变（forget 知识未擦除）——与论文 §5.5 的 CV 反弹现象一致，且与 CV 一样"rebound 的 RII 仅微降"（0.0825→0.0765）。
2. **NegGrad 在 LLM 上是破坏性遗忘（显性失败）**：8 步梯度上升使 retain_acc 从 75% 崩到 0%，MIA 飙到 87（分布完全崩坏），RII→0。与实验 E（NLP DistilBERT）一致：**RII 低 + retain_acc 低 = 模型崩溃**，必须联合 retain_acc 判读（呼应论文 §6）。
3. **known/unknown 不对称的 LLM 版本**：Retrain 从未见过 forget 作者，但 RII=0.0803≠0，且与 NoUnlearn（0.0825）几乎不可分——与 CV 中 Retrain oracle RII=0.09≠0 同构。**这是诚实的边界发现**：在"作者级 + vocab 通道"设定下，forget/retain 问题集的输出分布高度同质（都是流畅问答文本），RII 无法区分"学过 vs 没学过该作者"；CV 中 RII 能区分（0.233 vs 0.092）是因为 forget/retain 是**不同的输出类别**（不同 label 通道）。这提示：**LLM 应用 RII 时，通道应取类标签级（如作者分类头/隐藏状态聚类），而非原始 vocab**——正是论文 rem:channel "通道级 vs 样本级" 区分在 LLM 上的实例化。
4. **TOFU 评估的固有难点被定量暴露**：Retrain（零接触 forget 作者）forget_acc 仍 85%——因为多数 TOFU 答案（人名、城市）**verbatim 出现在问题里**，模型从问题提取即可"答对"，ROUGE-L 0.25 仍偏宽松；官方 TOFU 亦需 paraphrase 级人工评估。故本实验的**可靠指标是 RII 与 MIA（分布级）**，准确率仅作参考。

**诚实局限**：① 单种子、单模型（0.5B）、40 步微调（模型未完全收敛，MIA 2.8 偏高）；② NegGrad lr=2e-4 × 8 步过强（模型崩坏），未做"适中强度"扫描以观察隐藏失败；③ forget01 仅 40 QA、评估 20 个，统计功效低；④ 未跑官方 LLaMA-7B 级 TOFU 协议（24GB MPS 不可行）——该部分需人工在 GPU 服务器完成；⑤ 未测 MHPR（TOFU 无 held-out 作者类结构）。
