# 论文人工审稿指南（中文版）

> **用途**：供作者以审稿人视角，一小节一小节快速通读论文，逐节核对核心逻辑。
> **配套**：`ai_submission/manuscript.tex`（提交版，29 页）。本节指南中的"已修正"指已按此前审查修改过的内容。
> **阅读建议**：每节先看「核心主张」→ 再看「行文脉络」→ 最后用「审查建议」里的核查点逐条自问。标注 🔴 的是一旦被审稿人抓住会伤及可信度、必须能讲清楚的点；🟡 是可接受但最好有解释的点。

---

## 总览：论文到底讲了一个什么故事

**一句话**：模型输出通道矩阵（2×C）的第二个奇异值能量占比 ρ（RII）可以作为"遗忘是否成功"的黑盒谱证书；类级遗忘下由于"已知/未知类不对称"，标准 ρ 不够，用 MHPR（多留出类投影）修正；并给出理论保证（ρ=0 ⟺ 输出不可区分）与跨尺度验证（CIFAR-10/100、AG News、LLaMA-2-7B TOFU）。

**核心逻辑链**（审稿时盯住这条链是否每环都被支撑）：
1. RII 是 tuning-free 的输出级指标（无需训练/无超参）→ §2
2. ρ=0 ⟺ 输出通道独立（理论）→ §4
3. 类级遗忘有已知/未知不对称 → §3 LOCO 失败
4. MHPR 修正该不对称 → §3 + §5.6
5. 实验证明：RII 能区分真遗忘 vs 未遗忘，且揭示梯度上升"隐藏失败" → §5
6. MIA 与 RII 是正交两轴，需联合报告 → §6

---

## 摘要

**核心主张**：引入 RII 与 MHPR；证明 ρ=0 ⟺ 输出不可区分；在 CIFAR 家族 + TOFU/7B 上验证；在准确率饱和时 RII 是唯一区分 oracle 与未遗忘的指标；揭示梯度上升隐藏失败与 fine-tune 反弹。

**审查建议**：
- 🔴 摘要中 "RII is the only metric that separates the retraining oracle from no unlearning" —— 这是全篇最重的声明。核对：它**只**声称 oracle vs 未遗忘（Retrain vs NoUnlearn），**没有**声称区分 NegGrad（该差距在噪声内，正文已显式声明）。审稿人若问"为什么不提 NegGrad"，答案在 §5.3。
- 🟡 "MHPR resolves the class-level asymmetry" —— 注意与贡献 3 已统一为 "largely resolves"。若你觉得摘要措辞仍偏强，可自己斟酌。
- 核对关键词是否覆盖核心（RII/MHPR/谱证书/输出不可逆）。

---

## §1 Introduction（引言）

**行文脉络**：GDPR 需求 → 现有验证方法（MIA 混淆、DP 侵入）不够 → 提出谱条件 → 明确 Scope（输出空间、类级、非白盒安全）→ 5 条贡献列表。

**审查建议**：
- 🔴 Scope 段的"black-box certificate"与"Parameter-level white-box security is not asserted"是否自洽：不矛盾（黑盒=输出级审计工具，not asserted=不声称参数级安全）。若审稿人质疑，按此回答。
- 🔴 贡献 3 "largely resolves the known/unknown class asymmetry" —— 与 §5.4 benchmark 的 "narrows the gap"（oracle MHPR 0.428 仍不为 0）措辞统一。请自己判断"largely resolves"是否仍有过度承诺风险。
- 🟡 引言对"7B 实验"的定位（贡献 5）是否与 §5.3 一致：一致（oracle vs 未遗忘，准确率饱和）。

---

## §2 Problem Setup and Spectral Framework（问题设置与谱框架）

### 2.1 Problem Setup
- 符号：D_f（遗忘集）、D_r（保留集）、θ'（遗忘后参数）。**核对**：全文符号是否一致（尤其 D_f/D_r、μ_f/μ_r、σ₁/σ₂）。
- 声明聚焦"类级遗忘"，并说明"样本级遗忘会掩盖谱结构"。

### 2.2 The Empirical Output Channel Matrix（输出通道矩阵）
- 定义 M ∈ R^{2×C}，行 = 遗忘/保留集的平均 softmax 向量。
- **审查建议**：
  - 🟡 Def. M 无"等范数"假设；§2 说 "when ‖μ_f‖=‖μ_r‖, ‖μ_f−μ_r‖=√2σ₂"。**核对**：等范数不自动成立，正文是否只在明确限定时才用此关系（thm:mi_bound 已改为"条件渐近放宽"）。
  - 🔴 该矩阵只用**一阶矩（均值）**——这是本框架最大的理论局限，已由 rem:channel + PSSD（附录 A）处理。审稿人若问"均值相等≠分布相等"，回答指向 rem:channel 的 categorical-Y 论证 + PSSD。

### 2.3 The Residual Irreversibility Index（RII）
- 定义 ρ = σ₂²/(σ₁²+σ₂²) ∈ [0, 0.5]；性质（ρ=0 ⟺ 完美遗忘；尺度不变）。
- "What ρ is—and is not"：ρ 是谱残差能量，不是 MI 值；其信息论意义由 thm:mi_bound 给出。
- **审查建议**：
  - 🔴 为什么 ρ 上限是 0.5：因为 2×C 矩阵只有两个奇异值（σ₁≥σ₂，ρ≤0.5）。**核对**正文是否讲清（有）。
  - 🟡 ρ 是"群体级"统计量（平均分布），对单样本异常不敏感——与 PSSD 的分工要讲清。

---

## §3 Class-Level Forgetting and MHPR（类级遗忘与 MHPR）

### 3.1 LOCO Failure
- 单一留出类作为参照会失败：MNIST 例 ρ_LOCO≈0.32 > 标准 ρ≈0.14（方向反了）。
- **审查建议**：🟡 这里只用了一个 MNIST 例子支撑"LOCO 失败"的一般性。审稿人可能问"CIFAR-10 上 LOCO 也失败吗？"——回答：CIFAR-10 benchmark 的 held-out {7,8,9} 即多留出（K=3），其 MHPR 有效，间接支持；但没有 CIFAR 上的 LOCO 单类对照表。如被问，可承认这是 MNIST 演示。

### 3.2 Multi-Held-Out Projection Residual（MHPR）
- 定义 ρ_H = ‖μ_f − H⁺Hμ_f‖²/‖μ_f‖² ∈ [0,1]（遗忘类均值到留出类子空间的投影残差）。
- 属性：单调（K 大→ρ_H 小）、优于 LOCO、偏差/有限样本界。
- 温度缩放：低精度时留出均值共线→子空间退化；选最小 T 使 κ(H)<10。
- **审查建议**：
  - 🔴 ρ_H 与 ρ 范围不同（[0,1] vs [0,0.5]），正文已加一句解释归一化对象不同。**核对**自己读起来是否清楚。
  - 🔴 "ρ_H=0 ⟺ μ_f 在留出子空间内"——这不是"遗忘成功"的直接证明，而是"遗忘类像某个留出类"的证据。审稿人可能问"oracle 的 ρ_H=0.428 为什么不为 0"：因为 cat 与留出类 {7,8,9} 语义不近，投影残差大。**必须能讲清**（caveat 2 已提）。
  - 🟡 温度缩放的"选 T"准则（最小 T 使 κ(H)<10）是启发式，不是理论最优。若审稿人问"T 是数据窥探吗"——回答：有固定准则、且在补充实验 F 报告了 T 扫描全曲线（正文 §3.2 有一句）。
  - 🟡 K 的选择（推荐 K=3）依据 MNIST 表：K=4 更接近 0 但牺牲训练数据。**核对**这个权衡的表述。

### 3.3 Bridging Logit-State and Softmax RII（状态空间桥）
- 将 logits 量化（k-means，M 个状态）→ 状态分布 → 状态空间 RII 逼近 softmax RII；给出 χ² 界。
- **审查建议**：🟡 本节是"桥接/扩展"，正文承认细节在附录 B。审稿人若问"为什么需要状态空间"——回答：为 MHPR 提供干净的 χ² 界（thm:state_mhpr）与算子推广。

---

## §4 Theoretical Results（理论结果）

**行文脉络**：thm:perfect（ρ=0 ⟺ I(X;Y|θ')=0）→ prop:confidence（有限样本界）→ thm:finetune（O(λ²) 微调泄漏界）→ thm:mi_bound（谱 MI 界）→ thm:unified（两轴分解）。

**审查建议（重点，这是审稿人最挑剔的部分）**：
- 🔴 **thm:perfect**：两个方向都证了；关键论证是"rank-1 2×C 矩阵 + 概率向量行和为 1 → μ_f=λμ_r → λ=1"。**核对**：C=1 时怎么办（softmax 单类退化，论文默认 C≥2；若被问要能答）。"无需指数族/校准假设"是核心卖点，确认自己认同其论证（categorical Y 使行=条件分布）。
- 🔴 **thm:finetune**：界用 Tr(Σ_f)（logit 梯度协方差），更新用 loss 梯度 g。**已修正**：定理声明加了 "L-Lipschitz softmax map"，证明加了 "g 归一化到单位尺度（范数吸收进 λ）"。**核对**：你读证明（附录 C.4）时确认这两个修正是否让推导讲通了；若仍有疑问，这是最可能被揪的环节。
- 🔴 **thm:mi_bound**：等范数假设。**已修正**：括号从"the case used in Def. M"（错误）改为"条件渐近放宽"。**核对**：你能否向审稿人解释"为什么等范数条件可渐近放宽"（正文有，读一遍确认理解）。
- 🔴 **thm:unified**：信号模型 R=αX+βY+N，界 I(X;R) ≤ α²/(2σ²)Var(X) + η² + o(·)。**已验证**：β 项通过链式法则 I(X;R)≤I(X;Y)+I(X;R|Y) 在条件于 Y 后成为常数被吸收，正确。**核对**：你自己能否一句话复述这个"β 去哪了"的论证（审稿人必问）。
- 🟡 thm:unified 的 η²=Σ_{k≥2}σ_k² 与 ρ 的关系（ρ=σ₂²/(σ₁²+σ₂²)）——正文是否讲清"cover 泄漏与 RII 对应"（有，§4 + §6）。

---

## §5 Experimental Evaluation（实验）

### 5.1 Setup
- 数据集/模型/协议（CIFAR-10 主协议：训练 {0..6}，遗忘 cat，留出 {7,8,9}）；Key findings 预告三条。
- **审查建议**：🟡 CIFAR-100 用 SmallCNN（20.8%）与 ResNet-18（29.7%），无数据增强——已在表中标注 (SmallCNN)。若审稿人问"为什么 CIFAR-100 精度这么低"：无增强 + 轻量模型，是刻意的可复现设定，且正是论文要展示"ρ 与精度无关"的场景。

### 5.2 Baseline: Near-Rank-One（随机遗忘基线）
- 随机 5% 遗忘下 ρ<5×10⁻³（所有方法几乎相同）——控制实验，证明 RII 不退化。
- **审查建议**：
  - 🟡 MNIST 表只显示 NoUnlearn 一行，文本声称"Retrain/SISA/FineTune 也验证过"——**数据不在表内**，审稿人无法复现这句。若被问，如实说"完整数据在旧版/内部记录"，或考虑删掉这句声称。
  - 🔴 "across 10 random seeds, SD<3.5×10⁻⁴" 支撑"可单次审计"——但这个稳定性是**随机遗忘**下的；类级遗忘的稳定性另有 7B 多种子支撑。**核对**不要把它误读为"所有实验都单次即可"（caveat 4 已加说明）。

### 5.3 RII on a 7B-Parameter Language Model (TOFU)【主实验】
- 协议：LLaMA-2-7B + TOFU 作者级；LoRA；NoUnlearn/FineTune/NegGrad/Retrain。
- 四段：准确率饱和但 RII 区分；梯度上升无可检测擦除；FineTune 降低 RII；多种子稳定性；vocab 通道在 7B 可用；强度扫描（双向敏感）。
- **审查建议（本节是重头，请重点读）**：
  - 🔴 **本节的诚实边界**（已修正）：NegGrad 的 RII（0.117）与 NoUnlearn（0.116）**在噪声内不可区分**（差距 0.0006 vs std 0.002）。正文已把"隐藏失败复现"的主张从基础 NegGrad 移到了**强度扫描**（中等强度 RII 升到 0.1203）。**核对**：你读 §5.3 时确认两处表述不打架（"不区分 NegGrad/NoUnlearn"与"梯度上升无可检测擦除"是一致的，隐藏失败证据在强度扫描）。
  - 🔴 **Retrain 是"轻量 oracle"**（60 步 LoRA，retain_acc 65% 未收敛）——已标注。判别差距 1.45× 是**下界**；充分训练可能更强。若审稿人问"oracle 够格吗"：回答"轻量 oracle，判别是下界，方向对论文有利"。
  - 🟡 多种子（3 次子采样）std<3%：注意这是**评估集子采样**的波动，不是重训的种子波动。若审稿人问"为什么不是重训 3 次"：LoRA 训练有确定性 + 子采样波动已足以给误差棒；重训种子成本高。如实回答。
  - 🟡 vocab 通道在 7B 可用的结论依赖"模型容量"这个解释（0.5B 不行、7B 行）——这是事后解释，样本量只有 2 个模型。诚实表述已做（"capacity determines whether raw channels suffice"）。
  - 🔴 强度扫描 strong_x6 的"回落"只有一个数据点：0.1006 与 0.1168 差 0.016（远超 std 0.002，显著），但曲线形状单点支撑。若审稿人问"回落是噪声吗"——差距显著，但建议你把"回落=分布崩坏"的机制解释（正文已写）能用自己的话讲清。

### 5.4 Benchmark Comparison（CIFAR-10 七方法）
- 表：7 方法 × (retain/forget acc, RII, MHPR, MIA-loss, MMD, probe)；发现：已知/未知不对称、NegGrad 隐藏失败、一致性/正交、表示级残留、SISA caveat、跨数据集、CIFAR-100、跨模态。
- **审查建议（重点）**：
  - 🔴 **SISA 假阳性**（已修正）：SISA RII=0.064 最低（"最好"），但 retain_acc 仅 69.9%（模型劣化）。**已从排序结论中排除**，并加了一句"退化模型可把 RII 推向 0（与 7B/文本崩坏一致）"。**核对**：你读 caveat (1) 时确认自己认同这个"低 RII 的两种来源（真遗忘 vs 分布退化）"的统一解释——这是把"例外"变成"一致模式"的关键，也是审稿人最可能问的。
  - 🔴 **NegGrad 隐藏失败**：CIFAR-10 上 NegGrad forget_acc 67.9→63.4%（看似"有效遗忘"）但 RII 0.233→0.254 上升。**注意**：这是单次运行，无 std——你无法回答"这个上升是否显著"。若审稿人问，诚实说"CIFAR 主基准单次运行（成本），7B 上同类效应由强度扫描支撑（caveat 4）"。
  - 🔴 **RII 与 forget_acc 相关 0.93**：**已加解释**（相关由 Retrain 端点驱动；非 oracle 方法间二者分歧）。审稿人问"RII 是不是 forget_acc 的代理"——回答指向 NegGrad 分歧 + 7B 准确率饱和场景（那里 forget_acc 完全失效）。
  - 🟡 **相关性矩阵 n=7**（已修正）：表注已注明"7 个方法点、仅供参考"。不要再把它当统计检验用。
  - 🟡 表示级 probe AUC≈0.96 全部饱和（含 oracle）——这其实说明"输出级擦除≠表示级擦除"，论文已如实报告并引用了 RULER/Erased-not-gone。
  - 🟡 cross-modal（AG News DistilBERT）：NegGrad 崩坏（retain 37.5%，ρ≈0）= 显性失败，与 CIFAR 隐藏失败对照——**核对**你能否讲清"为什么文本上崩坏、图像上不崩坏"（论文没有给出机制解释，只说行为不同；如被问，诚实说留待未来）。

### 5.5 Dynamic Range and Gradient Sweep
- 动态范围（随机 1.2e-3 vs 类级 0.183，152×）；梯度上升步数扫描（ρ 单调下降）。
- **审查建议**：🟡 梯度扫描显示 ρ 随步数**下降**（0.183→2e-3），而 §5.4 说 NegGrad RII **上升**——**核对这个表面矛盾**：§5.5 是"步数越多模型越崩"（acc 25.4%），§5.4 的 NegGrad 是 10 步（acc 63.4% 未崩）且 RII 高。两处都是"梯度上升在未崩坏时 RII 高、崩坏时回落"的统一图景（与 7B 强度扫描完全一致）。**你必须能在审稿人面前把这个统一图景讲清**，否则会被认为前后矛盾。

### 5.6 MHPR Evaluation
- MNIST LOCO vs MHPR 表（K=3 → 0.046）；温度缩放（Fashion T=50 → 0.021）；合成验证（残差在子空间 → ρ_H=0）。
- **审查建议**：
  - 🔴 **MHPR 的"解决"证据主要在 MNIST**（0.046），CIFAR-10 上 oracle 是 0.428（只是缩小）。若审稿人问"MHPR 到底解决了吗"——答："在参照类与遗忘类语义相近时（MNIST）接近 0；在 CIFAR-10 因 cat 与 {7,8,9} 语义远而绝对值仍高，但相对 NoUnlearn 显著下降"。这是诚实边界，caveat 2 已提。请确认你认同此表述。
  - 🟡 合成"理想遗忘构造"（把 μ_f 放在子空间内 → ρ_H≈0）是自证，不是独立验证——它只是验证了定义，不要过度宣传。

### 5.7 FineTune Rebound and Sensitivity
- 反弹：ascent 后 fine-tune，ρ 从 0.105 升到 0.221（超过基线 0.183）；过拟合敏感性（ρ 涨 10×）；ResNet-18 架构独立性；per-sample MHPR gap。
- **审查建议**：
  - 🔴 **反弹机制**：论文说"ascent 降 η²，但 fine-tune 重塑决策边界再错位 forget 表示，抬高 σ₂"。**核对**：这机制解释是否让你信服？这是 thm:unified 的应用，但"重塑决策边界"是定性说法。若审稿人要求更定量，诚实说"机制解释是定性的，定量刻画留待未来"。
  - 🟡 "per-sample MHPR gap"用了 MNIST 的状态空间数字（0.99999 vs 0.99932）——数值很接近 1，读起来奇怪（gap 0.00066）。**核对**：这段是否值得保留（它是理论 prop 的验证，但绝对值接近饱和，可能让审稿人困惑）。

### 5.8 Large-Scale Validation and Sensitivity
- ResNet-18 表、过拟合敏感性表、10 种子 MNIST 稳定性。
- **审查建议**：🟡 本节（含 tab:resnet18、tab:sensitivity）是"附录级"的消融，放在正文 §5.8。若你想压缩篇幅，可整体移到附录（目前保留是因为架构独立性/过拟合敏感是审稿人常问）。

### 5.9 Comparison with Existing Metrics
- 表格对比 MIA/DP/Backdoor/表示级/审计 vs RII/MHPR（信息论性、tuning-free、复杂度）。
- **审查建议**：🟡 这个表把 RII 标为"Info.-theoretic: yes"——严格说 RII 是"谱残差能量、由 thm:mi_bound 提供信息论含义"（§2.3 已澄清"不是 MI 值"）。**核对**：表格的"yes"与正文"不是 MI 值"是否会被视为矛盾——建议你能解释"谱指标 + 信息论保证"的定位。

---

## §6 Reconciling MIA and RII（两轴调和）

**行文脉络**：MIA 测"直接泄漏"（train/test 位移，I_dir），RII 测"覆盖泄漏"（输出签名，I_cov）；两者可同时大、是独立轴；α-扰动实验验证不对称响应；"beyond output"（表示级残留需联合审计）。

**审查建议**：
- 🔴 **"RII correlates with MIA at r=0.81 (moderate-to-strong)"** 与"正交轴"主张的张力：0.81 是强相关，为什么还说"正交"？**核对**：论文的论点是"相关但不同轴"（r<1 + 排名反转 + α 实验的不对称响应），不是"不相关"。你要能讲清"部分正交/独立但相关"。
- 🟡 α-扰动实验（MNIST 5%）：α=0.20 时 RII 涨 9× 而 MIA 涨 6.8pp——这是合成扰动，不是真实遗忘方法。真实方法的证据是 benchmark 的排名反转（NegGrad 在 RII 最差、在 MIA 第二好）。
- 🟡 §6 提到 PSSD 是"per-sample 扩展"——PSSD 在附录 A，正文只在 rem:channel 和 §6 提及。若审稿人认为 PSSD 是重要贡献却埋在附录，你要能引用 advisor 的结构决定（或考虑未来版本调整）。

---

## §7 Related Work

**行文脉络**：机器学习遗忘 → 遗忘评估（MIA/DP/backdoor）→ 超越攻击的验证（RULER、Erased-not-gone、Fragile、bias）→ 信息论安全。

**审查建议**：
- 🔴 **引用真实性**：25 条全部已验证（含最新正式出版：Xue et al. ACM CSUR 58(12) Art.314, DOI 10.1145/3807451）。若审稿人问及 2026 年 arXiv 编号（2605/2606 开头）——当前就是 2026 年，正常。
- 🟡 RII 与 RULER/Erased-not-gone 的定位差异：RULER 是表示级、RII 是输出级——"输出级可与表示级残留共存"（probe AUC≈0.96）正是连接点，论文已引用。

---

## §8 Deployment Considerations（部署）

**行文脉络**：O(CN) 一次前向 → 子采样保证公式（prop:confidence）→ 推荐审计协议 4 步 → 直接泄漏参数 α 的估计（MIA 优势作下界）。

**审查建议**：
- 🔴 **N_min≈7×10⁷ 的数值**：正文说"最坏情况需 7×10⁷ 样本，实际每类 ~10⁴ 即可"。这个差距很大（10⁴ vs 10⁷）——**核对**你能否解释"为什么最坏情况界这么保守"（正文已说：union bound + 均值常数）。若审稿人算一下会觉得界很松——诚实承认是保守界。
- 🟡 α 不可观测 → 用 MIA AUC 作下界：这是合理的启发，但"两轴审计"的完整性依赖"α 由 MIA 下界 + η 由 RII 精确"，论文是否讲清这个分工（§6 + §8 都有）。

---

## §9 Conclusion

**行文脉络**：总结 → 5 条关键发现 → 实用建议 → 战斗口号（RII+MIA 联合报告）→ 未来工作。

**审查建议**：
- 🔴 结论 (v) 的"only metric separating retraining from no unlearning"与摘要/引言一致（都限定 oracle vs 未遗忘）——**核对**三处措辞统一。
- 🟡 结论 (iv) "AUC>0.94 across MLP-based datasets"——限定词 "MLP-based" 很重要（CIFAR-10 CNN 的 δ-MIA 只有 0.511）。若审稿人只读结论，会不会误读为"所有数据集"——措辞已有限定，但你可以考虑是否再加一句边界。

---

## 附录 A：PSSD（逐样本状态差异）

- 定义：逐样本 disparity δ(x)；聚合度量 Δ_f、Ψ；δ-MIA 保证（thm:pssd_mia）；实验表（MNIST/Fashion/CIFAR-10）。
- **审查建议**：
  - 🟡 PSSD 是"measure-then-average"的逐样本扩展，附录开头已声明"核心贡献不依赖此扩展"（advisor 决定）。若审稿人问"PSSD 为什么不在正文"——按此回答。
  - 🔴 **CIFAR-10 (CNN) 上 δ-MIA AUC=0.511 几乎随机**，而结论 (iv) 说 "AUC>0.94 across MLP-based"。**核对**：这个限定是否够诚实（附录 A.3 已详细解释 CNN 上优势消失的原因——softmax 置信本身已强 + Ψ≈0.001 极小）。若审稿人只看结论会误读，建议你考虑在结论 (iv) 再补一句边界（可选）。

---

## 附录 B：State-Space Bridge（状态空间桥）

- 量化误差引理、谱桥定理（状态 ρ 与 softmax ρ 等价）、softmax 偏差定理、算子推广。
- **审查建议**：🟡 这部分偏理论/扩展，正文只引用 3 个定理（thm:state_bridge、thm:softmax_deviation、lem:quantization 均已加引用）。若你不想被审稿人深究，可在回答问题时明确"这些是扩展性结果，不影响主线"。

---

## 附录 C：Key Proofs（关键证明）

- thm:perfect 证明（λ=1 论证）、prop:confidence、thm:mi_bound、thm:finetune（含 lem:cross_cov）、thm:unified 证明梗概、状态空间 MHPR 界、fisher 引理、MHPR 理论保证、温度缩放保信息。

**审查建议（逐条，这是你"身在此山中"最需要外部眼光的部分）**：
- 🔴 **thm:finetune 证明**（C.4）：已补"g 归一化"说明 + "L-Lipschitz softmax map"。**请重点通读**——这是最可能被审稿人推导验证的环节。核对：① Δμ_f 的一阶 Taylor 展开（用 J_softmax·J_θf·g）② "cross term with μ_f−μ_r vanishes by symmetry of balanced retain set"——这条对称性论证你认同吗？（若保留集平衡，μ_r 的位移一阶项抵消）③ 最后 ρ≈‖μ_f−μ_r‖²/(2σ₁²) 的近似（ρ≪1 时）。
- 🔴 **thm:mi_bound 证明**（C.3）：等范数下 MI 与 ρ 的链接。**核对**你能复述其结构（χ²-MI + 谱分解）。
- 🔴 **thm:unified 证明**（C.5）：链式法则 + HS 分解 + fisher 引理。**核对**"β 去哪了"（条件于 Y 吸收）你能一句话讲清。
- 🟡 **lem:cross_cov**：依赖"well-trained classifier（ε=O(λ)）"——即分类误差与步长同阶。这个假设在低精度模型（如 CIFAR-100 20.8%）下还成立吗？**核对**：若审稿人问，回答"该引理用于类级界，低精度场景未直接依赖"。
- 🟡 **prop:confidence**：N_min 公式的常数 16 与 ν——ν 的定义在正文有吗？**核对**符号一致性。
- 🟡 **状态空间 MHPR 界**（χ² 验证表 tab:chi2）：实验验证了界成立（保守 ~10²–10³ 倍），说明界很松但方向正确——诚实。

---

## 附：审稿人可能追问的 10 个问题（预答清单）

1. **RII 是不是 forget_acc 的代理？** → 相关 0.93 但由 Retrain 端点驱动；NegGrad 上二者分歧；7B 上 forget_acc 全饱和而 RII 仍判别。
2. **为什么 oracle 的 RII 不为 0？** → 已知/未知不对称；类级遗忘后 forget 类成为"未见类"，分布天然不同（§5.4）。
3. **SISA 为什么 RII 最低？** → utility-confounded（retain 69.9%），已排除出排序；且"退化→RII 假低"是统一模式（7B 强度扫描、文本崩坏）。
4. **NegGrad 到底算不算"隐藏失败"？7B 上它和 NoUnlearn 差不多。** → 基础 NegGrad 在 7B 与 NoUnlearn 噪声内不可区分（无擦除信号）；隐藏失败的强化证据在强度扫描（中等强度 RII 显著上升）。
5. **MIA 的 AUC=0.000 是不是 bug？** → 不是；oracle 从未训练 cat，loss 完美反向，MIA 将 forget 全判非成员——这是遗忘成功的分布级证据。
6. **ρ=0 为什么等于"输出不可区分"？均值相等不够吧。** → rem:channel：Y 是 categorical，P(Y|X=f) 就是 μ_f 向量本身，均值相等=分布相等；逐样本层面由 PSSD 处理。
7. **温度缩放是不是数据窥探？** → 有固定准则（最小 T 使 κ(H)<10）+ 补充实验 F 报告了全扫描。
8. **7B 的 oracle 够格吗？** → 轻量 60 步 LoRA oracle（retain 65% 未收敛），判别 1.45× 是下界，充分训练只会更强。
9. **CIFAR 主基准为什么单次运行？** → 训练成本；种子稳定性由随机基线（10 种子）与 7B（3 子采样）交叉支撑（caveat 4）。
10. **为什么文本上 NegGrad 崩坏、图像上隐藏失败？** → 行为不同但都是"RII 低=真遗忘或崩坏，须配 retain_acc 判读"的统一图景；机制差异留待未来。

---

*生成时间：2026-08-13。对应提交版 manuscript（commit 060e2cf）。*
