# 文献查重报告：RII（输出空间谱不可逆性）思路核查

> 日期：2026-08-11
> 核查对象：A Spectral Framework for Measuring Output-Space Irreversibility in Machine Unlearning（Applied Intelligence 投稿稿）
> 核心思路：2×C 混淆矩阵（forget/retain 平均 softmax 输出）SVD，RII ρ=1−σ₁²/(σ₁²+σ₂²)，rank-one ⟺ 输出级不可区分，MHPR 用于类级遗忘

---

## 一、结论（Bottom Line）

**未发现与本文完全相同的思路。** 具体验证：

| 检索词 | 来源 | 结果 |
|---|---|---|
| "confusion matrix" AND "machine unlearning" | arXiv 全文检索 | **0 条** |
| "Residual Information Index" machine unlearning | DuckDuckGo | **无结果** |
| "rank-one" "confusion matrix" unlearning "singular value" | DuckDuckGo | **无结果** |
| "machine unlearning" AND "singular value" | arXiv 全文检索 | 7 条，全部为**遗忘算法内部用 SVD**（非输出层评价指标） |
| "machine unlearning" AND spectral | arXiv 全文检索 | 8 条，全部为**算法方法**（非评价指标） |

**结论**：用"2×C 混淆矩阵 SVD 奇异值比"作为机器学习遗忘的输出级谱证书，且配有 RII/MHPR 理论（ρ=0 ⟺ I(X;Y|θ')=0、χ² 界、O(η²) 微调泄漏界），**在 arXiv + 通用网页范围内未检索到先例**。领域内最接近的工作均为"验证方法"或"评价指标"，但技术路线完全不同（见下文）。

⚠️ **注意**：本检索覆盖 arXiv（全文）、DuckDuckGo/Bing 通用网页、语义学者（API 被限流，未完成）。**未覆盖** IEEE Xplore / SpringerLink / ACM DL 的期刊专有库。建议投稿前再在上述库用关键词 "unlearning verification"、"forget set output distribution"、"spectral certificate unlearning" 复查一次。

---

## 二、最相关论文逐一核查（请在引言中引用并区分）

### A. 输出/行为层验证（与本文同为"输出层验证"，但方法不同）

| # | 论文 | 链接 | 主要方法 | 与本文区别 |
|---|---|---|---|---|
| 1 | **RULER: Representation-Level Verification of Machine Unlearning** (Cosma & Finke, 2026) | https://arxiv.org/abs/2605.27569 | 表示层验证：M2（oracle 对比，遗忘样本在重训练模型中的表示位置）、M4（无需重训练的残差检测，基于内部相似性结构 + 线性混合效应模型） | 工作在**中间表示层**（非输出层）；用统计假设检验而非谱分析；本文是黑盒输出层谱证书 |
| 2 | **Erased, but Not Gone: Output Forgetting Is Not True Forgetting** (Yong et al., 2026) | https://arxiv.org/abs/2606.25001 | 指出输出级成功可与表示层残差共存；用重训练一致性的表示遗忘来评估 | 论证输出层评估会高估遗忘成功——**可作为本文"输出层证书局限性"的引用**；方法在表示层，本文在输出层 |
| 3 | **Auditing Machine Unlearning: ... Whether Models Truly Forget** (Ye et al., 2026) | https://arxiv.org/abs/2606.16110 | 首个通用审计框架（proof of ignorance），无需重训练基线、无需影子模型、无需侵入训练 | 基于 influence/representation 的审计；非谱方法。结论（fine-tune 有效、de-optimization 无效）与本文 FineTune 现象可互为印证 |
| 4 | **RUB: Evaluating Residual Knowledge in Unlearned Models** (Xuan & Li, CVPR-W 2026) | https://arxiv.org/abs/2504.14798 | Robust Unlearning Benchmark，UMA（Unlearning Mapping Attack）检测残余知识，跨判别/生成任务 | 对抗攻击式基准（有攻击者模型）；本文是无攻击者的谱证书 |
| 5 | **TAPE: Tailored Posterior Difference for Auditing** (Wang et al., 2025) | https://arxiv.org/abs/2502.19770 | 用未学习前后 posterior 差异 + reconstructor 模型审计 | 需要训练 reconstructor/影子模型；本文 tuning-free 单次前向 |
| 6 | **EVE: Efficient Verification of Data Erasure** (Wang et al., 2026) | https://arxiv.org/abs/2602.03567 | 定制扰动数据使未学习前后预测改变，作为验证信号 | 需要扰动 forget 数据（侵入式）；本文无需扰动 |
| 7 | **SMS: Self-supervised Model Seeding for Verification** (Wang et al., 2025) | https://arxiv.org/abs/2509.25613 | 自监督种子嵌入（类后门法），验证真实样本遗忘 | 需要训练期种子嵌入；本文纯黑盒事后审计 |
| 8 | **Really Unlearned? IndirectVerify** (Xu et al., 2024) | https://arxiv.org/abs/2406.10953 | 影响样本对（trigger/reaction），扰动 trigger 观察 reaction 重分类 | 需构造扰动样本对；本文直接看输出分布 |
| 9 | **Robustness Verification Without Prior Modifications** (Xu et al., 2024) | https://arxiv.org/abs/2410.10120 | 优化法从模型参数恢复训练样本，比较遗忘前后 | 需要白盒参数；本文黑盒输出 |

### B. 证明/认证类（"certified" 路线）

| # | 论文 | 链接 | 主要方法 | 区别 |
|---|---|---|---|---|
| 10 | **Verification of Machine Unlearning is Fragile** (Zhang et al., ICML 2024) | https://arxiv.org/abs/2408.00929 | 证明验证可被对抗性遗忘绕过（两类 circumvention） | **建议引用**：说明任何验证都可能被对抗者规避，作为本文 Scope 讨论 |
| 11 | **Unlearning as Distribution Restoration** (Yang & Yeung, 2026) | https://arxiv.org/abs/2607.19442 | 无 oracle 认证的局限；分布恢复视角 | 讨论 oracle-free 认证边界，与本文 tuning-free 目标一致 |
| 12 | **Proof of Unlearning** (Weng et al., 2022) | https://arxiv.org/abs/2210.11334 | 密码学证明 | 密码学路线，非谱 |
| 13 | **Towards Probabilistic Verification** (Sommer et al., 2020) | https://arxiv.org/abs/2003.04247 | 概率验证框架 | 本文已引用（sommer2022towards） |

### C. 类级遗忘评价（与 MHPR 最相关）

| # | 论文 | 链接 | 主要方法 | 区别 |
|---|---|---|---|---|
| 14 | **Classification-Head Bias in Class-Level Machine Unlearning** (Zheng et al., 2026) | https://arxiv.org/abs/2605.08730 | 类级遗忘：分类头偏置捷径；提出 bias 指标 BSC/MBG/MBS | **建议引用**：同为类级遗忘评价，但分析的是分类头偏置而非输出分布谱；与 MHPR 视角互补 |

### D. 验证综述（务必引用）

| # | 论文 | 链接 | 内容 |
|---|---|---|---|
| 15 | **Towards Reliable Forgetting: A Survey on Machine Unlearning Verification** (Xue et al., ACM CSUR 2026) | https://arxiv.org/abs/2506.15115 | **首个 MU 验证综述**，分类：行为验证 vs 参数验证。建议在 Related Work 引用并说明本文属于"行为验证"中的输出层谱方法 |
| 16 | Machine Unlearning: A Comprehensive Survey (Wang et al.) | https://arxiv.org/abs/2405.07406 | 通用综述 |
| 17 | Machine Unlearning: Taxonomy, Metrics, Applications... (Li et al., 2024) | https://arxiv.org/abs/2403.08254 | 含 metrics 章节 |

### E. SVD/谱方法用于遗忘**算法**（非评价指标，无冲突）

- **SEMU** (Sendera et al., 2025) https://arxiv.org/abs/2502.07587 —— SVD 分解权重实现遗忘（算法）
- **SAP** (Kodge et al., 2024) https://arxiv.org/abs/2403.08618 —— SVD 激活投影修正性遗忘（算法）
- **Deep Unlearning** (Kodge et al., 2023) https://arxiv.org/abs/2312.00761 —— 无梯度类遗忘（算法）
- **QR-Erase** (Lizzo & Heck, 2026) https://arxiv.org/abs/2608.01422 —— 子空间法遗忘（算法）
- **Orthogonal Subspace Projection via SVD-LoRA** (Rahulamathavan et al., 2026) https://arxiv.org/abs/2604.12526 —— SVD-LoRA 连续遗忘（算法）
- **SAGE** (Zhang et al., 2026) https://arxiv.org/abs/2606.18309 —— 谱激活几何净化遗忘更新向量（算法，后处理）

> 这些均在**算法内部**用 SVD 消除信息，与本文"在输出空间用 SVD 度量遗忘质量"的评价指标定位不同，可放心。

---

## 三、名字查重

| 名称 | 结果 |
|---|---|
| Residual Information Index | 无同名论文 |
| Residual Irreversibility Index | 无同名论文 |
| RII（本领域语境） | 无冲突 |
| MHPR | 无同名论文 |
| "output distributional irreversibility" | 无结果 |

---

## 四、投稿前建议

1. **Related Work 必引**：#15（验证综述，ACM CSUR 2026）、#1 RULER、#3 Auditing、#10 Fragile、#14 类级偏置。这 5 篇最需要明确区分定位。
2. **区分话术**：现有验证方法 = 需要影子模型/重训练/后门/扰动/白盒参数；本文 = **纯黑盒、tuning-free、单次前向、输出层谱证书**。
3. #2（Output Forgetting Is Not True Forgetting）的批评与本论文 Scope（仅输出层、不声称白盒安全）直接相关——建议在 Introduction/Related Work 正面回应。
4. **检索局限**：IEEE Xplore/SpringerLink/ACM DL 期刊库未直接检索（无访问权限），建议投稿前由老师账号在库内用 "unlearning verification spectral / output distribution / forget set" 再跑一遍。
5. 风险提示：领域内"输出分布距离/不可区分"作为验证概念并不新奇（MIA、Sommer 2020 等），本文的新颖点应强调在 **2×C 谱分解的具体形式 + ρ⟺MI 的精确等价 + MHPR 类级修正 + 有限样本界**这一整套理论上。
