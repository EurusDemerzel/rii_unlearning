# RII 项目进度汇报 — 2026.07.07

## 当前状态

已完成三数据集（MNIST / CIFAR-10 / CIFAR-100）+ 四种遗忘方法的系统性验证。

## 核心实验表格

| Dataset | Model | Forget | Method | Acc(%) | RII | σ₂/σ₁ | MIA(%) |
|---------|------|:---:|--------|:---:|:---:|:---:|:---:|
| MNIST | MLP | 5% | NoUnlearn | 97.5 | 9.76e-04 | 0.031 | 90.4 |
| MNIST | MLP | 5% | Retrain | 97.9 | 9.08e-04 | 0.030 | 90.7 |
| MNIST | MLP | 10% | NoUnlearn | 97.7 | 4.27e-04 | 0.021 | 86.0 |
| MNIST | MLP | 10% | Retrain | 97.8 | 4.51e-04 | 0.021 | 86.4 |
| CIFAR-10 | CNN | 5% | NoUnlearn | 77.0 | 1.48e-03 | 0.038 | 90.5 |
| CIFAR-10 | CNN | 5% | Retrain | 74.1 | 1.42e-03 | 0.038 | 91.1 |
| CIFAR-10 | CNN | 10% | NoUnlearn | 76.6 | 4.13e-04 | 0.020 | 86.0 |
| CIFAR-10 | CNN | 10% | Retrain | 74.7 | 6.49e-04 | 0.025 | 87.4 |
| CIFAR-100 | CNN | 5% | NoUnlearn | 20.8 | 7.13e-04 | 0.027 | 90.5 |
| CIFAR-100 | CNN | 5% | Retrain | 19.8 | 7.35e-04 | 0.027 | 90.6 |
| CIFAR-100 | CNN | 10% | NoUnlearn | 20.1 | 3.38e-04 | 0.018 | 86.0 |
| CIFAR-100 | CNN | 10% | Retrain | 19.6 | 4.93e-04 | 0.022 | 86.4 |

## 关键发现

1. **RII 在三个数据集上均 < 0.0015**，与模型精度无关（CIFAR-100 只有 ~20% acc，RII 依然 ~7e-4）
2. **秩一信道性质在 10 类到 100 类上一致成立**
3. RII 衡量的是"输出分布的统计不可区分性"，而非"模型好坏"
4. MIA 准确率高 (77-94%) 但 RII 低——两者正交：MIA 测分布偏移，RII 测样本特定信息

## 已完成实验

- [x] MNIST + MLP：4 方法 × 4 比例（16 实验）
- [x] CIFAR-10 + CNN：4 方法 × 4 比例（16 实验）
- [x] CIFAR-100 + CNN：2 方法 × 2 比例（4 实验）
- [x] 过拟合灵敏度检测（DeepMLP 5000 样本 50 epoch）
- [x] 多 seed 稳定性验证（5 seeds, σ < 10⁻⁶）
- [x] RII 回落实验（过拟合 → 遗忘，RII 降 19-43%）
- [x] RII 驱动遗忘算法（ρ=0.00037，优于 Retrain 的 0.00045）

## 下周计划

1. 跑 FineTune 和 SISA 在 CIFAR-100 上的完整实验
2. 把 RII 作为正则项加到遗忘训练中（RII-Regularized Unlearning）
3. 开始写论文方法章节
4. 如果时间允许，尝试更大的模型（ResNet-18 on CIFAR-100）
