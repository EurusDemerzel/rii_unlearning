#!/usr/bin/env python3
"""
信息论度量：秩一信道模型下的机器遗忘残留信息分析
=====================================================

理论框架（秩一信道模型）：
  好的机器遗忘应使遗忘后模型在“遗忘集”与“保留集”上的输出分布在统计上
  不可区分。这对应于一个秩一信道：P(Y|X=forget) ≈ P(Y|X=retain)，
  即互信息 I(X;Y) ≈ 0。

度量指标：
  1. KL 散度：模型在遗忘集 vs 保留集上的 softmax 输出分布的 KL 散度
  2. 互信息 I(X;Y)：二值指示变量 X (forget/retain) 与模型预测 Y 之间的互信息
  3. 信道秩：2×10 信道矩阵的第二/第一奇异值比 σ₂/σ₁（秩一 ≈ 0）

对比方法：
  - No Unlearning (原始模型，无遗忘)
  - Retrain from Scratch (黄金标准)
  - SISA Unlearning

硬件：MacBook Pro (M5 Pro, 24 GB) — Apple Metal (MPS) 加速。
"""

import time
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")                   # 非交互后端，适合脚本运行
import matplotlib.pyplot as plt

# ============================================================================
# 0. 可复现性 & 设备
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
    DEVICE = torch.device("mps")
    print("✅  MPS 加速已启用。")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  MPS 不可用，回退到 CPU。")
print(f"   设备: {DEVICE}\n")

# ============================================================================
# 1. 超参数
# ============================================================================
BATCH_SIZE    = 64
HIDDEN_SIZE   = 128
LEARNING_RATE = 0.001
EPOCHS        = 10                # 每模型训练轮数

# SISA 参数
S_SISA = 5                        # 分片数
T_SISA = 10                       # 切片/分片
EPOCHS_PER_SLICE_SISA = 1         # 每切片训练轮数 (共 10 epochs)

# 实验参数
FORGET_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20]

# ============================================================================
# 2. 数据加载 — MNIST
# ============================================================================
print("📦 加载 MNIST 数据集...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset_full = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

N = len(train_dataset_full)
all_images = train_dataset_full.data.float() / 255.0
all_images = (all_images - 0.1307) / 0.3081
all_labels = train_dataset_full.targets
print(f"   训练样本: {N:,}  测试样本: {len(test_dataset):,}\n")

# ============================================================================
# 3. 模型定义
# ============================================================================
class SimpleMLP(nn.Module):
    """2 层 MLP: Flatten(784) → Linear(128) → ReLU → Linear(10)"""
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_model():
    return SimpleMLP(input_dim=784, hidden_dim=HIDDEN_SIZE, num_classes=10).to(DEVICE)


def clone_model(model):
    new_model = create_model()
    new_model.load_state_dict(copy.deepcopy(model.state_dict()))
    return new_model

# ============================================================================
# 4. 数据工具函数
# ============================================================================
def indices_to_loader(indices, batch_size=BATCH_SIZE, shuffle=True):
    """将索引列表转为 DataLoader。"""
    idx_list = list(indices) if isinstance(indices, set) else indices
    imgs = all_images[idx_list].unsqueeze(1)
    lbls = all_labels[idx_list]
    ds = TensorDataset(imgs, lbls)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, loader, epochs=EPOCHS, lr=LEARNING_RATE, quiet=True):
    """标准训练循环。"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in (tqdm(range(epochs), desc="   训练", leave=False) if not quiet else range(epochs)):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    return model

# ============================================================================
# 5. 信息论度量函数
# ============================================================================
def extract_outputs(model, loader):
    """
    从模型和数据加载器中提取两类输出：
      - softmax_probs: 每个样本的 softmax 概率分布 (N_samples, 10)
      - hard_preds:    每个样本的硬预测类别 (N_samples,)
    返回: (softmax_probs_np, hard_preds_np)
    """
    model.eval()
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_preds, axis=0)


def extract_sisa_outputs(models, loader):
    """
    SISA 聚合推理：平均所有分片模型的 softmax，再取 argmax。
    返回: (softmax_probs_np, hard_preds_np)
    """
    for m in models:
        m.eval()
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            probs = torch.zeros(images.size(0), 10, device=DEVICE)
            for m in models:
                probs += torch.softmax(m(images), dim=1)
            probs /= len(models)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_preds, axis=0)


def compute_kl_divergence(probs_forget, probs_retain, eps=1e-10):
    """
    计算两个经验 softmax 分布之间的 KL 散度。

    参数:
      probs_forget: (N_f, 10) — 遗忘集上每个样本的 softmax 输出
      probs_retain: (N_r, 10) — 保留集上每个样本的 softmax 输出

    返回:
      kl_forward:  KL(P_forget_avg || P_retain_avg)
      kl_backward: KL(P_retain_avg || P_forget_avg)
      kl_symmetric: (kl_forward + kl_backward) / 2
    """
    # 计算每个集合上的平均 softmax 分布
    p_forget = np.mean(probs_forget, axis=0) + eps
    p_retain = np.mean(probs_retain, axis=0) + eps

    # 归一化（防止数值误差）
    p_forget /= p_forget.sum()
    p_retain /= p_retain.sum()

    kl_forward  = np.sum(p_forget * np.log(p_forget / p_retain))
    kl_backward = np.sum(p_retain * np.log(p_retain / p_forget))
    kl_symmetric = (kl_forward + kl_backward) / 2.0

    return kl_forward, kl_backward, kl_symmetric


def compute_mutual_information(preds_forget, preds_retain, n_forget, n_retain,
                                num_classes=10, eps=1e-10):
    """
    计算二值变量 X (forget/retain) 与离散变量 Y (预测类别) 之间的互信息。

    I(X; Y) = H(Y) - H(Y|X)

    参数:
      preds_forget: (N_f,) — 遗忘集上的硬预测
      preds_retain: (N_r,) — 保留集上的硬预测
      n_forget, n_retain: 各自样本数
      num_classes: 类别数 (MNIST = 10)

    返回:
      mi: I(X; Y) 的估计值 (nats)
    """
    # P(Y|X=forget) — 遗忘集上各类别的经验频率
    p_y_given_forget = np.bincount(preds_forget, minlength=num_classes).astype(float)
    p_y_given_forget /= p_y_given_forget.sum()

    # P(Y|X=retain) — 保留集上各类别的经验频率
    p_y_given_retain = np.bincount(preds_retain, minlength=num_classes).astype(float)
    p_y_given_retain /= p_y_given_retain.sum()

    # P(X)
    total = n_forget + n_retain
    p_x_forget = n_forget / total
    p_x_retain = n_retain / total

    # P(Y) = P(X=forget)*P(Y|X=forget) + P(X=retain)*P(Y|X=retain)
    p_y = p_x_forget * p_y_given_forget + p_x_retain * p_y_given_retain

    # H(Y) = -Σ P(y) * log P(y)
    h_y = -np.sum(p_y * np.log(p_y + eps))

    # H(Y|X=forget)
    h_y_given_forget = -np.sum(p_y_given_forget * np.log(p_y_given_forget + eps))
    # H(Y|X=retain)
    h_y_given_retain = -np.sum(p_y_given_retain * np.log(p_y_given_retain + eps))

    # H(Y|X) = P(X=forget)*H(Y|X=forget) + P(X=retain)*H(Y|X=retain)
    h_y_given_x = p_x_forget * h_y_given_forget + p_x_retain * h_y_given_retain

    mi = h_y - h_y_given_x
    return max(mi, 0.0)  # 互信息非负（数值防护）


def compute_channel_rank(probs_forget, probs_retain):
    """
    计算 2×10 信道矩阵 P(Y|X) 的有效秩度量。

    信道矩阵行：
      Row 0: P(Y|X=retain) — 保留集上的平均 softmax
      Row 1: P(Y|X=forget) — 遗忘集上的平均 softmax

    完美遗忘 → 两行相同 → 秩 1 → σ₂/σ₁ ≈ 0

    返回:
      sigma_ratio: σ₂ / σ₁（越小越接近秩一）
      singular_values: [σ₁, σ₂]
    """
    p_retain = np.mean(probs_retain, axis=0)
    p_forget = np.mean(probs_forget, axis=0)

    channel = np.stack([p_retain, p_forget], axis=0)  # (2, 10)
    _, s, _ = np.linalg.svd(channel, full_matrices=False)
    sigma_ratio = s[1] / s[0] if s[0] > 1e-12 else 0.0
    return sigma_ratio, s

# ============================================================================
# 6. SISA 工具函数
# ============================================================================
def setup_sisa(forget_global_indices):
    """
    为给定的遗忘集设置 SISA 框架：
      - 将全部训练数据分成 S 个分片，每个分片 T 个切片
      - 增量训练每个分片（带检查点）
      - 执行 SISA 遗忘操作
    返回: (sisa_models, sisa_train_time, sisa_unlearn_time)
    """
    # 6a. 分片 & 切片
    indices = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))
    shard_size = N // S_SISA
    shard_indices = []
    for s in range(S_SISA):
        start = s * shard_size
        end   = start + shard_size if s < S_SISA - 1 else N
        shard_indices.append(indices[start:end])

    slice_indices = []
    for s in range(S_SISA):
        s_idx = shard_indices[s]
        s_len = len(s_idx)
        sl_size = s_len // T_SISA
        slices = []
        for t in range(T_SISA):
            start = t * sl_size
            end   = start + sl_size if t < T_SISA - 1 else s_len
            slices.append(s_idx[start:end])
        slice_indices.append(slices)

    # 6b. 找每个分片的最早受影响切片
    forget_set = set(forget_global_indices)
    shard_has_forget = [False] * S_SISA
    shard_first_affected = [T_SISA] * S_SISA

    for s in range(S_SISA):
        for t in range(T_SISA):
            if set(slice_indices[s][t].tolist()) & forget_set:
                shard_has_forget[s] = True
                if t < shard_first_affected[s]:
                    shard_first_affected[s] = t

    # 6c. 增量训练（带检查点）
    t_sisa_train_start = time.time()
    all_checkpoints = []
    sisa_models = []

    for s in range(S_SISA):
        checkpoints = []
        model = create_model()
        cumulative_indices = []
        for t in range(T_SISA):
            cumulative_indices.extend(slice_indices[s][t].tolist())
            loader = indices_to_loader(cumulative_indices, shuffle=True)
            model = train_model(model, loader, epochs=EPOCHS_PER_SLICE_SISA, quiet=True)
            checkpoints.append(clone_model(model))
        all_checkpoints.append(checkpoints)
        sisa_models.append(model)

    t_sisa_train = time.time() - t_sisa_train_start

    # 6d. SISA 遗忘
    t_sisa_unlearn_start = time.time()
    sisa_unlearned = []

    for s in range(S_SISA):
        if not shard_has_forget[s]:
            sisa_unlearned.append(clone_model(all_checkpoints[s][-1]))
            continue

        k = shard_first_affected[s]
        if k == 0:
            # 从头重训
            clean = []
            for t in range(T_SISA):
                sl = slice_indices[s][t].tolist()
                clean.extend([i for i in sl if i not in forget_set])
            model = create_model()
            loader = indices_to_loader(clean, shuffle=True)
            model = train_model(model, loader, epochs=EPOCHS_PER_SLICE_SISA * T_SISA, quiet=True)
            sisa_unlearned.append(model)
        else:
            # 回退到检查点 k-1
            model = clone_model(all_checkpoints[s][k - 1])
            clean = []
            for t in range(k, T_SISA):
                sl = slice_indices[s][t].tolist()
                clean.extend([i for i in sl if i not in forget_set])
            remaining = EPOCHS_PER_SLICE_SISA * (T_SISA - k)
            loader = indices_to_loader(clean, shuffle=True)
            model = train_model(model, loader, epochs=remaining, quiet=True)
            sisa_unlearned.append(model)

    t_sisa_unlearn = time.time() - t_sisa_unlearn_start

    return sisa_unlearned, t_sisa_train, t_sisa_unlearn

# ============================================================================
# 7. 实验循环
# ============================================================================
def run_experiments():
    """
    对不同遗忘比例和方法进行实验，收集信息论度量。
    返回 DataFrame 和用于绘图的结构化数据。
    """
    results = []
    all_forget_indices = {}   # forget_ratio → forget_global_indices (list)

    # 预先为每个遗忘比例生成遗忘集（保证可复现）
    for ratio in FORGET_RATIOS:
        n_forget = int(N * ratio)
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(SEED + int(ratio * 100)))
        all_forget_indices[ratio] = perm[:n_forget].tolist()

    # 训练一个"原始模型"（全量数据，无遗忘），供 "No Unlearning" 复用
    print("🔧 训练原始模型（全量数据，供 No Unlearning 复用）...")
    full_loader = indices_to_loader(list(range(N)), shuffle=True)
    model_original = create_model()
    model_original = train_model(model_original, full_loader, epochs=EPOCHS, quiet=False)
    print("   ✅ 原始模型训练完成。\n")

    # ------------------------------------------------------------------
    for ratio in FORGET_RATIOS:
        print(f"{'='*60}")
        print(f"  遗忘比例: {ratio*100:.0f}%")
        print(f"{'='*60}")

        forget_list = all_forget_indices[ratio]
        retain_list = list(set(range(N)) - set(forget_list))
        n_forget = len(forget_list)
        n_retain = len(retain_list)

        # 为当前遗忘比例构建 DataLoader
        forget_loader = indices_to_loader(forget_list, shuffle=False)
        retain_loader = indices_to_loader(retain_list, shuffle=False)

        # ----------------------------------------------------------------
        # 方法 A: No Unlearning（原始模型）
        # ----------------------------------------------------------------
        probs_f, preds_f = extract_outputs(model_original, forget_loader)
        probs_r, preds_r = extract_outputs(model_original, retain_loader)

        kl_fwd, kl_bwd, kl_sym = compute_kl_divergence(probs_f, probs_r)
        mi = compute_mutual_information(preds_f, preds_r, n_forget, n_retain)
        sigma_ratio, sv = compute_channel_rank(probs_f, probs_r)

        results.append({
            "forget_ratio": ratio, "method": "No Unlearning",
            "kl_symmetric": kl_sym, "kl_forward": kl_fwd, "kl_backward": kl_bwd,
            "mutual_information": mi, "sigma_ratio": sigma_ratio,
            "sigma_1": sv[0], "sigma_2": sv[1],
            "unlearn_time": 0.0,
            "n_forget": n_forget, "n_retain": n_retain,
        })
        print(f"   [No Unlearning]  KL_sym={kl_sym:.6f}  MI={mi:.6f}  σ₂/σ₁={sigma_ratio:.6f}")

        # ----------------------------------------------------------------
        # 方法 B: Retrain from Scratch（黄金标准）
        # ----------------------------------------------------------------
        t0 = time.time()
        model_retrained = create_model()
        retain_loader_shuf = indices_to_loader(retain_list, shuffle=True)
        model_retrained = train_model(model_retrained, retain_loader_shuf,
                                       epochs=EPOCHS, quiet=True)
        t_retrain = time.time() - t0

        probs_f2, preds_f2 = extract_outputs(model_retrained, forget_loader)
        probs_r2, preds_r2 = extract_outputs(model_retrained, retain_loader)

        kl_fwd2, kl_bwd2, kl_sym2 = compute_kl_divergence(probs_f2, probs_r2)
        mi2 = compute_mutual_information(preds_f2, preds_r2, n_forget, n_retain)
        sigma_ratio2, sv2 = compute_channel_rank(probs_f2, probs_r2)

        results.append({
            "forget_ratio": ratio, "method": "Retrain (Gold)",
            "kl_symmetric": kl_sym2, "kl_forward": kl_fwd2, "kl_backward": kl_bwd2,
            "mutual_information": mi2, "sigma_ratio": sigma_ratio2,
            "sigma_1": sv2[0], "sigma_2": sv2[1],
            "unlearn_time": t_retrain,
            "n_forget": n_forget, "n_retain": n_retain,
        })
        print(f"   [Retrain Gold]   KL_sym={kl_sym2:.6f}  MI={mi2:.6f}  σ₂/σ₁={sigma_ratio2:.6f}  time={t_retrain:.1f}s")

        # ----------------------------------------------------------------
        # 方法 C: SISA Unlearning
        # ----------------------------------------------------------------
        sisa_models, t_sisa_train, t_sisa_unlearn = setup_sisa(forget_list)

        probs_f3, preds_f3 = extract_sisa_outputs(sisa_models, forget_loader)
        probs_r3, preds_r3 = extract_sisa_outputs(sisa_models, retain_loader)

        kl_fwd3, kl_bwd3, kl_sym3 = compute_kl_divergence(probs_f3, probs_r3)
        mi3 = compute_mutual_information(preds_f3, preds_r3, n_forget, n_retain)
        sigma_ratio3, sv3 = compute_channel_rank(probs_f3, probs_r3)

        results.append({
            "forget_ratio": ratio, "method": "SISA",
            "kl_symmetric": kl_sym3, "kl_forward": kl_fwd3, "kl_backward": kl_bwd3,
            "mutual_information": mi3, "sigma_ratio": sigma_ratio3,
            "sigma_1": sv3[0], "sigma_2": sv3[1],
            "unlearn_time": t_sisa_unlearn,
            "n_forget": n_forget, "n_retain": n_retain,
        })
        print(f"   [SISA]           KL_sym={kl_sym3:.6f}  MI={mi3:.6f}  σ₂/σ₁={sigma_ratio3:.6f}  time={t_sisa_unlearn:.1f}s")
        print()

    return pd.DataFrame(results)

# ============================================================================
# 8. 可视化
# ============================================================================
def plot_results(df: pd.DataFrame):
    """绘制三张对比曲线图：互信息、KL 散度、信道秩度量。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = {"No Unlearning": "#E74C3C", "Retrain (Gold)": "#27AE60", "SISA": "#3498DB"}
    markers = {"No Unlearning": "s", "Retrain (Gold)": "o", "SISA": "^"}

    # --- 子图 1: 互信息 I(X;Y) ---
    ax = axes[0]
    for method in ["No Unlearning", "Retrain (Gold)", "SISA"]:
        sub = df[df["method"] == method]
        ax.plot(sub["forget_ratio"] * 100, sub["mutual_information"],
                color=colors[method], marker=markers[method], linewidth=2,
                markersize=8, label=method)
    ax.set_xlabel("Forget Ratio (%)", fontsize=12)
    ax.set_ylabel("Mutual Information I(X;Y) (nats)", fontsize=12)
    ax.set_title("Mutual Information vs Forget Ratio", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(FORGET_RATIOS) * 100 + 2)

    # --- 子图 2: 对称 KL 散度 ---
    ax = axes[1]
    for method in ["No Unlearning", "Retrain (Gold)", "SISA"]:
        sub = df[df["method"] == method]
        ax.plot(sub["forget_ratio"] * 100, sub["kl_symmetric"],
                color=colors[method], marker=markers[method], linewidth=2,
                markersize=8, label=method)
    ax.set_xlabel("Forget Ratio (%)", fontsize=12)
    ax.set_ylabel("Symmetric KL Divergence", fontsize=12)
    ax.set_title("KL(P_forget || P_retain) Symmetric", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(FORGET_RATIOS) * 100 + 2)

    # --- 子图 3: 信道秩度量 σ₂/σ₁ ---
    ax = axes[2]
    for method in ["No Unlearning", "Retrain (Gold)", "SISA"]:
        sub = df[df["method"] == method]
        ax.plot(sub["forget_ratio"] * 100, sub["sigma_ratio"],
                color=colors[method], marker=markers[method], linewidth=2,
                markersize=8, label=method)
    ax.set_xlabel("Forget Ratio (%)", fontsize=12)
    ax.set_ylabel("Singular Value Ratio sigma_2/sigma_1", fontsize=12)
    ax.set_title("Channel Rank Metric (sigma_2/sigma_1 -> 0 = Rank-1)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(FORGET_RATIOS) * 100 + 2)

    plt.tight_layout()
    out_path = "information_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"📈 图表已保存至: {out_path}")
    plt.close()

# ============================================================================
# 9. 主入口
# ============================================================================
def main():
    print("=" * 64)
    print("   信息论度量：秩一信道模型 — 机器遗忘残留信息分析")
    print("=" * 64)
    print()

    df = run_experiments()

    print("\n" + "=" * 64)
    print("                     📋  实 验 结 果 汇 总")
    print("=" * 64)
    print()
    print(df[["forget_ratio", "method", "kl_symmetric", "mutual_information",
              "sigma_ratio", "unlearn_time"]].to_string(index=False))
    print()

    plot_results(df)

    # 额外分析：秩一信道的理论解释
    print()
    print("─" * 64)
    print("  理论解释（秩一信道模型）：")
    print("  · 当 σ₂/σ₁ → 0 时，信道 P(Y|X) 近似秩一")
    print("  · 秩一信道意味着 P(Y|X=forget) ≈ P(Y|X=retain)")
    print("  · 此时 I(X;Y) ≈ 0：遗忘集与保留集在统计上不可区分")
    print("  · 有效的机器遗忘应使 σ₂/σ₁ 和 I(X;Y) 尽可能小")
    print("─" * 64)

    # 保存 CSV
    df.to_csv("information_metrics.csv", index=False)
    print("\n📁 详细数据已保存至: information_metrics.csv")


if __name__ == "__main__":
    main()
