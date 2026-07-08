#!/usr/bin/env python3
"""
SISA (Sharded, Isolated, Sliced, Aggregated) 机器遗忘框架
============================================================

基于 Bourtoule et al., "Machine Unlearning," IEEE S&P 2021.

SISA 将训练数据分成 S 个分片（Shard），每个分片独立训练一个模型；
每个分片内部再切成 T 个切片（Slice），训练过程中保存检查点。
遗忘时，只需回退到受影响的切片检查点并重新训练，无需重训全部模型。

与基线（完整重新训练）对比指标：
  - 遗忘后的测试准确率
  - 遗忘所需时间
  - 遗忘集 vs 保留集上的表现

硬件：MacBook Pro (M5 Pro, 24 GB) — Apple Metal (MPS) 加速。
"""

import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

# ============================================================================
# 0. 可复现性 & 设备
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅  使用 Apple Metal (MPS) 加速。")
elif torch.backends.mps.is_built():
    DEVICE = torch.device("cpu")
    print("⚠️  MPS 已编译但不可用 — 回退到 CPU。")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  MPS 不可用 — 回退到 CPU。")
print(f"   设备: {DEVICE}\n")

# ============================================================================
# 1. 超参数
# ============================================================================
BATCH_SIZE        = 64
HIDDEN_SIZE       = 128
LEARNING_RATE     = 0.001

S = 5                # 分片数（Shards）
T = 10               # 每个分片的切片数（Slices）
EPOCHS_PER_SLICE = 1 # 每个切片训练轮数 → 每分片共 10 epochs（与基线一致）

FORGET_RATIO      = 0.10  # 遗忘比例：10%
FORGET_SLICE_START = T - 3  # 遗忘数据仅从最后 3 个切片中选取（展示检查点优势）

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

# 提取全部训练数据为 tensor（便于手动分片）
N = len(train_dataset_full)
all_images = train_dataset_full.data.float() / 255.0           # (N, 28, 28)
all_images = (all_images - 0.1307) / 0.3081                     # 归一化
all_labels = train_dataset_full.targets                          # (N,)

print(f"   训练样本: {N:,}  测试样本: {len(test_dataset):,}\n")

# ============================================================================
# 3. 数据分片 & 切片
# ============================================================================
# 3a. 随机排列所有样本
indices = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))

# 3b. 分成 S 个分片（大小尽量均匀）
shard_size = N // S
shard_indices = []                # 每个分片的样本索引（全局索引）
for s in range(S):
    start = s * shard_size
    end   = start + shard_size if s < S - 1 else N
    shard_indices.append(indices[start:end])

# 3c. 每个分片内部分成 T 个切片
slice_indices = []                # slice_indices[shard][slice] = 该切片中的全局索引
for s in range(S):
    s_idx = shard_indices[s]
    s_len = len(s_idx)
    sl_size = s_len // T
    slices = []
    for t in range(T):
        start = t * sl_size
        end   = start + sl_size if t < T - 1 else s_len
        slices.append(s_idx[start:end])
    slice_indices.append(slices)

print(f"✂️  数据分片: S={S}, 每分片 {shard_size:,} 样本, T={T} 切片/分片\n")

# ============================================================================
# 4. 遗忘集选择（仅从每个分片的后 FORGET_SLICE_START 个切片中选取）
# ============================================================================
# 先收集所有"候选遗忘切片"中的样本（每个分片的后几个切片）
candidate_pool = []
for s in range(S):
    for t in range(FORGET_SLICE_START, T):
        candidate_pool.extend(slice_indices[s][t].tolist())

# 从候选池中随机选择 10% 总样本作为遗忘集
candidate_pool = torch.tensor(candidate_pool)
perm = torch.randperm(len(candidate_pool), generator=torch.Generator().manual_seed(SEED + 1))
forget_count = int(N * FORGET_RATIO)
forget_global_indices = set(candidate_pool[perm[:forget_count]].tolist())

# 标记每个 (分片, 切片) 是否包含遗忘数据
shard_has_forget = [False] * S
shard_first_affected_slice = [T] * S
shard_forget_indices = [set() for _ in range(S)]

for s in range(S):
    for t in range(T):
        slice_set = set(slice_indices[s][t].tolist())
        forget_in_slice = slice_set & forget_global_indices
        if forget_in_slice:
            shard_has_forget[s] = True
            if t < shard_first_affected_slice[s]:
                shard_first_affected_slice[s] = t
            shard_forget_indices[s] |= forget_in_slice

affected_shards = sum(shard_has_forget)
print(f"🔍 遗忘集: {forget_count:,} 样本 ({FORGET_RATIO*100:.0f}%)")
print(f"   受影响分片: {affected_shards}/{S}")
for s in range(S):
    if shard_has_forget[s]:
        t0 = shard_first_affected_slice[s]
        nf  = len(shard_forget_indices[s])
        print(f"   分片 {s}: {nf} 个遗忘点, 最早受影响切片 = {t0}")
print()

# ============================================================================
# 5. 模型定义（与基线相同）
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
    """工厂函数：创建置于 DEVICE 上的全新模型。"""
    return SimpleMLP(input_dim=784, hidden_dim=HIDDEN_SIZE, num_classes=10).to(DEVICE)


def clone_model(model):
    """深拷贝模型（state_dict 级别）。"""
    new_model = create_model()
    new_model.load_state_dict(copy.deepcopy(model.state_dict()))
    return new_model

# ============================================================================
# 6. 工具函数
# ============================================================================
def indices_to_loader(global_indices, batch_size=BATCH_SIZE, shuffle=True):
    """将全局索引列表转为 DataLoader。"""
    imgs = all_images[global_indices].unsqueeze(1)   # (k, 1, 28, 28)
    lbls = all_labels[global_indices]
    ds = TensorDataset(imgs, lbls)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epochs(model, loader, epochs, lr=LEARNING_RATE, desc="", quiet=False):
    """在给定 DataLoader 上训练 epochs 轮。返回平均损失。"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    pbar = tqdm(range(epochs), desc=desc, leave=False) if not quiet else range(epochs)
    for _ in pbar:
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg = running_loss / len(loader)
        if not quiet and isinstance(pbar, tqdm):
            pbar.set_postfix(loss=f"{avg:.4f}")
    return model


def evaluate_model(model, loader):
    """返回分类准确率（0-1 之间的浮点数）。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


def sisa_aggregated_predict(models, loader):
    """
    SISA 聚合推理：对 S 个分片模型的 softmax 概率取平均，再取 argmax。
    返回准确率。
    """
    for m in models:
        m.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # 聚合所有分片模型的概率
            probs = torch.zeros(images.size(0), 10, device=DEVICE)
            for m in models:
                probs += torch.softmax(m(images), dim=1)
            probs /= len(models)
            _, preds = torch.max(probs, 1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total

# ============================================================================
# 7. SISA 增量训练（带检查点）
# ============================================================================
def train_shard_incremental(shard_idx):
    """
    对分片 shard_idx 进行增量训练：
      - 从随机初始化开始
      - 逐切片训练，每完成一个切片保存检查点
      - 返回 (checkpoints[T], final_model)
        其中 checkpoints[t] 是训练完切片 0..t 后的模型
    """
    checkpoints = []               # checkpoints[t]: 训练完切片 0..t
    model = create_model()

    cumulative_indices = []
    for t in range(T):
        # 累积到目前为止的所有切片数据
        cumulative_indices.extend(slice_indices[shard_idx][t].tolist())
        loader = indices_to_loader(cumulative_indices, shuffle=True)

        desc = f"   分片 {shard_idx} 切片 {t}"
        model = train_epochs(model, loader, epochs=EPOCHS_PER_SLICE,
                             desc=desc, quiet=True)
        checkpoints.append(clone_model(model))

    return checkpoints, model

# ============================================================================
# 8. SISA 遗忘操作
# ============================================================================
def sisa_unlearn_shard(shard_idx, checkpoints):
    """
    对分片 shard_idx 执行 SISA 遗忘：
      - 找到最早受影响的切片 k
      - 如果 k=0：从头重训（无干净检查点可回退）
      - 如果 k>0：回退到 checkpoints[k-1]，用 k..T-1 的干净数据继续训练
      - 如果该分片无遗忘数据：直接返回原模型
    返回 (unlearned_model, retrained: bool)
    """
    if not shard_has_forget[shard_idx]:
        # 无遗忘数据，无需重训
        return clone_model(checkpoints[-1]), False

    k = shard_first_affected_slice[shard_idx]

    if k == 0:
        # 切片 0 就有遗忘数据 → 必须从头重训整个分片
        clean_indices = []
        for t in range(T):
            sl = slice_indices[shard_idx][t].tolist()
            clean_indices.extend([i for i in sl if i not in forget_global_indices])
        model = create_model()
        loader = indices_to_loader(clean_indices, shuffle=True)
        model = train_epochs(model, loader, epochs=EPOCHS_PER_SLICE * T,
                             desc=f"   ⟳ 分片 {shard_idx} (从头重训)", quiet=True)
        return model, True

    else:
        # 回退到检查点 k-1（切片 0..k-1 是干净的）
        model = clone_model(checkpoints[k - 1])

        # 收集切片 k..T-1 中的干净数据
        clean_indices = []
        for t in range(k, T):
            sl = slice_indices[shard_idx][t].tolist()
            clean_indices.extend([i for i in sl if i not in forget_global_indices])

        # 继续训练
        remaining_epochs = EPOCHS_PER_SLICE * (T - k)
        loader = indices_to_loader(clean_indices, shuffle=True)
        model = train_epochs(model, loader, epochs=remaining_epochs,
                             desc=f"   ⟳ 分片 {shard_idx} (回退切片 {k-1}→)", quiet=True)
        return model, True

# ============================================================================
# 9. 主实验
# ============================================================================
def main():
    print("=" * 64)
    print("    SISA 机器遗忘框架 — Bourtoule et al. (IEEE S&P 2021)")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # 9a. SISA 增量训练（所有分片）
    # ------------------------------------------------------------------
    print("🚀 阶段 1: SISA 增量训练（所有分片）")
    print("-" * 40)
    t_train_start = time.time()

    all_checkpoints = []    # all_checkpoints[s][t] = 分片 s 切片 t 后的检查点
    sisa_models = []        # 最终模型（训练完所有切片）

    for s in range(S):
        ckpts, final_model = train_shard_incremental(s)
        all_checkpoints.append(ckpts)
        sisa_models.append(final_model)

    t_train_total = time.time() - t_train_start
    print(f"   ✅ SISA 训练完成，总耗时: {t_train_total:.1f}s\n")

    # ------------------------------------------------------------------
    # 9b. 遗忘前评估
    # ------------------------------------------------------------------
    print("📊 阶段 2: 遗忘前评估")
    print("-" * 40)
    acc_pre_test   = sisa_aggregated_predict(sisa_models, test_loader)
    print(f"   SISA 聚合模型 — 测试准确率: {acc_pre_test*100:.2f}%")
    print()

    # ------------------------------------------------------------------
    # 9c. SISA 遗忘操作
    # ------------------------------------------------------------------
    print("🗑️  阶段 3: SISA 遗忘操作")
    print("-" * 40)
    t_unlearn_start = time.time()

    sisa_unlearned_models = []
    shards_retrained = 0

    for s in range(S):
        model, retrained = sisa_unlearn_shard(s, all_checkpoints[s])
        sisa_unlearned_models.append(model)
        if retrained:
            shards_retrained += 1

    t_unlearn_total = time.time() - t_unlearn_start
    print(f"   ✅ SISA 遗忘完成，重训了 {shards_retrained}/{S} 个分片")
    print(f"   遗忘耗时: {t_unlearn_total:.1f}s\n")

    # ------------------------------------------------------------------
    # 9d. 遗忘后评估
    # ------------------------------------------------------------------
    print("📊 阶段 4: 遗忘后评估")
    print("-" * 40)

    # 测试集
    acc_sisa_test = sisa_aggregated_predict(sisa_unlearned_models, test_loader)

    # 遗忘集
    forget_list = sorted(forget_global_indices)
    forget_loader = indices_to_loader(forget_list, shuffle=False)
    acc_sisa_forget = sisa_aggregated_predict(sisa_unlearned_models, forget_loader)

    # 保留集
    retain_list = sorted(set(range(N)) - forget_global_indices)
    retain_loader = indices_to_loader(retain_list, shuffle=False)
    acc_sisa_retain = sisa_aggregated_predict(sisa_unlearned_models, retain_loader)

    print(f"   SISA 遗忘后 — 测试准确率 : {acc_sisa_test*100:.2f}%")
    print(f"   SISA 遗忘后 — 遗忘集准确率: {acc_sisa_forget*100:.2f}%")
    print(f"   SISA 遗忘后 — 保留集准确率: {acc_sisa_retain*100:.2f}%")
    print()

    # ------------------------------------------------------------------
    # 9e. 黄金标准：完整重新训练（仅用保留集）
    # ------------------------------------------------------------------
    print("🥇 阶段 5: 黄金标准 — 完整重新训练（仅保留集）")
    print("-" * 40)
    t_gold_start = time.time()

    gold_model = create_model()
    retain_loader_shuffled = indices_to_loader(retain_list, shuffle=True)
    gold_model = train_epochs(gold_model, retain_loader_shuffled,
                               epochs=EPOCHS_PER_SLICE * T,
                               desc="   黄金标准重训", quiet=True)

    t_gold_total = time.time() - t_gold_start
    acc_gold_test   = evaluate_model(gold_model, test_loader)
    acc_gold_forget = evaluate_model(gold_model, forget_loader)
    acc_gold_retain = evaluate_model(gold_model, retain_loader)

    print(f"   ✅ 黄金标准重训完成，耗时: {t_gold_total:.1f}s")
    print(f"   黄金标准 — 测试准确率 : {acc_gold_test*100:.2f}%")
    print(f"   黄金标准 — 遗忘集准确率: {acc_gold_forget*100:.2f}%")
    print(f"   黄金标准 — 保留集准确率: {acc_gold_retain*100:.2f}%")
    print()

    # ------------------------------------------------------------------
    # 10. 综合对比报告
    # ------------------------------------------------------------------
    print("=" * 64)
    print("                     📋  综 合 对 比 报 告")
    print("=" * 64)
    print()
    print(f"  {'指标':<24} {'SISA 遗忘':>14} {'黄金标准（重训）':>16} {'差异':>10}")
    print(f"  {'─'*24} {'─'*14} {'─'*16} {'─'*10}")
    print(f"  {'遗忘前测试准确率':<24} {acc_pre_test*100:>13.2f}% {'—':>16} {'—':>10}")
    print(f"  {'遗忘后测试准确率':<24} {acc_sisa_test*100:>13.2f}% {acc_gold_test*100:>15.2f}% {(acc_sisa_test-acc_gold_test)*100:>+9.2f}%")
    print(f"  {'遗忘集准确率':<24} {acc_sisa_forget*100:>13.2f}% {acc_gold_forget*100:>15.2f}% {(acc_sisa_forget-acc_gold_forget)*100:>+9.2f}%")
    print(f"  {'保留集准确率':<24} {acc_sisa_retain*100:>13.2f}% {acc_gold_retain*100:>15.2f}% {(acc_sisa_retain-acc_gold_retain)*100:>+9.2f}%")
    print(f"  {'─'*24} {'─'*14} {'─'*16} {'─'*10}")
    print(f"  {'遗忘耗时':<24} {t_unlearn_total:>13.1f}s {t_gold_total:>15.1f}s {t_unlearn_total-t_gold_total:>+9.1f}s")
    print(f"  {'SISA 训练总耗时':<24} {t_train_total:>13.1f}s {'—':>16} {'—':>10}")
    print(f"  {'重训分片数':<24} {shards_retrained:>13}/{S}   {'全部 (等效)':>16} {'—':>10}")

    speedup = t_gold_total / max(t_unlearn_total, 0.001)
    print(f"  {'遗忘加速比':<24} {speedup:>13.1f}× {'—':>16} {'—':>10}")
    print()
    print("=" * 64)


if __name__ == "__main__":
    main()
