import torch
import numpy as np
import random
from typing import Tuple, List

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _build_cluster_capacities(num_items: int, K: int):
    """
    使前remainder个簇容量为base+1，其余为base，保证总和为num_items
    """
    base = num_items // K
    rem = num_items % K
    capacities = [base + (1 if k < rem else 0) for k in range(K)]
    return capacities

def BalancedKMeans(
    V: torch.Tensor,
    K: int,
    max_iter: int = 100,
    seed: int = None,
    verbose: bool = False
) -> Tuple[torch.Tensor, np.ndarray, List[List[int]]]:
    """
    平衡（容量约束）K-Means的贪心近似实现
    返回:
      centroids: (K, D) tensor
      labels: (N,) numpy array，对应每个样本的簇id
      assignments: list of lists，簇 -> 样本索引
    """
    if seed is not None:
        set_seed(seed)

    device = V.device
    N, D = V.shape
    if K > N:
        raise ValueError("K不能大于样本数")

    capacities = _build_cluster_capacities(N, K)  # 每簇最大容量（近似均衡）
    # 随机初始化中心
    initial_indices = random.sample(range(N), K)
    centroids = V[initial_indices].clone()

    # 记录上一轮标签
    prev_labels = None

    for it in range(max_iter):
        # 每轮重新分配
        assignments = [[] for _ in range(K)]
        remaining = set(range(N))

        # 逐簇贪心挑选容量内最近样本
        for k in range(K):
            if capacities[k] == 0 or len(remaining) == 0:
                continue
            # 取剩余向量
            idx_list = list(remaining)
            U = V[idx_list]  # (M, D)
            # 计算与当前质心距离
            dists = torch.norm(U - centroids[k], dim=1)
            # 选最小的top_c
            top_c = min(capacities[k], len(idx_list))
            top_vals, top_pos = torch.topk(dists, k=top_c, largest=False)
            chosen_global = [idx_list[p.item()] for p in top_pos]
            assignments[k].extend(chosen_global)
            for g in chosen_global:
                remaining.discard(g)
            # 更新质心
            if assignments[k]:
                centroids[k] = V[assignments[k]].mean(dim=0)

        # 若仍有剩余样本，分配给当前距离最近且尚未超容量的簇（或放宽容量）
        if remaining:
            if verbose:
                print(f"[Iter {it}] 仍有 {len(remaining)} 个剩余样本，进行补分配")
            rem_idx = list(remaining)
            rem_vecs = V[rem_idx]
            # 全部距离
            dmat = torch.cdist(rem_vecs, centroids) # (R, K)
            for i, g_idx in enumerate(rem_idx):
                # 依距离升序放入
                order = torch.argsort(dmat[i])
                placed = False
                for cand in order:
                    c = cand.item()
                    # 允许超容量1
                    if len(assignments[c]) < capacities[c] + 1:
                        assignments[c].append(g_idx)
                        placed = True
                        break
                if not placed:
                    # 强制放入最近
                    c = order[0].item()
                    assignments[c].append(g_idx)
            # 更新所有质心
            for k in range(K):
                if assignments[k]:
                    centroids[k] = V[assignments[k]].mean(dim=0)

        # 构造标签，对齐原始索引
        labels = np.empty(N, dtype=np.int32)
        for k, idxs in enumerate(assignments):
            for idx in idxs:
                labels[idx] = k

        # 收敛判定
        if prev_labels is not None and np.array_equal(labels, prev_labels):
            if verbose:
                print(f"Balanced K-Means 收敛于第 {it+1} 轮")
            break
        prev_labels = labels.copy()

    return centroids, labels, assignments

def residual_quantize(
    X: torch.Tensor,
    L: int = 3,
    K: int = 8,
    max_iter: int = 100,
    seed: int = None,
    return_residuals: bool = False,
    verbose: bool = False
):
    """
    多层残差量化:
      X: (N, D)
      返回:
        tokens: (N, L) numpy int
        codebooks: list[Tensor] 长度L，每个(K, D)
        residuals(optional): list[Tensor] 每层量化后残差
    """
    if seed is not None:
        set_seed(seed)
    device = X.device
    N, D = X.shape

    residual = X.clone()
    tokens = np.zeros((N, L), dtype=np.int32)
    codebooks = []
    residual_list = []

    for l in range(L):
        centroids, labels, _ = BalancedKMeans(
            residual, K=K, max_iter=max_iter, seed=None if seed is None else seed + l
        )
        codebooks.append(centroids)
        tokens[:, l] = labels
        # 选中对应质心（索引转torch）
        label_tensor = torch.from_numpy(labels).to(device=device, dtype=torch.long)
        selected = centroids[label_tensor] # (N, D)
        residual = residual - selected
        if return_residuals:
            residual_list.append(residual.clone())
        if verbose:
            with torch.no_grad():
                recon_error = torch.mean(torch.sum(residual ** 2, dim=1)).item()
                print(f"[Layer {l+1}] 平均残差能量 {recon_error:.6f}")

    if return_residuals:
        return tokens, codebooks, residual_list
    return tokens, codebooks
