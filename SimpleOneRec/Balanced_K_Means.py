import torch
import numpy as np
import random
from typing import Tuple, List, Optional

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _build_cluster_capacities(num_items: int, K: int):
    base = num_items // K
    rem = num_items % K
    return [base + (1 if k < rem else 0) for k in range(K)]

@torch.no_grad()
def BalancedKMeans(
    V: torch.Tensor,
    K: int,
    max_iter: int = 100,
    seed: Optional[int] = None,
    verbose: bool = False,
    use_squared_dist: bool = True
) -> Tuple[torch.Tensor, np.ndarray, List[List[int]]]:
    """
    平衡（容量约束）K-Means 贪心近似
    V: (N,D) 不参与梯度
    返回:
      centroids: (K,D) float32
      labels: (N,) np.int32
      assignments: list[list[int]]
    """
    if seed is not None:
        set_seed(seed)

    if V.requires_grad:
        V = V.detach()

    device = V.device
    N, D = V.shape
    if K > N:
        raise ValueError("K不能大于样本数")

    capacities = _build_cluster_capacities(N, K)
    initial_indices = random.sample(range(N), K)
    centroids = V[initial_indices].clone().float()  # (K,D)

    prev_labels = None

    for it in range(max_iter):
        assignments = [[] for _ in range(K)]
        remaining = set(range(N))

        for k in range(K):
            if capacities[k] == 0 or not remaining:
                continue
            idx_list = list(remaining)
            U = V[idx_list]  # (M,D)
            if use_squared_dist:
                # (M,)
                dists = torch.sum((U - centroids[k]) ** 2, dim=1)
            else:
                dists = torch.norm(U - centroids[k], dim=1)
            top_c = min(capacities[k], len(idx_list))
            _, top_pos = torch.topk(dists, k=top_c, largest=False)
            chosen_global = [idx_list[p.item()] for p in top_pos]
            assignments[k].extend(chosen_global)
            for g in chosen_global:
                remaining.discard(g)
            if assignments[k]:
                centroids[k] = V[assignments[k]].mean(dim=0)

        if remaining:
            if verbose:
                print(f"[Iter {it}] 仍有 {len(remaining)} 个剩余样本，补分配")
            rem_idx = list(remaining)
            rem_vecs = V[rem_idx]  # (R,D)
            dmat = torch.cdist(rem_vecs, centroids)  # (R,K)
            for i, g_idx in enumerate(rem_idx):
                order = torch.argsort(dmat[i])  # 距离升序
                placed = False
                for cand in order:
                    c = cand.item()
                    if len(assignments[c]) < capacities[c] + 1:
                        assignments[c].append(g_idx)
                        placed = True
                        break
                if not placed:
                    c = order[0].item()
                    assignments[c].append(g_idx)
            for k in range(K):
                if assignments[k]:
                    centroids[k] = V[assignments[k]].mean(dim=0)

        labels = np.empty(N, dtype=np.int32)
        for k, idxs in enumerate(assignments):
            for idx in idxs:
                labels[idx] = k

        if prev_labels is not None and np.array_equal(labels, prev_labels):
            if verbose:
                print(f"Balanced K-Means 收敛于第 {it+1} 轮")
            break
        prev_labels = labels.copy()

    return centroids.detach(), labels, assignments

@torch.no_grad()
def residual_quantize(
    X: torch.Tensor,
    L: int = 3,
    K: int = 8,
    max_iter: int = 100,
    seed: Optional[int] = None,
    return_residuals: bool = False,
    verbose: bool = False
):
    """
    多层残差量化
    X: (N,D)
    返回:
      tokens: (N,L) np.int32
      codebooks: list[Tensor] 每个 (K,D) float32
      residuals (可选)
    """
    if X.requires_grad:
        X = X.detach()
    if seed is not None:
        set_seed(seed)

    device = X.device
    N, D = X.shape
    residual = X.clone()
    tokens = np.zeros((N, L), dtype=np.int32)
    codebooks = []
    residual_list = []

    for l in range(L):
        # 为每一层单独设置不同 seed（可选）
        layer_seed = None if seed is None else seed + l
        centroids, labels, _ = BalancedKMeans(
            residual, K=K, max_iter=max_iter, seed=layer_seed, verbose=False
        )
        centroids = centroids.to(device)
        codebooks.append(centroids)
        tokens[:, l] = labels
        label_tensor = torch.from_numpy(labels).to(device=device, dtype=torch.long)
        selected = centroids[label_tensor]  # (N,D)
        residual = residual - selected
        if return_residuals:
            residual_list.append(residual.clone())
        if verbose:
            recon_error = torch.mean(torch.sum(residual ** 2, dim=1)).item()
            print(f"[Layer {l+1}] 平均残差能量 {recon_error:.6f}")

    if return_residuals:
        return tokens, codebooks, residual_list
    return tokens, codebooks
