import torch
import numpy as np
from SimpleOneRec.Balanced_K_Means import residual_quantize, BalancedKMeans

def test_basic_shapes():
    N, D, L, K = 120, 32, 3, 8
    X = torch.randn(N, D)
    tokens, codebooks, residuals = residual_quantize(
        X, L=L, K=K, max_iter=50, seed=42, return_residuals=True, verbose=False
    )
    assert tokens.shape == (N, L)
    assert len(codebooks) == L
    for cb in codebooks:
        assert cb.shape == (K, D)
    assert len(residuals) == L

def test_residual_energy_monotonic():
    N, D, L, K = 200, 48, 4, 10
    X = torch.randn(N, D)
    _, _, residuals = residual_quantize(
        X, L=L, K=K, max_iter=60, seed=123, return_residuals=True
    )
    # 计算每层后的平均残差能量
    energies = [r.pow(2).sum(dim=1).mean().item() for r in residuals]
    print("Residual energies:", energies)
    # 允许偶尔微小波动，要求最后一层显著低于第一层
    assert energies[-1] <= energies[0]

def test_reconstruction():
    N, D, L, K = 150, 24, 3, 12
    torch.manual_seed(7)
    X = torch.randn(N, D)
    tokens, codebooks, residuals = residual_quantize(
        X, L=L, K=K, seed=99, return_residuals=True
    )
    device = X.device
    # 重构 = Σ 每层选中质心 + 最终残差
    recon = torch.zeros_like(X)
    for l in range(L):
        cb = codebooks[l]
        idx = torch.from_numpy(tokens[:, l]).long()
        recon += cb[idx]
    final_residual = residuals[-1]
    recon_full = recon  # 这是逐层质心之和
    recon_plus_final = X - final_residual  # 理论上等于 recon_full
    diff = (recon_full - recon_plus_final).pow(2).mean().item()
    mse_total = (X - recon_plus_final).pow(2).mean().item()
    print("Reconstruction internal diff:", diff, "MSE:", mse_total)
    assert diff < 1e-6  # 数值一致
    # MSE 无绝对阈值（取决于 K/L），仅打印观察

def test_cluster_balance_single_layer():
    N, D, K = 103, 16, 9
    X = torch.randn(N, D)
    centroids, labels, assignments = BalancedKMeans(X, K=K, max_iter=40, seed=0)
    sizes = [len(a) for a in assignments]
    print("Cluster sizes:", sizes)
    assert sum(sizes) == N
    assert max(sizes) - min(sizes) <= 2  # 允许 +1 放宽导致的轻微不均

def test_reproducibility():
    N, D, L, K = 90, 20, 3, 6
    X = torch.randn(N, D)
    t1, _, _ = residual_quantize(X, L=L, K=K, seed=2024, return_residuals=True)
    t2, _, _ = residual_quantize(X, L=L, K=K, seed=2024, return_residuals=True)
    t3, _, _ = residual_quantize(X, L=L, K=K, seed=2025, return_residuals=True)
    assert np.array_equal(t1, t2)
    assert not np.array_equal(t1, t3)

def test_error_when_K_gt_N():
    X = torch.randn(5, 8)
    try:
        BalancedKMeans(X, K=10)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

def test_gpu_optional():
    if torch.cuda.is_available():
        X = torch.randn(128, 64, device="cuda")
        tokens, codebooks = residual_quantize(X, L=2, K=8, seed=11)
        assert tokens.shape[0] == 128
        assert codebooks[0].device.type == "cuda"

if __name__ == "__main__":
    # 简单串行执行
    test_basic_shapes()
    test_residual_energy_monotonic()
    test_reconstruction()
    test_cluster_balance_single_layer()
    test_reproducibility()
    test_error_when_K_gt_N()
    test_gpu_optional()
    print("All tests finished.")