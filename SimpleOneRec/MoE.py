import torch
import torch.nn as nn

class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.w2(self.gelu(self.w1(x)))

class MoELayer(nn.Module):
    """
    简化版 Top-K 路由 (token-level)
    """
    def __init__(self, d_model, d_ff, num_experts=24, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x):
        B,S,d = x.shape
        gate_scores = self.gate(x)  # (B,S,E)
        top_val, top_idx = torch.topk(gate_scores, k=self.top_k, dim=-1)  # (B,S,K)
        gate_prob = torch.softmax(top_val, dim=-1)  # (B,S,K)

        # 展平 tokens
        flat_x = x.view(B*S, d)
        flat_gate_scores = self.gate(x).view(B*S, self.num_experts)
        # 为每个 expert 收集被选中的 token 索引
        outputs = torch.zeros_like(x)

        for e in range(self.num_experts):
            # mask shape (B,S,K)
            match = (top_idx == e)
            if not match.any():
                continue
            # 取属于该 expert 的位置 (B,S,K) -> (B,S)
            token_mask = match.any(dim=-1)  # (B,S)
            gather_indices = torch.nonzero(token_mask, as_tuple=False)  # (M,2)
            if gather_indices.numel() == 0:
                continue
            sub_x = x[gather_indices[:,0], gather_indices[:,1]]  # (M,d)
            sub_out = self.experts[e](sub_x)  # (M,d)

            # 对每个 token 可能 top-k 中有该 expert 的权重
            for i,(b,s) in enumerate(gather_indices):
                k_mask = match[b,s]            # (K,)
                weights = gate_prob[b,s][k_mask].sum()  # 若重复只会一个 True
                outputs[b,s] += sub_out[i] * weights

        return outputs