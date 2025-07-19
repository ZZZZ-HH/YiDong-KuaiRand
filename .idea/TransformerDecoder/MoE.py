import torch
import torch.nn as nn

class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff) # 升维
        self.w2 = nn.Linear(d_ff, d_model) # 降维
        self.gelu = nn.GELU() # 激活函数

    def forward(self, x):
        return self.w2(self.gelu(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=24, top_k=2):
        super().__init__()
        # 创建专家池
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff) for _ in range(num_experts)
        ])

        # 门控网络
        self.gate = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k # 激活专家数

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            out: [batch, seq_len, d_model]
        """
        # 计算门控分数
        gate_scores = self.gate(x) # [batch, seq_len, num_experts]

        # TopK专家选择
        topk_val, topk_idx = torch.topk(
            gate_scores, k=self.top_k, dim=-1, sorted=False
        )

        # 稀疏门控计算
        sparse_gates = torch.softmax(topk_val, dim=-1)

        # 初始化输出
        out = torch.zeros_like(x)

        # 并行计算专家输出（仅激活TopK专家）
        for expert_id in range(self.num_experts):
            # 创建当前专家的mask
            expert_mask = (topk_idx == expert_id).any(dim=-1)

            if expert_mask.any():
                # 获取当前专家处理的token
                expert_input = x[expert_mask]

                # 计算专家输出
                expert_out = self.experts[expert_id](expert_input)

                # 加权门控值
                gate_vals = sparse_gates[expert_mask]
                gate_mask = (topk_idx[expert_mask] == expert_id)
                weighted_gate = torch.sum(gate_vals * gate_mask.float(), dim=-1, keepdim=True)

                # 累加输出
                out[expert_mask] += weighted_gate * expert_out

        # 残差连接
        return out + x