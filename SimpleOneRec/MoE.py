import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5Config, T5ForConditionalGeneration
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)  # 升维
        self.w2 = nn.Linear(d_ff, d_model)  # 降维
        self.gelu = nn.GELU()               # 激活函数

    def forward(self, x):
        return self.w2(self.gelu(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=24, top_k=2):
        super().__init__()
        # 专家池
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff) for _ in range(num_experts)
        ])

        # 门控网络
        self.gate = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            out: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算门控分数
        gate_scores = self.gate(x)  # [batch, seq_len, num_experts]

        # TopK专家选择
        topk_val, topk_idx = torch.topk(
            gate_scores, k=self.top_k, dim=-1, sorted=False
        )

        # 稀疏门控计算
        sparse_gates = torch.softmax(topk_val, dim=-1)

        # 初始化输出
        out = torch.zeros_like(x)

        # 并行计算专家输出（激活TopK专家）
        for expert_id in range(self.num_experts):
            # 创建当前专家的mask
            expert_mask = (topk_idx == expert_id).any(dim=-1)
            
            if expert_mask.any():
                # 获取选择当前专家的位置
                indices = torch.nonzero(expert_mask, as_tuple=True)
                
                # 计算专家权重
                expert_positions = (topk_idx == expert_id).float()
                # 稀疏门控与位置相乘得到权重
                expert_weights = sparse_gates * expert_positions
                # 移除零权重
                expert_weights = expert_weights[expert_weights > 0]
                
                # 提取需要处理的输入子集
                x_expert = x[indices]
                
                # 应用专家计算
                expert_output = self.experts[expert_id](x_expert)
                
                # 加权合并专家输出
                for i, (b, s) in enumerate(zip(*indices)):
                    weight_idx = 0
                    for k in range(self.top_k):
                        if topk_idx[b, s, k] == expert_id:
                            out[b, s] += expert_output[i] * sparse_gates[b, s, k]
                            weight_idx += 1

        return out