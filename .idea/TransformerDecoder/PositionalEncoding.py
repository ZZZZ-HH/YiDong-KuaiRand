import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置矩阵[max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)

        # 计算角度增量[d_model//2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # 初始化位置编码矩阵[max_len, d_model]
        pe = torch.zeros(max_len, 1, d_model)

        # 正弦函数应用于偶数索引
        pe[:, 0, 0::2] = torch.sin(position * div_term)

        # 余弦函数应用于奇数索引
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # 注册为固定参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + positional_encoding: [batch_size, seq_len, d_model]
        """
        # 添加位置编码（截取到当前序列长度）
        x = x + self.pe[:x.size(1), :].transpose(0, 1)

        # 应用dropout
        return self.dropout(x)