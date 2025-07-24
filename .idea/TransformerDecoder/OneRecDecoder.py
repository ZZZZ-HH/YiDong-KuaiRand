import torch
import torch.nn as nn
import math

from PositionalEncoding import PositionalEncoding
from SemanticEmbedding import SemanticEmbedding
from DecoderLayer import DecoderLayer

class OneRecDecoder(nn.Module):
    def __init__(self, num_layers, nhead, d_model,
                 num_codebooks, codebook_size,
                 num_experts=24, top_k=2):
        super().__init__()

        # 语义ID嵌入层
        self.embedding = SemanticEmbedding(
            num_codebooks, codebook_size, d_model
        )

        # 位置编码层
        self.pos_encoder = PositionalEncoding(d_model)

        # 解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, num_experts, top_k)
            for _ in range(num_layers)
        ])

        # 最终层归一化
        self.norm_out = nn.LayerNorm(d_model)

        # 输出投影层
        self.output_proj = nn.Linear(d_model, codebook_size)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """参数初始化方法"""
        # 线性层使用Xavier初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 输出层特殊初始化
        nn.init.normal_(self.output_proj.weight, mean=0, std=0.02)
        nn.init.constant_(self.output_proj.bias, 0)

    def generate_square_subsequent_mask(self, sz):
        """
        生成自回归掩码
        Args:
            sz: 序列长度
        Returns:
            mask: [sz, sz] 下三角为0，上三角为-inf
        """
        # 创建上三角矩阵（包含对角线）
        mask = (torch.triu(torch.ones(sz, sz)) == 1)

        # 转置使对角线下方为True
        mask = mask.transpose(0, 1)

        # 将False位置设为-inf，True位置设为0
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))

        return mask

    def forward(self, semantic_ids, encoder_output):
        """
        前向传播
        Args:
            semantic_ids: [batch_size, seq_len, num_codebooks]
            encoder_output: [batch_size, mem_len, d_model]
        Returns:
            logits: [batch_size, seq_len, codebook_size]
        """
        # 嵌入层
        # 将分层语义ID转换为连续向量
        x = self.embedding(semantic_ids) # [batch, seq_len, d_model]

        # 位置编码
        # 添加位置信息
        x = self.pos_encoder(x) # [batch, seq_len, d_model]

        # 生成自回归掩码
        # 创建因果掩码防止未来信息泄露
        tgt_mask = self.generate_square_subsequent_mask(
            semantic_ids.size(1) # 序列长度
        ) # [seq_len, seq_len]

        # 调整维度顺序
        # PyTorch Transformer要求序列维度在前
        x = x.permute(1, 0, 2) # [seq_len, batch, d_model]
        encoder_output = encoder_output.permute(1, 0, 2) # [mem_len, batch, d_model]

        # 逐层处理解码器
        for layer in self.layers:
            # 每层包含：自注意力+交叉注意力+MoE
            x = layer(
                tgt=x,
                memory=encoder_output,
                tgt_mask=tgt_mask
            ) # [seq_len, batch, d_model]

        # 最终层归一化
        x = self.norm_out(x) # [seq_len, batch, d_model]

        # 恢复维度顺序
        x = x.permute(1, 0, 2) # [batch, seq_len, d_model]

        # 输出投影
        # 预测每个位置的语义ID概率
        logits = self.output_proj(x) # [batch, seq_len, codebook_size]

        return logits