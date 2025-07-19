import torch
import torch.nn as nn

import MoE

class DecoderLayer(nn.TransformerDecoderLayer):
    """ 继承标准Transformer层，用MoE替换FFN """
    def __init__(self, d_model, nhead, num_experts=24, top_k=2, dim_feedforward=2048):
        super().__init__(d_model, nhead, dim_feedforward)
        # 用MoE替换原始FFN
        self.mlp = MoE.MoELayer(d_model, dim_feedforward, num_experts, top_k)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 自注意力
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(
            tgt2, tgt2, tgt2, attn_mask=tgt_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        # 编码器-解码器注意力
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(
            tgt2, memory, memory, attn_mask=memory_mask
        )
        tgt = tgt + self.dropout2(tgt2)

        # MoE前馈网络
        tgt2 = self.norm3(tgt)
        tgt2 = self.mlp(tgt2)
        tgt = tgt + self.dropout3(tgt2)

        return tgt