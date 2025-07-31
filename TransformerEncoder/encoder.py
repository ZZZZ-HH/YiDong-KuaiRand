import torch
import torch.nn as nn
import numpy as np
from Balanced_K_Means import BalancedKMeans, residual_quantize
from Multi_Head_Self_Attention import MultiHeadSelfAttention
from RMSNorm import RMSNorm
from Feed_Forward import PositionWiseFeedForward

#OneRec Encoder Single Layer
class OneRecEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, n_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

#OneRec Encoder
class OneRecEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.layers = nn.ModuleList([
            OneRecEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

        self.pad_idx = pad_idx

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)
        attention_mask = padding_mask & padding_mask.transpose(-1, -2)

        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        output = self.norm(x)
        return output

if __name__ == "__main__":
    # OneRec Encoder 参数
    VOCAB_SIZE = 10000
    D_MODEL = 256
    N_HEADS = 8
    D_FF = 1024
    NUM_LAYERS = 6
    DROPOUT = 0.1
    PAD_ID = 0
    SEP_ID = 1
    print("Initializing OneRec Encoder...")
    one_rec_encoder = OneRecEncoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=PAD_ID
    )

    print("OneRecEncoder 结构:")
    print(one_rec_encoder)

    # 假设我们想要对物品的嵌入进行聚类。
    # 最直接的方式是对 OneRec Encoder 的 `nn.Embedding` 层中的所有物品嵌入进行聚类。
    # 注意：这里我们使用的是 OneRec Encoder 内部的原始物品嵌入，而不是经过整个Encoder处理后的上下文嵌入。
    # 如果要对上下文嵌入聚类，您需要先运行Encoder得到H，然后从H中提取特定物品的表示。
    
    # 提取所有非 PAD/SEP 物品的嵌入向量作为聚类输入 V
    # 假设物品ID从 2 到 VOCAB_SIZE-1
    # V 应该是一个张量，其中每行是一个物品的嵌入向量
    # 这里我们排除 PAD_ID 和 SEP_ID，因为它们通常不作为常规“物品”进行聚类
    '''
    item_embeddings_for_clustering = one_rec_encoder.embedding.weight[2:] # 从ID 2 开始到最后
    print(f"item_embeddings_for_clustering.shape:{item_embeddings_for_clustering.shape}")  # 应该是 (num_items, d_model)
    print("\n--- 模拟 OneRec Encoder 运行 ---")
    dummy_input_ids = torch.tensor([
        [SEP_ID, 6, 1, 5, SEP_ID, 2, 1, 7, PAD_ID, PAD_ID],
        [SEP_ID, 10, 20, SEP_ID, 30, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID]
    ])
    print(f"输入序列 ID:\n{dummy_input_ids}")
    output_H = one_rec_encoder(dummy_input_ids)
    print(f"\nOneRec Encoder 输出 H 形状: {output_H.shape}")
    print("\n（注意：如果需要将 OneRec Encoder 的输出与聚类中心关联，需要额外的逻辑来映射。）")
    '''
    X = np.random.randint(0, 100, size=(100, 256))  # 模拟一些数据
    X = torch.tensor(X, dtype=torch.float32)
    tokens, labels = residual_quantize(X, L=10, K=8)
    print(tokens)
    print(labels)