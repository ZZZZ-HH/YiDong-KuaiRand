import torch
import torch.nn as nn

class SemanticEmbedding(nn.Module):
    def __init__(self, num_codebooks, codebook_size, embedding_dim):
        super().__init__()
        # 每层码本一个嵌入矩阵（L个码本）
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embedding_dim)
            for _ in range(num_codebooks)
        ])

    def forward(self, semantic_ids):
        """
        Args:
            semantic_ids: [batch, seq_len, num_codebooks]
        Returns:
            embeddings: [batch, seq_len, embedding_dim]
        """
        # 分层聚合嵌入
        embeddings = 0
        for i, emb_layer in enumerate(self.embeddings):
            embeddings += emb_layer(semantic_ids[..., i])
        return embeddings / len(self.embeddings)