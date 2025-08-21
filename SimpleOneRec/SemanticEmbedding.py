import torch
import torch.nn as nn

class SemanticEmbedding(nn.Module):
    def __init__(self, num_codebooks, codebook_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embedding_dim)
            for _ in range(num_codebooks)
        ])

    def forward(self, semantic_ids):
        """
        semantic_ids: (B,S,L)  L = num_codebooks
        return: (B,S,embedding_dim)
        """
        emb_sum = None
        for i, emb_layer in enumerate(self.embeddings):
            e = emb_layer(semantic_ids[..., i])  # (B,S,d)
            emb_sum = e if emb_sum is None else emb_sum + e
        return emb_sum / len(self.embeddings)