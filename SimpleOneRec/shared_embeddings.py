import torch
import torch.nn as nn

class SharedVideoEmbedding(nn.Module):
    def __init__(self, num_vids, base_dim=128, dtype=torch.float16):
        super().__init__()
        self.emb = nn.Embedding(num_vids, base_dim, dtype=dtype)
    def forward(self, vid_ids):
        return self.emb(vid_ids)