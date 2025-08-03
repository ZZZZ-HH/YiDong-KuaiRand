import torch
import torch.nn as nn

def prepare_decoder_inputs(semantic_ids):
    """添加BOS标记"""
    batch_size, num_items, num_codebooks = semantic_ids.shape
    bos_token = torch.full((batch_size, 1, num_codebooks), 0) # 假设0为BOS ID

    # 每个item前插入[BOS]
    repeated_bos = bos_token.repeat(1, num_items, 1)
    return torch.cat([repeated_bos, semantic_ids], dim=1)