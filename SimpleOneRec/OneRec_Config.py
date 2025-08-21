import torch
import os

class Config:
    data_path = "KuaiRand_subset.csv"
    max_seq_len = 1024
    semantic_id_layers = 3

    vocab_size = 128        # 与 K 一致
    d_model = 256
    num_layers = 3

    num_experts = 24
    top_k = 2

    batch_size = 16
    learning_rate = 2e-4
    num_epochs = 5
    K = 128
    L_code = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)