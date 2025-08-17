import torch
import os

class Config:
    data_path = "KuaiRand_subset.csv"
    max_seq_len = 256       # 最大序列长度
    semantic_id_layers = 3  # 语义ID层数
    
    vocab_size = 8192       # 语义ID数量
    d_model = 256           # 模型维度
    num_layers = 3          # 编码器/解码器层数
    
    num_experts = 24        # 专家数量
    top_k = 2               # 每个位置激活的专家数量

    batch_size = 16
    learning_rate = 2e-4
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)