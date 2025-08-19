import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5Config, T5ForConditionalGeneration
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

import MoE as MoE
import OneRec_Config as Config

# 数据预处理
class KuaiRecDataset(Dataset):
    def __init__(self, df, config):
        self.config = config
        self.user_sequences = self._process_data(df)
        self.samples = self._create_samples()
        
    def _process_data(self, df):
        print("处理数据...")
        # 生成语义ID映射(替换为真实生成逻辑)
        video_ids = df['video_id'].unique()
        semantic_id_map = {vid: np.random.randint(0, self.config.vocab_size, 
                            size=self.config.semantic_id_layers) 
                          for vid in video_ids}
        
        # 构建用户序列{user_id: [语义ID列表]}
        user_sequences = {}
        df_sorted = df.sort_values(by=['user_id', 'date'])
        
        for user_id, group in df_sorted.groupby('user_id'):
            seq = []
            for _, row in group.iterrows():
                sem_ids = semantic_id_map[row['video_id']]
                seq.append(sem_ids)
            user_sequences[user_id] = seq
        
        return user_sequences
    
    def _create_samples(self):
        samples = []
        for user_id, seq in self.user_sequences.items():
            # 展平语义ID序列[[id1,id2,id3],...] => [id1,id2,id3,id1,id2,id3,...]
            flat_seq = [item for sublist in seq for item in sublist]
            
            # 截断或填充到固定长度
            if len(flat_seq) > self.config.max_seq_len:
                flat_seq = flat_seq[-self.config.max_seq_len:]
            else:
                # 填充到最大长度
                pad_len = self.config.max_seq_len - len(flat_seq)
                flat_seq = [0] * pad_len + flat_seq
            
            samples.append(flat_seq)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)

# 简化版OneRec
class SimpleOneRec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        t5_config = T5Config(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_ff=config.d_model * 4,  # FFN维度
            num_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            decoder_start_token_id=0,   # [BOS] token ID
            pad_token_id=0              # 0作为padding
        )
        self.model = T5ForConditionalGeneration(t5_config)

        self._replace_ffn_with_moe(
            num_experts=config.num_experts if hasattr(config, 'num_experts') else 24,
            top_k=config.top_k if hasattr(config, 'top_k') else 2
        )
    
    def _replace_ffn_with_moe(self, num_experts=24, top_k=2):
        for i, layer in enumerate(self.model.decoder.block):
            # MoE层
            moe = MoE.MoELayer(
                d_model=self.config.d_model,
                d_ff=self.config.d_model * 4,
                num_experts=num_experts,
                top_k=top_k
            )
            
            # 保存原始层归一化
            layer_norm = layer.layer[2].layer_norm
            
            # 创建新MoE层
            class MoELayerWrapper(nn.Module):
                def __init__(self, layer_norm, moe, dropout_rate=0.1):
                    super().__init__()
                    self.layer_norm = layer_norm
                    self.moe = moe
                    self.dropout = nn.Dropout(dropout_rate)
                
                def forward(self, hidden_states, **kwargs):
                    norm_x = self.layer_norm(hidden_states)
                    moe_output = self.moe(norm_x)
                    return hidden_states + self.dropout(moe_output)
            
            # 替换原FFN层
            layer.layer[2] = MoELayerWrapper(layer_norm, moe)

    def forward(self, input_ids):
        """
        输入: input_ids [batch_size, seq_len]
        输出: NTP loss
        """
        # decoder输入(右移添加[BOS])
        labels = input_ids.clone()
        decoder_input_ids = torch.cat([
            torch.zeros_like(labels[:, :1]),  # [BOS] token
            labels[:, :-1]                    # 右移
        ], dim=1)
        
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        return outputs.loss

# 训练函数
def train_model(config):
    print("加载数据...")
    df = pd.read_csv(config.data_path)
    dataset = KuaiRecDataset(df, config)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    print(f"数据集大小: {len(dataset)} | 批次数量: {len(dataloader)}")
    
    # 初始化模型
    model = SimpleOneRec(config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 训练循环
    print("开始训练...")
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in progress_bar:
            batch = batch.to(config.device)
            
            # 前向传播
            loss = model(batch)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # 平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成 | 平均损失: {avg_loss:.4f}")
        
        # 保存检查点
        checkpoint_path = os.path.join(config.save_dir, f"onerec_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"模型保存至 {checkpoint_path}")
    
    print("训练完成！")

if __name__ == "__main__":
    print(f"PyTorch CUDA可用: {torch.cuda.is_available()}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    
    config = Config.Config()
    print(f"使用设备: {config.device}")
    print(f"配置参数: \n最大序列长度: {config.max_seq_len}\n词汇表大小: {config.vocab_size}")
    print(f"模型维度: {config.d_model}\n层数: {config.num_layers}")
    print(f"MoE专家数量: {config.num_experts}\nMoE激活专家数: {config.top_k}")
    print(f"批大小: {config.batch_size}\n学习率: {config.learning_rate}")
    
    train_model(config)