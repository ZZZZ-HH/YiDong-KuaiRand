import torch
import torch.nn as nn
import torch.nn.functional as F

class NTPLoss(nn.Module):
    """
    Next Token Prediction损失函数
    Args:
        num_codebooks: 码本层数(L)
        codebook_size: 每个码本的大小
        bos_id: BOS标记的ID(默认为0)
        ignore_index: 忽略的索引值(默认为-100)
    """
    def __init__(self, num_codebooks, codebook_size, bos_id=0, ignore_index=-100):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.bos_id = bos_id
        self.ignore_index = ignore_index

    def forward(self, logits, semantic_ids):
        """
        Args:
            logits: 模型输出[batch_size, seq_len, codebook_size]
            semantic_ids: 目标语义ID[batch_size, seq_len, num_codebooks]
        Returns:
            loss
        """
        batch_size, seq_len, _ = semantic_ids.shape

        # 1. 准备目标序列(移除BOS标记并创建移位目标)
        # 目标序列: 从第2个位置开始到末尾(移除开头的BOS)
        targets = semantic_ids[:, 1:, :]  # [batch, seq_len-1, num_codebooks]

        # 2. 准备预测序列(对齐目标序列)
        # 预测序列: 从第1个位置开始到倒数第2个位置(与目标对齐)
        pred_logits = logits[:, :-1, :]  # [batch, seq_len-1, codebook_size]

        # 3. 计算每个码本层的损失(分层预测)
        total_loss = 0.0
        valid_items = 0

        # 遍历每个码本层
        for layer_idx in range(self.num_codebooks):
            # 获取当前层的目标ID
            layer_targets = targets[..., layer_idx] # [batch, seq_len-1]

            # 计算当前层的交叉熵损失
            layer_loss = F.cross_entropy(
                pred_logits.reshape(-1, self.codebook_size),
                layer_targets.reshape(-1),
                ignore_index=self.ignore_index,
                reduction='sum'
            )

            # 计算有效token数 (忽略padding)
            valid_mask = (layer_targets != self.ignore_index)
            num_valid = valid_mask.sum().item()

            if num_valid > 0:
                total_loss += layer_loss / num_valid
                valid_items += 1

        # 计算平均损失 (跨所有码本层)
        if valid_items > 0:
            total_loss = total_loss / valid_items
        else:
            total_loss = torch.tensor(0.0, device=logits.device)

        return total_loss