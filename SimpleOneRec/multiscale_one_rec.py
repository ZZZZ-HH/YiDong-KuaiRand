import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from Balanced_K_Means import residual_quantize
from SemanticEmbedding import SemanticEmbedding
from MoE import MoELayer

class MultiScaleOneRec(nn.Module):
    """
    Encoder: 四通路拼接后的连续表示 -> 残差量化 -> 语义嵌入 + 路径类型 + 位置编码 -> 送入原生 T5 Encoder (不做 MoE)
    Decoder: 仅将原 T5 decoder 中的 FFN 替换为 MoE，使用 NTP 预测第 0 层 code
    Loss: T5 内置 cross entropy
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        t5_conf = T5Config(
            vocab_size=cfg.K,          # 词表与第0层 codebook 对齐
            d_model=cfg.d_model,
            d_ff=cfg.d_model * 4,
            num_layers=cfg.num_layers,
            num_decoder_layers=cfg.num_layers,
            decoder_start_token_id=0,
            pad_token_id=0
        )
        self.t5 = T5ForConditionalGeneration(t5_conf)

        # 语义 ID 多层平均嵌入
        self.semantic_embed = SemanticEmbedding(
            num_codebooks=cfg.L_code,
            codebook_size=cfg.K,
            embedding_dim=cfg.d_model
        )
        self.path_type_emb = nn.Embedding(4, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # 仅替换 decoder FFN -> MoE
        self._replace_decoder_ffn_with_moe(
            num_experts=getattr(cfg, "num_experts", 24),
            top_k=getattr(cfg, "top_k", 2)
        )

    def _replace_decoder_ffn_with_moe(self, num_experts=24, top_k=2):
        for blk in self.t5.decoder.block:
            ff_idx = -1  # decoder: [SelfAttn, CrossAttn, FF]
            ff_layer = blk.layer[ff_idx]
            orig_ln = ff_layer.layer_norm
            moe = MoELayer(
                d_model=self.t5.config.d_model,
                d_ff=self.t5.config.d_model * 4,
                num_experts=num_experts,
                top_k=top_k
            )

            class FFMoEWrapper(nn.Module):
                def __init__(self, ln, moe_mod, drop=0.1):
                    super().__init__()
                    self.layer_norm = ln
                    self.moe = moe_mod
                    self.dropout = nn.Dropout(drop)
                def forward(self, hidden_states, **kwargs):
                    x = self.layer_norm(hidden_states)
                    x = self.moe(x)
                    return hidden_states + self.dropout(x)

            blk.layer[ff_idx] = FFMoEWrapper(orig_ln, moe)

    @torch.no_grad()
    def quantize(self, tokens):
        """
        逐 forward 即时残差量化（计算量未优化）
        tokens: (B,S,d)
        return: codes (B,S,L_code)
        """
        B,S,d = tokens.shape
        flat = tokens.reshape(B*S, d)
        codes, _ = residual_quantize(
            flat,
            L=self.cfg.L_code,
            K=self.cfg.K,
            seed=getattr(self.cfg, "seed", None)
        )
        codes = torch.from_numpy(codes).to(tokens.device).view(B, S, self.cfg.L_code)
        return codes

    def forward(self, tokens, path_types):
        """
        tokens: (B,S,d_model)
        path_types: (B,S)
        """
        B,S,_ = tokens.shape
        if S > self.cfg.max_seq_len:
            raise ValueError(f"序列长度 {S} 超过 max_seq_len={self.cfg.max_seq_len}")

        # 残差量化
        codes = self.quantize(tokens)               # (B,S,L_code) int32
        codes = codes.long()                        # 统一转换为 int64 以适配 nn.Embedding 与 loss

        # 语义嵌入
        sem_vec = self.semantic_embed(codes)        # (B,S,d_model)

        # 编码增强
        pos_ids = torch.arange(S, device=tokens.device).unsqueeze(0).expand(B,S)
        sem_vec = sem_vec + self.path_type_emb(path_types) + self.pos_emb(pos_ids)

        # NTP 目标: 第0层 code
        labels = codes[:,:,0]                      # (B,S) long

        out = self.t5(
            input_ids=None,
            inputs_embeds=sem_vec,
            labels=labels
        )
        return out.loss, labels, codes, sem_vec