import os
import time
import torch
# 移除直接导入 autocast，改为兼容新旧接口动态选择
from pathways import (UserStaticPathway, ShortTermPathway,
                      PositiveFeedbackPathway, LifelongPathway)
from multiscale_one_rec import MultiScaleOneRec
from dataset_kuairand import build_dataloader, Ls, Lp, Ll
from shared_embeddings import SharedVideoEmbedding

class Cfg:
    d_model=512
    num_layers=4
    L_code=3
    K=512
    max_seq_len=1024
    seed=42
    lr=1e-4
    epochs=3
    num_experts=24
    top_k=2
    ckpt_dir="checkpoints_real"
    amp=True
    grad_clip=1.0
    log_interval=50

def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def assemble_tokens(batch, models, device):
    uid = batch['uid'].to(device)
    onehot = batch['onehot'].to(device)
    def unpack(tpl):
        vid, tag, ts, play, dur, label = tpl
        return (vid.to(device), tag.to(device), ts.to(device),
                play.to(device), dur.to(device), label.to(device))
    vid_s, tag_s, ts_s, play_s, dur_s, lab_s = unpack(batch['short'])
    vid_p, tag_p, ts_p, play_p, dur_p, lab_p = unpack(batch['pos'])
    vid_l, tag_l, ts_l, play_l, dur_l, lab_l = unpack(batch['life'])
    h_static = models['static'](uid, onehot)
    h_short  = models['short'](vid_s, tag_s, ts_s, play_s, dur_s, lab_s)
    h_pos    = models['pos'](vid_p, tag_p, ts_p, play_p, dur_p, lab_p)
    h_life   = models['life'](vid_l, tag_l, ts_l, play_l, dur_l, lab_l)
    tokens = torch.cat([h_static, h_short, h_pos, h_life], dim=1)
    B = tokens.size(0)
    path_types = torch.cat([
        torch.zeros(B, h_static.size(1), dtype=torch.long, device=device),
        torch.ones (B, h_short.size(1), dtype=torch.long, device=device),
        torch.full((B, h_pos.size(1)), 2, dtype=torch.long, device=device),
        torch.full((B, h_life.size(1)),3, dtype=torch.long, device=device),
    ], dim=1)
    return tokens, path_types

def build_optimizer(cfg, model, modules):
    seen = set()
    params = []
    def add_param(p):
        if id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    for p in model.parameters(): add_param(p)
    for m in modules:
        for p in m.parameters(): add_param(p)
    return torch.optim.Adam(params, lr=cfg.lr)

def save_ckpt(path, epoch, step, model, modules, optimizer, scaler, cfg):
    state = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": vars(cfg)
    }
    for k,v in modules.items():
        state[f"module_{k}"] = v.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)

if __name__ == "__main__":
    cfg = Cfg()
    set_seed(cfg.seed)
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        print("current device:", torch.cuda.current_device())
        print("device name:", torch.cuda.get_device_name(0))
        print("compute capability:", torch.cuda.get_device_capability(0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    base_dir = r"C:\Users\13438\Desktop\YiDong-KuaiRand\KuaiRand-1K\data"
    log_path        = os.path.join(base_dir, "log_random_4_22_to_5_08_1k.csv")
    user_feat_path  = os.path.join(base_dir, "user_features_1k.csv")
    video_basic_path= os.path.join(base_dir, "video_features_basic_1k.csv")
    video_stat_path = os.path.join(base_dir, "video_features_statistic_1k.csv")

    ds, dl = build_dataloader(log_path, user_feat_path, video_basic_path, video_stat_path,
                              batch_size=2, shuffle=True, device=device)

    print(f"users={ds.num_users} vids={ds.num_vids} tags={ds.num_tags} labels={ds.num_labels}")

    # 静态特征类别尺寸
    cat_sizes = []
    for col_idx in range(ds.user_static_feat.shape[1]):
        col_vals = ds.user_static_feat[:, col_idx].tolist()
        cat_sizes.append(max(col_vals)+1)

    # 共享视频 embedding
    shared_vid = SharedVideoEmbedding(ds.num_vids, base_dim=128).to(device)

    static = UserStaticPathway(ds.num_users, cat_sizes, d_model=cfg.d_model).to(device)
    common_kwargs = dict(shared_vid_emb=shared_vid, vid_base_dim=128, d_model=cfg.d_model)
    short = ShortTermPathway(ds.num_vids, ds.num_tags, ds.num_labels, **common_kwargs).to(device)
    pos   = PositiveFeedbackPathway(ds.num_vids, ds.num_tags, ds.num_labels, **common_kwargs).to(device)
    life  = LifelongPathway(ds.num_vids, ds.num_tags, ds.num_labels, Nq=128, **common_kwargs).to(device)

    model = MultiScaleOneRec(cfg).to(device)
    print("model first param device:", next(model.parameters()).device)

    # 参数内存统计
    total = 0
    for mod in [model, shared_vid, static, short, pos, life]:
        for p in mod.parameters():
            total += p.numel() * p.element_size()
    print("Param bytes =", total, "≈ %.2f GB" % (total/1024/1024/1024))

    # AMP Scaler 兼容新旧版本
    try:
        scaler = torch.amp.GradScaler(device_type='cuda', enabled=cfg.amp)
        use_new_amp = True
    except Exception:
        from torch.cuda.amp import GradScaler as OldGradScaler
        scaler = OldGradScaler(enabled=cfg.amp)
        use_new_amp = False
    print("AMP scaler new_api =", use_new_amp)

    # 动态获取 autocast 上下文
    if use_new_amp:
        def autocast_ctx():
            return torch.amp.autocast(device_type='cuda', enabled=cfg.amp)
    else:
        from torch.cuda.amp import autocast as legacy_autocast
        def autocast_ctx():
            return legacy_autocast(enabled=cfg.amp)

    optimizer = build_optimizer(cfg, model, [shared_vid, static, short, pos, life])

    model.train(); static.train(); short.train(); pos.train(); life.train(); shared_vid.train()

    global_step = 0
    for epoch in range(1, cfg.epochs+1):
        t0 = time.time()
        for step, batch in enumerate(dl):
            tokens, path_types = assemble_tokens(batch, {
                'static': static, 'short': short, 'pos': pos, 'life': life
            }, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                loss, labels, codes, _ = model(tokens, path_types)

            if cfg.amp:
                scaler.scale(loss).backward()
                if cfg.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            global_step += 1
            if global_step % cfg.log_interval == 0:
                print(f"Epoch {epoch} Step {global_step} loss={loss.item():.4f} S={tokens.size(1)}")
                if torch.cuda.is_available():
                    print(f"GPU alloc MB={torch.cuda.memory_allocated()/1024/1024:.1f} reserved MB={torch.cuda.memory_reserved()/1024/1024:.1f}")

        dur = time.time() - t0
        print(f"[Epoch {epoch}] last_loss={loss.item():.4f} time={dur:.1f}s")
        ckpt_path = os.path.join(cfg.ckpt_dir, f"epoch{epoch}.pt")
        save_ckpt(ckpt_path, epoch, global_step, model,
                  {'static': static, 'short': short, 'pos': pos, 'life': life, 'shared_vid': shared_vid},
                  optimizer, scaler if cfg.amp else None, cfg)
        print("Saved:", ckpt_path)