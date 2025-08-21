import torch
import torch.nn as nn

class UserStaticPathway(nn.Module):
    def __init__(self, num_users, num_categories_list, d_model=512, emb_dim=64):
        super().__init__()
        self.uid_embedding = nn.Embedding(num_users, emb_dim)
        self.embeddings = nn.ModuleList([nn.Embedding(nc, emb_dim) for nc in num_categories_list])
        input_dim = emb_dim * (len(num_categories_list) + 1)
        self.fc1 = nn.Linear(input_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.LeakyReLU()
    def forward(self, uid, onehot_feats):
        uid_emb = self.uid_embedding(uid)
        feat_embs = [emb(onehot_feats[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([uid_emb] + feat_embs, dim=-1)
        x = self.act(self.fc1(x))
        return self.fc2(x).unsqueeze(1)

class _SeqPathBase(nn.Module):
    def __init__(self,
                 num_vids, num_tags, num_labels,
                 d_model=512, proj_dim=128,
                 shared_vid_emb: nn.Module = None,
                 vid_base_dim=128):
        super().__init__()
        self.use_shared_vid = shared_vid_emb is not None
        if self.use_shared_vid:
            self.shared_vid = shared_vid_emb
            vid_dim = vid_base_dim
        else:
            self.vid_emb = nn.Embedding(num_vids, d_model)
            vid_dim = d_model
        self.tag_emb = nn.Embedding(num_tags, proj_dim)
        self.label_emb = nn.Embedding(num_labels, proj_dim)
        self.ts_fc = nn.Linear(1, proj_dim)
        self.play_fc = nn.Linear(1, proj_dim)
        self.dur_fc = nn.Linear(1, proj_dim)
        input_dim = vid_dim + proj_dim * 5  # vid + tag + label + ts + play + dur
        self.fc1 = nn.Linear(input_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.LeakyReLU()
    def fuse(self, vid, tag, ts, play, dur, label):
        e_vid = self.shared_vid(vid).to(self.fc1.weight.dtype) if self.use_shared_vid else self.vid_emb(vid)
        e_tag = self.tag_emb(tag)
        e_lab = self.label_emb(label)
        e_ts  = self.ts_fc(ts.unsqueeze(-1))
        e_play= self.play_fc(play.unsqueeze(-1))
        e_dur = self.dur_fc(dur.unsqueeze(-1))
        x = torch.cat([e_vid, e_tag, e_lab, e_ts, e_play, e_dur], dim=-1)
        x = self.act(self.fc1(x))
        return self.fc2(x)
    def forward(self, vid, tag, ts, play, dur, label):
        return self.fuse(vid, tag, ts, play, dur, label)

class ShortTermPathway(_SeqPathBase): pass
class PositiveFeedbackPathway(_SeqPathBase): pass

class LifelongPathway(_SeqPathBase):
    def __init__(self, num_vids, num_tags, num_labels,
                 d_model=512, Nq=128, **kw):
        super().__init__(num_vids, num_tags, num_labels, d_model, **kw)
        self.Nq = Nq
        self.query = nn.Parameter(torch.randn(Nq, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8,
                                               dim_feedforward=d_model*4,
                                               batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
    def forward(self, vid, tag, ts, play, dur, label):
        seq = self.fuse(vid, tag, ts, play, dur, label)
        B = seq.size(0)
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        out = self.transformer(torch.cat([q, seq], dim=1))[:, :self.Nq]
        return out