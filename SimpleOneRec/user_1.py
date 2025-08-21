import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing.pre_kuairand import pre_kuairand
from preprocessing.cal_baseline_label import cal_baseline_label
from preprocessing.cal_ground_truth import cal_ground_truth
import pandas as pd
import numpy as np


class UserStaticPathway(nn.Module):
    """
    用户静态特征通道
    输入: 
        - uid: 用户ID
        - onehot_feats (feat_num,)，每个元素是类别索引
    输出: h_u (1, d_model)
    """

    def __init__(self, num_users, num_categories_list, d_model=512, emb_dim=64):
        super().__init__()
        # 用户ID的embedding层
        self.uid_embedding = nn.Embedding(num_users, emb_dim)

        # 其他类别特征的embedding层
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim) for num_cat in num_categories_list
        ])

        # 计算总的输入维度（uid embedding + 其他特征的embedding）
        input_dim = emb_dim * (len(num_categories_list) + 1)
        self.fc1 = nn.Linear(input_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.LeakyReLU()

    def forward(self, uid, onehot_feats):
        """
        uid: LongTensor, shape (1,)
        onehot_feats: LongTensor, shape (feat_num,)
        """
        # 获取uid的embedding
        uid_emb = self.uid_embedding(uid)
        # 获取其他特征的embedding
        embs = [uid_emb]  # 将uid_emb作为第一个元素

        for i, emb_layer in enumerate(self.embeddings):
            emb = emb_layer(onehot_feats[i])  # (1, emb_dim)
            embs.append(emb)
        f_u = torch.cat(embs, dim=-1)  # (1, (feat_num+1) * emb_dim)

        x = self.fc1(f_u)
        x = self.act(x)
        h_u = self.fc2(x)

        return h_u.unsqueeze(0)


class ShortTermPathway(nn.Module):
    """
    短期行为特征通道 (Ls=20)
    输入: 最近 20 条交互的多种特征
    输出: h_s (batch, Ls, d_model)
    """

    def __init__(self,
                 num_vids, num_aids, num_tags, num_labels,
                 d_model=512, Ls=20):
        super().__init__()
        # Embedding 层
        self.vid_emb = nn.Embedding(num_vids, d_model)  # vid -> d_model
        self.aid_emb = nn.Embedding(num_aids, 512)  # aid -> 512
        self.tag_emb = nn.Embedding(num_tags, 128)  # tag -> 128
        self.label_emb = nn.Embedding(num_labels, 128)  # label -> 128

        # 连续特征（时间戳, 播放时长, 视频时长）用 Linear 投到对应维度
        self.ts_fc = nn.Linear(1, 128)  # timestamp -> 128
        self.playtime_fc = nn.Linear(1, 128)  # playtime -> 128
        self.dur_fc = nn.Linear(1, 128)  # duration -> 128

        # 两层 MLP
        input_dim = d_model + 512 + 128 * 5  # vid + aid + tag/ts/playtime/dur/label
        self.fc1 = nn.Linear(input_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.LeakyReLU()

    def forward(self, vid, aid, tag, ts, playtime, dur, label):
        e_vid = self.vid_emb(vid)  # (batch, Ls, d_model)
        e_aid = self.aid_emb(aid)  # (batch, Ls, 512)
        e_tag = self.tag_emb(tag)  # (batch, Ls, 128)
        e_label = self.label_emb(label)  # (batch, Ls, 128)

        e_ts = self.ts_fc(ts.unsqueeze(-1))  # (batch, Ls, 128)
        e_playtime = self.playtime_fc(playtime.unsqueeze(-1))  # (batch, Ls, 128)
        e_dur = self.dur_fc(dur.unsqueeze(-1))  # (batch, Ls, 128)
        #print(e_vid.shape, e_aid.shape, e_tag.shape, e_ts.shape, e_playtime.shape, e_dur.shape, e_label.shape)
        # 拼接所有特征
        f_s = torch.cat([e_vid, e_aid, e_tag, e_ts, e_playtime, e_dur, e_label], dim=-1)

        # MLP
        x = self.fc1(f_s)
        x = self.act(x)
        h_s = self.fc2(x)  # (batch, Ls, d_model)

        return h_s.squeeze(0)


def contain_ls(ls):
    result_ls = []
    for x in ls:
        result_ls.extend(x)
    return result_ls


def compare_max(cat_ls, frac_dict):
    frac_ls = np.array([frac_dict[c] for c in cat_ls])
    cat_ls = np.array(cat_ls)
    frac_sort_cat_ls = cat_ls[np.argsort(frac_ls)][::-1]
    return frac_sort_cat_ls[0]


class PositiveFeedbackPathway(nn.Module):
    """
    正反馈行为特征通道 (Lp=256)
    """

    def __init__(self, num_vids, num_aids, num_tags, num_labels,
                 d_model=512, Lp=256):
        super().__init__()
        self.vid_emb = nn.Embedding(num_vids, d_model)
        self.aid_emb = nn.Embedding(num_aids, 512)
        self.tag_emb = nn.Embedding(num_tags, 128)
        self.label_emb = nn.Embedding(num_labels, 128)

        self.ts_fc = nn.Linear(1, 128)
        self.playtime_fc = nn.Linear(1, 128)
        self.dur_fc = nn.Linear(1, 128)

        input_dim = d_model + 512 + 128 * 5
        self.fc1 = nn.Linear(input_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.LeakyReLU()

    def forward(self, vid, aid, tag, ts, playtime, dur, label):
        e_vid = self.vid_emb(vid)
        e_aid = self.aid_emb(aid)
        e_tag = self.tag_emb(tag)
        e_label = self.label_emb(label)
        e_ts = self.ts_fc(ts.unsqueeze(-1))
        e_playtime = self.playtime_fc(playtime.unsqueeze(-1))
        e_dur = self.dur_fc(dur.unsqueeze(-1))

        f_p = torch.cat([e_vid, e_aid, e_tag, e_ts, e_playtime, e_dur, e_label], dim=-1)
        x = self.fc1(f_p)
        x = self.act(x)
        h_p = self.fc2(x)
        return h_p



class LifelongPathway(nn.Module):
    """
    长期行为特征通道 (Ll=2000 → Nq=128)
    参考报告中 Lifelong Pathway 描述实现
    """
    def __init__(self, num_vids, num_aids, num_tags, num_labels,
                 d_model=512, Ll=2000, Nq=128, num_layers=2):
        super().__init__()
        # Embedding 层
        self.vid_emb = nn.Embedding(num_vids, d_model)
        self.aid_emb = nn.Embedding(num_aids, 512)
        self.tag_emb = nn.Embedding(num_tags, 128)
        self.label_emb = nn.Embedding(num_labels, 128)

        # 连续特征投影
        self.ts_fc = nn.Linear(1, 128)
        self.playtime_fc = nn.Linear(1, 128)
        self.dur_fc = nn.Linear(1, 128)

        # MLP 映射到 d_model
        input_dim = d_model + 512 + 128*5
        self.fc1 = nn.Linear(input_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.LeakyReLU()

        # QFormer Queries
        self.query = nn.Parameter(torch.randn(Nq, d_model))
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model*4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

    def forward(self, vid, aid, tag, ts, playtime, dur, label):
        # 如果输入是一维 (Ll,)，自动加 batch 维
        if vid.dim() == 1:
            vid = vid.unsqueeze(0)
            aid = aid.unsqueeze(0)
            tag = tag.unsqueeze(0)
            ts = ts.unsqueeze(0)
            playtime = playtime.unsqueeze(0)
            dur = dur.unsqueeze(0)
            label = label.unsqueeze(0)

        # 嵌入
        e_vid = self.vid_emb(vid)
        e_aid = self.aid_emb(aid)
        e_tag = self.tag_emb(tag)
        e_label = self.label_emb(label)
        e_ts = self.ts_fc(ts.unsqueeze(-1))
        e_playtime = self.playtime_fc(playtime.unsqueeze(-1))
        e_dur = self.dur_fc(dur.unsqueeze(-1))

        # 拼接并过 MLP
        f_l = torch.cat([e_vid, e_aid, e_tag, e_ts, e_playtime, e_dur, e_label], dim=-1)
        v_l = self.fc1(f_l)
        v_l = self.act(v_l)
        v_l = self.fc2(v_l)  # (B, Ll, d_model)

        # QFormer 压缩
        B = v_l.size(0)
        queries = self.query.unsqueeze(0).expand(B, -1, -1)  # (B, Nq, d_model)
        for layer in self.cross_attn_layers:
            # 将 queries 与 v_l 合并成一个序列，过 Transformer
            seq = torch.cat([queries, v_l], dim=1)  # (B, Nq+Ll, d_model)
            seq = layer(seq)
            queries = seq[:, :self.query.size(0), :]  # 取前 Nq 个

        return queries  # (B, Nq, d_model)





if __name__ == "__main__":
    
    #df_kuaiRand_usr_fe = pd.read_csv('rec_datasets/KuaiRand-Pure/data/user_features_pure.csv')
    #df_kuaiRand_log = pd.read_csv('rec_datasets/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv')
    #df_kuaiRand_vid_fe = pd.read_csv('rec_datasets/KuaiRand-Pure/data/video_features_basic_pure.csv')
    #df_kuaiRand_usr_fe = pd.read_csv('../data/KuaiRand-Pure/data/user_features_pure.csv')
    #df_kuaiRand_log = pd.read_csv('../data/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv')
    #df_kuaiRand_vid_fe = pd.read_csv('../data/KuaiRand-Pure/data/video_features_basic_pure.csv')
    df_kuaiRand_usr_fe = pd.read_csv('../data/KuaiRand-1K/data/user_features_1k.csv')
    df_kuaiRand_log = pd.read_csv('../data/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv')
    df_kuaiRand_vid_fe = pd.read_csv('../data/KuaiRand-1K/data/video_features_basic_1k.csv')
    df = pd.merge(df_kuaiRand_log, df_kuaiRand_usr_fe, on='user_id')
    df = pd.merge(df, df_kuaiRand_vid_fe, on='video_id')

    # -------------------- 构造统一映射 --------------------
    unique_vids = sorted(df['video_id'].unique())
    vid_mapping = {raw_id: i for i, raw_id in enumerate(unique_vids)}
    vids_num = len(unique_vids)

    unique_aids = sorted(df['author_id'].unique())
    aids_mapping = {raw_id: i for i, raw_id in enumerate(unique_aids)}
    aids_num = len(unique_aids)

    # tag 预处理
    df['tag_ls'] = df['tag'].apply(lambda x: str(x).split(','))
    total_ls = contain_ls(df['tag_ls'].values)
    stat_series = pd.Series(total_ls).value_counts()
    count_info = dict(zip(stat_series.index, stat_series.values))
    df['tag_pop'] = df['tag_ls'].apply(lambda x: compare_max(x, count_info))
    count_info = {k: v for k, v in count_info.items() if k != 'nan'}

    # 替换 nan/无效值为 0
    df['tag_pop'] = df['tag_pop'].apply(lambda v: 0 if pd.isna(v) or str(v) == 'nan' else int(v))
    tags_num = int(df['tag_pop'].max()) + 1

    labels_num = df['is_like'].nunique()

    # -------------------- 选择一个有正反馈的用户 --------------------
    valid_users = df.groupby('user_id').filter(lambda x: (x['is_like'] == 1).sum() > 0)['user_id'].unique()
    uid = int(valid_users[0])
    print(uid)

    # -------------------- 1. User Static Pathway --------------------
    onehot_cols = [f'onehot_feat{i}' for i in range(18)]
    num_categories_list = [int(df[f'onehot_feat{i}'].max()) + 1 for i in range(18)]
    d_model = 512
    emb_dim = 64
    model = UserStaticPathway(len(df_kuaiRand_usr_fe['user_id'].unique()), num_categories_list, d_model, emb_dim)

    onehot_feats = df_kuaiRand_usr_fe.loc[df_kuaiRand_usr_fe['user_id'] == uid, onehot_cols].iloc[0].tolist()
    onehot_feats = [int(x) for x in onehot_feats]
    h_u = model(torch.tensor(uid, dtype=torch.long), torch.tensor(onehot_feats, dtype=torch.long))

    # -------------------- 2. Short-term Pathway --------------------
    Ls = 20
    log = df[df['user_id'] == uid].head(Ls)
    vid = torch.tensor([vid_mapping[v] for v in log['video_id']], dtype=torch.long)
    aid_indices = torch.tensor([aids_mapping[a] for a in log['author_id']], dtype=torch.long)
    tag = torch.tensor([int(x) for x in log['tag_pop']], dtype=torch.long)
    ts = torch.tensor(log['time_ms'].values, dtype=torch.float)
    playtime = torch.tensor(log['play_time_ms'].values, dtype=torch.float)
    dur = torch.tensor(log['duration_ms'].values, dtype=torch.float)
    label = torch.tensor(log['is_like'].values, dtype=torch.long)

    short_term_model = ShortTermPathway(
        num_vids=vids_num,
        num_aids=aids_num,
        num_tags=tags_num,
        num_labels=labels_num,
        d_model=512,
        Ls=Ls
    )
    h_s = short_term_model(vid, aid_indices, tag, ts, playtime, dur, label)

    # -------------------- 3. Positive-feedback Pathway --------------------
    Lp = 256
    log_pos = df[(df['user_id'] == uid) & (df['is_like'] == 1)].head(Lp)

    # 如果不足 Lp 条，padding
    pad_len = Lp - len(log_pos)
    if pad_len > 0:
        pad_df = pd.DataFrame({
            'video_id': [unique_vids[0]] * pad_len,
            'author_id': [unique_aids[0]] * pad_len,
            'tag_pop': [0] * pad_len,
            'time_ms': [0.0] * pad_len,
            'play_time_ms': [0.0] * pad_len,
            'duration_ms': [0.0] * pad_len,
            'is_like': [0] * pad_len
        })
        log_pos = pd.concat([log_pos, pad_df], ignore_index=True)

    vid_p = torch.tensor([vid_mapping[v] for v in log_pos['video_id']], dtype=torch.long)
    aid_p = torch.tensor([aids_mapping[a] for a in log_pos['author_id']], dtype=torch.long)
    tag_p = torch.tensor([int(x) for x in log_pos['tag_pop']], dtype=torch.long)
    ts_p = torch.tensor(log_pos['time_ms'].values, dtype=torch.float)
    playtime_p = torch.tensor(log_pos['play_time_ms'].values, dtype=torch.float)
    dur_p = torch.tensor(log_pos['duration_ms'].values, dtype=torch.float)
    label_p = torch.tensor(log_pos['is_like'].values, dtype=torch.long)

    pos_model = PositiveFeedbackPathway(
        num_vids=vids_num,
        num_aids=aids_num,
        num_tags=tags_num,
        num_labels=labels_num,
        d_model=512,
        Lp=Lp
    )
    h_p = pos_model(vid_p, aid_p, tag_p, ts_p, playtime_p, dur_p, label_p)

    # -------------------- 4. Lifelong Pathway --------------------
    Ll = 2000
    log_long = df[df['user_id'] == uid].head(Ll)
    vid_l = torch.tensor([vid_mapping[v] for v in log_long['video_id']], dtype=torch.long)
    aid_l = torch.tensor([aids_mapping[a] for a in log_long['author_id']], dtype=torch.long)
    tag_l = torch.tensor([int(x) for x in log_long['tag_pop']], dtype=torch.long)
    ts_l = torch.tensor(log_long['time_ms'].values, dtype=torch.float)
    playtime_l = torch.tensor(log_long['play_time_ms'].values, dtype=torch.float)
    dur_l = torch.tensor(log_long['duration_ms'].values, dtype=torch.float)
    label_l = torch.tensor(log_long['is_like'].values, dtype=torch.long)

    life_model = LifelongPathway(
        num_vids=vids_num,
        num_aids=aids_num,
        num_tags=tags_num,
        num_labels=labels_num,
        d_model=512,
        Ll=Ll,
        Nq=128
    )
    h_l = life_model(vid_l, aid_l, tag_l, ts_l, playtime_l, dur_l, label_l)

    # -------------------- 打印结果 --------------------
    print("User Static Pathway:", h_u.shape)
    print("Short-term Pathway:", h_s.shape)
    print("Positive-feedback Pathway:", h_p.shape)
    print("Lifelong Pathway:", h_l.shape)