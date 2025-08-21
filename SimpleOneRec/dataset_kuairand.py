import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime

# ----------------- 配置常量（可外部传入） -----------------
Ls = 20
Lp = 32
Ll = 2000   # 终身路径截断长度

def parse_timestamp(date_int, hourmin_int):
    """
    date: 20220430  hourmin: 1800
    转为 int 排序；也可转为 unix 时间戳
    """
    ds = str(int(date_int))
    hs = f"{int(hourmin_int):04d}"
    dt = datetime.strptime(ds + hs, "%Y%m%d%H%M")
    return int(dt.timestamp())

def extract_first_tag(tag_val):
    if pd.isna(tag_val):
        return 0
    # 可能含空格 / 逗号
    s = str(tag_val)
    for sep in [',', ' ', ';', '|']:
        if sep in s:
            part = s.split(sep)
            for x in part:
                x = x.strip()
                if x.isdigit():
                    return int(x)
            return 0
    # 单个值
    try:
        return int(float(s))
    except:
        return 0

def is_positive(row):
    # 简化规则，可自行强化
    return int((row.get("comment_stay_time", 0) > 0) or (row.get("is_profile_enter", 0) == 1))

class KuaiRandFourPathDataset(Dataset):
    """
    每个样本 = 一个用户的四通路表示 (静态 / 短期 / 正反馈 / 终身聚合输入)
    只保留 user_id 与 video_id 以及 video tag/duration 等
    """
    def __init__(self,
                 log_path,
                 user_feat_path,
                 video_basic_path,
                 video_stat_path,
                 device='cpu'):
        super().__init__()
        self.device = device

        # 读取
        self.log = pd.read_csv(log_path)
        self.user_feat = pd.read_csv(user_feat_path)
        self.v_basic = pd.read_csv(video_basic_path)
        self.v_stat = pd.read_csv(video_stat_path)

        # ID 映射
        self.user2id = {u: i for i, u in enumerate(sorted(self.user_feat['user_id'].unique()))}
        self.video2id = {v: i for i, v in enumerate(sorted(self.v_basic['video_id'].unique()))}

        # 视频特征
        self.v_basic['video_tag_first'] = self.v_basic['tag'].apply(extract_first_tag).astype(int)
        self.v_basic = self.v_basic.fillna(0)
        keep_cols_basic = ['video_id', 'video_duration', 'video_tag_first']
        vb_small = self.v_basic[keep_cols_basic]
        vs_small = self.v_stat[['video_id']]  # 可追加更多统计列
        video_feat = pd.merge(vb_small, vs_small, on='video_id', how='left')

        video_feat['vid_idx'] = video_feat['video_id'].map(self.video2id)
        self.video_feat_df = video_feat.set_index('vid_idx')

        # 过滤日志
        self.log = self.log[self.log['video_id'].isin(self.video2id.keys()) &
                            self.log['user_id'].isin(self.user2id.keys())].copy()
        self.log['uid_idx'] = self.log['user_id'].map(self.user2id)
        self.log['vid_idx'] = self.log['video_id'].map(self.video2id)

        # 时间戳
        self.log['timestamp'] = self.log.apply(lambda r: parse_timestamp(r['date'], r['hourmin']), axis=1)

        # 正反馈标志
        self.log['pos_flag'] = self.log.apply(is_positive, axis=1)

        # 目标列
        target_col = 'tab'
        if target_col not in self.log.columns:
            raise ValueError(f"日志缺少标签列 {target_col}")
        self.log['label'] = self.log[target_col].astype(int)

        # 排序分组
        self.user_groups = self.log.sort_values('timestamp').groupby('uid_idx')

        # 用户静态特征 (user_active_degree + onehot_feat*)
        self.user_feat = self.user_feat.fillna(0)
        if 'user_active_degree' in self.user_feat.columns:
            uad_map = {v: i for i, v in enumerate(self.user_feat['user_active_degree'].unique())}
            self.user_feat['user_active_degree_idx'] = self.user_feat['user_active_degree'].map(uad_map)
        else:
            self.user_feat['user_active_degree_idx'] = 0

        self.onehot_cols = [c for c in self.user_feat.columns if c.startswith('onehot_feat')]
        for c in self.onehot_cols:
            self.user_feat[c] = self.user_feat[c].fillna(0).astype(int)

        feat_mat = []
        for _, row in self.user_feat.iterrows():
            vals = [row['user_active_degree_idx']] + [int(row[c]) for c in self.onehot_cols]
            feat_mat.append(vals)
        self.user_static_feat = torch.tensor(feat_mat, dtype=torch.long)

        # vocab 尺寸
        self.num_users = len(self.user2id)
        self.num_vids = len(self.video2id)
        self.num_tags = int(self.v_basic['video_tag_first'].max()) + 1
        self.num_labels = int(self.log['label'].max()) + 1

        self.user_id_list = list(self.user_groups.groups.keys())

    def __len__(self):
        return len(self.user_id_list)

    def _pad_or_trunc(self, arr, L, pad_val=0):
        if len(arr) >= L:
            return arr[-L:]
        return [pad_val] * (L - len(arr)) + arr

    def _build_sequence_block(self, sub_df, L):
        """
        返回 dict: vid, aid, tag, ts, play, dur, label (长度 L)
        play: 使用 comment_stay_time 代替
        dur: 来自 video_basic (若无则 0)
        """
        vids = sub_df['vid_idx'].tolist()
        labels = sub_df['label'].tolist()
        ts_list = sub_df['timestamp'].tolist()
        play_list = sub_df.get('comment_stay_time', pd.Series([0] * len(sub_df))).tolist()

        tag_list = []
        dur_list = []
        for vid in vids:
            vf = self.video_feat_df.loc[vid]
            tag_list.append(int(vf['video_tag_first']))
            dur_list.append(float(vf.get('video_duration', 0.0)))

        vids = self._pad_or_trunc(vids, L)
        tag_list = self._pad_or_trunc(tag_list, L)
        ts_list = self._pad_or_trunc(ts_list, L)
        play_list = self._pad_or_trunc(play_list, L)
        dur_list = self._pad_or_trunc(dur_list, L)
        labels = self._pad_or_trunc(labels, L)

        return dict(
            vid=torch.tensor(vids, dtype=torch.long),
            tag=torch.tensor(tag_list, dtype=torch.long),
            ts=torch.tensor(ts_list, dtype=torch.float32),
            play=torch.tensor(play_list, dtype=torch.float32),
            dur=torch.tensor(dur_list, dtype=torch.float32),
            label=torch.tensor(labels, dtype=torch.long)
        )

    def __getitem__(self, idx):
        uid = self.user_id_list[idx]
        uid_tensor = torch.tensor(uid, dtype=torch.long)
        static_feats = self.user_static_feat[uid]

        g = self.user_groups.get_group(uid)
        # 短期
        short = g.tail(Ls)
        # 正反馈
        pos_rows = g[g['pos_flag'] == 1].tail(Lp)
        # 终身 (截断)
        lifelong = g.tail(Ll)

        short_block = self._build_sequence_block(short, Ls)
        pos_block = self._build_sequence_block(pos_rows, Lp)
        life_block = self._build_sequence_block(lifelong, Ll)  # pad 到 Ll

        sample = dict(
            uid=uid_tensor,
            static_feats=static_feats,
            short=short_block,
            pos=pos_block,
            life=life_block
        )
        return sample

def collate_fn(batch):
    # B
    B = len(batch)
    # 静态
    uids = torch.stack([b['uid'] for b in batch], dim=0)
    onehot_feats = torch.stack([b['static_feats'] for b in batch], dim=0)  # (B,F)

    def stack_block(key):
        return (
            torch.stack([b[key]['vid'] for b in batch], 0),
            torch.stack([b[key]['tag'] for b in batch], 0),
            torch.stack([b[key]['ts'] for b in batch], 0),
            torch.stack([b[key]['play'] for b in batch], 0),
            torch.stack([b[key]['dur'] for b in batch], 0),
            torch.stack([b[key]['label'] for b in batch], 0),
        )
    short = stack_block('short')
    pos = stack_block('pos')
    life = stack_block('life')
    return dict(
        uid=uids,
        onehot=onehot_feats,
        short=short,
        pos=pos,
        life=life
    )

def build_dataloader(log_path, user_feat_path, video_basic_path, video_stat_path,
                     batch_size=4, shuffle=True, device='cpu'):
    ds = KuaiRandFourPathDataset(log_path, user_feat_path, video_basic_path, video_stat_path, device=device)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)