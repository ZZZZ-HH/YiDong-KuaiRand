import pandas as pd
from preprocessing.pre_kuairand import pre_kuairand
import numpy as np
import torch
from TransformerEncoder.Balanced_K_Means import residual_quantize
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


def preprocess():
    df_kuaiRand_vid_fe_bas = pd.read_csv('rec_datasets/KuaiRand-Pure/data/video_features_basic_pure.csv')
    df_kuaiRand_vid_fe_stat = pd.read_csv('rec_datasets/KuaiRand-Pure/data/video_features_statistic_pure.csv')
    df = pd.merge(df_kuaiRand_vid_fe_bas, df_kuaiRand_vid_fe_stat, on='video_id')
     # preprocess the video feature
    dic_video_type = {'NORMAL':1,'AD':0,'UNKNOWN':0}
    df['video_type'] = df['video_type'].apply(lambda x: dic_video_type[x])

    dic_upload_type = dict(zip(df['upload_type'].unique().tolist(),list(range(len(df['upload_type'].unique())))))
    df['upload_type'] = df['upload_type'].apply(lambda x: dic_upload_type[x])

    df['tag_ls'] = df['tag'].apply(lambda x: str(x).split(','))

    total_ls = contain_ls(df['tag_ls'].values)
    stat_series = pd.Series(total_ls).value_counts()
    count_info = dict(zip(stat_series.index,stat_series.values))

    df['tag_pop'] = df['tag_ls'].apply(lambda x: compare_max(x, count_info))
    df['tag_pop'] = df['tag_pop'].replace('nan', np.nan).fillna(0) 
    df['tag_pop'] = df['tag_pop'].astype(int)

    df = df[['video_type', 'upload_type', 'video_duration', 'server_width', 'server_height',
        'music_type', 'tag_pop', 'counts', 'show_cnt', 'show_user_num',
       'play_cnt', 'play_user_num', 'play_duration', 'complete_play_cnt',
       'complete_play_user_num', 'valid_play_cnt', 'valid_play_user_num',
       'long_time_play_cnt', 'long_time_play_user_num', 'short_time_play_cnt',
       'short_time_play_user_num', 'play_progress', 'comment_stay_duration',
       'like_cnt', 'like_user_num', 'click_like_cnt', 'double_click_cnt',
       'cancel_like_cnt', 'cancel_like_user_num', 'comment_cnt',
       'comment_user_num', 'direct_comment_cnt', 'reply_comment_cnt',
       'delete_comment_cnt', 'delete_comment_user_num', 'comment_like_cnt',
       'comment_like_user_num', 'follow_cnt', 'follow_user_num',
       'cancel_follow_cnt', 'cancel_follow_user_num', 'share_cnt',
       'share_user_num', 'download_cnt', 'download_user_num', 'report_cnt',
       'report_user_num', 'reduce_similar_cnt', 'reduce_similar_user_num',
       'collect_cnt', 'collect_user_num', 'cancel_collect_cnt',
       'cancel_collect_user_num', 'direct_comment_user_num',
       'reply_comment_user_num', 'share_all_cnt', 'share_all_user_num',
       'outsite_share_all_cnt']]
    return df
if __name__ == "__main__":
    #df_kuaiRand_vid_fe_bas = pd.read_csv('../data/KuaiRand-1K/data/video_features_basic_1k.csv')
    #df_kuaiRand_vid_fe_stat = pd.read_csv('../data/KuaiRand-1K/data/video_features_statistic_1k.csv')
    #df = pd.merge(df_kuaiRand_vid_fe_bas, df_kuaiRand_vid_fe_stat, on='video_id')
    
    df = preprocess()
    df = df.fillna(0)
    #print(df)

    df_tensor = torch.tensor(df.values, dtype=torch.float32)
    #print(df_tensor)
    tokens, codebooks = residual_quantize(df_tensor, 3, 100)
    for token in tokens:
        print(token)