import pandas as pd
import numpy as np


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


def pre_kuairand():
    print("0 读入原始数据")
    # KuaiRand-1K
    df_kuaiRand_interaction_1 = pd.read_csv('../data/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv')
    df_kuaiRand_interaction_2 = pd.read_csv('../data/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv')
    df_kuaiRand_interaction_standard = pd.concat([df_kuaiRand_interaction_1, df_kuaiRand_interaction_2], axis=0)
    df_kuaiRand_video_feature_basic = pd.read_csv('../data/KuaiRand-1K/data/video_features_basic_1k.csv')
    df_kuaiRand_user_features = pd.read_csv('../data/KuaiRand-1K/data/user_features_1k.csv')

    # 筛选
    df_kuaiRand_interaction_standard['play_time_truncate'] = df_kuaiRand_interaction_standard.apply(
        lambda row: row['play_time_ms'] if row['play_time_ms'] < row['duration_ms'] else row['duration_ms'], axis=1)

    # 毫秒值转为秒
    df_kuaiRand_interaction_standard['duration_ms'] = np.round(
        df_kuaiRand_interaction_standard['duration_ms'].values / 1e3)

    df_kuaiRand_interaction_standard['play_time_ms'] = np.round(
        df_kuaiRand_interaction_standard['play_time_ms'].values / 1e3)

    df_kuaiRand_interaction_standard['play_time_truncate'] = np.round(
        df_kuaiRand_interaction_standard['play_time_truncate'].values / 1e3)

    # todo 2 处理 user_features
    print("2 开始处理 user_features")
    # 2.1 user_active_degree 字段映射到 0-4 并替换原始字段
    dic_user_active_degree = {'full_active': 4, 'high_active': 3, 'middle_active': 2, '2_14_day_new': 0,
                              'low_active': 1, 'single_low_active': 1, '30day_retention': 0, 'day_new': 0, 'UNKNOWN': 0}
    df_kuaiRand_user_features['user_active_degree'] = df_kuaiRand_user_features['user_active_degree'].apply(
        lambda x: dic_user_active_degree[x])

    # 2.2 follow_user_num_range 字段映射到 0-6 并替换原始字段
    dic_follow_user_num_range = {'0': 0, '(0,10]': 1, '(10,50]': 2, '(50,100]': 3, '(100,150]': 4, '(150,250]': 5,
                                 '(250,500]': 6, '500+': 7}
    df_kuaiRand_user_features['follow_user_num_range'] = df_kuaiRand_user_features['follow_user_num_range'].apply(
        lambda x: dic_follow_user_num_range[x])

    # 2.3 将 fans_user_num_range 字段映射到 0-6 并替换原始字段
    dic_fans_user_num_range = {'0': 0, '[1,10)': 1, '[10,100)': 2, '[100,1k)': 3, '[1k,5k)': 4, '[5k,1w)': 5,
                               '[1w,10w)': 6, '[10w,100w)': 6, '[100w,1000w)': 6}
    df_kuaiRand_user_features['fans_user_num_range'] = df_kuaiRand_user_features['fans_user_num_range'].apply(
        lambda x: dic_fans_user_num_range[x])

    # 2.4 将 friend_user_num_range 字段映射到 0-6 并替换原始字段
    dic_friend_user_num_range = {'0': 0, '[1,5)': 1, '[5,30)': 2, '[30,60)': 3, '[60,120)': 4, '[120,250)': 5,
                                 '250+': 6}
    df_kuaiRand_user_features['friend_user_num_range'] = df_kuaiRand_user_features['friend_user_num_range'].apply(
        lambda x: dic_friend_user_num_range[x])

    # 2.5 将 register_days_range 字段映射到 0-6 并替换原始字段
    dic_register_days_range = {'8-14': 0, '15-30': 0, '31-60': 1, '61-90': 2, '91-180': 3, '181-365': 4, '366-730': 5,
                               '730+': 6}
    df_kuaiRand_user_features['register_days_range'] = df_kuaiRand_user_features['register_days_range'].apply(
        lambda x: dic_register_days_range[x])

    # 2.6 将 is_lowactive_period 字段的 非法值 赋值为 0 并替换原始的 is_lowactive_period
    df_kuaiRand_user_features['is_lowactive_period'] = np.nan_to_num(
        df_kuaiRand_user_features['is_lowactive_period'].values, nan=0.0,
        posinf=0.0, neginf=0.0)

    # todo 3 处理 video_features
    print("3 开始处理 video_feature_basic")

    # 3.1 将 video_type 映射到 0-1 并替换原始字段
    dic_video_type = {'NORMAL': 1, 'AD': 0, 'UNKNOWN': 0}
    df_kuaiRand_video_feature_basic['video_type'] = df_kuaiRand_video_feature_basic['video_type'].apply(
        lambda x: dic_video_type[x])

    # 3.2 upload_type 字段的种类太多了 自动映射
    dic_upload_type = dict(zip(df_kuaiRand_video_feature_basic['upload_type'].unique().tolist(),
                               list(range(len(df_kuaiRand_video_feature_basic['upload_type'].unique())))))
    df_kuaiRand_video_feature_basic['upload_type'] = df_kuaiRand_video_feature_basic['upload_type'].apply(
        lambda x: dic_upload_type[x])

    df_kuaiRand_video_feature_basic['tag_ls'] = df_kuaiRand_video_feature_basic['tag'].apply(
        lambda x: str(x).split(','))

    total_ls = contain_ls(df_kuaiRand_video_feature_basic['tag_ls'].values)
    stat_series = pd.Series(total_ls).value_counts()
    count_info = dict(zip(stat_series.index, stat_series.values))

    df_kuaiRand_video_feature_basic['tag_pop'] = df_kuaiRand_video_feature_basic['tag_ls'].apply(
        lambda x: compare_max(x, count_info))

    # merge the dataframe
    df_kuaiRand_interaction_standard = pd.merge(df_kuaiRand_interaction_standard, df_kuaiRand_user_features,
                                                on=['user_id'],
                                                how='left')
    df_kuaiRand_interaction_standard = pd.merge(df_kuaiRand_interaction_standard, df_kuaiRand_video_feature_basic,
                                                on=['video_id'], how='left')

    print("开始筛选")
    # 根据时长筛选  然后挑选部分字段
    # todo 可以缩小范围 避免内存或显存不足
    df_select_data = df_kuaiRand_interaction_standard[(df_kuaiRand_interaction_standard['duration_ms'] >= 300) & (
            df_kuaiRand_interaction_standard['duration_ms'] <= 400)]
    df_select_data = df_select_data[
        ['date', 'user_id', 'video_id', 'author_id', 'music_id', 'tag_pop', 'video_type', 'upload_type',
         'tab', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_profile_enter', 'is_hate',
         'profile_stay_time', 'comment_stay_time', 'follow_user_num_range', 'register_days_range',
         'fans_user_num_range', 'friend_user_num_range', 'user_active_degree', 'duration_ms', 'play_time_truncate',
         'play_time_ms']]
    df_select_data['tag_pop'] = df_select_data['tag_pop'].apply(lambda x: 999 if pd.isna(x) else x)

    user_id_map = dict(zip(np.sort(df_select_data['user_id'].unique()), range(len(df_select_data['user_id'].unique()))))
    video_id_map = dict(zip(np.sort(df_select_data['video_id'].unique()), range(len(df_select_data['video_id'].unique()))))
    author_id_map = dict(zip(np.sort(df_select_data['author_id'].unique()), range(len(df_select_data['author_id'].unique()))))
    music_id_map = dict(zip(np.sort(df_select_data['music_id'].unique()), range(len(df_select_data['music_id'].unique()))))
    tag_pop_map = dict(zip(np.sort(df_select_data['tag_pop'].unique()), range(len(df_select_data['tag_pop'].unique()))))
    upload_type_map = dict(
        zip(np.sort(df_select_data['upload_type'].unique()), range(len(df_select_data['upload_type'].unique()))))

    df_select_data['user_id'] = df_select_data['user_id'].apply(lambda x: user_id_map[x])
    df_select_data['video_id'] = df_select_data['video_id'].apply(lambda x: video_id_map[x])
    df_select_data['author_id'] = df_select_data['author_id'].apply(lambda x: author_id_map[x])
    df_select_data['music_id'] = df_select_data['music_id'].apply(lambda x: music_id_map[x])
    df_select_data['tag_pop'] = df_select_data['tag_pop'].apply(lambda x: tag_pop_map[x])
    df_select_data['upload_type'] = df_select_data['upload_type'].apply(lambda x: upload_type_map[x])

    return df_select_data
