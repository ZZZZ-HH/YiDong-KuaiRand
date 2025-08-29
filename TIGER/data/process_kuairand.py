import pandas as pd
dataset_name = "KuaiRand-1K"
input_file1 = 'video_features_basic_1k.csv'
input_file2 = 'video_features_statistic_1k.csv'
df_input_file1 = pd.read_csv(f"{dataset_name}/data/{input_file1}")
df_input_file2 = pd.read_csv(f"{dataset_name}/data/{input_file2}")
df = pd.merge(df_input_file1, df_input_file2, on='video_id')
video_info = {}
print(df.columns)

for index, row in df.iterrows():
    videoID = row['video_id']
    video_info[videoID] = {
        'video_type': row['video_type'] if row['video_type'] is not None else None,
        'upload_type': row['upload_type'] if row['upload_type'] is not None else None,
        'video_duration': row['video_duration'] if row['video_duration'] is not None else None,
        'server_width': row['server_width'] if row['server_width'] is not None else None,
        'server_height': row['server_height'] if row['server_height'] is not None else None,
        'music_type': row['music_type'] if row['music_type'] is not None else None,
        'tag': row['tag'] if row['tag'] is not None else None,
    }
for videoID, info in list(video_info.items())[:5]:
    print(f"videoID: {videoID}, Info: {info}")

from sentence_transformers import SentenceTransformer

# download sentence-t5-base model if you haven't downloaded it yet
model = SentenceTransformer('./sentence-t5-base')

video_embeddings = []
ans = 0
length = len(video_info)
for videoID, info in video_info.items():
    ans = ans + 1
    if ans % 1000 == 0:
        print(f"{ans}/{length}")
    semantics = f"'video_type':{info.get('video_type', '')}\n 'upload_type':{info.get('upload_type', '')}\n 'video_duration':{info.get('video_duration', '')}\n 'server_width':{info.get('server_width', '')}\n 'server_height':{info.get('server_height', '')}\n 'music_type':{info.get('music_type', '')}\n 'tag':{info.get('tag', '')}"
    embedding = model.encode(semantics)
    video_embeddings.append({'videoID': videoID, 'embedding': embedding.tolist()})

video_emb_df = pd.DataFrame(video_embeddings)

print("\nItem embeddings DataFrame shape:", video_emb_df.shape)
print("The first 3 rows of item embeddings DataFrame:\n", video_emb_df.head(3))

video_emb_df.to_parquet(f'./{dataset_name}/video_emb.parquet', index=False)

print("Item embeddings saved to item_emb.parquet.")