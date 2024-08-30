import os
import sqlite3
import pandas as pd

# 获取脚本所在的目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 构建相对路径
input_db_path = os.path.join(base_dir, '..', '..', 'data', 'songsdata_bytrackid_onlyone.db')
genre_file_path = os.path.join(base_dir, '..', '..', 'data', 'msd_tagtraum_cd2c.cls')
output_db_path = os.path.join(base_dir, '..', '..', 'data', 'songsdata_genre_full_bytrackid.db')

# 读取 songsdata_bytrackid_onlyone.db 数据库
conn = sqlite3.connect(input_db_path)
songsdata_bytrackid_onlyone_df = pd.read_sql_query("SELECT * FROM songsdata_bytrackid_onlyone", conn)

# 读取 msd_tagtraum_cd2c.cls 文件
genre_df = pd.read_csv(genre_file_path, sep='\t', names=['track_id', 'majority_genre', 'minority_genre'])

# 通过 track_id 匹配并添加 genre 信息
result_df = pd.merge(songsdata_bytrackid_onlyone_df, genre_df[['track_id', 'majority_genre']], on='track_id', how='left')

# 重命名列
result_df.rename(columns={'majority_genre': 'genre'}, inplace=True)

# 删除 genre 列为空的行
result_df = result_df.dropna(subset=['genre'])

# 将结果写入新的数据库文件
conn_output = sqlite3.connect(output_db_path)
result_df.to_sql('songsdata_genre_full_bytrackid', conn_output, if_exists='replace', index=False)

# 关闭数据库连接
conn.close()
conn_output.close()

print("Generated songsdata_genre_full_bytrackid.db successfully.")
