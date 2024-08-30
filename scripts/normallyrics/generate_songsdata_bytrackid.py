import os
import sqlite3
import pandas as pd

# 获取脚本所在的目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 构建相对路径
songsdata_path = os.path.join(base_dir, '..', '..', 'data', 'songsdata.csv')
songs_dataframe_db_path = os.path.join(base_dir, '..', '..', 'data', 'songs_dataframe.db')
output_db_path = os.path.join(base_dir, '..', '..', 'data', 'songsdata_bytrackid.db')

# 读取 songsdata.csv 文件
songsdata_df = pd.read_csv(songsdata_path)

# 连接到 songs_dataframe.db 数据库
conn = sqlite3.connect(songs_dataframe_db_path)

# 使用正确的表名 'songs' 进行查询
songs_dataframe_df = pd.read_sql_query("SELECT * FROM songs", conn)

# 通过 title 和 artist_name (songsdata.csv) 匹配 title 和 artist (songs_dataframe.db)
matched_df = pd.merge(songs_dataframe_df, songsdata_df, left_on=['title', 'artist'], right_on=['title', 'artist_name'], how='inner')

# 提取所需的列并重命名
result_df = matched_df[['track_id', 'title', 'lyrics', 'artist_name', 'tempo', 'loudness', 'mode', 'year', 'genius_link', 'category_ids', 'category_names']]

# 将结果写入新的数据库文件
conn_output = sqlite3.connect(output_db_path)
result_df.to_sql('songsdata_bytrackid', conn_output, if_exists='replace', index=False)

# 关闭数据库连接
conn.close()
conn_output.close()

print("Generated songsdata_bytrackid.db successfully.")
