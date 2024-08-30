import os
import sqlite3
import pandas as pd

# 获取脚本所在的目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 构建相对路径
input_db_path = os.path.join(base_dir, '..', '..', 'data', 'songsdata_genre_full_bytrackid.db')
output_db_path = os.path.join(base_dir, '..', '..', 'data', 'songsdata_genre_lyrics_bytrackid.db')

# 读取 songsdata_genre_full_bytrackid.db 数据库
conn = sqlite3.connect(input_db_path)
songsdata_genre_full_df = pd.read_sql_query("SELECT track_id, title, lyrics, genre FROM songsdata_genre_full_bytrackid", conn)

# 将结果写入新的数据库文件
conn_output = sqlite3.connect(output_db_path)
songsdata_genre_full_df.to_sql('songsdata_genre_lyrics_bytrackid', conn_output, if_exists='replace', index=False)

# 关闭数据库连接
conn.close()
conn_output.close()

print("Generated songsdata_genre_lyrics_bytrackid.db successfully.")


