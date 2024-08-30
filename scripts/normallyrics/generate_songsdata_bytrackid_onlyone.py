import os
import sqlite3
import pandas as pd

# 获取脚本所在的目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 构建相对路径
input_db_path = os.path.join(base_dir, '..', '..', 'data', 'songsdata_bytrackid.db')
output_db_path = os.path.join(base_dir, '..', '..', 'data', 'songsdata_bytrackid_onlyone.db')

# 连接到 songsdata_bytrackid.db 数据库
conn = sqlite3.connect(input_db_path)

# 读取数据
df = pd.read_sql_query("SELECT * FROM songsdata_bytrackid", conn)

# 移除 mode 列
df = df.drop(columns=['mode'])

# 删除 title 和 artist_name 同时相同的重复行，只保留一行
df_unique = df.drop_duplicates(subset=['title', 'artist_name'])

# 将结果写入新的数据库文件
conn_output = sqlite3.connect(output_db_path)
df_unique.to_sql('songsdata_bytrackid_onlyone', conn_output, if_exists='replace', index=False)

# 关闭数据库连接
conn.close()
conn_output.close()

print("Generated songsdata_bytrackid_onlyone.db successfully.")
