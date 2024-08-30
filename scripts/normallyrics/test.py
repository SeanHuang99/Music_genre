import os
import sqlite3
import pandas as pd

# 获取脚本所在的目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 构建数据库的路径
songs_dataframe_db_path = os.path.join(base_dir, '..', '..', 'data', 'songs_dataframe.db')

# 连接到数据库
conn = sqlite3.connect(songs_dataframe_db_path)

# 查询所有表名
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables in the database:", tables['name'].tolist())

# 关闭数据库连接
conn.close()
