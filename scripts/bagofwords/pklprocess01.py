import pandas as pd
import sqlite3

# 加载 .pkl 文件
df = pd.read_pickle('../../data/songs_dataframe.pkl')

# 确保显示所有列
pd.set_option('display.max_columns', None)

# 将列表列转换为字符串，以便插入 SQLite 数据库
df['category_ids'] = df['category_ids'].apply(lambda x: str(x))
df['category_names'] = df['category_names'].apply(lambda x: str(x))

# 连接到 SQLite 数据库（如果数据库不存在，会自动创建）
conn = sqlite3.connect('../../data/songs_dataframe.db')
cursor = conn.cursor()

# 创建表结构
cursor.execute('''
    CREATE TABLE IF NOT EXISTS songs (
        id INTEGER PRIMARY KEY,
        title TEXT,
        artist TEXT,
        year INTEGER,
        genius_link TEXT,
        lyrics TEXT,
        category_ids TEXT,
        category_names TEXT
    )
''')

# 将 DataFrame 数据插入到数据库中
for _, row in df.iterrows():
    cursor.execute('''
        INSERT INTO songs (id, title, artist, year, genius_link, lyrics, category_ids, category_names)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (row['id'], row['title'], row['artist'], row['year'], row['genius_link'], row['lyrics'], row['category_ids'], row['category_names']))

# 提交更改并关闭连接
conn.commit()
conn.close()
