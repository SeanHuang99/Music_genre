import pandas as pd
import sqlite3

# Load the provided pickle file
df = pd.read_pickle('../../data/songs_dataframe.pkl')
pd.set_option('display.max_columns', None)
# Display the first few rows of the dataframe to understand its structure
print(df.head(30))


# Re-import necessary libraries and ensure correct execution of previous steps

# 将字典数据转为DataFrame
df_2 = pd.DataFrame('../data/songs_dataframe.pkl')

# The data is already loaded into the dataframe 'df', so I'll proceed directly to creating the database

# Create and connect to SQLite database
conn = sqlite3.connect('/mnt/data/songs.db')
cursor = conn.cursor()

# Create the table structure
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

# Insert data into the table
for _, row in df_2.iterrows():
    cursor.execute('''
        INSERT INTO songs (id, title, artist, year, genius_link, lyrics, category_ids, category_names)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (row['id'], row['title'], row['artist'], row['year'], row['genius_link'], row['lyrics'], row['category_ids'], row['category_names']))

# Commit changes and close the connection
conn.commit()
conn.close()

# Provide the database file path
db_file_path = '../../data/songs_dataframe.db'
db_file_path
