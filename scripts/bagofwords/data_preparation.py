import pandas as pd
import sqlite3
import os
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from collections import Counter

print("Current working directory: ", os.getcwd())

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_genre_data():
    # 读取歌曲类别数据
    genre_df = pd.read_csv('../../data/msd_tagtraum_cd2c.cls', sep='\t', comment='#', header=None,
                           names=['trackId', 'majority_genre', 'minority_genre'])
    return genre_df

def load_lyrics_data():
    # 连接到mxm_dataset.db数据库并提取歌词词袋数据
    conn = sqlite3.connect('../../data/mxm_dataset.db')
    query = "SELECT track_id, word, count FROM lyrics"
    lyrics_df = pd.read_sql_query(query, conn)
    conn.close()

    return lyrics_df

def preprocess_words(df, stop_words):
    # Remove stopwords
    df['word'] = df['word'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    # Calculate word frequencies
    word_counts = Counter(" ".join(df['word']).split())

    # Define a threshold for rare words, e.g., less than 5 occurrences
    rare_words = set([word for word, count in word_counts.items() if count < 5])

    # Remove rare words
    df['word'] = df['word'].apply(lambda x: ' '.join([word for word in x.split() if word not in rare_words]))

    return df

def prepare_dataset():
    # 加载数据
    genre_df = load_genre_data()
    lyrics_df = load_lyrics_data()

    # 合并数据，只保留majority_genre列
    merged_df = pd.merge(lyrics_df, genre_df[['trackId', 'majority_genre']], left_on='track_id', right_on='trackId',
                         how='inner')

    # 删除不必要的列
    merged_df.drop(columns=['trackId'], inplace=True)

    # 重命名列
    merged_df.rename(columns={'track_id': 'trackId', 'majority_genre': 'genre'}, inplace=True)

    # Preprocess words to remove stopwords and rare words
    merged_df = preprocess_words(merged_df, stop_words)

    # 将词袋形式的单词合并成句子形式
    merged_df_grouped = merged_df.groupby(['trackId', 'genre'])['word'].apply(lambda x: ' '.join(x)).reset_index()

    # 按照每个类别进行等比例分割训练集和测试集
    train_df, test_df = train_test_split(merged_df_grouped, test_size=0.2, stratify=merged_df_grouped['genre'], random_state=42)

    # 添加分割标记
    train_df['is_split'] = 'TRAIN'
    test_df['is_split'] = 'TEST'

    # 合并训练集和测试集
    final_df = pd.concat([train_df, test_df])

    # 保存处理后的数据
    final_df.to_csv('../../data/mxm_msd_genre_pro.cls', index=False)
    print("New dataset created and saved as 'mxm_msd_genre_pro.cls'")

if __name__ == "__main__":
    prepare_dataset()


