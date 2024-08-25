import pandas as pd
import sqlite3
import os
print("Current working directory: ", os.getcwd())


def load_genre_data():
    # 读取歌曲类别数据
    genre_df = pd.read_csv('../data/msd_tagtraum_cd2c.cls', sep='\t', comment='#', header=None,
                           names=['trackId', 'majority_genre', 'minority_genre'])

    # 读取训练/测试划分数据
    split_df = pd.read_csv('../data/msd_tagtraum_cd2c_stratified_train55.cls', sep='\t', comment='#', header=None,
                           names=['trackId', 'split'])

    return genre_df, split_df


def load_lyrics_data():
    # 连接到mxm_dataset.db数据库并提取歌词词袋数据
    conn = sqlite3.connect('../data/mxm_dataset.db')
    query = "SELECT track_id, word, count FROM lyrics"
    lyrics_df = pd.read_sql_query(query, conn)
    conn.close()

    return lyrics_df


def prepare_dataset():
    # 加载数据
    genre_df, split_df = load_genre_data()
    lyrics_df = load_lyrics_data()

    # 合并数据
    merged_df = pd.merge(lyrics_df, genre_df[['trackId', 'majority_genre']], left_on='track_id', right_on='trackId',
                         how='inner')
    merged_df = pd.merge(merged_df, split_df, on='trackId', how='inner')

    # 删除不必要的列
    merged_df.drop(columns=['trackId'], inplace=True)

    # 重命名列
    merged_df.rename(columns={'track_id': 'trackId', 'majority_genre': 'genre', 'split': 'is_split'}, inplace=True)

    # 保存处理后的数据
    merged_df.to_csv('../data/processed_data.csv', index=False)
    print("New dataset created and saved as 'processed_data.csv'")


if __name__ == "__main__":
    prepare_dataset()
