# import pandas as pd
# import sqlite3
# import os
# print("Current working directory: ", os.getcwd())
#
#
# def load_genre_data():
#     # 读取歌曲类别数据
#     genre_df = pd.read_csv('../../data/msd_tagtraum_cd2c.cls', sep='\t', comment='#', header=None,
#                            names=['trackId', 'majority_genre', 'minority_genre'])
#
#     # 读取训练/测试划分数据
#     split_df = pd.read_csv('../../data/msd_tagtraum_cd2c_stratified_train55.cls', sep='\t', comment='#', header=None,
#                            names=['trackId', 'split'])
#
#     return genre_df, split_df
#
#
# def load_lyrics_data():
#     # 连接到mxm_dataset.db数据库并提取歌词词袋数据
#     conn = sqlite3.connect('../../data/mxm_dataset.db')
#     query = "SELECT track_id, word, count FROM lyrics"
#     lyrics_df = pd.read_sql_query(query, conn)
#     conn.close()
#
#     return lyrics_df
#
#
# def prepare_dataset():
#     # 加载数据
#     genre_df, split_df = load_genre_data()
#     lyrics_df = load_lyrics_data()
#
#     # 合并数据
#     merged_df = pd.merge(lyrics_df, genre_df[['trackId', 'majority_genre']], left_on='track_id', right_on='trackId',
#                          how='inner')
#     merged_df = pd.merge(merged_df, split_df, on='trackId', how='inner')
#
#     # 删除不必要的列
#     merged_df.drop(columns=['trackId'], inplace=True)
#
#     # 重命名列
#     merged_df.rename(columns={'track_id': 'trackId', 'majority_genre': 'genre', 'split': 'is_split'}, inplace=True)
#
#     # 保存处理后的数据
#     merged_df.to_csv('../../data/processed_data.csv', index=False)
#     print("New dataset created and saved as 'processed_data.csv'")
#
#
# if __name__ == "__main__":
#     prepare_dataset()


import pandas as pd
import sqlite3
import os
from sklearn.model_selection import train_test_split

print("Current working directory: ", os.getcwd())

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
    final_df.to_csv('../../data/mxm_msd_genre.cls', index=False)
    print("New dataset created and saved as 'mxm_msd_genre.cls'")

if __name__ == "__main__":
    prepare_dataset()


# def prepare_dataset():
#     # 加载数据
#     genre_df = load_genre_data()
#     lyrics_df = load_lyrics_data()
#
#     # 合并数据，只保留majority_genre列
#     merged_df = pd.merge(lyrics_df, genre_df[['trackId', 'majority_genre']], left_on='track_id', right_on='trackId',
#                          how='inner')
#
#     # 删除不必要的列
#     merged_df.drop(columns=['trackId'], inplace=True)
#
#     # 重命名列
#     merged_df.rename(columns={'track_id': 'trackId', 'majority_genre': 'genre'}, inplace=True)
#
#     # 不再合并单词，将数据直接保存
#     merged_df.to_csv('../../data/mxm_msd_genre.cls', index=False)
#     print("New dataset created and saved as 'mxm_msd_genre.cls'")
#
# if __name__ == "__main__":
#     prepare_dataset()
