import dask.dataframe as dd
import pandas as pd
import sqlite3
import os
from sklearn.model_selection import train_test_split

# 打印当前工作目录
print("Current working directory: ", os.getcwd())

# 自定义简单的停用词列表
simple_stop_words = {"you", "and", "i", "the", "to", "of", "a", "it", "in"}

def load_genre_data():
    # 读取歌曲类别数据
    genre_df = pd.read_csv('../../data/msd_tagtraum_cd2c.cls', sep='\t', comment='#', header=None,
                           names=['trackId', 'majority_genre', 'minority_genre'])
    return genre_df

def load_lyrics_data():
    # 连接到 mxm_dataset.db 数据库并提取歌词词袋数据
    conn = sqlite3.connect('../../data/mxm_dataset.db')
    query = "SELECT track_id, word, count FROM lyrics"
    lyrics_df = pd.read_sql_query(query, conn)
    conn.close()

    return lyrics_df

# 修改 preprocess_words 函数，保留 'trackId' 和 'genre' 列
def preprocess_words(df, stop_words):
    words_list = []
    counts_list = []

    for index, row in df.iterrows():
        words = row['word'].split()
        counts = str(row['count']).split()  # 确保 count 是字符串

        # 过滤掉自定义 stopwords，并保持 word 和 count 同步
        filtered_words_and_counts = [(word, count) for word, count in zip(words, counts) if
                                     word.lower() not in stop_words]

        if filtered_words_and_counts:
            words_filtered, counts_filtered = zip(*filtered_words_and_counts)
            words_list.append(' '.join(words_filtered))
            counts_list.append(' '.join(counts_filtered))
        else:
            words_list.append('')
            counts_list.append('')

    # 返回一个 DataFrame，同时保留 'trackId' 和 'genre'
    result = pd.DataFrame({'trackId': df['trackId'], 'genre': df['genre'], 'word': words_list, 'count': counts_list})
    return result

def prepare_dataset():
    # 加载数据
    genre_df = load_genre_data()
    lyrics_df = load_lyrics_data()

    # 合并数据时，保留 'trackId' 列
    merged_df = pd.merge(lyrics_df, genre_df[['trackId', 'majority_genre']], left_on='track_id', right_on='trackId',
                         how='inner')

    # 删除重复的 'trackId' 列，保留其中一个
    merged_df = merged_df.drop(columns=['track_id'])  # 删除 'track_id' 列，保留 'trackId'

    # 强制将 'trackId' 列转换为字符串类型
    merged_df['trackId'] = merged_df['trackId'].astype(str)

    # 检查合并后的列名
    print("Columns after merge:", merged_df.columns)

    # 将 pandas DataFrame 转换为 dask DataFrame
    ddf = dd.from_pandas(merged_df, npartitions=4)

    # 不删除 'trackId' 列，确保它保留到后面的 groupby 操作
    ddf = ddf.rename(columns={'majority_genre': 'genre'})

    # 再次检查列名
    print("Columns in Dask DataFrame:", ddf.columns)

    # 提供 `meta` 信息，包括 'trackId' 和 'genre'
    meta = pd.DataFrame({'trackId': pd.Series(dtype='str'), 'genre': pd.Series(dtype='str'),
                         'word': pd.Series(dtype='str'), 'count': pd.Series(dtype='str')})

    # 并行处理，移除简单停用词并保持同步，保留 'trackId' 和 'genre'
    ddf = ddf.map_partitions(lambda df: preprocess_words(df, simple_stop_words), meta=meta)

    # 使用 Dask 进行并行 groupby 处理
    def combine_words_and_counts(group):
        words = group['word'].values
        counts = group['count'].values
        # 过滤掉空值以及空的括号对，移除没有词的计数项
        combined = ' '.join([f"{w}({c})" for w, c in zip(words, counts) if
                             pd.notna(w) and pd.notna(c) and w.strip() != '' and c.strip() != ''])
        return combined

    # 确保 'trackId' 和 'genre' 列存在，并且类型正确
    ddf_grouped = ddf.groupby(['trackId', 'genre']).apply(combine_words_and_counts, meta=('x', 'str')).reset_index()

    # 将 Dask DataFrame 转换为 pandas DataFrame
    merged_df_grouped = ddf_grouped.compute()

    # 按照每个类别进行等比例分割训练集和测试集
    train_df, test_df = train_test_split(merged_df_grouped, test_size=0.2, stratify=merged_df_grouped['genre'],
                                         random_state=42)

    # 添加分割标记
    train_df['is_split'] = 'TRAIN'
    test_df['is_split'] = 'TEST'

    # 合并训练集和测试集
    final_df = pd.concat([train_df, test_df])

    # 保存处理后的数据
    final_df.to_csv('../../data/mxm_msd_genre_pro_simple_stopwords.cls', index=False)
    print("New dataset created and saved as 'mxm_msd_genre_pro_simple_stopwords.cls'")

if __name__ == "__main__":
    prepare_dataset()
