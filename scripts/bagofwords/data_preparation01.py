import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder


def extract_words_with_count(text):
    """从 'word(count)' 格式中提取单词并基于 count 重复单词"""
    if isinstance(text, str):  # Only process if text is a string
        # 匹配 'word(count)' 格式的单词和计数
        matches = re.findall(r'(\w+)\((\d+)\)', text)
        # 将单词重复 'count' 次，形成最终的词汇列表
        words = [word for word, count in matches for _ in range(int(count))]
        return words
    else:
        return []  # Return an empty list if the input is not valid


def prepare_data():
    # 读取CSV文件
    df = pd.read_csv('../data/mxm_msd_genre_pro_no_stopwords.cls')

    # 提取出 `word_with_count` 字段中的单词部分，并根据 count 重复单词
    df['word'] = df['x'].apply(extract_words_with_count)

    # 将词袋形式的单词合并成句子形式
    df_grouped = df.groupby(['trackId', 'genre', 'is_split'])['word'].apply(
        lambda x: [word for sublist in x for word in sublist]).reset_index()

    # 使用预训练的Word2Vec模型或自训练模型
    # 加载预训练的Word2Vec模型（示例中使用Google的模型）
    # model = KeyedVectors.load_word2vec_format('path_to_pretrained_word2vec.bin', binary=True)

    # 使用自训练模型
    sentences = df_grouped['word'].tolist()
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    def get_word_vectors(words, model, vector_size=100):
        """将单词列表转换为词向量矩阵"""
        vectors = [model.wv[word] for word in words if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(vector_size)
        return np.mean(vectors, axis=0)

    # 转换文本为词向量
    X = np.array([get_word_vectors(words, model) for words in df_grouped['word']])

    # 将分类标签转化为数字编码
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_grouped['genre'])

    # 分割训练集和测试集
    X_train = X[df_grouped['is_split'] == 'TRAIN']
    X_test = X[df_grouped['is_split'] == 'TEST']
    y_train = y[df_grouped['is_split'] == 'TRAIN']
    y_test = y[df_grouped['is_split'] == 'TEST']

    # 将数据转为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 使用DataLoader加载数据
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, X_train.shape[1], len(label_encoder.classes_)


# 如果这个文件是直接运行的，那么可以如下执行数据准备
if __name__ == "__main__":
    train_loader, test_loader, input_size, num_classes = prepare_data()
