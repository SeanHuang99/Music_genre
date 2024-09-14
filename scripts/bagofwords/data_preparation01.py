import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import DataLoader, TensorDataset
import re  # 用于提取单词


def extract_words_with_count(text):
    """从 'word(count)' 格式中提取单词并基于 count 重复单词"""
    if isinstance(text, str):  # Only process if text is a string
        # 匹配 'word(count)' 格式的单词和计数
        matches = re.findall(r'(\w+)\((\d+)\)', text)
        # 将单词重复 'count' 次，形成最终的词汇列表
        words = [word for word, count in matches for _ in range(int(count))]
        return ' '.join(words)
    else:
        return ''  # Return an empty string if the input is not valid


def prepare_data():
    # 读取CSV文件
    df = pd.read_csv('../data/mxm_msd_genre_pro.cls')

    # 提取出 `word_with_count` 字段中的单词部分，并根据 count 重复单词
    df['word'] = df['x'].apply(extract_words_with_count)

    # 将词袋形式的单词合并成句子形式
    df_grouped = df.groupby(['trackId', 'genre', 'is_split'])['word'].apply(lambda x: ' '.join(x)).reset_index()

    # 使用TfidfVectorizer将文本转化为TF-IDF特征
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_grouped['word'])

    # 将分类标签转化为数字编码
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_grouped['genre'])

    # 分割训练集和测试集
    X_train = X[df_grouped['is_split'] == 'TRAIN']
    X_test = X[df_grouped['is_split'] == 'TEST']
    y_train = y[df_grouped['is_split'] == 'TRAIN']
    y_test = y[df_grouped['is_split'] == 'TEST']

    # 将稀疏矩阵转为密集张量
    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 使用DataLoader加载数据
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, len(vectorizer.get_feature_names_out()), len(label_encoder.classes_)


# 如果这个文件是直接运行的，那么可以如下执行数据准备
if __name__ == "__main__":
    train_loader, test_loader, input_size, num_classes = prepare_data()
