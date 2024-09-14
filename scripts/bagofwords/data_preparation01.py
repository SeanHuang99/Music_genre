import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

import torch
from torch.utils.data import DataLoader, TensorDataset


def prepare_data():
    # 读取CSV文件
    df = pd.read_csv('.././data/mxm_msd_genre.cls')

    # 将词袋形式的单词合并成句子形式
    df_grouped = df.groupby(['trackId', 'genre', 'is_split'])['word'].apply(lambda x: ' '.join(x)).reset_index()

    # 使用TfidfVectorizer将文本转化为TF-IDF特征
    vectorizer = TfidfVectorizer()
    # vectorizer = CountVectorizer()
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