import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# 获取当前脚本所在的目录
curPath = os.path.abspath(os.path.dirname(__file__))

# 找到项目根目录
rootPath = os.path.split(curPath)[0]

# 将项目根目录插入到 sys.path 的第一个位置
sys.path.insert(0, rootPath)

# 配置日志记录
log_file = os.path.join(rootPath, 'svm_train.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# 配置同时输出到控制台
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# 现在可以进行模块导入
from scripts.bagofwords.data_preparation01 import prepare_data
from scripts.pushbullet_notify import send_pushbullet_notification


def train_and_evaluate():
    logging.info("Starting training and evaluation...")

    # Data preparation
    df = pd.read_csv('../data/mxm_msd_genre.cls')

    # Print out the column names to verify
    logging.info(f"Columns in the dataset: {df.columns}")

    X = df['word']
    y = df['genre']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline that vectorizes the data and then applies SVM
    model = make_pipeline(CountVectorizer(), SVC(kernel='linear', probability=True))

    # Training
    model.fit(X_train, y_train)
    logging.info("Model training completed.")

    # Evaluation on the test set
    predictions = model.predict(X_test)

    test_acc = (predictions == y_test).mean() * 100
    logging.info(f'Test Accuracy: {test_acc:.2f}%')

    # Generate classification report and confusion matrix
    logging.info("\n" + classification_report(y_test, predictions, target_names=label_encoder.classes_))

    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(rootPath, 'svm_confusion_matrix.png'), dpi=500)
    plt.show()


if __name__ == "__main__":
    train_and_evaluate()
    logging.info("Training and evaluation completed. Sending notification...")
    send_pushbullet_notification("Task completed", "Your task on the server has finished.")
    logging.info("Notification sent successfully.")
