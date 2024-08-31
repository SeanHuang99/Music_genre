import pandas as pd
import os
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

# 获取当前脚本所在的目录
curPath = os.path.abspath(os.path.dirname(__file__))

# 找到项目根目录
rootPath = os.path.split(curPath)[0]

# 将项目根目录插入到 sys.path 的第一个位置
sys.path.insert(0, rootPath)

# 配置日志记录
log_file = os.path.join(rootPath, 'dummy_classifier_train.log')
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

    try:
        # 加载数据
        train_loader, test_loader, input_size, num_classes = prepare_data()
        logging.info(f"Data loaded successfully with input size: {input_size} and number of classes: {num_classes}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    try:
        # 在这里重新创建一个 LabelEncoder 并对数据进行拟合，以恢复类别名称
        # df = pd.read_csv(os.path.join(rootPath, 'data/processed_data.csv'))

        df = pd.read_csv(os.path.join(rootPath, 'data/mxm_msd_genre.cls'))

        label_encoder = LabelEncoder()
        label_encoder.fit(df['genre'])
        logging.info("LabelEncoder fitted successfully.")
    except Exception as e:
        logging.error(f"Error processing labels: {e}")
        return

    try:
        # 准备数据
        X = df.drop(columns=['genre'])
        y = df['genre']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Data split into train and test sets successfully.")

        # 定义 DummyClassifier
        model = DummyClassifier(strategy="most_frequent")

        # 训练模型
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")

        # 评估模型
        predictions = model.predict(X_test)
        test_acc = model.score(X_test, y_test) * 100
        logging.info(f'Test Accuracy: {test_acc:.2f}%')
    except Exception as e:
        logging.error(f"Error during model training or evaluation: {e}")
        return

    try:
        # 生成分类报告和混淆矩阵
        logging.info("\n" + classification_report(y_test, predictions, target_names=label_encoder.classes_))

        conf_matrix = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(rootPath, 'dummy_confusion_matrix.png'), dpi=500)
        plt.show()
        logging.info("Confusion matrix plotted and saved successfully.")
    except Exception as e:
        logging.error(f"Error generating reports or plots: {e}")
        return


if __name__ == "__main__":
    try:
        train_and_evaluate()
        logging.info("Training and evaluation completed successfully.")
        send_pushbullet_notification("Task completed", "Your task on the server has finished.")
        logging.info("Notification sent successfully.")
    except Exception as e:
        logging.error(f"Error in the main execution: {e}")
