import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import timedelta
import sys
import logging

# 获取当前脚本所在的目录
curPath = os.path.abspath(os.path.dirname(__file__))

# 找到项目根目录
rootPath = os.path.split(curPath)[0]

# 将项目根目录插入到 sys.path 的第一个位置
sys.path.insert(0, rootPath)

# 现在可以安全地导入项目中的模块
from scripts.pushbullet_notify import send_pushbullet_notification


def setup_logger(rank, rootPath):
    log_file = os.path.join(rootPath, f'dummy_classifier_train_rank_{rank}.log')

    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format=f'%(asctime)s - %(levelname)s - [Rank {rank}] - %(message)s (%(filename)s:%(lineno)d)',
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - [Rank {rank}] - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logger = logging.getLogger(f'Rank {rank}')
    return logger


def setup(rank, world_size, logger):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6006'

    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5)
    )
    torch.cuda.set_device(rank)
    logger.info(f"Process group initialized with rank {rank} and world size {world_size}")


def cleanup(logger):
    dist.destroy_process_group()
    logger.info("Cleanup complete")


def main_worker(rank, world_size):
    logger = setup_logger(rank, rootPath)
    setup(rank, world_size, logger)

    logger.info(f"Running on device: {torch.cuda.get_device_name(rank)} with PyTorch version {torch.__version__}")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.join(base_dir, '.', 'plt')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_path = os.path.join(base_dir, '..', 'data', 'processed_data.csv')
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded from {data_path}")

    train_data = df[df['is_split'] == 'TRAIN']
    test_data = df[df['is_split'] == 'TEST']
    logger.info(
        f"Loaded dataset from {data_path}, training samples: {len(train_data)}, testing samples: {len(test_data)}")

    le = LabelEncoder()
    y_train = le.fit_transform(train_data['genre'])
    y_test = le.transform(test_data['genre'])

    most_frequent_class = train_data['genre'].mode()[0]
    most_frequent_class_index = le.transform([most_frequent_class])[0]
    logger.info(f"Most frequent class is '{most_frequent_class}'")

    X_train = torch.tensor(train_data[['word', 'count']].values).float().to(rank)
    X_test = torch.tensor(test_data[['word', 'count']].values).float().to(rank)

    y_train_tensor = torch.tensor(y_train).to(rank)
    y_test_tensor = torch.tensor(y_test).to(rank)

    batch_size = 8
    train_dataset = TensorDataset(X_train, y_train_tensor)
    test_dataset = TensorDataset(X_test, y_test_tensor)

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    logger.info(f"DataLoader created with batch size {batch_size} and {len(train_loader)} batches per epoch")

    model = lambda x: torch.full_like(x[:, 0], most_frequent_class_index, dtype=torch.long)

    train_loss_values, train_acc_values = [], []
    val_loss_values, val_acc_values = [], []

    start_time = time.time()

    for epoch in range(5):
        logger.info(f"Epoch {epoch + 1}/5 started")
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (b_inputs, b_labels) in enumerate(train_loader):
            predictions = model(b_inputs)
            correct_predictions += torch.sum(predictions == b_labels)
            total_predictions += b_labels.size(0)

        train_acc = correct_predictions.double() / total_predictions
        train_acc_values.append(train_acc.cpu().numpy())
        logger.info(f"Epoch {epoch + 1} Training Accuracy: {train_acc:.4f}")

        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (b_inputs, b_labels) in enumerate(test_loader):
            predictions = model(b_inputs)
            correct_predictions += torch.sum(predictions == b_labels)
            total_predictions += b_labels.size(0)

        val_acc = correct_predictions.double() / total_predictions
        val_acc_values.append(val_acc.cpu().numpy())
        logger.info(f"Epoch {epoch + 1} Validation Accuracy: {val_acc:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Training completed in {total_time // 60:.0f} minutes {total_time % 60:.0f} seconds")

    if rank == 0:
        predictions = []
        true_labels = []

        for b_inputs, b_labels in test_loader:
            preds = model(b_inputs).cpu().numpy()
            labels = b_labels.cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels)

        logger.info("\n" + classification_report(true_labels, predictions, target_names=le.classes_))

        conf_matrix = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(save_dir, 'dummy_confusion_matrix.png'), dpi=500)
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(train_acc_values, label='Training Accuracy')
        plt.plot(val_acc_values, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'dummy_training_validation_accuracy.png'), dpi=500)
        plt.show()

    cleanup(logger)


def merge_logs(rootPath, world_size):
    merged_log_file = os.path.join(rootPath, 'dummy_classifier_train_merged.log')
    with open(merged_log_file, 'w') as outfile:
        for rank in range(world_size):
            log_file = os.path.join(rootPath, f'dummy_classifier_train_rank_{rank}.log')
            with open(log_file, 'r') as infile:
                outfile.write(infile.read())
                outfile.write("\n")
    print(f"Logs merged into {merged_log_file}")


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

    merge_logs(rootPath, world_size)


if __name__ == "__main__":
    main()
    send_pushbullet_notification("Task completed", "Your task on the server has finished.")
