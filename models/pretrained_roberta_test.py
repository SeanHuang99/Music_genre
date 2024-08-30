import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import RobertaTokenizer, RobertaForSequenceClassification
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
    """
    设置logger, 每个进程有独立的日志文件.
    """
    log_file = os.path.join(rootPath, f'roberta_base_test_rank_{rank}.log')

    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format=f'%(asctime)s - %(levelname)s - [Rank {rank}] - %(message)s (%(filename)s:%(lineno)d)',
        filemode='w'
    )

    # 配置同时输出到控制台
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

    test_data = df[df['is_split'] == 'TEST']
    logger.info(f"Loaded testing samples: {len(test_data)}")

    model_path = './pretrained_roberta'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)

    model.to(rank)
    model = DDP(model, device_ids=[rank])

    le = LabelEncoder()
    y_test = le.fit_transform(test_data['genre'])

    def encode_texts(texts):
        inputs = tokenizer(texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
        return inputs['input_ids'], inputs['attention_mask']

    X_test_ids, X_test_mask = encode_texts(test_data['word'] + ' ' + test_data['count'].astype(str))
    logger.debug(f"Encoded {len(test_data)} test texts into input IDs and attention masks")

    y_test_tensor = torch.tensor(y_test)

    batch_size = 8
    test_dataset = TensorDataset(X_test_ids, X_test_mask, y_test_tensor)
    test_sampler = DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    logger.info(f"DataLoader created with batch size {batch_size} and {len(test_loader)} batches")

    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(rank) for t in batch)
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = b_labels.cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)

    if rank == 0:
        logger.info("\n" + classification_report(true_labels, predictions, target_names=le.classes_))

        conf_matrix = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(save_dir, 'roberta_base_test_confusion_matrix.png'), dpi=500)
        plt.show()

        logger.info(f"Test results saved to {save_dir}")

    cleanup(logger)

def merge_logs(rootPath, world_size):
    merged_log_file = os.path.join(rootPath, 'roberta_base_test_merged.log')
    with open(merged_log_file, 'w') as outfile:
        for rank in range(world_size):
            log_file = os.path.join(rootPath, f'roberta_base_test_rank_{rank}.log')
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
    send_pushbullet_notification("Testing completed", "Your task on the server has finished.")
