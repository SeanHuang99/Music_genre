import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
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
    log_file = os.path.join(rootPath, f'roberta_base_train_rank_{rank}.log')

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


def save_checkpoint(model, optimizer, scaler, epoch, loss, save_path='./checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved at {save_path} for epoch {epoch + 1}")


def load_checkpoint(model, optimizer, scaler, save_path='./checkpoint.pth'):
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        logging.info(f"Checkpoint loaded from {save_path}, starting from epoch {start_epoch}")
        return start_epoch, loss
    else:
        logging.info(f"No checkpoint found at {save_path}, starting from scratch")
        return 0, None


def save_and_cleanup_checkpoints(model, optimizer, scaler, epoch, loss, save_dir='./checkpoints', max_checkpoints=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存新检查点
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
    save_checkpoint(model, optimizer, scaler, epoch, loss, save_path=checkpoint_path)

    # 清理旧检查点，保持最新的max_checkpoints个
    checkpoint_files = sorted(os.listdir(save_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(checkpoint_files) > max_checkpoints:
        os.remove(os.path.join(save_dir, checkpoint_files[0]))


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

    model_path = '/root/autodl-tmp/DissertationProject/models/roberta_base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=len(df['genre'].unique()))

    logger.info(f"Loaded RoBERTa model from {model_path} with {model.num_parameters()} parameters")

    model.to(rank)
    model = DDP(model, device_ids=[rank])

    le = LabelEncoder()
    y_train = le.fit_transform(train_data['genre'])
    y_test = le.transform(test_data['genre'])

    def encode_texts(texts):
        inputs = tokenizer(texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
        return inputs['input_ids'], inputs['attention_mask']

    X_train_ids, X_train_mask = encode_texts(train_data['word'] + ' ' + train_data['count'].astype(str))
    logger.debug(f"Encoded {len(train_data)} training texts into input IDs and attention masks")

    X_test_ids, X_test_mask = encode_texts(test_data['word'] + ' ' + test_data['count'].astype(str))
    logger.debug(f"Encoded {len(test_data)} test texts into input IDs and attention masks")

    y_train_tensor = torch.tensor(y_train)
    y_test_tensor = torch.tensor(y_test)

    batch_size = 8
    train_dataset = TensorDataset(X_train_ids, X_train_mask, y_train_tensor)
    test_dataset = TensorDataset(X_test_ids, X_test_mask, y_test_tensor)

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    logger.info(f"DataLoader created with batch size {batch_size} and {len(train_loader)} batches per epoch")

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    loss_fn = CrossEntropyLoss()
    scaler = GradScaler()

    train_loss_values, train_acc_values = [], []
    val_loss_values, val_acc_values = [], []

    start_epoch, _ = load_checkpoint(model, optimizer, scaler, save_path='./checkpoints/latest_checkpoint.pth')

    start_time = time.time()

    model.train()
    for epoch in range(start_epoch, 5):  # 从检查点的 epoch 开始继续训练
        logger.info(f"Epoch {epoch + 1}/5 started")
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            b_input_ids, b_input_mask, b_labels = tuple(t.to(rank) for t in batch)

            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += torch.sum(predictions == b_labels)
                total_predictions += b_labels.size(0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Modify this logging statement to include epoch information
            logger.debug(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)} processed with loss {loss.item():.4f}")
        avg_train_loss = total_loss / len(train_loader)
        train_loss_values.append(avg_train_loss)
        train_acc = correct_predictions.double() / total_predictions
        train_acc_values.append(train_acc.cpu().numpy())

        logger.info(
            f"Epoch {epoch + 1} finished. Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_acc:.4f}")

        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in test_loader:
                b_input_ids, b_input_mask, b_labels = tuple(t.to(rank) for t in batch)

                with autocast():
                    outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs.loss
                    logits = outputs.logits
                    val_loss += loss.item()

                    predictions = torch.argmax(logits, dim=-1)
                    correct_predictions += torch.sum(predictions == b_labels)
                    total_predictions += b_labels.size(0)

        avg_val_loss = val_loss / len(test_loader)
        val_loss_values.append(avg_val_loss)
        val_acc = correct_predictions.double() / total_predictions
        val_acc_values.append(val_acc.cpu().numpy())

        logger.info(f"Validation finished. Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        model.train()

        # 保存检查点
        save_and_cleanup_checkpoints(model, optimizer, scaler, epoch, avg_train_loss, save_dir='./checkpoints',
                                     max_checkpoints=5)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"训练完成，总运行时间: {total_time // 60:.0f} 分 {total_time % 60:.0f} 秒")

    if rank == 0:
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

        logger.info("\n" + classification_report(true_labels, predictions, target_names=le.classes_))

        conf_matrix = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(save_dir, 'roberta_base_confusion_matrix.png'), dpi=500)
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(train_acc_values, label='Training Accuracy')
        plt.plot(val_acc_values, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'roberta_base_training_validation_accuracy.png'), dpi=500)
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(train_loss_values, label='Training Loss')
        plt.plot(val_loss_values, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'roberta_base_training_validation_loss.png'), dpi=500)
        plt.show()

        model.module.save_pretrained('./pretrained_roberta')
        tokenizer.save_pretrained('./pretrained_roberta')
        torch.save(optimizer.state_dict(), './pretrained_roberta/optimizer.pt')
        torch.save(scaler.state_dict(), './pretrained_roberta/scaler.pt')
        logger.info(f"Model and tokenizer saved to './pretrained_roberta'")
        logger.info(f"Optimizer and scaler states saved")

    cleanup(logger)



# def merge_logs(rootPath, world_size):
#     merged_log_file = os.path.join(rootPath, 'roberta_base_train_merged.log')
#     with open(merged_log_file, 'w') as outfile:
#         for rank in range(world_size):
#             log_file = os.path.join(rootPath, f'roberta_base_train_rank_{rank}.log')
#             with open(log_file, 'r') as infile:
#                 outfile.write(infile.read())
#                 outfile.write("\n")
#     print(f"Logs merged into {merged_log_file}")

def merge_logs(rootPath, world_size):
    merged_log_file = os.path.join(rootPath, 'roberta_base_train_merged.log')
    with open(merged_log_file, 'w') as outfile:
        for rank in range(world_size):
            log_file = os.path.join(rootPath, f'roberta_base_train_rank_{rank}.log')
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