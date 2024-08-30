import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import timedelta
import sys
from sklearn.preprocessing import LabelEncoder

# 获取当前脚本所在的目录
curPath = os.path.abspath(os.path.dirname(__file__))

# 找到项目根目录
rootPath = os.path.split(curPath)[0]

# 将项目根目录插入到 sys.path 的第一个位置
sys.path.insert(0, rootPath)

# 现在可以进行模块导入
from scripts.bagofwords.data_preparation01 import prepare_data
from scripts.pushbullet_notify import send_pushbullet_notification

def setup_logger(rank, rootPath):
    """
    设置日志记录器，每个进程有独立的日志文件。
    """
    log_file = os.path.join(rootPath, f'fcnn_train_rank_{rank}.log')

    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format=f'%(asctime)s - %(levelname)s - [Rank {rank}] - %(message)s (%(filename)s:%(lineno)d)',
        filemode='w'
    )

    # 配置同时输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f'%(asctime)s - %(levellevel)s - [Rank {rank}] - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logger = logging.getLogger(f'Rank {rank}')
    return logger

def setup(rank, world_size, logger):
    """
    初始化分布式进程组。
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6006'

    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5)
    )
    torch.cuda.set_device(rank)
    logger.info(f"已初始化进程组，rank={rank}，总进程数={world_size}")

def cleanup(logger):
    """
    清理分布式进程组。
    """
    dist.destroy_process_group()
    logger.info("进程组已清理")

class FCNN(nn.Module):
    """
    全连接神经网络 (Fully Connected Neural Network) 模型定义。
    """
    def __init__(self, input_size, num_classes):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

def plot_metrics(train_acc, val_acc, train_loss, val_loss, save_dir):
    """
    绘制训练和验证过程中的准确率和损失曲线，并保存图片。
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'fcnn_training_validation_accuracy.png'), dpi=500)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'fcnn_training_validation_loss.png'), dpi=500)
    plt.close()

def save_checkpoint(model, optimizer, scaler, epoch, loss, save_path='./checkpoint.pth'):
    """
    保存模型检查点。
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    logging.info(f"检查点已保存至 {save_path}，轮次 {epoch + 1}")

def load_checkpoint(model, optimizer, scaler, save_path='./checkpoint.pth'):
    """
    加载模型检查点。
    """
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        logging.info(f"已从 {save_path} 加载检查点，开始于轮次 {start_epoch}")
        return start_epoch, loss
    else:
        logging.info(f"在 {save_path} 没有找到检查点，从头开始训练")
        return 0, None

def save_and_cleanup_checkpoints(model, optimizer, scaler, epoch, loss, save_dir='./checkpoints', max_checkpoints=5):
    """
    保存最新检查点，并清理旧的检查点文件，保留最多 max_checkpoints 个。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, f'fcnn_checkpoint_epoch_{epoch + 1}.pth')
    save_checkpoint(model, optimizer, scaler, epoch, loss, save_path=checkpoint_path)
    checkpoint_files = sorted(os.listdir(save_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(checkpoint_files) > max_checkpoints:
        os.remove(os.path.join(save_dir, checkpoint_files[0]))

def main_worker(rank, world_size):
    """
    主要工作进程，每个进程对应一个GPU。
    """
    logger = setup_logger(rank, rootPath)
    setup(rank, world_size, logger)

    logger.info(f"运行设备: {torch.cuda.get_device_name(rank)}，PyTorch 版本: {torch.__version__}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, '.', 'fcnn_metrics')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_loader, test_loader, input_size, num_classes = prepare_data()

    # 对类别进行编码
    le = LabelEncoder()
    le.fit([label for _, label in train_loader.dataset])

    model = FCNN(input_size, num_classes).to(rank)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler()

    best_acc = 0.0
    train_acc_values = []
    val_acc_values = []
    train_loss_values = []
    val_loss_values = []

    start_epoch, _ = load_checkpoint(model, optimizer, scaler, save_path='./checkpoints/latest_checkpoint.pth')

    start_time = time.time()

    for epoch in range(start_epoch, 30):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total

        train_loss_values.append(epoch_loss)
        train_acc_values.append(epoch_acc)

        logger.info(f'Epoch {epoch+1}/{30}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(rank), labels.to(rank)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader.dataset)
        val_acc = 100 * correct / total

        val_loss_values.append(val_loss)
        val_acc_values.append(val_acc)

        logger.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'fcnn_best_model.pth')
            logger.info(f"New best model saved with accuracy: {best_acc:.2f}%")

        save_and_cleanup_checkpoints(model, optimizer, scaler, epoch, epoch_loss, save_dir='./checkpoints')

    total_time = time.time() - start_time
    logger.info(f"Training completed in: {total_time // 60:.0f} minutes {total_time % 60:.0f} seconds")
    logger.info(f'Best Validation Accuracy: {best_acc:.2f}%')

    if rank == 0:
        model.eval()
        predictions, true_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(rank), labels.to(rank)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        logger.info("\n" + classification_report(true_labels, predictions, target_names=le.classes_))

        conf_matrix = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title("FCNN Confusion Matrix")
        plt.savefig(os.path.join(save_dir, 'fcnn_confusion_matrix.png'), dpi=500)
        plt.show()

        plot_metrics(train_acc_values, val_acc_values, train_loss_values, val_loss_values, save_dir)

    cleanup(logger)

def merge_logs(rootPath, world_size):
    """
    合并日志文件。
    """
    merged_log_file = os.path.join(rootPath, 'fcnn_train_merged.log')
    with open(merged_log_file, 'w') as outfile:
        for rank in range(world_size):
            log_file = os.path.join(rootPath, f'fcnn_train_rank_{rank}.log')
            with open(log_file, 'r') as infile:
                outfile.write(infile.read())
                outfile.write("\n")
    print(f"Logs merged into {merged_log_file}")

def main():
    """
    主函数，启动多进程分布式训练。
    """
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
    merge_logs(rootPath, world_size)

if __name__ == "__main__":
    main()
    send_pushbullet_notification("Task completed", "Your FCNN task on the server has finished.")
