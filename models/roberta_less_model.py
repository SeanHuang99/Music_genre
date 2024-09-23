# -*- coding: utf-8 -*-

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import timedelta
import logging
import re


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, logger=None):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if logger:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# Set up the logger
def setup_logger(rank, rootPath):
    logger = logging.getLogger(f'Rank {rank}')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_file = os.path.join(rootPath, f'roberta_base_train_rank_{rank}.log')

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - [Rank {rank}] - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - [Rank {rank}] - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


# Set up the distributed training
def setup(rank, world_size, logger):
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'

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


# Save logs, checkpoints, and figures
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


def save_and_cleanup_checkpoints(model, optimizer, scaler, epoch, loss, save_dir='./checkpoints', max_checkpoints=5,
                                 logger=None):
    if logger is None:
        print("Logger is None, using print instead.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
    save_checkpoint(model, optimizer, scaler, epoch, loss, save_path=checkpoint_path)

    checkpoint_files = sorted(os.listdir(save_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if len(checkpoint_files) > max_checkpoints:
        file_to_remove = os.path.join(save_dir, checkpoint_files[0])
        if os.path.exists(file_to_remove):
            try:
                os.remove(file_to_remove)
                if logger is not None:
                    logger.info(f"Removed old checkpoint: {file_to_remove}")
                else:
                    print(f"Removed old checkpoint: {file_to_remove}")
            except FileNotFoundError:
                if logger is not None:
                    logger.warning(f"File not found when attempting to delete: {file_to_remove}")
                else:
                    print(f"File not found when attempting to delete: {file_to_remove}")
        else:
            if logger is not None:
                logger.warning(f"File not found: {file_to_remove}, skipping deletion.")
            else:
                print(f"File not found: {file_to_remove}, skipping deletion.")
    else:
        if logger is not None:
            logger.info("No checkpoints to remove.")
        else:
            print("No checkpoints to remove.")


# Function to extract words based on count from the 'x' column
def extract_words_with_count(text):
    """From 'word(count)' format, extract words and repeat them based on count."""
    if isinstance(text, str):  # Only process if text is a string
        matches = re.findall(r'(\w+)\((\d+)\)', text)
        words = [word for word, count in matches for _ in range(int(count))]
        return ' '.join(words)
    return ''  # Return an empty string if the input is not valid


# Prepare the data for RoBERTa
def prepare_data(tokenizer):
    # Load the dataset
    df = pd.read_csv('../data/mxm_msd_genre_pro_no_stopwords.cls')

    # Process the 'x' column to extract words and counts
    df['word'] = df['x'].apply(extract_words_with_count)

    # Split data into training and testing
    train_data = df[df['is_split'] == 'TRAIN']
    test_data = df[df['is_split'] == 'TEST']

    # Encode texts using RobertaTokenizer
    def encode_texts(texts):
        return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')

    X_train = encode_texts(train_data['word'])
    X_test = encode_texts(test_data['word'])

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data['genre'])
    y_test = label_encoder.transform(test_data['genre'])

    y_train_tensor = torch.tensor(y_train)
    y_test_tensor = torch.tensor(y_test)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train['input_ids'], X_train['attention_mask'], y_train_tensor)
    test_dataset = TensorDataset(X_test['input_ids'], X_test['attention_mask'], y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, len(label_encoder.classes_), label_encoder


# Main worker function
def main_worker(rank, world_size):
    logger = setup_logger(rank, os.getcwd())
    setup(rank, world_size, logger)
    logger.info(f"Running on device: {torch.cuda.get_device_name(rank)} with PyTorch version {torch.__version__}")

    model_path = './roberta_base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    train_loader, test_loader, num_classes, label_encoder = prepare_data(tokenizer)

    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)

    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False

    for param in model.roberta.encoder.layer[:8].parameters():
        param.requires_grad = False

    for param in model.roberta.encoder.layer[8:].parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=0.01)
    loss_fn = CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    scaler = GradScaler()

    train_loss_values, train_acc_values = [], []
    val_loss_values, val_acc_values = [], []

    early_stopping = EarlyStopping(patience=5, verbose=True)

    start_epoch, _ = load_checkpoint(model, optimizer, scaler)

    for epoch in range(start_epoch, 50):
        logger.info(f"Epoch {epoch + 1}/50 started")
        total_loss, correct_predictions, total_predictions = 0, 0, 0

        model.train()
        # Training Loop
        for step, batch in enumerate(train_loader):
            b_input_ids, b_attention_mask, b_labels = tuple(t.to(rank) for t in batch)

            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
                loss = outputs.loss / 4  # Gradient accumulation steps

            scaler.scale(loss).backward()
            if (step + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * 4
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += torch.sum(predictions == b_labels)
            total_predictions += b_labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_predictions.double() / total_predictions
        train_loss_values.append(avg_train_loss)
        train_acc_values.append(train_acc.cpu().numpy())

        logger.info(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_acc:.4f}")

        # Validation Loop
        val_loss, correct_predictions, total_predictions = 0, 0, 0
        all_true_labels = []
        all_predictions = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                b_input_ids, b_attention_mask, b_labels = tuple(t.to(rank) for t in batch)

                with autocast():
                    outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
                    loss = outputs.loss
                    val_loss += loss.item()

                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct_predictions += torch.sum(predictions == b_labels)
                    total_predictions += b_labels.size(0)

                    # Collect true labels and predictions
                    all_true_labels.extend(b_labels.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        val_acc = correct_predictions.double() / total_predictions
        val_loss_values.append(avg_val_loss)
        val_acc_values.append(val_acc.cpu().numpy())

        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Generate classification report and confusion matrix
        logger.info("\n" + classification_report(all_true_labels, all_predictions, target_names=label_encoder.classes_))

        conf_matrix = confusion_matrix(all_true_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(os.getcwd(), 'roberta_confusion_matrix.png'), dpi=500)
        plt.show()

        # Plot training/validation accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(train_acc_values, label='Training Accuracy')
        plt.plot(val_acc_values, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), 'roberta_training_validation_accuracy.png'), dpi=500)
        plt.show()

        # Plot training/validation loss
        plt.figure(figsize=(8, 6))
        plt.plot(train_loss_values, label='Training Loss')
        plt.plot(val_loss_values, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), 'roberta_training_validation_loss.png'), dpi=500)
        plt.show()

        # Save checkpoint
        save_and_cleanup_checkpoints(model, optimizer, scaler, epoch, avg_val_loss, logger=logger)

        # Early stopping
        early_stopping(avg_val_loss, logger)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

        scheduler.step(avg_val_loss)
        model.train()

    logger.info("Training complete")
    cleanup(logger)



# Merge logs
def merge_logs(rootPath, world_size):
    merged_log_file = os.path.join(rootPath, 'roberta_base_merged.log')
    with open(merged_log_file, 'w') as outfile:
        for rank in range(world_size):
            log_file = os.path.join(rootPath, f'roberta_base_train_rank_{rank}.log')
            if os.path.exists(log_file):
                with open(log_file, 'r') as infile:
                    outfile.write(f"\n----- Logs from Rank {rank} -----\n")
                    outfile.write(infile.read())
                    outfile.write("\n")
    print(f"Logs merged into {merged_log_file}")


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
    merge_logs(os.getcwd(), world_size)


if __name__ == "__main__":
    main()
