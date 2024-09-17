import logging
import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from torchtext.vocab import GloVe
from torch.cuda.amp import GradScaler, autocast
import multiprocessing

# Setting up directories
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.insert(0, rootPath)

# Logging setup
log_file = os.path.join(rootPath, 'lstm_train.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Dataset preparation function
from scripts.bagofwords.data_preparation01 import prepare_data

# LSTM model class with bidirectional LSTM and GloVe embeddings
class LSTMWithPretrainedEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(LSTMWithPretrainedEmbeddings, self).__init__()
        # Embedding layer using pre-trained GloVe embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply hidden size by 2 due to bidirectional
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take output of the last time step
        out = self.fc(self.relu(out))
        return out

def train_and_evaluate(num_epochs, device_ids):
    logging.info("Starting training and evaluation...")

    # Prepare data
    train_loader, test_loader, input_size, num_classes = prepare_data()

    # Load the dataset to fit the label encoder
    df = pd.read_csv('../data/mxm_msd_genre.cls')
    label_encoder = LabelEncoder()
    label_encoder.fit(df['genre'])

    # Load pre-trained GloVe embeddings
    glove = GloVe(name="6B", dim=300)

    # Define device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    vocab_size = len(glove.stoi)
    embedding_dim = glove.dim
    hidden_size = 128  # You can experiment with this

    # Model initialization with DataParallel for multi-GPU support
    model = LSTMWithPretrainedEmbeddings(vocab_size, embedding_dim, hidden_size, num_classes)
    model = nn.DataParallel(model, device_ids=device_ids)  # Multi-GPU
    model = model.to(device)

    # Initialize model's embedding layer with pre-trained GloVe weights
    model.module.embedding.weight.data.copy_(glove.vectors)

    # Define adaptive optimizer (AdamW) and learning rate scheduler (ReduceLROnPlateau)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Mixed Precision Training
    scaler = GradScaler()

    best_val_loss = float('inf')
    no_improvement_epochs = 0
    early_stop_patience = 5

    # Tracking performance metrics
    train_loss_values, train_acc_values = [], []
    val_loss_values, val_acc_values = [], []

    # Automatically determine the number of CPU cores for num_workers
    cpu_count = multiprocessing.cpu_count()  # Get the number of CPU cores
    logging.info(f'Using {cpu_count} CPU cores for DataLoader.')

    # Set the DataLoader with optimized number of workers
    train_loader = DataLoader(train_loader.dataset, batch_size=16, shuffle=True, num_workers=cpu_count)
    test_loader = DataLoader(test_loader.dataset, batch_size=16, shuffle=False, num_workers=cpu_count)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():  # Mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total

        train_loss_values.append(epoch_loss)
        train_acc_values.append(epoch_acc)

        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        # Validation step
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(test_loader.dataset)
        val_epoch_acc = 100 * val_correct / val_total

        val_loss_values.append(val_epoch_loss)
        val_acc_values.append(val_epoch_acc)

        logging.info(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%')

        # Adjust learning rate if validation loss plateaus
        scheduler.step(val_epoch_loss)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')
            logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= early_stop_patience:
                logging.info("Early stopping triggered.")
                break

    # Load the best model and evaluate
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    model.eval()
    correct, total = 0, 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    test_acc = 100 * correct / total
    logging.info(f'Test Accuracy: {test_acc:.2f}%')

    # Generate classification report and confusion matrix
    logging.info("\n" + classification_report(true_labels, predictions, target_names=label_encoder.classes_))

    conf_matrix = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(rootPath, 'lstm_confusion_matrix.png'), dpi=500)
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc_values, label='Training Accuracy')
    plt.plot(val_acc_values, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(rootPath, 'lstm_training_validation_accuracy.png'), dpi=500)
    plt.show()

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(rootPath, 'lstm_training_validation_loss.png'), dpi=500)
    plt.show()

if __name__ == "__main__":
    # Detect multiple GPUs and set the device_ids
    num_gpus = torch.cuda.device_count()
    device_ids = list(range(num_gpus))  # Automatically assigns all available GPUs

    train_and_evaluate(num_epochs=50, device_ids=device_ids)
    logging.info("Training and evaluation completed.")
