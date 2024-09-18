import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Get the current script's directory
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.insert(0, rootPath)

# Set up logging
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

from scripts.bagofwords.data_preparation01 import prepare_data
from scripts.pushbullet_notify import send_pushbullet_notification


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence length dimension (batch_size, 1, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(self.relu(out))
        return out


def train_and_evaluate(num_epochs):
    logging.info("Starting training and evaluation...")
    train_loader, test_loader, input_size, num_classes = prepare_data()

    df = pd.read_csv('../data/mxm_msd_genre_pro_no_stopwords.cls')
    label_encoder = LabelEncoder()
    label_encoder.fit(df['genre'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = LSTMModel(input_size=input_size, hidden_size=128, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    best_val_loss = float('inf')
    no_improvement_epochs = 0
    early_stop_patience = 5

    train_loss_values, train_acc_values = [], []
    val_loss_values, val_acc_values = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total

        train_loss_values.append(epoch_loss)
        train_acc_values.append(epoch_acc)

        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        scheduler.step()

        # Validate the model
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

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

    logging.info(f'Best Validation Accuracy: {best_acc:.2f}%')

    # Testing the model
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    model.eval()
    correct = 0
    total = 0
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

    # Classification report and confusion matrix
    logging.info("\n" + classification_report(true_labels, predictions, target_names=label_encoder.classes_))

    conf_matrix = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(rootPath, 'lstm_confusion_matrix.png'), dpi=500)
    plt.show()

    # Accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc_values, label='Training Accuracy')
    plt.plot(val_acc_values, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(rootPath, 'lstm_training_validation_accuracy.png'), dpi=500)
    plt.show()

    # Loss plot
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
    train_and_evaluate(num_epochs=50)
    logging.info("Training and evaluation completed. Sending notification...")
    send_pushbullet_notification("Task completed", "Your task on the server has finished.")
    logging.info("Notification sent successfully.")
