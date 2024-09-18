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
from transformers import BertTokenizer, BertModel
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

from scripts.pushbullet_notify import send_pushbullet_notification

# BERT tokenizer and model setup
model_path = '/mnt/parscratch/users/acr23sh/DissertationProject/models/pretrained_bert_base_uncased/'
tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertModel.from_pretrained(model_path)

class LSTMWithBERTEmbeddings(nn.Module):
    def __init__(self, bert_model, hidden_size, num_classes, num_layers=2):
        super(LSTMWithBERTEmbeddings, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(bert_model.config.hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # hidden_size * 2 because of bidirectional LSTM
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        # Freeze the BERT layers if needed
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        out, _ = self.lstm(hidden_state)  # Pass BERT embeddings through LSTM
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(self.relu(out))
        return out


def prepare_data(file_path, tokenizer):
    # Load dataset
    df = pd.read_csv(file_path)

    # Extract text and labels
    texts = df['x'].tolist()
    labels = df['genre'].tolist()

    # Tokenize using BERT tokenizer
    inputs = tokenizer(texts, return_tensors='pt', max_length=512, padding=True, truncation=True)

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Create TensorDataset
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels_encoded))

    # Train-test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, test_loader, len(label_encoder.classes_), label_encoder


def train_and_evaluate(num_epochs):
    logging.info("Starting training and evaluation...")

    # Prepare data
    train_loader, test_loader, num_classes, label_encoder = prepare_data('../data/mxm_msd_genre_pro_no_stopwords.cls', tokenizer)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize model
    model = LSTMWithBERTEmbeddings(bert_model, hidden_size=128, num_classes=num_classes).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

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

        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total

        train_loss_values.append(epoch_loss)
        train_acc_values.append(epoch_acc)

        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        scheduler.step()

        # Validation step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in test_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * input_ids.size(0)
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

    # Final model evaluation
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    model.eval()
    correct = 0
    total = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
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
    plt.savefig('lstm_confusion_matrix.png', dpi=500)
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc_values, label='Training Accuracy')
    plt.plot(val_acc_values, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('lstm_training_validation_accuracy.png', dpi=500)
    plt.show()

    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('lstm_training_validation_loss.png', dpi=500)
    plt.show()

if __name__ == "__main__":
    train_and_evaluate(num_epochs=50)
    logging.info("Training and evaluation completed.")
    send_pushbullet_notification("Task completed", "Your task on the server has finished.")
