import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel, BertConfig
import pandas as pd
import re

# Set up paths for logging and saving plots
root_path = os.path.abspath('.')
log_file = os.path.join(root_path, 'bert_cnn_train.log')

# Configure logging
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

# Define a function to extract words based on count
def extract_words_with_count(text):
    """Extract words from 'word(count)' format."""
    if isinstance(text, str):
        matches = re.findall(r'(\w+)\((\d+)\)', text)
        words = [word for word, count in matches for _ in range(int(count))]
        return ' '.join(words)
    return ''  # If invalid input, return empty string

# Define the BERT + CNN model
class BertCNN(nn.Module):
    def __init__(self, bert_model, cnn_output_size, num_classes):
        super(BertCNN, self).__init__()
        self.bert = bert_model
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * cnn_output_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

        x = last_hidden_state.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Update prepare_data function with extraction
def prepare_data(file_path, tokenizer):
    # Load dataset
    df = pd.read_csv(file_path)

    # Process the 'x' column and generate 'word'
    df['word'] = df['x'].apply(extract_words_with_count)

    # Extract text and labels
    texts = df['word'].tolist()
    labels = df['genre'].tolist()

    # Tokenize the texts
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

    num_workers = 48  # Use all CPU cores
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, label_encoder

# Training and evaluation function
def train_and_evaluate(model, train_loader, test_loader, num_epochs, device, label_encoder):
    logging.info("Starting training and evaluation...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    train_loss_values, train_acc_values = [], []
    val_loss_values, val_acc_values = [], []

    best_val_loss = float('inf')
    early_stop_patience = 5
    no_improvement_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        train_loss_values.append(epoch_loss)
        train_acc_values.append(epoch_acc)

        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in test_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / len(test_loader.dataset)
        val_epoch_acc = 100 * val_correct / val_total
        val_loss_values.append(val_epoch_loss)
        val_acc_values.append(val_epoch_acc)

        logging.info(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%')

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), 'best_bert_cnn_model.pth')
            logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= early_stop_patience:
                logging.info("Early stopping triggered.")
                break

        scheduler.step()

    # Test model and log results
    model.load_state_dict(torch.load('best_bert_cnn_model.pth'))
    model.eval()

    correct, total = 0, 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    test_acc = 100 * correct / total
    logging.info(f'Test Accuracy: {test_acc:.2f}%')

    # Generate classification report and confusion matrix
    predicted_labels = label_encoder.inverse_transform(predictions)
    true_labels_mapped = label_encoder.inverse_transform(true_labels)
    logging.info("\n" + classification_report(true_labels_mapped, predicted_labels))
    conf_matrix = confusion_matrix(true_labels_mapped, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.savefig('bert_cnn_confusion_matrix.png', dpi=500)
    plt.show()

    # Plot accuracy and loss curves
    plt.figure()
    plt.plot(train_acc_values, label='Training Accuracy')
    plt.plot(val_acc_values, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('bert_cnn_accuracy.png', dpi=500)
    plt.show()

    plt.figure()
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('bert_cnn_loss.png', dpi=500)
    plt.show()

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('./pretrained_bert_base_uncased/config.json')
    bert_model = BertModel.from_pretrained('./pretrained_bert_base_uncased/model.safetensors', config=config)

    train_loader, test_loader, label_encoder = prepare_data('../data/mxm_msd_genre_pro_simple_stopwords.cls', tokenizer)
    model = BertCNN(bert_model, cnn_output_size=128, num_classes=len(label_encoder.classes_))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model)
    model.to(device)

    train_and_evaluate(model, train_loader, test_loader, num_epochs=50, device=device, label_encoder=label_encoder)
