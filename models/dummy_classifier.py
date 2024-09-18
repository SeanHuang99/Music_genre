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
from sklearn.feature_extraction.text import CountVectorizer
import re  # For extracting words from 'x' column
import numpy as np

# Extract words from 'word(count)' format
def extract_words_with_count(text):
    """Extract words and repeat them based on their counts."""
    if isinstance(text, str):
        matches = re.findall(r'(\w+)\((\d+)\)', text)
        words = [word for word, count in matches for _ in range(int(count))]
        return ' '.join(words)
    return ''  # Return an empty string if the input is not valid

# Setting up directories
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.insert(0, rootPath)

# Logging setup
log_file = os.path.join(rootPath, 'dummy_classifier_train.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Stream logs to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

from scripts.pushbullet_notify import send_pushbullet_notification

def train_and_evaluate():
    logging.info("Starting training and evaluation...")

    try:
        # Load the dataset
        data_file_path = os.path.join(rootPath, 'data/mxm_msd_genre_pro_no_stopwords.cls')
        df = pd.read_csv(data_file_path)

        # Extract words from the 'x' column and prepare features
        df['word'] = df['x'].apply(extract_words_with_count)

        # Split into training and testing data
        X = df['word']  # Text data
        y = df['genre']  # Labels

        # Use CountVectorizer to transform text data into feature vectors
        vectorizer = CountVectorizer()
        X_transformed = vectorizer.fit_transform(X)

        # Encode the genre labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split the dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.2, random_state=42)
        logging.info(f"Data split into train and test sets. Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

        # Define DummyClassifier
        model = DummyClassifier(strategy="most_frequent")

        # Train the model
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")

        # Evaluate the model
        predictions = model.predict(X_test)
        test_acc = model.score(X_test, y_test) * 100
        logging.info(f'Test Accuracy: {test_acc:.2f}%')
    except Exception as e:
        logging.error(f"Error during training or evaluation: {e}")
        return

    try:
        # Generate classification report
        class_report = classification_report(y_test, predictions, target_names=label_encoder.classes_, output_dict=True)
        logging.info("\n" + classification_report(y_test, predictions, target_names=label_encoder.classes_))

        # Plot the precision, recall, and F1-scores
        metrics = ['precision', 'recall', 'f1-score']
        df_metrics = pd.DataFrame(class_report).transpose()[metrics].iloc[:-3]  # Exclude avg/total rows

        plt.figure(figsize=(12, 8))
        df_metrics.plot(kind='bar', figsize=(12, 8))
        plt.title('Classification Report Metrics per Class')
        plt.xticks(rotation=45, horizontalalignment="right")
        plt.ylabel('Score')
        plt.tight_layout()
        plt.savefig(os.path.join(rootPath, 'classification_report_metrics.png'), dpi=300)
        plt.show()
        logging.info("Classification report metrics plotted and saved successfully.")

        # Plot confusion matrix
        conf_matrix = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
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
