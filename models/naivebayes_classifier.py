import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# Setup paths for logging and output
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.insert(0, rootPath)

log_file = os.path.join(rootPath, 'nb_train.log')
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

# Import any necessary utility functions
from scripts.pushbullet_notify import send_pushbullet_notification

# Main function for training and evaluation
def train_and_evaluate():
    logging.info("Starting training and evaluation...")

    # Data preparation
    df = pd.read_csv('../data/mxm_msd_genre_pro_no_stopwords.cls')

    # Print out the column names to verify
    logging.info(f"Columns in the dataset: {df.columns}")

    # Splitting the 'word' and 'genre' columns
    X = df['x']  # Assuming 'x' holds the word(count) data after extraction
    y = df['genre']

    # Label encoding for the genre column
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Splitting into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline to vectorize the data and then apply Naive Bayes
    model = make_pipeline(CountVectorizer(), MultinomialNB())

    # Training the model
    model.fit(X_train, y_train)
    logging.info("Model training completed.")

    # Evaluating on the test set
    predictions = model.predict(X_test)
    test_acc = (predictions == y_test).mean() * 100
    logging.info(f'Test Accuracy: {test_acc:.2f}%')

    # Generate classification report and confusion matrix
    report = classification_report(y_test, predictions, target_names=label_encoder.classes_)
    logging.info("\n" + report)
    print(report)

    # Confusion matrix generation
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Naive Bayes - Confusion Matrix")
    plt.savefig(os.path.join(rootPath, 'naive_bayes_confusion_matrix.png'), dpi=500)
    plt.show()

    # Plotting the classification report values (precision, recall, f1-score)
    report_dict = classification_report(y_test, predictions, output_dict=True)
    categories = list(report_dict.keys())[:-3]  # Avoid 'accuracy', 'macro avg', and 'weighted avg'

    precision = [report_dict[cat]['precision'] for cat in categories]
    recall = [report_dict[cat]['recall'] for cat in categories]
    f1_score = [report_dict[cat]['f1-score'] for cat in categories]

    plt.figure(figsize=(12, 8))
    x = range(len(categories))
    plt.bar(x, precision, width=0.2, label='Precision', align='center')
    plt.bar([p + 0.2 for p in x], recall, width=0.2, label='Recall', align='center')
    plt.bar([p + 0.4 for p in x], f1_score, width=0.2, label='F1-Score', align='center')
    plt.xlabel('Categories')
    plt.ylabel('Scores')
    plt.xticks([p + 0.2 for p in x], categories, rotation=45, ha="right")
    plt.title('Naive Bayes - Precision, Recall, and F1-Score per Category')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(rootPath, 'naive_bayes_classification_report.png'), dpi=500)
    plt.show()


if __name__ == "__main__":
    train_and_evaluate()
    logging.info("Training and evaluation completed. Sending notification...")
    send_pushbullet_notification("Task completed", "Naive Bayes task on the server has finished.")
    logging.info("Notification sent successfully.")
