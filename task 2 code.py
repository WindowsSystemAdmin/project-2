import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load and preprocess data (with a larger subset)
print("Loading data...")
df = pd.read_csv("amazon_reviews.csv")
df = df[['reviewText', 'overall']].dropna()
df['overall'] = df['overall'].astype(str)

# Use a larger subset for better results (adjust as needed)
df = df.sample(n=3000, random_state=42)  # Increased from 1000 to 3000

# Split into train and test sets
print("Splitting data...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['reviewText'].tolist(), df['overall'].tolist(), test_size=0.2, random_state=42
)

# Load BERT model and tokenizer
print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()


# Process text in batches to improve efficiency
def get_bert_embeddings_batch(texts, batch_size=16):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize the batch
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                           max_length=128, padding="max_length")

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract [CLS] token embeddings
        batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)


# Extract embeddings with batching
print("Extracting BERT embeddings for training data...")
train_embeddings = get_bert_embeddings_batch(train_texts)

print("Extracting BERT embeddings for test data...")
test_embeddings = get_bert_embeddings_batch(test_texts)

# Convert labels to integers
train_labels_int = [int(float(label)) - 1 for label in train_labels]
test_labels_int = [int(float(label)) - 1 for label in test_labels]

# Print class distribution to understand imbalance
print("Class distribution in test set:")
for i in range(5):
    print(f"Class {i + 1}: {test_labels_int.count(i)}")

# Train a classifier on the embeddings
print("Training classifier...")
classifier = LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                                class_weight='balanced')  # Added class_weight parameter

# Important: Actually fit the classifier - this line was missing before
classifier.fit(train_embeddings, train_labels_int)

# Evaluate
print("Evaluating...")
predictions = classifier.predict(test_embeddings)
print(classification_report(test_labels_int, predictions, digits=4))