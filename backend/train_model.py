import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB

print("Starting Training...")

# Load dataset
df = pd.read_csv("../dataset/spam.csv", encoding="latin-1")

# Keep only needed columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Features and labels
X = df['message']
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Dataset Split Completed")

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1,2)
)

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

print("Vectorization Completed")

# Train model
model = MultinomialNB()

print("Training Model...")

model.fit(X_train_vector, y_train)

print("Training Finished")

# Predictions
y_pred = model.predict(X_test_vector)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Save fresh model
joblib.dump(model, "../model/spam_model.pkl")

# Save fresh vectorizer
joblib.dump(vectorizer, "../model/vectorizer.pkl")

print("Fresh Model Saved Successfully")


import joblib

# Load trained model
model = joblib.load("../model/spam_model.pkl")

# Load vectorizer
vectorizer = joblib.load("../model/vectorizer.pkl")

print("Model Loaded Successfully")

# Test messages
messages = [
    "WIN CASH NOW!!!",
    "Free lottery entry available",
    "Claim your reward now",
    "Call me when you reach home",
    "Meeting at 5 PM today",
    "URGENT! You won a free iPhone",
    "Hey bro where are you?"
]

# Convert text into vectors
message_vectors = vectorizer.transform(messages)

# Predict
predictions = model.predict(message_vectors)

print("\nPredictions Started...\n")

# Show results
for i in range(len(messages)):

    print("Message:")
    print(messages[i])

    print("Raw Prediction:", predictions[i])

    if int(predictions[i]) == 1:
        print("Prediction: SPAM")

    else:
        print("Prediction: HAM")

    print("----------------------")
