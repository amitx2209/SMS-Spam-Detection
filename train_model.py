import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from features import get_vectorizer

# Load dataset (correct encoding)
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only useful columns and rename them
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

# Remove empty messages
data = data.dropna(subset=["message"])

# Inputs and labels
X_text = data["message"]
y = data["label"]

# Convert text to numbers
vectorizer = get_vectorizer()
X = vectorizer.fit_transform(X_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
