import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from features import get_vectorizer


# -----------------------------
# Load & Merge Datasets
# -----------------------------
# Original dataset
data1 = pd.read_csv("spam.csv", encoding="latin-1")
data1 = data1[["v1", "v2"]]
data1.columns = ["label", "message"]

# New phishing dataset
data2 = pd.read_csv("phishing_data.csv")

# Combine both
data = pd.concat([data1, data2], ignore_index=True)

# Clean
data = data.dropna(subset=["message"])


# -----------------------------
# Prepare Data
# -----------------------------
X_text = data["message"]
y = data["label"]

vectorizer = get_vectorizer()
X = vectorizer.fit_transform(X_text)


# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# -----------------------------
# Train Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)


# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -----------------------------
# Save Model
# -----------------------------
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved ✅")