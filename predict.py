import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from features import get_vectorizer

# Load and prepare dataset
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

X_text = data["message"]
y = data["label"]

# Train model
vectorizer = get_vectorizer()
X = vectorizer.fit_transform(X_text)

model = MultinomialNB()
model.fit(X, y)

# Test with your own message
while True:
    msg = input("\nEnter SMS (or type exit): ")
    if msg.lower() == "exit":
        break

    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)[0]

    print("Prediction:", prediction.upper())
