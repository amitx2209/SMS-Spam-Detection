import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


# -----------------------------
# Load & Merge Data
# -----------------------------
data1 = pd.read_csv("spam.csv", encoding="latin-1")
data1 = data1[["v1", "v2"]]
data1.columns = ["label", "message"]

data2 = pd.read_csv("phishing_data.csv")

data = pd.concat([data1, data2], ignore_index=True)

# Convert labels
data["label"] = data["label"].map({"ham": 0, "spam": 1})

data = data.dropna(subset=["message"])


# -----------------------------
# Prepare Data
# -----------------------------
X = data["message"]
y = data["label"]


# -----------------------------
# Tokenization
# -----------------------------
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100)


# -----------------------------
# Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42
)


# -----------------------------
# Model
# -----------------------------
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64))
model.add(LSTM(64))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# Train
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# -----------------------------
# Evaluate
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("\nAccuracy:", acc)


# -----------------------------
# Save
# -----------------------------
model.save("lstm_model.h5")
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

print("\nLSTM model saved ✅")