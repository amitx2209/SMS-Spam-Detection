from flask import Flask, request, jsonify
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load models
# -----------------------------
nb_model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

lstm_model = load_model("lstm_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# -----------------------------
# Create app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Prediction function
# -----------------------------
def predict_message(message):
    # NB
    vec = vectorizer.transform([message.lower()])
    nb_pred = nb_model.predict(vec)[0]
    nb_prob = nb_model.predict_proba(vec)[0]
    nb_conf = nb_prob[list(nb_model.classes_).index(nb_pred)] * 100

    # LSTM
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=100)

    lstm_prob = lstm_model.predict(padded)[0][0]
    lstm_pred = "spam" if lstm_prob > 0.5 else "ham"
    lstm_conf = lstm_prob * 100 if lstm_pred == "spam" else (1 - lstm_prob) * 100

    # Final decision
    final_pred = lstm_pred if lstm_conf > nb_conf else nb_pred

    return {
        "naive_bayes": {
            "prediction": nb_pred,
            "confidence": round(nb_conf, 2)
        },
        "lstm": {
            "prediction": lstm_pred,
            "confidence": round(lstm_conf, 2)
        },
        "final_prediction": final_pred
    }

# -----------------------------
# API Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    result = predict_message(message)

    return jsonify(result)

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)