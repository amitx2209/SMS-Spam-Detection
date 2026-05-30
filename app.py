import streamlit as st
import pickle
import numpy as np
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Intelligent SMS Threat Detection System",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------

# Load Models

# -----------------------------

@st.cache_resource
def load_models():

    import os
    import tensorflow as tf

    st.sidebar.write(f"TensorFlow: {tf.__version__}")
    st.sidebar.write(f"Working Dir: {os.getcwd()}")

    # Verify files exist
    required_files = [
        "spam_model.pkl",
        "tfidf_vectorizer.pkl",
        "tokenizer.pkl",
        "lstm_model.h5"
    ]

    for file in required_files:
        if not os.path.exists(file):
            st.error(f"Missing file: {file}")
            st.stop()

    # Load Naive Bayes components
    nb_model = pickle.load(open("spam_model.pkl", "rb"))

    vectorizer = pickle.load(
        open("tfidf_vectorizer.pkl", "rb")
    )

    tokenizer = pickle.load(
        open("tokenizer.pkl", "rb")
    )

    # Load LSTM model
    try:
        lstm_model = load_model(
            "lstm_model.h5",
            compile=False
        )

    except Exception as e:
        st.error("Failed to load LSTM model")
        st.exception(e)
        st.stop()

    return nb_model, vectorizer, lstm_model, tokenizer

nb_model, vectorizer, lstm_model, tokenizer = load_models()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🔎 Message Insights")

    if "msg" in st.session_state:
        msg = st.session_state.msg

        st.markdown("### 📌 Message Stats")
        st.metric("Words", len(msg.split()))
        st.metric("Digits", sum(c.isdigit() for c in msg))
        st.metric("Links", len(re.findall(r'http[s]?://', msg)))

        if "spam_prob" in st.session_state:
            st.markdown("### 🚨 Risk Level")
            st.progress(int(st.session_state.spam_prob))

        if "agreement" in st.session_state:
            st.markdown("### 🤝 Model Agreement")
            if st.session_state.agreement:
                st.success("Models agree")
            else:
                st.warning("Models disagree")

# -----------------------------
# Explainability
# -----------------------------
def get_top_words(message):
    vector = vectorizer.transform([message.lower()])
    if vector.nnz == 0:
        return []

    feature_names = np.array(vectorizer.get_feature_names_out())
    indices = vector.nonzero()[1]

    spam_probs = nb_model.feature_log_prob_[1]
    ham_probs = nb_model.feature_log_prob_[0]

    words = []
    for idx in indices:
        word = feature_names[idx]

        if spam_probs[idx] > ham_probs[idx]:
            words.append((word, "red", "spam-leaning"))
        else:
            words.append((word, "green", "ham-leaning"))

    return words[:5]

# -----------------------------
# UI
# -----------------------------
st.title("📩 Intelligent SMS Threat Detection System")

st.markdown("""
This system combines **Machine Learning (Naive Bayes)** and  
**Deep Learning (LSTM)** to detect spam and phishing messages.

It provides:
- 🔍 Multi-model prediction comparison  
- ⚠️ Ambiguity detection  
- 🧠 Explainable AI outputs  
- 🎯 Confidence-based final decision  

Designed for **real-world fraud detection and secure communication systems**.
""")

st.markdown("---")

# -----------------------------
# Model Info
# -----------------------------
st.markdown("### 📘 Model Overview")

col1, col2 = st.columns(2)

with col1:
    st.info("""
**Naive Bayes (ML)**  
• TF-IDF based word probability model  
• Fast and efficient  
• Works best on known keyword patterns  
""")

with col2:
    st.info("""
**LSTM (Deep Learning)**  
• Learns sequential patterns  
• Detects contextual meaning  
• Strong for phishing detection  
""")

st.markdown("---")

# -----------------------------
# Model Performance
# -----------------------------
st.markdown("### 📊 Model Performance")

st.write("""
Naive Bayes Accuracy: ~98%  
LSTM Accuracy: ~98.3%  

Both models perform strongly, with LSTM showing better capability in detecting
contextual and modern phishing patterns.
""")

st.markdown("---")

# -----------------------------
# Examples
# -----------------------------
st.markdown("### ✉️ Try Realistic Examples")

col1, col2 = st.columns(2)

if col1.button("💳 Banking Phishing"):
    st.session_state.msg = (
        "URGENT: Your SBI account is suspended. Verify now at "
        "http://secure-update-bank.com to restore access."
    )

if col2.button("🎁 Lottery Scam"):
    st.session_state.msg = (
        "Congratulations! You won ₹5,00,000. Call now to claim your reward."
    )

col3, col4 = st.columns(2)

if col3.button("💬 Casual Chat"):
    st.session_state.msg = (
        "Hey bro, I’ll reach by 6. Let’s meet near the tea stall."
    )

if col4.button("📅 Work Reminder"):
    st.session_state.msg = (
        "Reminder: Meeting at 10 AM tomorrow. Please review the slides."
    )

# -----------------------------
# Input
# -----------------------------
message = st.text_area(
    "Enter your message:",
    value=st.session_state.get("msg", ""),
    height=120
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Analyze Message"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message")

    else:
        st.session_state.msg = message

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

        st.session_state.agreement = (nb_pred == lstm_pred)
        st.session_state.spam_prob = max(nb_conf, lstm_conf)

        # -----------------------------
        # Results
        # -----------------------------
        st.markdown("### 🔍 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🤖 Naive Bayes")
            st.write("🚨 SPAM" if nb_pred == "spam" else "✅ HAM")
            st.write(f"Confidence: {nb_conf:.2f}%")

        with col2:
            st.markdown("#### 🧠 LSTM")
            st.write("🚨 SPAM" if lstm_pred == "spam" else "✅ HAM")
            st.write(f"Confidence: {lstm_conf:.2f}%")

        # -----------------------------
        # Ambiguity + Reason
        # -----------------------------
        if nb_pred != lstm_pred:
            st.markdown("### ⚠️ Ambiguous Case Detected")

            st.warning("Models disagree — message may contain advanced phishing structure.")

            if lstm_conf > nb_conf:
                st.info("🧠 LSTM is more confident due to contextual understanding (sequence + intent).")
            else:
                st.info("🤖 Naive Bayes is more confident due to strong keyword patterns.")

        # -----------------------------
        # Final Decision
        # -----------------------------
        final_pred = lstm_pred if lstm_conf > nb_conf else nb_pred

        st.markdown("### 🎯 Final Decision")
        st.success(final_pred.upper())

        if lstm_conf > nb_conf:
            st.caption("Decision based on higher confidence from LSTM model")
        else:
            st.caption("Decision based on higher confidence from Naive Bayes model")

        # -----------------------------
        # Explainability
        # -----------------------------
        st.markdown("### 🧠 Key Influencing Words")

        st.caption("These words influenced the prediction based on their statistical importance in training data.")

        for word, color, label in get_top_words(message):
            st.markdown(
                f"<span style='color:{color}; font-weight:bold'>{word}</span> "
                f"<span style='color:gray'>({label})</span>",
                unsafe_allow_html=True
            )

st.markdown("---")

# -----------------------------
# Real World Applications
# -----------------------------
st.markdown("### 🌍 Real-World Applications")

st.write("""
• SMS fraud and phishing detection  
• Banking security systems  
• Email spam filtering  
• Telecom threat monitoring  
""")

st.markdown("---")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div style='text-align: center'>

### 📚 System Summary

A hybrid AI system combining Machine Learning and Deep Learning  
to deliver accurate, explainable spam detection.

✔ Model comparison  
✔ Ambiguity reasoning  
✔ Confidence-based decision  

</div>
""", unsafe_allow_html=True)