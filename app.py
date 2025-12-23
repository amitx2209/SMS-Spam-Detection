import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from features import get_vectorizer

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="SMS Spam Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# CSS – HARD RESET
# -----------------------------
st.markdown(
    """
    <style>
    /* Hide Streamlit header & footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Remove ALL padding */
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Full viewport */
    html, body, [class*="css"]  {
        height: 100%;
        overflow: hidden;
    }

    /* Background */
    .stApp {
        background-image: url("https://png.pngtree.com/thumb_back/fw800/background/20250828/pngtree-wireframe-mesh-with-glowing-nodes-on-dark-background-futuristic-digital-network-image_18256772.webp");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    /* Center container */
    .center {
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Glass card */
    .card {
        background: rgba(0, 0, 0, 0.65);
        backdrop-filter: blur(14px);
        padding: 36px;
        border-radius: 16px;
        width: 90%;
        max-width: 620px;
        color: white;
        box-shadow: 0 15px 40px rgba(0,0,0,0.8);
        border: 1px solid rgba(255,255,255,0.15);
    }

    textarea {
        background-color: #111 !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* Button */
    div.stButton > button {
        background-color: #0f172a;
        color: white;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 0.6em 1.6em;
        transition: 0.3s;
        box-shadow: 0 0 12px rgba(59,130,246,0.5);
    }

    div.stButton > button:hover {
        background-color: #2563eb;
        box-shadow: 0 0 25px rgba(59,130,246,1);
        transform: translateY(-2px);
    }

    .footer-text {
        margin-top: 18px;
        font-size: 0.85rem;
        color: #c7d2fe;
        text-align: center;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Model
# -----------------------------
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

X_text = data["message"]
y = data["label"]

vectorizer = get_vectorizer()
X = vectorizer.fit_transform(X_text)

model = MultinomialNB()
model.fit(X, y)

# -----------------------------
# UI
# -----------------------------
st.markdown("<div class='center'>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.title("SMS Spam Detection")
st.write(
    "This application uses Machine Learning to classify SMS messages as **Spam** or **Ham (Not Spam)**."
)

message = st.text_area("Enter an SMS message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict(vectorizer.transform([message]))[0]
        if prediction == "spam":
            st.error("🚨 This message is SPAM.")
        else:
            st.success("✅ This message is HAM (Not Spam).")

st.markdown(
    """
    <div class="footer-text">
        Project by <b>Amit Sharma</b><br>
        GitHub: https://github.com/amitx2209/SMS-Spam-Detection
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
