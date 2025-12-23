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
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    /* Remove default padding */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }

    /* Background */
    .stApp {
        background-image: url("https://png.pngtree.com/thumb_back/fw800/background/20250828/pngtree-wireframe-mesh-with-glowing-nodes-on-dark-background-futuristic-digital-network-image_18256772.webp");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    /* Center everything */
    .center-container {
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Glass card */
    .card {
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 40px;
        border-radius: 16px;
        width: 100%;
        max-width: 620px;
        color: white;
        box-shadow: 0 12px 35px rgba(0,0,0,0.7);
        border: 1px solid rgba(255,255,255,0.15);
    }

    /* Text area */
    textarea {
        background-color: #121212 !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* Neon button */
    div.stButton > button {
        background-color: #111827;
        color: #ffffff;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(59,130,246,0.4);
    }

    div.stButton > button:hover {
        background-color: #2563eb;
        color: white;
        box-shadow: 0 0 20px rgba(59,130,246,0.9);
        transform: translateY(-1px);
    }

    /* Footer */
    .footer {
        margin-top: 25px;
        font-size: 0.85rem;
        color: #cbd5f5;
        text-align: center;
        opacity: 0.9;
    }

    .footer a {
        color: #60a5fa;
        text-decoration: none;
    }

    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load & train model
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
st.markdown("<div class='center-container'>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.title("SMS Spam Detection")
st.write(
    "This application uses Machine Learning to determine whether an SMS message is **Spam** or **Ham (Not Spam)**."
)

user_input = st.text_area("Enter an SMS message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == "spam":
            st.error("🚨 This message is classified as SPAM.")
        else:
            st.success("✅ This message is classified as HAM (Not Spam).")

# -----------------------------
# Footer (Verification)
# -----------------------------
st.markdown(
    """
    <div class="footer">
        Project by <b>Amit Sharma</b><br>
        GitHub: <a href="https://github.com/amitx2209/SMS-Spam-Detection" target="_blank">
        SMS Spam Detection Repository</a><br>
        Streamlit App: <i>Use this page to verify live deployment</i>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
