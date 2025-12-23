import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from features import get_vectorizer

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="SMS Spam Detection",
    layout="centered"
)

# -----------------------------
# Background + styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://png.pngtree.com/thumb_back/fw800/background/20250828/pngtree-wireframe-mesh-with-glowing-nodes-on-dark-background-futuristic-digital-network-image_18256772.webp");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    .overlay {
        background-color: rgba(0, 0, 0, 0.65);
        min-height: 100vh;
        padding: 60px 20px;
    }

    .card {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 35px;
        border-radius: 12px;
        max-width: 600px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load and train model
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
# UI Layout
# -----------------------------
st.markdown("<div class='overlay'>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.title("SMS Spam Detection")
st.write(
    "This application uses Machine Learning to determine whether an SMS message "
    "is **Spam** or **Ham (Not Spam)**."
)

user_input = st.text_area("Enter an SMS message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == "spam":
            st.error("This message is classified as SPAM.")
        else:
            st.success("This message is classified as HAM (Not Spam).")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
