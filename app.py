import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from features import get_vectorizer

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detection",
    layout="centered"
)

# Background image styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-vector/network-mesh-wire-digital-technology-background_1017-27428.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    .content-box {
        background-color: rgba(255, 255, 255, 0.92);
        padding: 25px;
        border-radius: 10px;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load dataset and train model
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

X_text = data["message"]
y = data["label"]

vectorizer = get_vectorizer()
X = vectorizer.fit_transform(X_text)

model = MultinomialNB()
model.fit(X, y)

# App UI
st.markdown('<div class="content-box">', unsafe_allow_html=True)

st.title("SMS Spam Detection")
st.write("Enter an SMS message below to check whether it is spam or not.")

user_input = st.text_area("Your SMS message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == "spam":
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")

st.markdown("</div>", unsafe_allow_html=True)
