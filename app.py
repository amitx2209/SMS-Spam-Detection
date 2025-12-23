import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from features import get_vectorizer
import pandas as pd

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

# Streamlit interface
st.title("SMS Spam Detection")
st.write("Enter an SMS message and check if it's SPAM or HAM.")

user_input = st.text_area("Your SMS here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        if prediction == "spam":
            st.error("🚨 SPAM")
        else:
            st.success("✅ HAM")
