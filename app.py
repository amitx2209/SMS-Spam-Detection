
import pickle
import numpy as np
import streamlit as st
import pandas as pd


def tokens_to_dataframe(tokens, label):
    return pd.DataFrame({
        "Token": tokens[label],
        "Score": list(range(len(tokens[label]), 0, -1))
    })




# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="smslogo.jpeg",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown("☰ **Menu** → Open sidebar for model details")

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------

with st.sidebar:
    st.title("ℹ️ Model Information")
    st.markdown(
        """
        • **Deployed Model:** Multinomial Naive Bayes  
        • **Experimented Models:** Logistic Regression, SVM, Random Forest  
        • **Deployment Rationale:** Low inference time and reliable performance
        """
    )


st.sidebar.title("ℹ️ Project Information")

st.sidebar.subheader("Dataset")
st.sidebar.write("""
- SMS Spam Collection (UCI Repository)  
- 5,572 messages  
- Classes: Spam / Ham  
""")

st.sidebar.subheader("Preprocessing")
st.sidebar.write("""
- Lowercasing  
- Punctuation removal  
- Text cleaning  
""")

st.sidebar.subheader("Feature Engineering")
st.sidebar.write("""
- TF-IDF Vectorization  
- Unigrams & Bigrams  
- Top 1000 features  
""")

st.sidebar.subheader("Workflow")
st.sidebar.write("""
1. User enters SMS  
2. TF-IDF transformation  
3. Naive Bayes prediction  
4. Confidence & explanation  
""")

st.sidebar.subheader("Limitations")
st.sidebar.write("""
- May misclassify ambiguous messages  
- Depends on historical patterns  
""")





# -------------------------------------------------
# Minimal dark UI (no image, no scroll)
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* Safe Streamlit styling – sidebar friendly */

    /* App background */
    .stApp {
        background-color: #0b0f14;
    }

    /* Main content padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Card UI */
    .card {
        background-color: #111827;
        padding: 28px 32px;
        border-radius: 14px;
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
        color: #e5e7eb;
        border: 1px solid rgba(255,255,255,0.08);
    }

/* Footer text */
.meta {
    font-size: 0.85rem;
    color: #9ca3af;
    margin-top: 16px;
    text-align: center;
}

    </style>
    """,
    unsafe_allow_html=True
)


    
# -------------------------------------------------
# Safe model loading & training
# -------------------------------------------------



def get_top_tokens(model, vectorizer, n=10):
    feature_names = vectorizer.get_feature_names_out()
    class_labels = model.classes_

    top_tokens = {}

    for i, label in enumerate(class_labels):
        top_indices = np.argsort(model.feature_log_prob_[i])[::-1][:n]
        tokens = [feature_names[j] for j in top_indices]
        top_tokens[label] = tokens

    return top_tokens


@st.cache_resource
def load_artifacts():
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_artifacts()
top_tokens = get_top_tokens(model, vectorizer, n=10)

st.sidebar.subheader("Token Frequency Visualization")

spam_df = tokens_to_dataframe(top_tokens, "spam")
ham_df  = tokens_to_dataframe(top_tokens, "ham")

st.sidebar.markdown("🚨 **Top Spam Tokens**")
st.sidebar.bar_chart(
    spam_df.set_index("Token")["Score"]
)

st.sidebar.markdown("✅ **Top Ham Tokens**")
st.sidebar.bar_chart(
    ham_df.set_index("Token")["Score"]
)




# -------------------------------------------------
# UI
# -------------------------------------------------
st.markdown("<div class='center'>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)


st.title("SMS Spam Detection")
st.write(
    "This application uses a Machine Learning model to classify SMS messages "
    "as **Spam** or **Ham (Not Spam)**."
)

st.subheader("Example Messages")
st.write("""
- **Spam:** Win a free prize now! Call urgently.  
- **Ham:** Are we meeting tomorrow at 10 AM?  
""")


message = st.text_area("Enter an SMS message", height=120)

# Add JS to select all text when the textarea is clicked or focused
st.markdown(
    """
    <script>
    const textarea = window.parent.document.querySelector('textarea[key="sms_input"]');
    if(textarea){
        textarea.addEventListener('focus', function() {
            this.select();
        });
        textarea.addEventListener('click', function() {
            this.select();
        });
    }
    </script>
    """,
    unsafe_allow_html=True
)


predict_clicked = st.button(
    "Predict"
)

if predict_clicked and model is not None:
    with st.spinner("Analyzing message..."):
        input_vector = vectorizer.transform([message])
        prediction = model.predict(input_vector)[0]
        confidence = max(model.predict_proba(input_vector)[0]) * 100

    # -------------------------------
    # Show prediction result
    # -------------------------------
    if prediction == "spam":
        st.error(f" 🚨 Prediction: SPAM \n       📊 Confidence: {confidence:.2f}%")
        bar_color = "#dc2626"
    else:
        st.success(f" ✅ Prediction: HAM (Not Spam) \n       📊 Confidence: {confidence:.2f}%")
        bar_color = "#16a34a"

    # -------------------------------
    # Confidence bar
    # -------------------------------
    st.markdown(f"""
    <div style="background-color:#1f2937; border-radius:8px; padding:2px; width:100%;">
        <div style="background-color:{bar_color}; width:{confidence}%; padding:6px;
                    border-radius:6px; text-align:right; color:#ffffff; font-weight:bold;">
            {confidence:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------
    # Confidence interpretation
    # -------------------------------
    if confidence >= 80:
        st.success("🔍 High confidence prediction – the model is very certain.")
    elif confidence >= 60:
        st.info("🔍 Moderate confidence prediction – the model is fairly confident.")
    else:
        st.warning("🔍 Low confidence prediction – the message may be ambiguous.")




# -------------------------------------------------
# Model info (verification-friendly)
# -------------------------------------------------
st.markdown("---")
st.subheader("Model Information")
st.write(f"- Algorithm: Multinomial Naive Bayes")
st.write(f"- Vectorization: TF-IDF")
st.write(f"- Dataset size: 5,572 SMS messages")
st.write("- Test Accuracy: **~98% (on held-out test set)**") 






st.markdown(
    """
    <div class="meta">
        Developed by Amit Sharma<br>
        GitHub Repository:
        <a href="https://github.com/amitx2209/SMS-Spam-Detection.git"
           target="_blank"
           style="color:#60a5fa; text-decoration:none;">
           https://github.com/amitx2209/SMS-Spam-Detection.git
        </a><br>
        Verified via Streamlit Cloud deployment
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
