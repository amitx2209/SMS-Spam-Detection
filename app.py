import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import get_vectorizer

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# Minimal dark UI (no image, no scroll)
# -------------------------------------------------
st.markdown(
    """
    <style>
    header, footer {visibility: hidden;}

    html, body {
        height: 100%;
        background-color: #0b0f14;
        overflow: hidden;
    }

    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    .center {
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .card {
        background-color: #111827;
        padding: 28px 32px;
        border-radius: 14px;
        width: 100%;
        max-width: 520px;
        color: #e5e7eb;
        box-shadow: 0 12px 30px rgba(0,0,0,0.7);
        border: 1px solid rgba(255,255,255,0.08);
    }

    textarea {
        background-color: #020617 !important;
        color: #e5e7eb !important;
        border-radius: 8px !important;
        border: 1px solid #1f2937 !important;
    }

    div.stButton > button {
        width: 100%;
        background-color: #020617;
        color: #e5e7eb;
        border: 1px solid #2563eb;
        border-radius: 8px;
        padding: 0.6em;
        font-size: 1rem;
    }

    div.stButton > button:hover {
        background-color: #1e3a8a;
    }

    .meta {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 12px;
        text-align: center;
    }

    /* Media query for mobile screens */
    @media (max-width: 480px) {
        .card {
            padding: 16px 12px;
        }
        h1, h2, h3, .stMarkdown {
            font-size: 0.95rem !important;
        }
        div.stButton > button {
            font-size: 0.9rem !important;
            padding: 0.45em;
        }
        textarea {
            height: 100px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)
    
# -------------------------------------------------
# Safe model loading & training
# -------------------------------------------------
@st.cache_resource
def load_model():
    try:
        data = pd.read_csv("spam.csv", encoding="latin-1")
        data = data[["v1", "v2"]]
        data.columns = ["label", "message"]

        X_train, X_test, y_train, y_test = train_test_split(
            data["message"], data["label"], test_size=0.25, random_state=42
        )

        vectorizer = get_vectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test_vec))
        return model, vectorizer, accuracy

    except Exception:
        return None, None, None


model, vectorizer, accuracy = load_model()

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

    if prediction == "spam":
        st.error(f" 🚨 Prediction: SPAM \n       📊 Confidence: {confidence:.2f}%")
        bar_color = "#dc2626"  # Tailwind red-600
    else:
        st.success(f" ✅ Prediction: HAM (Not Spam) \n       📊 Confidence: {confidence:.2f}%")
        bar_color = "#16a34a"  # Tailwind green-600

  # Custom colored confidence bar
    st.markdown(f"""
    <div style="background-color:#1f2937; border-radius:8px; padding:2px; width:100%;">
        <div style="background-color:{bar_color}; width:{confidence}%; padding:6px; border-radius:6px; text-align:right; color:#ffffff; font-weight:bold;">
            {confidence:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# Model info (verification-friendly)
# -------------------------------------------------
st.markdown("---")
st.subheader("Model Information")
st.write(f"- Algorithm: Multinomial Naive Bayes")
st.write(f"- Vectorization: TF-IDF")
st.write(f"- Dataset size: 5,572 SMS messages")
st.write(f"- Test accuracy: **{accuracy:.4f}**")

st.markdown(
    """
    <div class="meta">
        Developed by Amit Sharma<br>
        Verified via Streamlit Cloud deployment
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
