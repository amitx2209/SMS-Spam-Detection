from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_text

def get_vectorizer():
    return TfidfVectorizer(
        preprocessor=clean_text,
        max_features=1000
    )
