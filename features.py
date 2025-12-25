from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_text

def get_vectorizer():
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        max_features=1000
    )
