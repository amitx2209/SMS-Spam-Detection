import string

# Simple stopword list (enough for beginners)
STOPWORDS = {
    "the", "is", "in", "and", "to", "a", "of", "for", "on",
    "with", "as", "by", "this", "that", "are", "was", "were",
    "be", "have", "has", "had", "you", "your", "i", "me", "my"
}

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]

    return " ".join(words)
