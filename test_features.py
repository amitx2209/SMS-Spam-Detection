from features import get_vectorizer

texts = [
    "Hello how are you",
    "Congratulations you won a prize"
]

vectorizer = get_vectorizer()
X = vectorizer.fit_transform(texts)

print(X.shape)
print(vectorizer.get_feature_names_out())
