import joblib
from preprocess import clean_text

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_spam(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    return "SPAM" if model.predict(vec)[0] == 1 else "NOT SPAM"

print(predict_spam("Congratulations! You won a free ticket"))


