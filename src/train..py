import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from preprocess import clean_text
import os
import pandas as pd

df = pd.read_csv(
    "/Users/krishnasoni/Documents/spam classifier/data/spam.csv",
    encoding="latin-1"
)

df = df.rename(columns={
    "v1": "label",
    "v2": "message"
})

df = df[["label", "message"]]

print(df.head())
print(df.columns)

df["message"] = df["message"].apply(clean_text)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
import os

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
