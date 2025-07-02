import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text)).lower().split()
    text = [word for word in text if word not in stop_words and len(word) > 2]
    return " ".join(text)

# Load and label data
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")
df_fake["label"] = 0
df_real["label"] = 1

# Combine and shuffle
df = pd.concat([df_fake, df_real])
df = df.sample(frac=1).reset_index(drop=True)
df["text"] = (df["title"] + " " + df["text"]).apply(clean)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Vectorize and train model
vectorizer = TfidfVectorizer(max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Accuracy
acc = model.score(X_test_vec, y_test)
print(f"Accuracy: {acc:.2f}")

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
