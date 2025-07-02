
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib


nltk.download('stopwords')

df_fake = pd.read_csv("Fake.csv")  # Contains fake news
df_real = pd.read_csv("True.csv")  # Contains real news


df_fake["label"] = 0
df_real["label"] = 1


df = pd.concat([df_fake, df_real])
df = df[['title', 'text', 'label']]  # Keep only relevant columns
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset


def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))  # Remove punctuation/numbers
    text = text.lower().split()
    text = [word for word in text if word not in stopwords.words('english')]
    return ' '.join(text)


df['text'] = df['text'].apply(clean_text)


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])

y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train, y_train)


joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model and Vectorizer saved successfully!")
