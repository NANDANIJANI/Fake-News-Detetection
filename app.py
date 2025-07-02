import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from newspaper import Article

nltk.download('stopwords')

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower().split()
    text = [word for word in text if word not in stopwords.words('english')]
    return " ".join(text)

st.title("📰 Real-Time Fake News Detector")

option = st.radio("Choose input type:", ("Paste News Text", "Paste Article Link"))

if option == "Paste News Text":
    user_input = st.text_area("Paste news content or headline here")
    if st.button("Check News"):
        cleaned = clean(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        st.subheader("Result:")
        st.success("✅ Real News" if prediction == 1 else "❌ Fake News")

elif option == "Paste Article Link":
    url = st.text_input("Paste full news article URL here")
    if st.button("Check Link"):
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            cleaned = clean(text)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            st.subheader("Extracted Article:")
            st.write(article.title)
            st.write(text[:500] + "...")
            st.subheader("Result:")
            st.success("✅ Real News" if prediction == 1 else "❌ Fake News")
        except:
            st.error("⚠️ Failed to extract article. Make sure the link is valid and public.")
