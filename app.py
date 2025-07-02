import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from newspaper import Article

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

TRUSTED_DOMAINS = ["thehindu.com", "ndtv.com", "bbc.com", "reuters.com", "indiatoday.in", "toi.in"]

def clean(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text)).lower().split()
    text = [word for word in text if word not in stop_words and len(word) > 2]
    return " ".join(text)

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Real-Time Fake News Detector")

option = st.radio("Choose input type:", ("Paste News Text", "Paste Article Link"))

if option == "Paste News Text":
    user_input = st.text_area("Paste news content or headline here")
    if st.button("Check News"):
        if len(user_input.split()) < 5:
            st.warning("⚠️ Please enter a longer news statement.")
        else:
            cleaned = clean(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            confidence = model.predict_proba(vector)[0][prediction]

            st.subheader("Result:")
            st.success("✅ Real News" if prediction == 1 else "❌ Fake News")
            st.caption(f"🧠 Confidence: {confidence:.2f}")

elif option == "Paste Article Link":
    url = st.text_input("Paste full news article URL here")
    if st.button("Check Link"):
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            if not text.strip():
                raise ValueError("No content found in article.")
            cleaned = clean(text)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            confidence = model.predict_proba(vector)[0][prediction]

            st.subheader("Extracted Article:")
            st.write(article.title)
            st.write(text[:500] + "...")

            st.subheader("Result:")

            if any(domain in url for domain in TRUSTED_DOMAINS):
                st.success("✅ Real News (Trusted Source)")
            else:
                st.success("✅ Real News" if prediction == 1 else "❌ Fake News")
            st.caption(f"🧠 Confidence: {confidence:.2f}")
        except Exception as e:
            st.error("⚠️ Could not extract or analyze the article.")
            st.caption(f"Details: {str(e)}")
