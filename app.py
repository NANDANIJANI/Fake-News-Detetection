import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from newspaper import Article, Config

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Trusted sources that usually allow article extraction
TRUSTED_DOMAINS = ["ndtv.com", "bbc.com", "reuters.com", "indiatoday.in", "thehindu.com"]

# Fake browser header to avoid 403 error
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
config = Config()
config.browser_user_agent = user_agent

def clean(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text)).lower().split()
    text = [word for word in text if word not in stop_words and len(word) > 2]
    return " ".join(text)

# Streamlit UI
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
            try:
                confidence = model.predict_proba(vector)[0][prediction]
                confidence_msg = f"🧠 Confidence: {confidence:.2f}"
            except:
                confidence_msg = "⚠️ Confidence score not available"

            st.subheader("Result:")
            st.success("✅ Real News" if prediction == 1 else "❌ Fake News")
            st.caption(confidence_msg)

elif option == "Paste Article Link":
    url = st.text_input("Paste full news article URL here")

    if st.button("Check Link"):
        if not any(domain in url for domain in TRUSTED_DOMAINS):
            st.warning("⚠️ This website may block article downloads. Use NDTV, BBC, Reuters, or The Hindu.")

        try:
            article = Article(url, config=config)
            article.download()
            article.parse()
            text = article.text
            title = article.title

            if not text.strip():
                raise ValueError("No article content found.")

            cleaned = clean(text)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            try:
                confidence = model.predict_proba(vector)[0][prediction]
                confidence_msg = f"🧠 Confidence: {confidence:.2f}"
            except:
                confidence_msg = "⚠️ Confidence score not available"

            st.subheader("Extracted Article:")
            st.write(f"**{title}**")
            st.write(text[:600] + "...")

            st.subheader("Result:")
            st.success("✅ Real News" if prediction == 1 else "❌ Fake News")
            st.caption(confidence_msg)

        except Exception as e:
            if "403" in str(e):
                st.error("⚠️ This website blocks article extraction (403 Forbidden). Try a different link.")
            else:
                st.error("⚠️ Could not extract or analyze the article.")
            st.caption(f"Details: {str(e)}")
