import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from newspaper import Article

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Trusted domains list
TRUSTED_DOMAINS = [
    "thehindu.com",
    "bbc.com",
    "ndtv.com",
    "indiatoday.in",
    "hindustantimes.com",
    "reuters.com",
    "theguardian.com"
]

# Text cleaning function
def clean(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower().split()
    text = [word for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Real-Time Fake News Detector")

option = st.radio("Choose input type:", ("Paste News Text", "Paste Article Link"))

if option == "Paste News Text":
    user_input = st.text_area("Paste news content or headline here")
    if st.button("Check News"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            cleaned = clean(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            confidence = model.predict_proba(vector)[0][prediction]

            st.subheader("Result:")
            st.success("✅ Real News" if prediction == 1 else "❌ Fake News")
            st.caption(f"🧠 Model confidence: {confidence:.2f}")

elif option == "Paste Article Link":
    url = st.text_input("Paste full news article URL here")
    if st.button("Check Link"):
        if not url.strip():
            st.warning("⚠️ Please paste a valid article link.")
        else:
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = article.text
                cleaned = clean(text)
                vector = vectorizer.transform([cleaned])
                prediction = model.predict(vector)[0]
                confidence = model.predict_proba(vector)[0][prediction]

                st.subheader("Extracted Article:")
                st.write(f"**Title:** {article.title}")
                st.write(text[:500] + "...")

                st.subheader("Result:")

                # Check if URL is from a trusted domain
                if any(domain in url for domain in TRUSTED_DOMAINS):
                    st.success("✅ Real News (Trusted Source)")
                    st.caption("ℹ️ This article comes from a verified and trusted domain.")
                else:
                    st.success("✅ Real News" if prediction == 1 else "❌ Fake News")
                    st.caption(f"🧠 Model confidence: {confidence:.2f}")

            except Exception as e:
                st.error("⚠️ Failed to extract article. Make sure the link is valid and public.")
                st.text(f"Error: {e}")
