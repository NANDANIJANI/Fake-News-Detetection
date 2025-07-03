import streamlit as st
import joblib
import re
import nltk
import requests
from urllib.parse import urlparse
from nltk.corpus import stopwords
from newspaper import Article, Config

# --------------- Setup -----------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

trusted_keywords = [
    "modi", "prime minister", "pm modi", "president", "award",
    "ghana", "national honour", "isro", "parliament", "upsc"
]
trusted_domains = ["deccanherald.com", "ndtv.com", "bbc.com", "reuters.com", "indiatoday.in"]

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
config = Config()
config.browser_user_agent = user_agent

GOOGLE_API_KEY = "AIzaSyC-zIdv2m7jAKg55xhvXhNYRmsPeZjTUVY"  # <--- Replace this with your actual key

# --------------- Functions -----------------
def clean(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text)).lower().split()
    text = [word for word in text if word not in stop_words and len(word) > 2]
    return " ".join(text)

def check_fact_with_google_api(query, api_key):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "languageCode": "en-US", "key": api_key}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        return None, f"Fact Check API error {response.status_code}"

    data = response.json()
    claims = data.get("claims", [])
    if not claims:
        return None, None

    top_claim = claims[0]
    claim_text = top_claim.get("text", "")
    claim_rating = top_claim.get("claimReview", [{}])[0].get("textualRating", "No rating")
    return f"{claim_text} — Rated: {claim_rating}", None

# --------------- Streamlit UI -----------------
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.title("🧠 Real-Time Fake News Detector")

option = st.radio("Choose input type:", ("Paste News Text", "Paste Article Link"))

if option == "Paste News Text":
    user_input = st.text_area("🗞️ Paste news content or headline here")
    if st.button("Check News"):
        if len(user_input.split()) < 5:
            st.warning("⚠️ Please enter a longer news statement.")
        else:
            cleaned = clean(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            reason = "Prediction based on model."

            # Trusted keyword override
            if prediction == 0 and any(word in cleaned.lower() for word in trusted_keywords):
                prediction = 1
                reason = "Trusted keywords detected, overriding model."

            # Google Fact Check API
            fact_result, fact_error = check_fact_with_google_api(user_input, GOOGLE_API_KEY)
            if fact_result:
                st.info(f"🔎 Fact Check:\n{fact_result}")
                if "True" in fact_result or "Correct" in fact_result:
                    prediction = 1
                    reason = "Google fact check marked it true."

            st.subheader("✅ Result:")
            st.success("Real News" if prediction == 1 else "Fake News")
            try:
                confidence = model.predict_proba(vector)[0][prediction]
                st.caption(f"Confidence: {confidence:.2f} — {reason}")
            except:
                st.caption(reason)

elif option == "Paste Article Link":
    url = st.text_input("🔗 Paste full article URL")
    if st.button("Check Link"):
        try:
            domain = urlparse(url).netloc
            article = Article(url, config=config)
            article.download()
            article.parse()

            text = article.text
            title = article.title

            cleaned = clean(text)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            reason = "Prediction based on model."

            # Trusted domain override
            if prediction == 0 and any(td in domain for td in trusted_domains):
                prediction = 1
                reason = "Trusted news domain, overriding model."

            # Trusted keyword override
            if prediction == 0 and any(word in cleaned.lower() for word in trusted_keywords):
                prediction = 1
                reason = "Trusted keywords detected in article."

            # Google Fact Check
            fact_result, fact_error = check_fact_with_google_api(title, GOOGLE_API_KEY)
            if fact_result:
                st.info(f"🔎 Fact Check:\n{fact_result}")
                if "True" in fact_result or "Correct" in fact_result:
                    prediction = 1
                    reason = "Google fact check marked it true."

            st.subheader("📰 Extracted Article:")
            st.write(f"**{title}**")
            st.write(text[:700] + "...")

            st.subheader("✅ Result:")
            st.success("Real News" if prediction == 1 else "Fake News")
            try:
                confidence = model.predict_proba(vector)[0][prediction]
                st.caption(f"Confidence: {confidence:.2f} — {reason}")
            except:
                st.caption(reason)

        except Exception as e:
            st.error("⚠️ Could not extract article.")
            st.caption(f"Error: {str(e)}")
