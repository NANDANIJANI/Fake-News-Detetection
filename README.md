
# 📰 Fake News Detection using Machine Learning

A capstone project that detects whether a news article or headline is **fake or real**, using machine learning, natural language processing (NLP), and optional integration with Google Fact Check API.

> Built with Python, Streamlit, and scikit-learn  
> Smart logic: adjusts prediction using trusted domains & keywords

---

## 🔍 Features

- ✅ Paste text or URL of a news article
- 🤖 Uses trained ML model to classify news
- 🧠 Shows model confidence
- 🔒 Trusted source/domain detection (e.g. `ndtv.com`, `bbc.com`)
- 📝 Keyword-based override for real news
- 🌐 Optional Google Fact Check API integration
- 🚀 Deployed on Streamlit Cloud *(if available)*

---
---

## 📂 Project Structure

```
📁 fake-news-detector/
├── app.py                   # Streamlit app
├── train.py                 # Training script for model
├── fake_news_model.pkl      # Trained ML model (Logistic Regression)
├── vectorizer.pkl           # Trained TF-IDF vectorizer
├── requirements.txt         # All dependencies
└── README.md                # Project documentation
```

---

## 🛠 Tech Stack

- Python 3.10+
- Streamlit (UI)
- Scikit-learn (ML Model)
- NLTK (Text Preprocessing)
- Newspaper3k (Web article extraction)
- Google Fact Check Tools API (optional)
- Joblib (Model serialization)

---

## ⚙️ How It Works

1. **Train Model:** `train.py` uses a cleaned dataset (`Fake.csv`, `True.csv`)
2. **Build TF-IDF vectorizer**
3. **Train Logistic Regression** (supports confidence)
4. **In `app.py`:**
   - Takes user input (text or article URL)
   - Cleans and vectorizes
   - Predicts with model
   - Adjusts based on trusted domains or keywords
   - Shows result + explanation

---

## 🔐 Trusted Overrides (Smart Logic)

- Domains like `ndtv.com`, `bbc.com`, `deccanherald.com` force a **Real News** label if model says otherwise.
- Keywords like `modi`, `isro`, `president`, `national honour` also help override false negatives.

---

## 📦 Installation

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Training the Model

```bash
python train.py
```

Ensure `Fake.csv` and `True.csv` are in the same folder.

---

## 🔑 Optional: Setup Google Fact Check API

1. Go to: [Google Cloud Console](https://console.cloud.google.com/)
2. Enable **Fact Check Tools API**
3. Create API key
4. Paste it into `app.py`:

```python
GOOGLE_API_KEY = "your-api-key"
```

---

## 📊 Dataset Used

- Source: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Balanced: 50% real, 50% fake
- Preprocessed and cleaned in training

---

## 📈 Future Scope

- Use BERT or transformer-based models for semantic understanding
- Add feedback loop for user flagging
- Add browser plugin for live news classification
- Expand to Hindi/Gujarati language support

---

## 👩‍💻 Developed By

**Nandani Jani**  
Final Year B.E. (Computer Engineering)  
[nandan17072005@gmail.com] |

---

## 📄 License

This project is for academic and learning purposes.
