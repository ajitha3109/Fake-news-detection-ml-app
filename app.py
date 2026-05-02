import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("📰 Fake News Detection System")
st.write("Detect whether a news article is **Real or Fake** using Machine Learning & NLP.")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer_news.pkl")

# =========================
# CLEAN FUNCTION (IMPROVED)
# =========================
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =========================
# SIDEBAR INFO
# =========================
st.sidebar.title("ℹ️ About Project")
st.sidebar.info(
    """
    🧠 Fake News Detection AI

    🔹 Model: Logistic Regression  
    🔹 Technique: TF-IDF NLP  
    🔹 UI: Streamlit Web App  

    Detects fake vs real news using machine learning patterns.
    """
)

# =========================
# INPUT
# =========================
news_text = st.text_area("📝 Enter News Article", height=200)

# =========================
# BUTTON
# =========================
if st.button("🔍 Analyze News"):

    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")

    else:

        # =========================
        # PREPROCESSING
        # =========================
        cleaned = clean(news_text)
        vec = vectorizer.transform([cleaned])

        # =========================
        # PREDICTION
        # =========================
        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        fake_prob = prob[1] * 100
        real_prob = prob[0] * 100
        confidence = max(fake_prob, real_prob)

        # =========================
        # RESULT
        # =========================
        st.subheader("📊 Result")

        if fake_prob > 80:
            st.error("🚨 Fake News Detected (High Confidence)")
        elif fake_prob > 60:
            st.warning("⚠️ Likely Fake News")
        elif fake_prob > 40:
            st.info("⚖️ Uncertain / Mixed Signals")
        else:
            st.success("✅ Likely Real News")

        # =========================
        # UNCERTAINTY WARNING
        # =========================
        if confidence < 60:
            st.warning("⚠️ Low confidence — model prediction may be unreliable")

        # =========================
        # RISK LEVEL
        # =========================
        if fake_prob > 80:
            risk = "🔴 High"
        elif fake_prob > 60:
            risk = "🟡 Medium"
        else:
            risk = "🟢 Low"

        st.write(f"📌 Risk Level: {risk}")
        st.write(f"🎯 Confidence Score: {confidence:.2f}%")

        st.markdown("---")

        # =========================
        # PROBABILITY DISPLAY (IMPROVED UI)
        # =========================
        st.markdown("### 📈 Probability Breakdown")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("🔴 Fake Probability", f"{fake_prob:.2f}%")

        with col2:
            st.metric("🟢 Real Probability", f"{real_prob:.2f}%")

        st.progress(int(confidence))

        st.markdown("---")

        # =========================
        # AI EXPLANATION (IMPROVED)
        # =========================
        st.markdown("### 🧠 AI Explanation")

        if fake_prob > real_prob:
            st.write(
                "The system detected linguistic patterns commonly associated with "
                "misleading or sensationalized news, based on training data distribution "
                "and word usage patterns."
            )
        else:
            st.write(
                "The text follows structured journalistic patterns such as factual reporting, "
                "neutral tone, and institutional references."
            )

        st.markdown("---")

        # =========================
        # KEY INDICATORS (FIXED)
        # =========================
        st.markdown("### 🔎 Key Indicators")

        words = [w for w in cleaned.split() if w not in ENGLISH_STOP_WORDS]
        top_words = list(set(words))[:10]

        st.write(", ".join(top_words))

        st.markdown("---")

        # =========================
        # FOOTER
        # =========================
        st.caption("Built using Machine Learning & NLP | Project by Kalai Ajitha M 🚀")