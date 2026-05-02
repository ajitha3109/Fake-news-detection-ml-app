# 📰 Fake News Detection System

An AI-powered web application that classifies news articles as **Real or Fake** using Machine Learning and Natural Language Processing (NLP).

---

## 🚀 Live Features

- 🧠 Fake vs Real News Classification  
- 📊 Probability Score Prediction  
- ⚠️ Risk Level Indicator  
- 🧾 AI Explanation of Prediction  
- 🔎 Key Word Indicators  
- 🌐 Interactive Streamlit Web App  

---

## 🧠 How It Works

1. User enters a news article  
2. Text is cleaned and preprocessed  
3. TF-IDF converts text into numerical features  
4. Logistic Regression model predicts output  
5. System displays:
   - Fake / Real label  
   - Confidence score  
   - Risk level  
   - Explanation  

---

## 🛠️ Tech Stack

- Python 🐍  
- Scikit-learn 🤖  
- Pandas 📊  
- Numpy 🔢  
- Streamlit 🌐  
- NLP (TF-IDF Vectorization)

---

## 📂 Project Structure
Fake-News-Detection/
│
├── app.py # Streamlit application
├── fake_news_model.pkl # Trained ML model
├── vectorizer_news.pkl # TF-IDF vectorizer
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ⚙️ How to Run Locally

pip install -r requirements.txt
streamlit run app.py

---

📊 Model Details
Algorithm: Logistic Regression
Feature Extraction: TF-IDF Vectorizer
Task: Binary Classification (Fake / Real News)
Dataset: Fake and Real News Dataset

---

📌 Key Highlights

✔ Real-time prediction
✔ Explainable AI output
✔ Clean Streamlit UI
✔ Confidence-based decision system

👩‍💻 Author

Kalai Ajitha M
AI & Data Science Student


---

# 📦 4. requirements.txt (name + description not needed but here)


streamlit
scikit-learn
pandas
numpy
joblib


---

# 🧠 5. FILE NAMING (IMPORTANT)

Make sure:

| File | Purpose |
|------|--------|
| app.py | main Streamlit app |
| fake_news_model.pkl | trained model |
| vectorizer_news.pkl | TF-IDF vectorizer |
| requirements.txt | dependencies |
| README.md | project explanation |

---
