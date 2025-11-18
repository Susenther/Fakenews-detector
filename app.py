import streamlit as st
import joblib
from utils import preprocess_text

# Load the saved models
try:
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    model = joblib.load('models/lr_model.pkl')
except FileNotFoundError:
    st.error("Models not found! Please run 'train_classical.py' first.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# App Title and Description
st.title("📰 Fake News Detector")
st.markdown("Enter a news article below to check if it's **Real** or **Fake**.")

# Text Input
news_text = st.text_area("News Text", height=200, placeholder="Paste the news article here...")

# Prediction Logic
if st.button("Analyze News"):
    if news_text.strip():
        with st.spinner("Analyzing..."):
            # 1. Preprocess
            processed_text = preprocess_text(news_text)
            
            # 2. Vectorize
            vectorized_text = vectorizer.transform([processed_text])
            
            # 3. Predict
            prediction = model.predict(vectorized_text)[0]
            
            # 4. Display Result
            if prediction == 1:
                st.success("✅ Prediction: **REAL News**")
            else:
                st.error("🚨 Prediction: **FAKE News**")
    else:
        st.warning("Please enter some text to analyze.")