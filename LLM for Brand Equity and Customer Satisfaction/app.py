import streamlit as st
import joblib
import numpy as np
import re
import string

# Load Models
rf_be = joblib.load("rf_be_model.pkl")
rf_ce = joblib.load("rf_ce_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    return " ".join(words)

# Streamlit App
st.title("📊 Customer Satisfaction & Brand Equity Predictor")

# User Inputs
company_input = st.selectbox("Select a Company", label_encoder.classes_)
text_input = st.text_area("Enter a Customer Review")

if st.button("Predict"):
    if text_input:
        # Encode company
        company_encoded = label_encoder.transform([company_input])[0]

        # Process text
        text_features = tfidf_vectorizer.transform([preprocess_text(text_input)]).toarray()

        # Combine features
        input_features = np.hstack((text_features, [[company_encoded]]))

        # Make Predictions
        be_pred = rf_be.predict(input_features)[0]
        ce_pred = rf_ce.predict(input_features)[0]

        # Display Results
        st.success(f"🔵 Predicted Brand Equity: {be_pred:.2f}%")
        st.success(f"🟢 Predicted Customer Satisfaction: {ce_pred:.2f}%")
    else:
        st.warning("⚠️ Please enter a customer review!")
