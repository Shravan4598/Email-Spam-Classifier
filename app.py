import streamlit as st
import os
import sys
import numpy as np
from src.utils import load_object
from src.exception import CustomException

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="ClassifAI - Email Spam Classifier",
    layout="centered"
)

st.title("ClassifAI - Email Spam Classifier")
st.write(
    "This application demonstrates **Logistic Regression** for "
    "email spam classification."
)

# ---------------- Load Model & Preprocessor ----------------
try:
    model = load_object(os.path.join("artifacts", "model.pkl"))
    preprocessor = load_object(os.path.join("artifacts", "preprocessor.pkl"))
except Exception as e:
    st.error("❌ Failed to load model artifacts.")
    raise CustomException(e, sys)

# ---------------- User Input ----------------
email_text = st.text_area(
    "Enter Email Text",
    placeholder="Type or paste the email content here...",
    height=200
)

# ---------------- Prediction ----------------
if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter valid email text.")
    else:
        try:
            # Preprocess (same as training)
            processed_text = preprocessor.transform([email_text]).toarray()

            # Predict
            prediction = model.predict(processed_text)[0]
            probability = model.predict_proba(processed_text)[0]

            # Decode label safely
            if prediction == 1:
                label = "Spam"
            else:
                label = "Not Spam"

            # Confidence score
            confidence_score = float(np.max(probability))

            # Display result
            st.subheader("Prediction Result")

            if label == "Spam":
                st.error("🚫 Classification: Spam")
            else:
                st.success("✅ Classification: Not Spam")

            st.write(f"**Confidence Score:** {confidence_score:.2f}")

        except Exception as e:
            st.error("❌ An unexpected error occurred during prediction.")
            raise CustomException(e, sys)

# ---------------- Educational Section ----------------
st.markdown("---")
st.subheader("About Logistic Regression & Spam Detection")

st.write(
    """
    Logistic Regression is a supervised machine learning algorithm used for
    binary classification problems such as spam detection.

    In this project, email text is converted into numerical features using
    **TF-IDF Vectorization**, and a trained Logistic Regression model predicts
    whether the email is **Spam** or **Not Spam** along with a confidence score.
    """
)

