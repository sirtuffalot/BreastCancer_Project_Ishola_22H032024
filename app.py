import streamlit as st
import joblib
import numpy as np
import os

# Page Config
st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺")

st.title("🩺 Breast Cancer Prediction System")
st.write("Name: ISHOLA OLUFEMI | Matric: 22H032024")
st.write("Enter tumor details below. The model uses Logistic Regression with Feature Scaling.")

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'breast_cancer_model.pkl')
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("System Status: Model Loaded Successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

# User Inputs (Matches the 5 features in training)
st.subheader("Tumor Features")

col1, col2 = st.columns(2)

with col1:
    radius = st.number_input("Mean Radius", min_value=0.0, value=14.0, format="%.2f", help="Average distance from center to points on the perimeter")
    texture = st.number_input("Mean Texture", min_value=0.0, value=20.0, format="%.2f", help="Standard deviation of gray-scale values")
    perimeter = st.number_input("Mean Perimeter", min_value=0.0, value=90.0, format="%.2f", help="Size of the core tumor")

with col2:
    area = st.number_input("Mean Area", min_value=0.0, value=600.0, format="%.2f")
    smoothness = st.number_input("Mean Smoothness", min_value=0.0, value=0.1, format="%.4f", step=0.0001, help="Local variation in radius lengths")

# Predict
if st.button("Analyze Tumor"):
    # Create input array
    features = np.array([[radius, texture, perimeter, area, smoothness]])
    
    # Predict
    # The pipeline will automatically SCALE this input before predicting.
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]
    
    # Logic: 0 = Malignant, 1 = Benign
    if prediction == 0:
        st.error(f"Prediction: MALIGNANT (Confidence: {probability:.2%})")
        st.write("⚠️ The model detected patterns consistent with malignancy.")
    else:
        st.balloons()
        st.success(f"Prediction: BENIGN (Confidence: {probability:.2%})")
        st.write("✅ The tumor appears to be safe.")
