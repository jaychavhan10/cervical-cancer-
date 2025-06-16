import streamlit as st
import numpy as np
import pickle
from PIL import Image
import os
from utils import extract_features  


import joblib
model = joblib.load('models/model1.pkl')


st.set_page_config(page_title="Cervical Cancer Detection", layout="centered")

# Main Page
st.title("🧬 Cervical Cancer Detection from Cell Image")
st.markdown("""
Upload a **cervical cell image**, and the app will:
- Extract shape-based features
- Display the extracted parameters
- Predict whether the case is high risk or low risk
""")

# File uploader
uploaded_file = st.file_uploader("Upload a Cervical Cell Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    # Show image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save image to disk (if needed by extract_features)
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.read())

    # Extract features
    try:
        features_dict = extract_features("temp_image.png")
        st.subheader("🔍 Extracted Features")
        for key, value in features_dict.items():
            st.write(f"**{key}**: {value}")

        # Prepare feature vector
        input_vector = np.array(list(features_dict.values())).reshape(1, -1)

        # Predict
        prediction = model.predict(input_vector)[0]
        label = "🔴 High Risk (Class 1)" if prediction == 1 else "🟢 Low Risk (Class 0)"

        st.subheader("📊 Prediction Result")
        st.success(label)
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
