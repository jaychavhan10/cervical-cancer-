import streamlit as st
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from utils2 import extract_features
from streamlit_echarts import st_echarts

# --- Page Config ---
st.set_page_config(
    page_title="Cervical Cancer Detection",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="auto"
)
MODEL_ACCURACY = 0.92  # Replace with your actual model accuracy
# --- Custom CSS for style ---
st.markdown("""
    <style>
    .main { background-color: #f7f9fa; }
    .stButton>button { background-color: #4F8BF9; color: white; font-weight: bold; }
    .st-bb { background: #e3f2fd; border-radius: 10px; padding: 1em; }
    .st-cf { background: #fff3e0; border-radius: 10px; padding: 1em; }
    .stResult { font-size: 1.3em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🧬 Cervical Cancer Detection from Cell Image")
st.markdown("""
<div class="st-bb">
    <h4>How it works:</h4>
    <ul>
        <li>Upload a cervical cell image (JPG, PNG, BMP).</li>
        <li>The system will <b>segment the nucleus and cytoplasm</b>, extract shape-based features, and use a trained AI model to predict risk.</li>
        <li>All processing is done locally for privacy.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --- Sidebar for info ---
with st.sidebar:
    st.image("https://cdn.pixabay.com/photo/2017/01/31/13/14/analysis-2025785_1280.png", use_column_width=True)
    st.markdown("""
    ### About This App
    - **Purpose:** Early detection of cervical cancer risk using cell morphology.
    - **Model:** Logistic Regression (shape features).
    - **Disclaimer:** This tool is for research/educational use only. Not a substitute for medical diagnosis.
    """)
    st.markdown("---")
    st.markdown("**Developed by:** Your Lab/Team Name")
    st.markdown("[GitHub](https://github.com/) | [Contact](mailto:your@email.com)")

# --- File Upload ---
uploaded_file = st.file_uploader("📷 Upload a Cervical Cell Image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with st.spinner("🔬 Processing image..."):
        # Save uploaded image temporarily
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            features_dict, nucleus_mask, cytoplasm_mask = extract_features("temp_image.png")

            # --- Show segmentation masks side by side ---
            st.subheader("🖼️ Segmentation Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(nucleus_mask * 255, caption="Nucleus Mask", use_column_width=True, channels="GRAY")
            with col2:
                st.image(cytoplasm_mask * 255, caption="Cytoplasm Mask", use_column_width=True, channels="GRAY")

            # --- Show extracted features in a nice table ---
            st.subheader("🧪 Extracted Shape Features")
            st.dataframe(
                { "Feature": list(features_dict.keys()), "Value": [round(v, 4) for v in features_dict.values()] },
                hide_index=True,
                use_container_width=True
            )

            # --- Prepare input vector in correct order ---
            feature_order = [
                "Nucleus Area",                # Kerne_A
                "Cytoplasm Area",              # Cyto_A
                "Nucleus/Cytoplasm Area Ratio",# K/C
                "Nucleus Minor Axis Length",   # KerneShort
                "Cytoplasm Minor Axis Length", # CytoShort
                "Nucleus Major Axis Length",   # KerneLong
                "Cytoplasm Major Axis Length", # CytoLong
                "Nucleus Elongation",          # KerneElong
                "Nucleus Roundness",           # KerneRund
                "Cytoplasm Roundness",         # CytoRund
                "Cytoplasm Elongation",        # CytoElong
                "Nucleus Perimeter",           # KernePeri
                "Cytoplasm Perimeter"          # CytoPeri
            ]
            input_vector = np.array([features_dict[key] for key in feature_order]).reshape(1, -1)

            # --- Load model and predict ---
            model = joblib.load('models/model1.pkl')
            prediction = model.predict(input_vector)[0]
            proba = model.predict_proba(input_vector)[0][prediction] if hasattr(model, "predict_proba") else None

            # --- Show result ---
            st.subheader("📊 Prediction Result")
            if prediction == 1:
                st.markdown(f'<div class="stResult" style="color:#d32f2f;">🔴 <b>High Risk (Class 1)</b></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="stResult" style="color:#388e3c;">🟢 <b>Low Risk (Class 0)</b></div>', unsafe_allow_html=True)
            if proba is not None:
                st.markdown(f"**Confidence:** {proba*100:.1f}%")

            st.markdown("#### Model Accuracy")
            option = {
                "series": [
                    {
                        "type": "gauge",
                        "startAngle": 210,
                        "endAngle": -30,
                        "progress": {"show": True, "width": 18},
                        "axisLine": {"lineStyle": {"width": 18}},
                        "pointer": {"show": True},
                        "detail": {
                            "valueAnimation": True,
                            "formatter": "{value}%",
                            "fontSize": 28,
                            "color": "#1976d2"
                        },
                        "data": [{"value": round(MODEL_ACCURACY*100, 1), "name": "Accuracy"}],
                        "min": 0,
                        "max": 100,
                        "title": {"fontSize": 18}
                    }
                ]
            }
            st_echarts(options=option, height="250px")

            # --- More info ---
            st.markdown("""
            <div class="st-cf">
            <b>What does this mean?</b><br>
            - <b>High Risk:</b> The cell's shape features are similar to those seen in high-risk cases.<br>
            - <b>Low Risk:</b> The cell's features are typical of low-risk cases.<br>
            <br>
            <i>For any medical concerns, consult a healthcare professional.</i>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")

else:
    st.info("Please upload a cervical cell image to begin analysis.")