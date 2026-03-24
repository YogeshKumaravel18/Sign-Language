import streamlit as st
import requests
from PIL import Image
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Sign Language Detection", layout="centered")

st.title("🖐️ Sign Language Detection")
st.write("Capture image → send to AI backend → get prediction")

# 🔥 CHANGE THIS AFTER DEPLOYMENT
API_URL = "https://sign-language-vifu.onrender.com"

# ---------------- CAMERA ----------------
img_file_buffer = st.camera_input("📸 Take a picture")

if img_file_buffer is not None:
    st.image(img_file_buffer)

    if st.button("🔍 Predict"):
        with st.spinner("Processing..."):

            response = requests.post(
                API_URL,
                files={"file": img_file_buffer.getvalue()}
            )

            result = response.json()

            if "prediction" in result:
                st.success(
                    f"Prediction: {result['prediction']} ({result['confidence']:.2f})"
                )
            else:
                st.error("❌ No hand detected")
