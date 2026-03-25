import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model("frontend/CNNmodel.h5", compile=False)

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

st.title("🧠 Sign Language Detection")
st.write("Upload an image or take a photo")

img_file = st.camera_input("Take a picture")

if img_file is not None:
    bytes_data = img_file.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    st.image(img, caption="Captured Image")

    # ✅ ROI Crop (same logic you used before)
    h, w, _ = img.shape
    img = img[int(h*0.3):int(h*0.8), int(w*0.3):int(w*0.8)]

    # ✅ Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ✅ Gaussian Blur (VERY IMPORTANT)
    img = cv2.GaussianBlur(img, (15, 15), 0)

    # ✅ Resize
    img = cv2.resize(img, (28, 28))

    # ✅ Normalize
    img = img / 255.0

    # ✅ Reshape
    img = np.reshape(img, (1, 28, 28, 1))

    # Debug (optional)
    st.image(img.reshape(28,28), caption="Processed Image")

    # Predict
    prediction = model.predict(img)
    index = np.argmax(prediction)
    result = chr(index + 65)
    confidence = np.max(prediction)

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}")
