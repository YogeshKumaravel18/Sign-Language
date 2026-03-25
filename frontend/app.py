import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model safely
model = load_model("CNNmodel.h5", compile=False)

# Labels (A-Z)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

st.title("🧠 Sign Language Detection")

st.write("Upload an image or take a photo")

# Camera input
img_file = st.camera_input("Take a picture")

if img_file is not None:
    # Convert image to OpenCV format
    bytes_data = img_file.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Show image
    st.image(img, caption="Captured Image", use_column_width=True)

    # Preprocess (IMPORTANT - match your model)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.reshape(img, (1, 64, 64, 3))

    # Predict
    prediction = model.predict(img)
    index = np.argmax(prediction)
    result = labels[index]
    confidence = np.max(prediction)

    # Output
    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}")
