import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="Sign Language Detection", layout="centered")

st.title("🖐️ Sign Language Detection")
st.write("Capture an image and detect the hand sign")

# Load model
model = load_model("sign_language_model.h5")

# Labels (same order as training)
labels = ['enna pannura', 'vanakam']

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# 📸 Camera input (browser-based)
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Convert image to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Flip image (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert to RGB for Mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Extract landmarks
        landmark_list = []
        for lm in hand.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])

        # Prediction
        prediction = model.predict(np.array([landmark_list]))
        class_id = np.argmax(prediction)
        label = labels[class_id]

        # Show result
        st.success(f"Prediction: {label}")

        # Put text on image
        cv2.putText(frame, label, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        st.warning("No hand detected ❌")

    # Show image
    st.image(frame, channels="BGR")