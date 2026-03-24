import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sign Language Detection", layout="centered")

st.title("🖐️ Sign Language Detection")
st.write("Capture an image and detect the hand sign")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_my_model():
    try:
        model = load_model("sign_language_model.h5")
        return model
    except:
        return None

model = load_my_model()

if model is None:
    st.error("❌ Model file not found! Please add 'sign_language_model.h5'")
    st.stop()

# ---------------- LABELS ----------------
labels = ['enna pannura', 'vanakam']

# ---------------- MEDIAPIPE SETUP ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA INPUT ----------------
img_file_buffer = st.camera_input("📸 Take a picture")

if img_file_buffer is not None:
    # Convert image to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # ---------------- EXTRACT LANDMARKS ----------------
        landmark_list = []
        for lm in hand.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])

        # Convert to numpy
        landmarks = np.array(landmark_list)

        # ---------------- NORMALIZATION ----------------
        landmarks = landmarks - np.min(landmarks)
        if np.max(landmarks) != 0:
            landmarks = landmarks / np.max(landmarks)

        # Reshape for model
        landmarks = landmarks.reshape(1, -1)

        # ---------------- PREDICTION ----------------
        prediction = model.predict(landmarks)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        label = labels[class_id]

        # ---------------- DISPLAY RESULT ----------------
        st.success(f"Prediction: {label} ({confidence:.2f})")

        # Put text on image
        cv2.putText(frame, f"{label} ({confidence:.2f})",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

    else:
        st.warning("❌ No hand detected")

    # Show image
    st.image(frame, channels="BGR")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + MediaPipe + TensorFlow")
