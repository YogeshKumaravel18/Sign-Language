import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Sign Language Detection (Vanakkam / Hi / Bye)")

# Dummy prediction function (replace with your ML model later)
def predict_sign(frame):
    # Convert to grayscale (just example processing)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Fake logic for demo
    avg_pixel = np.mean(gray)

    if avg_pixel < 80:
        return "Vanakkam 🙏"
    elif avg_pixel < 150:
        return "Hi 👋"
    else:
        return "Bye 👋"

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        prediction = predict_sign(img)

        # Display prediction on screen
        cv2.putText(img, prediction, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        return img

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
