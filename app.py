import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

st.title("Sign Detection (Test Mode)")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        st.success("Hand detected ✅")
    else:
        st.warning("No hand detected ❌")

    st.image(frame, channels="BGR")
