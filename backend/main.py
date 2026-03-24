from fastapi import FastAPI, File, UploadFile
import numpy as np

from PIL import Image
import io
import mediapipe as mp
print(mp.__file__)
print(dir(mp))

app = FastAPI()

import tensorflow as tf

model = tf.keras.models.load_model(
    "model/sign_language_model.h5",
    compile=False
)

labels = ['enna pannura', 'vanakam']

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

@app.get("/")
def home():
    return {"message": "API Running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    frame = np.array(image)

    rgb = frame.copy()
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        landmark_list = []
        for lm in hand.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmark_list)

        # Normalize
        landmarks = landmarks - np.min(landmarks)
        if np.max(landmarks) != 0:
            landmarks = landmarks / np.max(landmarks)

        landmarks = landmarks.reshape(1, -1)

        prediction = model.predict(landmarks)
        class_id = np.argmax(prediction)
        confidence = float(np.max(prediction))

        return {
            "prediction": labels[class_id],
            "confidence": confidence
        }

    return {"error": "No hand detected"}
