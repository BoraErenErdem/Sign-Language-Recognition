

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib

model = load_model('asl_model.h5')
le = joblib.load('label_encoder.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

captures = cv2.VideoCapture(0)

while True:
    ret, frame = captures.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        base = hand.landmark[0]
        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([lm.x - base.x, lm.y - base.y, lm.z - base.z])

        x = np.array(landmarks).reshape(1, -1)
        pred = model.predict(x, verbose=0)
        letter = le.inverse_transform([np.argmax(pred)])[0] # sayıları harfe geri dönüştürüyorum..!

        cv2.putText(frame, f'{letter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 4)

    cv2.imshow('ASL Prediction', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

captures.release()
cv2.destroyAllWindows()