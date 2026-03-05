

import mediapipe as mp
import cv2
import numpy as np
import os
import csv


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

captures = cv2.VideoCapture(0)

current_label = input(f'Kayıt edilecek harf:').upper()

with open('dataset/asl_landmarks.csv', mode='a', newline='') as f:
    writer = csv.writer(f)

    while True:
        ret, frame = captures.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = []

            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            row = landmarks + [current_label]
            writer.writerow(row)

            cv2.putText(frame, f'label: {current_label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('collecting', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

captures.release()
cv2.destroyAllWindows()