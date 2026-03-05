

import time
from collections import deque
from pathlib import Path
from tensorflow.keras.models import load_model
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf


MODEL_PATH = 'models/cnn_bilstm_attention_model_v5.h5'
SCALER_PATH = 'models/standardscaler.save'
SPLITS_DIR = Path(r'C:\Projects\sign_language\ASL_Citizen\splits')
TARGET_FRAMES = 30
FEATURE_DIM = 33 * 4 + 21 * 3 + 21 * 3  # pose(132) + left(63) + right(63) = 258
TOP_K = 5              # top5 değerlendirmesini al
STABILITY_COUNT = 5    # üst üste 5 aynı tahmin sayısı varsa kelime eklenir
CONFIDENCE_THR = 0.60 # minimum güven eşiği
LOCK_DURATION = 1.5  # kelime eklendikten sonra 1.5 saniye kelime kitlenir (ard arda aynı şeyi eklemesin diye..!)


# label mapping
# landmarks_extract.py ile aynı yani tüm glossları toplayıp alfabetik sıraya göre 0 tabanlı integer'a tersine çevirilip map ediliyor..! (int -> gloss)
def build_label_to_gloss():
    all_glosses = set()
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(SPLITS_DIR / f'{split}.csv')
        all_glosses.update(df['Gloss'].tolist())
    gloss_to_label = {gloss: idx for idx, gloss in enumerate(sorted(all_glosses))}
    return {idx: gloss for gloss, idx in gloss_to_label.items()}

label_to_gloss = build_label_to_gloss()
print(f'label mapping yüklendi -> {len(label_to_gloss)} class')


# model v5 ve standardscaler
print('model V5 + standardscaler')
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# medipipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# landmark extract kısmında normalizasyon burada da uygulanmalı yoksa model yanlış tahmin yapıyor..! (shoulder_mid, hip_mid, torso_scale, wrist_relative)
def extract_landmarks_frame(results):
    pose = np.zeros(33 * 4)
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pose[i * 4: (i + 1) * 4] = [lm.x, lm.y, lm.z, lm.visibility]

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            left_hand[i * 3: (i + 1) * 3] = [lm.x, lm.y, lm.z]

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            right_hand[i * 3: (i + 1) * 3] = [lm.x, lm.y, lm.z]

    if results.pose_landmarks:
        ls = pose[11 * 4: 11 * 4 + 3]
        rs = pose[12 * 4: 12 * 4 + 3]
        shoulder_mid = (ls + rs) / 2.0

        lh = pose[23 * 4: 23 * 4 + 3]
        rh = pose[24 * 4: 24 * 4 + 3]
        hip_mid = (lh + rh) / 2.0

        torso_scale = np.linalg.norm(shoulder_mid[:2] - hip_mid[:2])
        if torso_scale < 0.000001:
            torso_scale = 1.0

        for i in range(33):
            pose[i * 4: i * 4 + 3] = (pose[i * 4: i * 4 + 3] - shoulder_mid) / torso_scale

        if results.left_hand_landmarks:
            wrist = left_hand[0:3].copy()
            for i in range(21):
                left_hand[i * 3: i * 3 + 3] = (left_hand[i * 3: i * 3 + 3] - wrist) / torso_scale
            left_hand[0:3] = (wrist - shoulder_mid) / torso_scale

        if results.right_hand_landmarks:
            wrist = right_hand[0:3].copy()
            for i in range(21):
                right_hand[i * 3: i * 3 + 3] = (right_hand[i * 3: i * 3 + 3] - wrist) / torso_scale
            right_hand[0:3] = (wrist - shoulder_mid) / torso_scale

    return np.concatenate([pose, left_hand, right_hand])  # (258,) (pose + left + right)


# 30 frame scale edip modele veriliyor. zero masking uygulanıyor. tamamen 0 olan frameler scale sonrası tekrar 0'a dönüyor. (masking layer onları atlıyor)
def predict_sequence(frame_buffer):
    flat = np.array(frame_buffer)          # (30, 258)
    zero_mask = (flat == 0.0).all(axis=1)  # hangi framelerin tamamen 0.0 olduğunu buluyor

    flat_scaled = scaler.transform(flat)
    flat_scaled[zero_mask] = 0.0           # sıfır frame'leri geri sıfırlar

    x = flat_scaled[np.newaxis, ...]       # (1, 30, 258)
    probs = model.predict(x, verbose=0)[0] # (2731,)

    top_k_idx = np.argsort(probs)[::-1][:TOP_K]
    return [(label_to_gloss[i], float(probs[i])) for i in top_k_idx]


# mediapipe için çizim kısmı
def draw_predictions(frame, predictions):
    panel_h = 25 + len(predictions) * 35 + 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (330, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, 'Tahminler:', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)

    for i, (gloss, conf) in enumerate(predictions):
        y = 50 + i * 35
        color = (0, 255, 100) if i == 0 else (190, 190, 190)
        cv2.putText(frame, f'{i+1}. {gloss}  {conf:.1%}', (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def draw_sentence(frame, sentence):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 55), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if sentence:
        text = ' '.join(sentence)
        color = (255, 255, 255)
    else:
        text = 'Space: temizle  |  Backspace: son kelimeyi sil'
        color = (110, 110, 110)

    cv2.putText(frame, text, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def draw_status(frame, buffer_len, fps):
    h, w = frame.shape[:2]
    cv2.putText(frame, f'FPS: {fps:.1f}', (w - 140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    label = f'Buffer {buffer_len}/{TARGET_FRAMES}'
    cv2.putText(frame, label, (w - 195, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    bar_x0, bar_x1 = w - 205, w - 10
    bar_fill = int((buffer_len / TARGET_FRAMES) * (bar_x1 - bar_x0))
    cv2.rectangle(frame, (bar_x0, 65), (bar_x1, 80), (70, 70, 70), -1)
    if bar_fill > 0:
        cv2.rectangle(frame, (bar_x0, 65), (bar_x0 + bar_fill, 80), (0, 210, 255), -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('kamera başlatılamadı.')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # maxlen=30 olduğundan buffer otomatik kayar ??
    frame_buffer = deque(maxlen=TARGET_FRAMES)
    predictions = []
    prev_time = time.time()
    frame_count = 0
    sentence = []
    pred_history = deque(maxlen=8) # son 8 tahmin
    last_added = None
    last_added_time = 0.0

    print('kamera açıldı. çıkmak için -> Q\n')

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=False, refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # min_detection_confidence=0.5 -> %50'lik güven duyarsa o zaman tahmin et
        # min_tracking_confidence=0.5 -> takip edilen landmark'ın hala güvenilir olup olmadığını tahmin eder (bozulma var mı yok mu vb.)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # +fps
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # landmark extract -> buffer
            lm = extract_landmarks_frame(results)
            frame_buffer.append(lm)
            frame_count += 1

            # buffer dolduğunda her 3 frame'de bir tahmin yapar (her seferinde predict() çağırıp tahmin yapınca fps çok düşüyor o yüzden 3 frame'de bir tahmin yapar..!)
            if len(frame_buffer) == TARGET_FRAMES and frame_count % 3 == 0:
                predictions = predict_sequence(frame_buffer)

                # cümle oluşturma için son STABILITY_COUNT tahmini aynı kelime + yüksek güven varsa eklenir
                top_gloss, top_conf = predictions[0]
                pred_history.append((top_gloss, top_conf))

                if len(pred_history) >= STABILITY_COUNT:
                    recent = list(pred_history)[-STABILITY_COUNT:]
                    all_same = all(g == top_gloss for g, _ in recent)
                    high_conf = all(c >= CONFIDENCE_THR for _, c in recent)
                    not_locked = (top_gloss != last_added) or (time.time() - last_added_time > LOCK_DURATION)

                    if all_same and high_conf and not_locked:
                        sentence.append(top_gloss)
                        last_added = top_gloss
                        last_added_time = time.time()

            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76),  thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # basic ui
            if predictions:
                draw_predictions(frame, predictions)
            draw_status(frame, len(frame_buffer), fps)
            draw_sentence(frame, sentence)

            cv2.imshow('Sign Language Recognition model V5', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '): # space = cüme + geçmişi siler
                sentence.clear()
                pred_history.clear()
            elif key == 8:  # backspace = son kelimeyi siler
                if sentence:
                    sentence.pop()

    cap.release()
    cv2.destroyAllWindows()
    print('kamera kapatıldı.')


if __name__ == '__main__':
    main()