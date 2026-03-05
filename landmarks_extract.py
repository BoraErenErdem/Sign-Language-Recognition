

import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path


asl_citizen_dir = Path(r'C:\Projects\sign_language\ASL_Citizen')
video_dir = asl_citizen_dir / 'videos'
splits_dir = asl_citizen_dir / 'splits'
out_dir = Path(r'C:\Projects\sign_language\features')

target_frames = 30
feature_dim = 33 * 4 + 21 * 3 + 21 * 3  # pose, sağ el ve sol el için landmark boyutları ->    pose -> 33 * 4 (x,y,z,visibility)   el -> 21 * 3 (x,y,z)

mp_holistic = mp.solutions.holistic


def build_gloss_to_label():
    # tüm splitlerdeki unique gloss'ları toplayıp alfabetik sıraya göre 0 tabanlı index'e ata..!
    # sorted() -> her çalıştırmada aynı mapping elde edilmesi için kullandım..!
    all_glosses = set()
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(splits_dir / f'{split}.csv')
        all_glosses.update(df['Gloss'].tolist())
    return {gloss: idx for idx, gloss in enumerate(sorted(all_glosses))}


gloss_to_label = build_gloss_to_label()
print(f'Toplam class sayısı: {len(gloss_to_label)}')


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

    ### bu kısımda vücuda bağlı normalizasyon yapıyorum. omzun orta noktasını referans alıp torso (omuz-kalça) yüksekliğiyle scale ediyorum. böylece videodakiler nerede durursa dursun koordinatlar sabit kalıyor ###
    if results.pose_landmarks:
        # omuz orta noktası (landmark 11 = sol omuz, landmark 12 = sağ omuz)
        ls = pose[11 * 4: 11 * 4 + 3] # (x, y, z)
        rs = pose[12 * 4: 12 * 4 + 3]
        shoulder_mid = (ls + rs) / 2.0 # omuz ortası

        # kalça orta noktası (landmark 23 = sol kalça, landmark 24 = sağ kalça) -> torso scale
        lh = pose[23 * 4: 23 * 4 + 3] # (x, y, z)
        rh = pose[24 * 4: 24 * 4 + 3]
        hip_mid = (lh + rh) / 2.0 # kalça ortası

        torso_scale = np.linalg.norm(shoulder_mid[:2] - hip_mid[:2])  # 2D mesafe (x, y)
        if torso_scale < 0.000001:
            torso_scale = 1.0

        # pose koordinatlarının normalize edilmesi (visibility değişmez, sadece x,y,z normalize edilir..!)
        for i in range(33):
            pose[i * 4: i * 4 + 3] = (pose[i * 4: i * 4 + 3] - shoulder_mid) / torso_scale

        # sol el için önce wrist (bilek) (index 0) referansla el şeklini alır sonrasında vücuda göre konumlandırır
        if results.left_hand_landmarks:
            wrist = left_hand[0:3].copy()
            for i in range(21):
                left_hand[i * 3: i * 3 + 3] = (left_hand[i * 3: i * 3 + 3] - wrist) / torso_scale
            left_hand[0:3] = (wrist - shoulder_mid) / torso_scale # wrist konumunu vücuda bağlı olarak ilk landmarka yazar

        # sağ el için önce wrist (bilek) (index 0) referansla el şeklini alır sonra vücuda göre konumlandırır
        if results.right_hand_landmarks:
            wrist = right_hand[0:3].copy()
            for i in range(21):
                right_hand[i * 3: i * 3 + 3] = (right_hand[i * 3: i * 3 + 3] - wrist) / torso_scale
            right_hand[0:3] = (wrist - shoulder_mid) / torso_scale

    return np.concatenate([pose, left_hand, right_hand]) # en son pose, left hand ve right hand için olan landmarkları birleştiriyorum. böylece yapım -> (pose, left, right)


def temporal_resize(sequence, target_len): # frame sayısı target_len'e normalize ediliyor..! (örn. 30 frame)
    if len(sequence) == 0:
        return np.zeros((target_len, feature_dim))
    idxs = np.linspace(0, len(sequence) - 1, target_len).astype(int)
    return np.array([sequence[i] for i in idxs])


def process_split(split, selected_glosses=None):
    df = pd.read_csv(splits_dir / f'{split}.csv')

    # top 100 class ile hızlı test edip prototip çıkarmak için..!
    if selected_glosses is not None:
        df = df[df['Gloss'].isin(selected_glosses)].reset_index(drop=True)

    out_split_dir = out_dir / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    total = len(df)
    processed = 0
    skip_miss = 0
    skip_empty = 0
    done = 0

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=False, refine_face_landmarks=False) as holistic:
        for _, row in tqdm(df.iterrows(), total=total, desc=f'process {split}'):
            gloss = row['Gloss']
            video_name = row['Video file']
            label = gloss_to_label[gloss]

            video_path = video_dir / video_name
            out_path = out_split_dir / str(label) / (Path(video_name).stem + '.npy')

            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                done += 1
                continue

            if not video_path.exists():
                skip_miss += 1
                continue

            captures = cv2.VideoCapture(str(video_path)) # videodan landmarkları çıkarmak için videoları açar
            frames = []

            try:
                while True:
                    ret, frame = captures.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # mediapipe rgb formatında veri bekler o yüzden frameler rgb'ye dönüştürülür
                    results = holistic.process(frame_rgb) # holistic.preprocess() ile frame_rgb'ler mediapipe holistic modelinde ön işlenir ve landmark tespiti olur
                    lm = extract_landmarks_frame(results) # extract_landmarks_frame() fonksiyonuyla results kısmında ön işlenen landmarklar çıkarılır tek boyutlu vektör olur (258,)
                    frames.append(lm) # frames listesine çıkarılan landmarklar eklenir
            finally:
                captures.release()

            if len(frames) == 0:
                skip_empty += 1
                continue

            frames = temporal_resize(frames, target_frames) # zaman boyutunu normalize eder. (30, 258) frames -> videodaki frame sayısı   target_frames -> hedeflenen frame (30)
            np.save(str(out_path), frames)
            processed += 1

    print(f'\n{"=" * 50}')
    print(f'{split.upper()} özet')
    print(f'{"=" * 50}')
    print(f'toplam: {total}')
    print(f'işlendi: {processed}')
    print(f'mevcut (atlandı): {done}')
    print(f'video bulunamadı: {skip_miss}')
    print(f'boş video: {skip_empty}')
    print(f'{"=" * 50}\n')


def main():
    print(f'feature dim: {feature_dim}')
    print(f'target frames: {target_frames}')
    print(f'output shape: ({target_frames}, {feature_dim})\n')

    # tam dataset için selected_glosses=None yap..!
    # top 100 class ile hızlı test için -> selected_glosses = set(sorted(gloss_to_label.keys())[:100])
    selected_glosses = None

    for split in ['train', 'val', 'test']:
        process_split(split, selected_glosses=selected_glosses)

    print('tüm splitler tamamlandı')


if __name__ == '__main__':
    main()