

import numpy as np
from pathlib import Path


feature_dir = Path(r'C:\Projects\sign_language\features')


# mediapipe pose için sol ve sağ landmark çiftleri
POSE_LR_PAIRS = [
    (1, 4),    # sol iç göz, sağ iç göz
    (2, 5),    # sol göz, sağ göz
    (3, 6),    # sol dış göz, sağ dış göz
    (7, 8),    # sol kulak, sağ kulak
    (9, 10),   # ağız sol, ağız sağ
    (11, 12),  # sol omuz, sağ omuz
    (13, 14),  # sol dirsek, sağ dirsek
    (15, 16),  # sol wrist, sağ wrist
    (17, 18),
    (19, 20),  # sol index, sağ index
    (21, 22),
    (23, 24),  # sol kalça, sağ kalça
    (25, 26),  # sol diz, sağ diz
    (27, 28),
    (29, 30),
    (31, 32),
]


def mirror_augmentation(sequence): # tüm dataset'te 2731 class için ortalama 15 örnek var. overfitting olmasın diye aynalama yöntemini deniyorum.
    # sol ve sağ flip -> x koordinatlarını negatifler, sol ve sağ eli değiştirir
    # feature layout -> pose(132) left_hand(63) right_hand(63)
    seq = sequence.copy()

    # pose flip
    for i in range(33):
        seq[:, i * 4] *= -1

    # pose lr swap
    for l, r in POSE_LR_PAIRS:
        l_slice = slice(l * 4, l * 4 + 4)
        r_slice = slice(r * 4, r * 4 + 4)
        tmp = seq[:, l_slice].copy()
        seq[:, l_slice] = seq[:, r_slice]
        seq[:, r_slice] = tmp

    # hand flip (sol ve sağ)
    left = seq[:, 132:195].copy()
    right = seq[:, 195:258].copy()
    for i in range(21):
        left[:, i * 3] *= -1
        right[:, i * 3] *= -1
    seq[:, 132:195] = right # sol -> sağ
    seq[:, 195:258] = left # sağ -> sol

    return seq


def landmark_augmentation(sequence): # landmark augmentation (30, 258) yani (timesteps, features_dim) uygulanıyor..!
    seq = sequence.copy()

    zero_mask = (seq == 0.0).all(axis=1)  # masking koruması için 0 olan frame'leri kaydediyorum.
    noise = np.random.normal(0, 0.005, seq.shape)
    seq = seq + noise
    seq[zero_mask] = 0.0  # sıfır frame'leri geri yükle

    # temporal shift edge pad
    shift = np.random.randint(-1, 2) # -1 ve +1 frame arasında kaydırsın
    if shift > 0:
        pad = np.repeat(seq[0:1], shift, axis=0)
        seq = np.concatenate([pad, seq[:-shift]], axis=0)
    elif shift < 0:
        pad = np.repeat(seq[-1:], -shift, axis=0)
        seq = np.concatenate([seq[-shift:], pad], axis=0)

    # linear interpolasyon
    if np.random.rand() < 0.5:
        speed_factor = np.random.uniform(0.9, 1.1)
        new_len = int(30 * speed_factor)
        old_idx = np.linspace(0, 29, 30)
        new_idx = np.linspace(0, 29, new_len)
        stretched = np.array([np.interp(new_idx, old_idx, seq[:, f]) for f in range(seq.shape[1])]).T
        resample_idx = np.linspace(0, new_len - 1, 30)
        seq = np.array([np.interp(resample_idx, np.arange(new_len), stretched[:, f]) for f in range(seq.shape[1])]).T

    return seq


def load_split(split, augment=False, selected_classes=None):  # train, val, test için X ve y verisi oluşturur. X -> (batch_size, 30, 258) y -> (batch_size,)
    # cache dosya adı -> split + augment + selected_classes bilgisini içerir
    cache_suffix = '_aug' if augment else ''
    if selected_classes is not None:
        cache_suffix += f'_top{len(selected_classes)}'
    cache_path = feature_dir / f'cache_{split}{cache_suffix}.npz'

    if cache_path.exists():
        print(f'{split} cache\'den yükleniyor: {cache_path.name}')
        cached = np.load(cache_path)
        X, y = cached['X'], cached['y']
        print(f'{split} loaded: X={X.shape}, y={y.shape}')
        return X, y

    # cache yoksa .npy dosyalarından yükler
    split_dir = feature_dir/split

    X = []
    y = []

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()],key=lambda x: int(x.name)) # train, val, test klasörlerindeki bütün sayı olan klasörleri (yani classları) label olarak alır..!

    for class_dir in class_dirs:
        label = int(class_dir.name)

        ### DEBUG için sadece seçilen class'ları alır ###
        if selected_classes is not None and label not in selected_classes:
            continue

        npy_files = list(class_dir.glob('*.npy'))

        for npy_path in npy_files:
            data = np.load(npy_path)

            if data.shape != (30, 258):
                continue

            X.append(data)
            y.append(label)

            if augment:
                # orijinal için 4 farklı augmented kopya -> toplam 6x veri (1 orijinal + 4 kopya + 1 mirror)
                # kopya farklı random seed ile farklı augmentation alır..!
                for _ in range(4):
                    X.append(landmark_augmentation(data))
                    y.append(label)
                X.append(mirror_augmentation(data))  # sol ve sağ flip'i signer çeşitliliği için yaptım (patlayabilir..!)
                y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # bir sonraki çalıştırma için cache'e kayıt eder
    print(f'{split} cache kaydediliyor: {cache_path.name} ...')
    np.savez_compressed(cache_path, X=X, y=y)
    print(f'{split} loaded: X={X.shape}, y={y.shape}')
    return X, y


if __name__ == '__main__':
    X_train, y_train = load_split('train')
    X_val, y_val = load_split('val')
    X_test, y_test = load_split('test')