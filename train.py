

import joblib
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'GPU memory growth enabled..!')
    except RuntimeError as e:
        print(f'{e} GPU memory growth disabled..!')

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from load_dataset import load_split
from base_models import base_lstm_model, bidirectional_lstm_model, bidirectional_lstm_cnn_model, cnn_bilstm_attention_model


# tam dataset için = None (2731 class)
# top 100 testi için list(range(100)) -> alfabetik sırayla ilk 100 gloss (landmarks_extract ile aynı mapping) (modellerin varyasyonlarını bununla oluşturup test ettim)
selected_classes = None

X_train, y_train = load_split('train', augment=True, selected_classes=selected_classes)
X_val, y_val = load_split('val', augment=False, selected_classes=selected_classes)
X_test, y_test = load_split('test', augment=False, selected_classes=selected_classes)

# label remapping ile seçili class'ların label'larını 0,..n-1'e dönüştürüyorum..! (classları daha güvenli çalıştırması için)
unique_labels = sorted(set(np.unique(y_train)) | set(np.unique(y_val)) | set(np.unique(y_test)))
label_map = {old: new for new, old in enumerate(unique_labels)}
y_train = np.array([label_map[l] for l in y_train], dtype=np.int32)
y_val = np.array([label_map[l] for l in y_val], dtype=np.int32)
y_test = np.array([label_map[l] for l in y_test], dtype=np.int32)

num_classes = len(unique_labels)
print(f'train shape -> {X_train.shape}, {y_train.shape}')
print(f'val shape -> {X_val.shape}, {y_val.shape}')
print(f'num_classes -> {num_classes}')
print(f'unique train labels (remapped): {np.unique(y_train)}')

# standardscaler() ile (N, T, F) -> (N*T, F) şeklinde normalize ediyorum
scaler = StandardScaler()
X_train_2d = X_train.reshape(-1, X_train.shape[-1])
X_val_2d = X_val.reshape(-1, X_val.shape[-1])
X_test_2d = X_test.reshape(-1, X_test.shape[-1])

# scale öncesi sıfır frame'leri işaretliyorum (landmark tespiti olmayan frameler -> tüm 258 değer 0.0)
# standardscaler 0'ları (0-mean)/std'ye dönüştürür -> Masking layer artık bu frame'leri tanıyamaz..!
# scale sonrası bu frameleri tekrar 0.0'a döndürerek Masking'in çalışmasını ve algılamasını sağlıyorum..!
zero_mask_train = (X_train_2d == 0.0).all(axis=1)
zero_mask_val = (X_val_2d == 0.0).all(axis=1)
zero_mask_test = (X_test_2d == 0.0).all(axis=1)

X_train_2d_scaled = scaler.fit_transform(X_train_2d)
X_val_2d_scaled = scaler.transform(X_val_2d)
X_test_2d_scaled = scaler.transform(X_test_2d)

X_train_2d_scaled[zero_mask_train] = 0.0
X_val_2d_scaled[zero_mask_val] = 0.0
X_test_2d_scaled[zero_mask_test] = 0.0

X_train = X_train_2d_scaled.reshape(X_train.shape)
X_val = X_val_2d_scaled.reshape(X_val.shape)
X_test = X_test_2d_scaled.reshape(X_test.shape)

### eğitim sırasında yaklaşık 20 gb ram kullandı. bu da train ederken OOM hatasına sebep oldu. bu yüzden geçici değişkenleri görevi bittikten sonra bellekten siliyorum. yaklaşık 8.5gb yer açılıyor..!
del X_train_2d, X_train_2d_scaled
del X_val_2d, X_val_2d_scaled
del X_test_2d, X_test_2d_scaled

joblib.dump(scaler, 'models/standardscaler.save')

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=10, min_lr=1e-6, mode='max', verbose=1)

# CosineDecay lr başta 0.001'den başlar ve 100 epoch sonunda 0'a iner (planlanmış, reducelronplateu gibi reaktif değil..!) (!!!v2, v3, v4 için denedim ama sonuç daha kötü!!!)
# steps_per_epoch = len(X_train) // 64
# lr_schedule = CosineDecay(initial_learning_rate=0.0003, decay_steps=100 * steps_per_epoch, alpha=0.0) # final lr = 0

cnn_bilstm_attention = cnn_bilstm_attention_model(num_classes, learning_rate=0.0003)
history_cnn_bilstm_attention = cnn_bilstm_attention.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr], class_weight=class_weights_dict)
cnn_bilstm_attention.save('models/cnn_bilstm_attention_model_v5.h5')

test_loss, test_accuracy = cnn_bilstm_attention.evaluate(X_test, y_test, verbose=0)
print(f'cnn_bilstm_attention -> test_loss: {test_loss:.4f}, test_accuracy: {test_accuracy:.4f}')