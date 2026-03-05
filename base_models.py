

from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, Masking, Dropout, Conv1D, BatchNormalization, LayerNormalization, MultiHeadAttention, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf


def base_lstm_model(num_classes):
    inputs = Input(shape=(30, 258))
    x = Masking(mask_value=0.0)(inputs) # landmark çıkarırken 30 frame'den daha az olan videoları 0 ile doldurdum. lstm bu sahte 0'ları gerçek frame sanmasın diye 0'ları maskeliyorum..!
    x = LSTM(128, activation='tanh', return_sequences=True)(x) # return_sequences=True -> (batch_size, timesteps, features) => 2.lstm katmanına gireceği için bu formatta yaptım.
    x = LSTM(128, activation='tanh', return_sequences=False)(x) # return_sequences=False -> (batch_size, features) => Dense katmanına gireceği için bu formatta yaptım.
    # ilk LSTM katmanında zamansal dizinin korunması için return_sequences=True kullanılmış, ikinci LSTM katmanı ise tüm zaman adımlarını özetleyen tek bir vektör return_sequences=False ile üretilmiştir.
    x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    base_lstm = Model(inputs, outputs)
    base_lstm.compile(optimizer=Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    base_lstm.summary() # etiketi integer formatta kullandığımdan 'sparse_categorical_crossentropy' kullandım..!
    return base_lstm


def bidirectional_lstm_model(num_classes):
    inputs = Input(shape=(30, 258))
    x = Masking(mask_value=0.0)(inputs)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    bidirectional_lstm = Model(inputs, outputs)
    bidirectional_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    bidirectional_lstm.summary()
    return bidirectional_lstm


def bidirectional_lstm_cnn_model(num_classes):
    inputs = Input(shape=(30, 258))
    # Masking katmanı Conv1D üzerinden mask propagation yapmaz (Keras limiti).
    # Masking burada kullanılmıyor ve sıfır frame'ler zaten train.py'de korunuyor.
    x = Conv1D(64, 3, activation=None, kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(64, 3, activation=None, kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Bidirectional(LSTM(64, activation='tanh', return_sequences=False, recurrent_dropout=0.0))(x)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    bidirectional_lstm_cnn = Model(inputs, outputs)
    bidirectional_lstm_cnn.compile(optimizer=Adam(learning_rate=0.0003), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    bidirectional_lstm_cnn.summary()
    return bidirectional_lstm_cnn


def cnn_bilstm_attention_model(num_classes, learning_rate=0.001): # en iyi model ve varyasyon
    inputs = Input(shape=(30, 258))
    x = Conv1D(64, 3, activation=None, kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(64, 3, activation=None, kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Bidirectional(LSTM(128, activation='tanh', return_sequences=True, recurrent_dropout=0.0))(x) # return_sequences=True -> tüm frame çıktıları korunur (batch_size, 30, 256)
    x = Dropout(0.3)(x) # bilstm çıktısını düzenlileştirmek için ekledim..! (0.3 -> 0.4)
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(x, x) # self attention -> her frame diğer frame'lere bakarak ağırlık hesaplıyor..!
    x = LayerNormalization()(x + attention)  # residual connection ile bilstm çıktısını (x) ve attention çıktısını toplayarak koruyorum..!
    x = tf.reduce_mean(x, axis=1) # 30 frame'i tek vektöre indiriyor. (256,)
    x = Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    cnn_bilstm_attention = Model(inputs, outputs)

    cnn_bilstm_attention.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_bilstm_attention.summary()
    return cnn_bilstm_attention