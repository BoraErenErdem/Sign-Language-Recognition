

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import joblib



df = pd.read_csv('dataset/asl_landmarks.csv', header=None)

X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

X = X.reshape(-1, 21, 3) # bileğe göre normalizasyon..!
base = X[:, 0:1, :]
X = X - base
X = X.reshape(-1, 63)

le = LabelEncoder() # sinir ağlarına sayı olarak verebilmek için label encoder ile harfleri sayıya çevirdim..!
y_encoder = le.fit_transform(y)
y_categorical = to_categorical(y_encoder) # multi class (softmax) olduğu için sayıları one-hot vektörüne çevirdim..!

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, stratify=y_encoder, random_state=42)

inputs = Input(shape=(63,))
x = Dense(256, activation='relu', kernel_initializer='he_uniform')(inputs)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
x = Dropout(0.4)(x)
outputs = Dense(y_categorical.shape[1], activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

model.save('asl_model.h5')
joblib.dump(le, 'label_encoder.pkl')