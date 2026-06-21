import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

DATA_DIR = './hand_data'

sign_a = np.load(os.path.join(DATA_DIR, 'Sign_A.npy'))
sign_b = np.load(os.path.join(DATA_DIR, 'Sign_B.npy'))

X = np.concatenate([sign_a, sign_b], axis=0)

y = np.array([0] * len(sign_a) + [1] * len(sign_b))

print(f"Data shape X: {X.shape}")  # This should now perfectly match the length of y
print(f"Labels shape y: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(42, 2)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print("Training custom CNN model...")
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

model.save('sign_language_cnn.h5')
print("Model saved as sign_language_cnn.h5!")