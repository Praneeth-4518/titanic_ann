import tensorflow as tf
import numpy as np

# --------------------------------
# SAMPLE TRAINING DATA
# --------------------------------

# Features:
# [Pclass, Age, Fare]

X = np.array([
    [1, 25, 100],
    [3, 40, 10],
    [2, 30, 50],
    [1, 22, 80],
    [3, 35, 15],
    [2, 28, 40],
    [1, 19, 120],
    [3, 50, 7]
], dtype=float)

# Labels:
# 1 = Survived
# 0 = Not Survived

y = np.array([
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    0
])

# --------------------------------
# NORMALIZATION
# --------------------------------

X[:, 0] = (X[:, 0] - 1) / 2
X[:, 1] = X[:, 1] / 100
X[:, 2] = X[:, 2] / 150

# --------------------------------
# BUILD MODEL
# --------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# --------------------------------
# COMPILE MODEL
# --------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --------------------------------
# TRAIN MODEL
# --------------------------------

model.fit(X, y, epochs=100, verbose=1)

# --------------------------------
# SAVE MODEL
# --------------------------------

model.save("titanic_ann_model.keras")

print("Model saved successfully!")