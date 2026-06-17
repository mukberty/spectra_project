import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# =====================================================
# LOAD DATA
# =====================================================

X_train = np.load("processed/X_train.npy")
X_test = np.load("processed/X_test.npy")

y_train = np.load("processed/y_train.npy")
y_test = np.load("processed/y_test.npy")

# =====================================================
# ADD CHANNEL DIMENSION
# =====================================================

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(X_train.shape)

# =====================================================
# MODEL
# =====================================================

model = models.Sequential([

    layers.Conv1D(
        32,
        7,
        activation="relu",
        input_shape=X_train.shape[1:]
    ),

    layers.MaxPooling1D(2),

    layers.Conv1D(
        64,
        5,
        activation="relu"
    ),

    layers.MaxPooling1D(2),

    layers.Conv1D(
        128,
        5,
        activation="relu"
    ),

    layers.GlobalAveragePooling1D(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        y_train.shape[1],
        activation="sigmoid"
    )
])

# =====================================================
# COMPILE
# =====================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# TRAIN
# =====================================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=20,
    batch_size=32
)

# =====================================================
# SAVE
# =====================================================

model.save("processed/raman_cnn.keras")

print("\nModel saved.")
