import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading data...")

spectra = np.load(
    "processed/spectral_embeddings.npy"
)

fingerprints = np.load(
    "processed/morgan_fingerprints.npy"
)

print("Spectra shape:", spectra.shape)
print("Fingerprints shape:", fingerprints.shape)

# =====================================================
# NORMALIZE
# =====================================================

spectra = spectra.astype("float32")
fingerprints = fingerprints.astype("float32")

# =====================================================
# MODEL DIMENSIONS
# =====================================================

EMBED_DIM = 128

# =====================================================
# SPECTRUM ENCODER
# =====================================================

spectrum_input = layers.Input(
    shape=(spectra.shape[1],)
)

x = layers.Dense(
    256,
    activation="relu"
)(spectrum_input)

x = layers.Dense(
    128,
    activation="relu"
)(x)

spectrum_output = layers.Dense(
    EMBED_DIM
)(x)

spectrum_encoder = Model(
    spectrum_input,
    spectrum_output
)

# =====================================================
# MOLECULE ENCODER
# =====================================================

molecule_input = layers.Input(
    shape=(fingerprints.shape[1],)
)

y = layers.Dense(
    512,
    activation="relu"
)(molecule_input)

y = layers.Dense(
    256,
    activation="relu"
)(y)

molecule_output = layers.Dense(
    EMBED_DIM
)(y)

molecule_encoder = Model(
    molecule_input,
    molecule_output
)

# =====================================================
# OPTIMIZER
# =====================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-3
)

# =====================================================
# TRAINING LOOP
# =====================================================

BATCH_SIZE = 256
EPOCHS = 20

dataset = tf.data.Dataset.from_tensor_slices(
    (spectra, fingerprints)
)

dataset = dataset.shuffle(
    10000
).batch(BATCH_SIZE)

# =====================================================
# TRAIN
# =====================================================

print("\nTraining contrastive model...")

for epoch in range(EPOCHS):

    epoch_loss = 0.0
    steps = 0

    for batch_spectra, batch_mols in dataset:

        with tf.GradientTape() as tape:

            spec_embed = spectrum_encoder(
                batch_spectra,
                training=True
            )

            mol_embed = molecule_encoder(
                batch_mols,
                training=True
            )

            # Normalize embeddings
            spec_embed = tf.math.l2_normalize(
                spec_embed,
                axis=1
            )

            mol_embed = tf.math.l2_normalize(
                mol_embed,
                axis=1
            )

            # Similarity matrix
            logits = tf.matmul(
                spec_embed,
                mol_embed,
                transpose_b=True
            )

            labels = tf.range(
                tf.shape(logits)[0]
            )

            loss1 = tf.keras.losses.sparse_categorical_crossentropy(
                labels,
                logits,
                from_logits=True
            )

            loss2 = tf.keras.losses.sparse_categorical_crossentropy(
                labels,
                tf.transpose(logits),
                from_logits=True
            )

            loss = (
                tf.reduce_mean(loss1) +
                tf.reduce_mean(loss2)
            ) / 2

        variables = (
            spectrum_encoder.trainable_variables +
            molecule_encoder.trainable_variables
        )

        grads = tape.gradient(
            loss,
            variables
        )

        optimizer.apply_gradients(
            zip(grads, variables)
        )

        epoch_loss += float(loss)
        steps += 1

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"- Loss: {epoch_loss/steps:.4f}"
    )

# =====================================================
# SAVE
# =====================================================

spectrum_encoder.save(
    "processed/spectrum_encoder.keras"
)

molecule_encoder.save(
    "processed/molecule_encoder.keras"
)

print("\nSaved encoders.")
