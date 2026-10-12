import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# LOAD
# =====================================================

df = pd.read_parquet(
    "processed/continuous_spectra.parquet"
)

spectra = np.load(
    "processed/spectral_embeddings.npy"
).astype("float32")

# =====================================================
# LOAD ENCODER
# =====================================================

encoder = tf.keras.models.load_model(
    "processed/spectrum_encoder.keras"
)

# =====================================================
# EMBEDDINGS
# =====================================================

embeddings = encoder.predict(
    spectra,
    batch_size=256
)

embeddings = (
    embeddings /
    np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )
)

# =====================================================
# SIMILARITY
# =====================================================

similarity = cosine_similarity(
    embeddings
)

# =====================================================
# INSPECT RANDOM MOLECULES
# =====================================================

np.random.seed(42)

indices = np.random.choice(
    len(df),
    10,
    replace=False
)

for idx in indices:

    print("\n" + "="*70)

    query_name = df.iloc[idx]["molecule_name"]

    print(f"\nQUERY:")
    print(query_name)

    print("\nLABELS:")
    print(df.iloc[idx]["labels"])

    ranking = np.argsort(
        similarity[idx]
    )[::-1]

    print("\nNEAREST NEIGHBORS:\n")

    for rank in range(1, 6):

        nidx = ranking[rank]

        print(f"{rank}.")

        print(
            "Name:",
            df.iloc[nidx]["molecule_name"]
        )

        print(
            "Labels:",
            df.iloc[nidx]["labels"]
        )

        print(
            "Similarity:",
            round(
                similarity[idx][nidx],
                4
            )
        )

        print()
