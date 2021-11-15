################################################################################
#       Given one .wav file + trained model, return one embedding.
################################################################################

# TODO(phil)
import CONSTANTS
import numpy as np
import os

def get_embedding_from_wav(filepath: str) -> np.array:
    # Load .wav file from local filesystem. (or on a held out wav)

    # Create spectogram from .wav.

    # Model (tentative):
        # Define model (same as slim/train.py model).
        # Load .pb file of weights.

    # Run spectogram through model -> embedding Tensor 

    # Return embeddings
    return np.zeros((1,100))

if __name__ == "__main__":
    # Demo .wav test
    embedding = get_embedding_from_wav(CONSTANTS.DEMO_WAV_PATH)
    assert(embedding != np.zeros((1,100)))
    print(embedding)
