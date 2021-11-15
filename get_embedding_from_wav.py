# TODO(phil)
import numpy as np

def get_embedding_from_wav(filename: str) -> np.array:
    return np.zeros((1,100))

if __name__ == "__main__":
    # Demo .wav test
    demo_wav = "demo_wav_path"  # TODO(phil): add demo wav
    embedding = get_embedding_from_wav(demo_wav)
    print(embedding)

    ################################################################################
    #       Given one .wav file + trained model, return one embedding.
    ################################################################################
    # Load .wav file from local filesystem. (or on a held out wav)

    # Create spectogram from .wav.

    # Model (tentative):
        # Define model (same as slim/train.py model).
        # Load .pb file of weights.

    # Run spectogram through model -> embedding Tensor 

    # Return embeddings