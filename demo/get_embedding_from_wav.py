################################################################################
#       Given one .wav file + trained model, return one embedding.
################################################################################

# TODO(phil)
import CONSTANTS
from demo.CONSTANTS import MODEL_CHECKPOINT_PATH
import numpy as np
import os
import subprocess

def get_embedding_from_wav(filepath: str) -> np.array:
    command = f"python {CONSTANTS.REPO_FILEPATH}tensorflow/tensorflow/tensorflow/examples/speech_commands/train.py\
        --model_architecture=mobilenet_embedding \
        --data_dir={CONSTANTS.REPO_FILEPATH}/dataset \
        --batch_size 5 \
        --inference=True \
        --inference_checkpoint_path={MODEL_CHECKPOINT_PATH} \
        --embedding_size=2 \
        --query_file={filepath}"

    result = subprocess.check_output(command, shell=True)
    result = str(result)
    embedding_idx = result.find("[") + len("array([[") + 1
    post_array_str = result[embedding_idx:]
    end_idx = post_array_str.find("]")
    array_str = post_array_str[:end_idx]
    array_strs = array_str.strip().split(', ')
    array = [float(num) for num in array_strs]

    # Return embeddings
    return np.array(array)

if __name__ == "__main__":
    # Demo .wav test
    embedding = get_embedding_from_wav(CONSTANTS.DEMO_WAV_PATH)
    print(embedding)
