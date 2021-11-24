################################################################################
#       Given one .wav file + trained model, return one embedding.
################################################################################

# TODO(phil)
import CONSTANTS
from demo.CONSTANTS import MODEL_CHECKPOINT_PATH
import numpy as np
import os
import subprocess
import argparse

def get_embedding_from_wav(filepath: str, embedding_size: int) -> np.array:
    command = f"python {CONSTANTS.REPO_FILEPATH}tensorflow/tensorflow/tensorflow/examples/speech_commands/train.py\
        --model_architecture=mobilenet_embedding \
        --data_dir={CONSTANTS.REPO_FILEPATH}/dataset \
        --batch_size 5 \
        --inference=True \
        --inference_checkpoint_path={MODEL_CHECKPOINT_PATH} \
        --embedding_size={embedding_size} \
        --query_file={filepath}"

    result = subprocess.check_output(command, shell=True)
    result = str(result)
    embedding_idx = result.find("[") + len("array([[") + 1
    post_array_str = result[embedding_idx:]
    end_idx = post_array_str.find("]")
    array_str = post_array_str[:end_idx]
    array_str = array_str.replace(" ", "").replace("\\", "").replace("n", "").replace("\n", "")
    array_strs = array_str.strip().split(',')
    array = [float(num) for num in array_strs]

    # Return embeddings
    return np.array(array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embedding_size',
        help='Size of embeddings used for this training run.')
    FLAGS, unparsed = parser.parse_known_args()
    
    
    # Demo .wav test
    embedding = get_embedding_from_wav(CONSTANTS.DEMO_WAV_PATH, FLAGS.embedding_size)
    print(embedding)
