################################################################################
#       Given one .wav file + trained model, return one embedding.
################################################################################

# TODO(phil)
import CONSTANTS
import numpy as np
import os
import subprocess

def get_embedding_from_wav(filepath: str) -> np.array:
    command = f"python {CONSTANTS.REPO_FILEPATH}/tensorflow/tensorflow/tensorflow/examples/speech_commands/train.py"
    #  \
    #     --model_architecture=mobilenet_embedding \
    #     --embedding_size=2 \
    #     --data_dir=/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/edna_mode/dataset \
    #     --batch_size 5 \
    #     --inference=True \
    #     --inference_checkpoint_path=/tmp/speech_commands_train/mobilenet_embedding.ckpt-0) \
    #     --query_file={filepath}"

    result = subprocess.check_output(command, shell=True)
    
    # Return embeddings
    return result

if __name__ == "__main__":
    # Demo .wav test
    embedding = get_embedding_from_wav(CONSTANTS.DEMO_WAV_PATH)
    assert(embedding != np.zeros((1,100)))
    print(embedding)
