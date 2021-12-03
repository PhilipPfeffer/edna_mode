################################################################################
#                           Create Mean Embeddings Phase
################################################################################
# Modify CONSTANTS.MODEL_CHECKPOINT_PATH if needed, then
# Call:
#   python demo/create_mean_embeddings.py --embedding_size N  
#
# Description:
#   Loop over all examples in each label, call get_embedding_from_wavs()
#   Calculate mean embedding for each label.
#   Calculate threshold: largest pairwise distance between embeddings of each label.
#   Store mean embeddings to csv.
#   Store threshold to csv.
################################################################################

import CONSTANTS
import get_embedding_from_wav
import os
import numpy as np
import argparse
import inference

def create_mean_embeddings(embedding_size: int):
    mean_embeddings = {}
    max_distances = {}

    #   Loop over all examples in each label
    for label in os.scandir(CONSTANTS.DATASET_FILEPATH):
        if label.name in CONSTANTS.LABELS: # Only consider real labels, i.e. not __backgorund_noise__
            # Concat embedding for each wav file.
            label_embeddings = np.array([])
            for wav_file in os.scandir(label):
                if len(wav_file.name.split('.')) == 2:  # Only consider .wav files, not .wav.old
                    new_embedding = get_embedding_from_wav.get_embedding_from_wav(wav_file.path, embedding_size)
                    label_embeddings = np.concatenate((label_embeddings, new_embedding),axis=0)

            # Calculate mean embedding for each label.
            label_embeddings = label_embeddings.reshape(-1, embedding_size)
            mean_embedding = np.average(label_embeddings, axis=0)
            mean_embeddings[label.name] = mean_embedding
            print(f"Mean embedding for {label.name}: {mean_embedding}")

            # Calculate pairwise distances.
            label_distances = []
            for idx_i, embedding_i in enumerate(label_embeddings):
                for idx_j, embedding_j in enumerate(label_embeddings):
                    if idx_i != idx_j:
                        dist_ij = inference.dist(embedding_i, embedding_j)
                        label_distances.append(dist_ij)
            max_dist = np.max(label_distances)
            max_distances[label.name] = max_dist
            print(f"Max distance for {label.name}: {max_dist}")

    # Store mean embeddings to csv.
    with open(CONSTANTS.MEAN_EMBEDDINGS_PATH, 'w') as f:
        for key in mean_embeddings.keys():
            f.write("%s"%(key))
            for i in range(embedding_size):
                f.write(",%s"%(mean_embeddings[key][i]))
            f.write("\n")
    
    # Store threshold to csv.
    with open(CONSTANTS.THRESHOLD_EMBEDDINGS_PATH, 'w') as f:
        for key in max_distances.keys():
            f.write("%s"%(key))
            f.write(",%s"%(max_distances[key]))
            f.write("\n")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embedding_size',
        help='Size of embeddings used for this training run.')
    FLAGS, unparsed = parser.parse_known_args()
    
    prediction = create_mean_embeddings(int(FLAGS.embedding_size))