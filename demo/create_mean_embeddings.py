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
#   Store mean embeddings to csv.
################################################################################

import CONSTANTS
import get_embedding_from_wav
import os
import numpy as np
import argparse
import csv

def create_mean_embeddings(embedding_size: int):
    embeddings = {}
    for label in os.scandir(CONSTANTS.DATASET_FILEPATH):
        if label.name in CONSTANTS.LABELS: # Only consider real labels, i.e. not __backgorund_noise__
            label_embeddings = np.array([])
            for wav_file in os.scandir(label):
                if len(wav_file.name.split('.')) == 2:  # Only consider .wav files, not .wav.old
                    new_embedding = get_embedding_from_wav.get_embedding_from_wav(wav_file.path, embedding_size)
                    label_embeddings = np.concatenate((label_embeddings, new_embedding),axis=0)
            
            reshaped = label_embeddings.reshape(-1, embedding_size)
            mean_embedding = np.average(reshaped, axis=0)
            print(mean_embedding)
            embeddings[label.name] = mean_embedding

    with open(CONSTANTS.MEAN_EMBEDDINGS_PATH, 'w') as f:
        for key in embeddings.keys():
            f.write("%s"%(key))
            for i in range(embedding_size):
                f.write(",%s"%(embeddings[key][i]))
            f.write("\n")

def print_mean_embeddings():
    label_string = 'const char *mean_embeddings_labels[num_labels] = {'
    mean_embedding_string = 'const float mean_embeddings[num_labels][embedding_size] = {\n'
    labels = []
    with open(CONSTANTS.MEAN_EMBEDDINGS_PATH, mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            label = row[0]
            embedding = [str(num_str) for num_str in row[1:]]
            labels.append(f'"{label}"')
            mean_embedding_string += '{' + ','.join(embedding) + '},\n'
            

    label_string += ','.join(labels) + '};'
    mean_embedding_string += '};'
    print(label_string)
    print(mean_embedding_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embedding_size',
        help='Size of embeddings used for this training run.')
    FLAGS, unparsed = parser.parse_known_args()
    
    create_mean_embeddings(int(FLAGS.embedding_size))
    print_mean_embeddings()
