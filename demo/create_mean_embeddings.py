# TODO(greg)

import CONSTANTS
import get_embedding_from_wav
import os
import numpy as np
import csv


if __name__ == "__main__":
    embeddings = {}
    for label in os.scandir(CONSTANTS.DATASET_FILEPATH):
        if label.name in CONSTANTS.LABELS: # Only consider real labels, i.e. not __backgorund_noise__
            label_embeddings = np.array([])
            for wav_file in os.scandir(label):
                if len(wav_file.name.split('.')) == 2:  # Only consider .wav files, not .wav.old
                    # print(wav_file.path)
                    new_embedding = get_embedding_from_wav.get_embedding_from_wav(wav_file.path)
                    label_embeddings = np.concatenate((label_embeddings, new_embedding),axis=0)
                    print(label_embeddings)
            reshaped = label_embeddings.reshape(-1,2)
            mean_embedding = np.average(reshaped, axis=0)
            print(mean_embedding)
            embeddings[label.name] = mean_embedding

    with open('/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/edna_mode/demo/mean_embeddings.csv', 'w') as f:
        for key in embeddings.keys():
            f.write("%s,%s\n"%(key, embeddings[key]))


    





    ################################################################################
    #                           Create Mean Embeddings Phase
    ################################################################################
    # Loop over all examples in each label, call get_embedding_from_wavs()

    # Calculate mean embedding for each label.

    # Store mean embeddings to csv.

