# TODO(greg or phil)

import CONSTANTS
import get_embedding_from_wav
import csv
import numpy as np
import ast

def similarity(a, b):
    a_norm = np.array(a) * 1/float(np.linalg.norm(a))
    b_norm = np.array(b) * 1/float(np.linalg.norm(b))
    return np.abs(a_norm.dot(b))

if __name__ == "__main__":
    # TODO: Write a simple test that checks implementation.
    
    ################################################################################
    #                           Inference of new, unseen example
    ################################################################################
    # Call get_embedding_from_wavs
    embedding = get_embedding_from_wav.get_embedding_from_wav(CONSTANTS.DEMO_WAV_PATH)

    # Load all embeddings from CSV.
    mydict = {}
    with open('/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/edna_mode/demo/mean_embeddings.csv', mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            k = row[0]
            string = row[1].replace(" ", ",")
            v = np.array(ast.literal_eval(string))
            mydict[k] = v

    # Compute distance of input_embedding to each mean_embedding.
    similarities = []
    for label in CONSTANTS.LABELS:
        similarities.append(similarity(embedding, mydict[label]))
    similarities = np.array(similarities)

    # Check if min distance is to admin or other.
    prediction = CONSTANTS.LABELS[np.argmax(similarities)]
    print(f"PREDICTION: {prediction}")


