################################################################################
#                           Inference of new, unseen example
################################################################################

import CONSTANTS
import get_embedding_from_wav
import csv
import numpy as np
import ast
import argparse

def similarity(a, b):
    a_norm = np.array(a) * 1/float(np.linalg.norm(a))
    b_norm = np.array(b) * 1/float(np.linalg.norm(b))
    return np.abs(a_norm.dot(b))

def inference(input_path: str) -> str:
    # Call get_embedding_from_wavs
    embedding = get_embedding_from_wav.get_embedding_from_wav(input_path)

    # Load all embeddings from CSV.
    mydict = {}
    with open(CONSTANTS.MEAN_EMBEDDINGS_PATH, mode='r') as infile:
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
    return prediction

if __name__ == "__main__":
    # TODO: Write a simple test that checks implementation.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        default=CONSTANTS.DEMO_WAV_PATH,
        help='Get prediction for wav file found at input_path.')
    FLAGS, unparsed = parser.parse_known_args()
    
    prediction = inference(FLAGS.input_path)
    print("========================")
    print(f"PREDICTION: {prediction}")
    print("========================")
