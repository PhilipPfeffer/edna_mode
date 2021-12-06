################################################################################
#                           Inference of new, unseen example
################################################################################
# Call:
#   python demo/inference.py --input_path=PATH --embedding_size=N  
################################################################################

import CONSTANTS
import get_embedding_from_wav
import csv
import numpy as np
import ast
import argparse
import re

def similarity(a, b):
    a_norm = np.array(a) * 1/float(np.linalg.norm(a))
    b_norm = np.array(b) * 1/float(np.linalg.norm(b))
    return np.abs(a_norm.dot(b_norm))

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def inference(query_embedding: np.ndarray) -> str:
    # Load all embeddings from CSV.
    mean_embeddings = {}
    with open(CONSTANTS.MEAN_EMBEDDINGS_PATH, mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            label = row[0]
            embedding = [float(num_str) for num_str in row[1:]]
            mean_embeddings[label] = embedding

    # Compute distance of input_embedding to each mean_embedding.
    distances = []
    for person, mean_embedding in mean_embeddings.items():
        distances.append((dist(query_embedding, mean_embedding), person))
    print(distances)

    # Check if min distance is to admin or other.
    prediction = min(distances, key=lambda x: x[0])
    return prediction[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        default=CONSTANTS.DEMO_WAV_PATH,
        help='Get prediction for wav file found at input_path.')
    parser.add_argument(
        '--embedding_size',
        help='Size of embeddings used for this training run.')
    FLAGS, unparsed = parser.parse_known_args()    

    # Call get_embedding_from_wavs
    query_embedding = get_embedding_from_wav.get_embedding_from_wav(FLAGS.input_path, FLAGS.embedding_size)
    prediction = inference(query_embedding)
    print("========================")
    print(f"PREDICTION: {prediction}")
    print("========================")
