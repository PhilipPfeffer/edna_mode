import os
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

import umap
import umap.plot

import CONSTANTS
import get_embedding_from_wav

def visualize_test_embs(embedding_size: int):
    labels = []
    column_names = [f"emb_{i}" for i in range(embedding_size)]
    df = pd.DataFrame(columns=column_names)

    for label in os.scandir(CONSTANTS.DATASET_FILEPATH):
        if label.name in CONSTANTS.LABELS: # Only consider real labels, i.e. not __backgorund_noise__
            for wav_file in os.scandir(label):
                if len(wav_file.name.split('.')) == 2:  # Only consider .wav files, not .wav.old
                    new_embedding = get_embedding_from_wav.get_embedding_from_wav(wav_file.path, embedding_size)
                    new_emb_dict = {f"emb_{i}": val for i, val in enumerate(new_embedding)}
                    df = df.append(new_emb_dict, ignore_index=True)
                    labels.append(label.name)

    print(df)            
    mapper = umap.UMAP().fit(df.values)
    umap.plot.points(mapper, labels=np.array(labels))
    plt.title("UMAP Embeddings of Testing Data")
    plt.show()
    plt.savefig('umap.png')

def visualize_mean_embs(embedding_size: int):
    # Load all embeddings from CSV.
    mean_embeddings = []
    mean_embeddings_labels = []
    with open(CONSTANTS.MEAN_EMBEDDINGS_PATH, mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            label = row[0]
            embedding = [float(num_str) for num_str in row[1:]]
            mean_embeddings.append(embedding)
            mean_embeddings_labels.append(label)
            
    labels = []
    column_names = [f"emb_{i}" for i in range(embedding_size)]
    df = pd.DataFrame(columns=column_names)
    for label in os.scandir(CONSTANTS.DATASET_FILEPATH):
        if label.name in CONSTANTS.LABELS: # Only consider real labels, i.e. not __backgorund_noise__
            for wav_file in os.scandir(label):
                if len(wav_file.name.split('.')) == 2:  # Only consider .wav files, not .wav.old
                    new_embedding = get_embedding_from_wav.get_embedding_from_wav(wav_file.path, embedding_size)
                    new_emb_dict = {f"emb_{i}": val for i, val in enumerate(new_embedding)}
                    df = df.append(new_emb_dict, ignore_index=True)
                    labels.append(label.name)
        

    pca = PCA(n_components=3)
    pca.fit(df.values)
    # pca.fit(np.array(mean_embeddings)) 
    mean_embeddings_pca = pca.transform(mean_embeddings)
    print(f"mean embeddings after PCA:\n{mean_embeddings_pca}")
    reference_embeddings_pca = pca.transform(df.values)
    print(f"reference embeddings after PCA:\n{reference_embeddings_pca}")

    # Plot mean embeddings
    origin = np.array([[0, 0, 0],[0, 0, 0], [0, 0, 0]]) # origin point
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_title(f"PCA Decomposition of {embedding_size}-d Mean Embeddings")

    color_map = ['r', 'g', 'b']
    for i in range(len(mean_embeddings_pca)):
        ax.quiver(*origin, mean_embeddings_pca[i,0], mean_embeddings_pca[i,1], mean_embeddings_pca[i,2], color=color_map[i], normalize=True)
    ax.legend(mean_embeddings_labels)

    # Plot reference embeddings
    label_to_color_map = {label: color_map[i] for i, label in enumerate(mean_embeddings_labels)}
    for i, label in enumerate(labels):
        ax.scatter(reference_embeddings_pca[i,0], reference_embeddings_pca[i,1], reference_embeddings_pca[i,2], marker='o', color=label_to_color_map[label])

    plt.show()
    plt.savefig('mean_embeddings_pca.png')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embedding_size', type=int,
        help='Size of embeddings used for this training run.')
    parser.add_argument('--visualize_mean_embs', dest='visualize_mean_embs', action='store_true', default=False)
    parser.add_argument('--visualize_test_embs', dest='visualize_test_embs', action='store_true', default=False)

    FLAGS, unparsed = parser.parse_known_args()
    
    if FLAGS.visualize_mean_embs:
        print("Visualizing mean embeddings using PCA...")
        visualize_mean_embs(FLAGS.embedding_size)
    
    if FLAGS.visualize_test_embs:
        print("Visualizing test embeddings using UMAP...")
        visualize_test_embs(FLAGS.embedding_size)