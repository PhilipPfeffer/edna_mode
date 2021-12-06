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
import os
import numpy as np
import argparse
import csv
import inference
import tensorflow as tf

from get_embedding_from_wav import get_embedding_from_wav
from test_tflite import get_test_data

def create_mean_embeddings_quant(embedding_size: int, test_data: np.ndarray, test_labels):
    label_to_embeddings = {}
    mean_embeddings = {}
    max_distances = {}

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(os.path.join(CONSTANTS.TFLITE_MODEL_SAVE_PATH, 'model.tflite'))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # For quantized models, manually quantize the input data from float to integer:
    input_scale, input_zero_point = input_details["quantization"]
    test_data = test_data / input_scale + input_zero_point
    test_data = test_data.astype(input_details["dtype"])

    for data, label in zip(test_data, test_labels):
        interpreter.set_tensor(input_details["index"], data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        if label not in label_to_embeddings:
            label_to_embeddings[label] = [output]
        else:
            label_to_embeddings[label].append(output)
    
    for label, embeddings in label_to_embeddings.items():
        mean_embedding = np.average(np.vstack(embeddings), axis=0)
        mean_embeddings[label] = mean_embedding
        print(f"Mean embedding for {label}: {mean_embedding}")

        label_distances = []
        for idx_i, embedding_i in enumerate(embeddings):
            for idx_j, embedding_j in enumerate(embeddings):
                if idx_i != idx_j:
                    dist_ij = inference.dist(embedding_i, embedding_j)
                    label_distances.append(dist_ij)
        max_dist = np.max(label_distances)
        max_distances[label] = max_dist
        print(f"Max distance for {label}: {max_dist}")
 
    # Store mean embeddings to csv.
    with open(CONSTANTS.MEAN_EMBEDDINGS_QUANT_PATH, 'w') as f:
        for key in mean_embeddings.keys():
            f.write("%s"%(key))
            for i in range(embedding_size):
                f.write(",%s"%(mean_embeddings[key][i]))
            f.write("\n")
    
    # Store threshold to csv.
    with open(CONSTANTS.THRESHOLD_EMBEDDINGS_QUANT_PATH, 'w') as f:
        for key in max_distances.keys():
            f.write("%s"%(key))
            f.write(",%s"%(max_distances[key]))
            f.write("\n")

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
                    new_embedding = get_embedding_from_wav(wav_file.path, embedding_size)
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
    with open(CONSTANTS.MEAN_EMBEDDINGS_FLOAT_PATH, 'w') as f:
        for key in mean_embeddings.keys():
            f.write("%s"%(key))
            for i in range(embedding_size):
                f.write(",%s"%(mean_embeddings[key][i]))
            f.write("\n")
    
    # Store threshold to csv.
    with open(CONSTANTS.THRESHOLD_EMBEDDINGS_FLOAT_PATH, 'w') as f:
        for key in max_distances.keys():
            f.write("%s"%(key))
            f.write(",%s"%(max_distances[key]))
            f.write("\n")

def print_mean_embeddings(run_quantized: bool):
    label_string = 'const char *mean_embeddings_labels[num_labels] = {'
    mean_embedding_string = 'const float mean_embeddings[num_labels][embedding_size] = {\n'
    threshold_string = 'const float thresholds[num_labels] = {'
    labels = []
    keys = []
    
    MEAN_EMBEDDINGS_PATH = CONSTANTS.MEAN_EMBEDDINGS_QUANT_PATH if run_quantized else CONSTANTS.MEAN_EMBEDDINGS_FLOAT_PATH
    THRESHOLD_EMBEDDINGS_PATH = CONSTANTS.THRESHOLD_EMBEDDINGS_QUANT_PATH if run_quantized else CONSTANTS.THRESHOLD_EMBEDDINGS_FLOAT_PATH

    with open(MEAN_EMBEDDINGS_PATH, mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            label = row[0]
            embedding = [str(num_str) for num_str in row[1:]]
            labels.append(f'"{label}"')
            keys.append(label)
            mean_embedding_string += '{' + ','.join(embedding) + '},\n'
    
    threshold_dict = {}
    with open(THRESHOLD_EMBEDDINGS_PATH, mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            label = row[0]
            threshold = row[1]
            threshold_dict[label] = threshold

    label_string += ','.join(labels) + '};'
    mean_embedding_string += '};'
    threshold_string += ','.join([threshold_dict[label] for label in keys]) + '};'

    print()
    
    print('const char *unknown_label = "UNKNOWN USER";')
    print(label_string)
    print(mean_embedding_string)
    print(threshold_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embedding_size',
        help='Size of embeddings used for this training run.')
    parser.add_argument(
      '--run_quantized',
      action="store_true",
      help='Run quantized model.',
      default=False)
    FLAGS, unparsed = parser.parse_known_args()
    
    test_data = None
    if FLAGS.run_quantized:
        test_data, test_labels = get_test_data(CONSTANTS.DATASET_FILEPATH, FLAGS.embedding_size)
        test_data = np.expand_dims(test_data, axis=1).astype(np.float32)
        create_mean_embeddings_quant(int(FLAGS.embedding_size), test_data, test_labels)
    else: 
        create_mean_embeddings(int(FLAGS.embedding_size))

    print_mean_embeddings(FLAGS.run_quantized)
