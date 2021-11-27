################################################################################
#                   Convert a frozen model to TF Lite.
################################################################################
#   Call:
#       python demo/convert_to_tflite.py --data_dir=PATH --input_model_dir=PATH --embedding_size=N
#
#   e.g. --data_dir=/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/cs329e/edna_mode/dataset
#   e.g. --input_model_dir=/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/cs329e/edna_mode/demo/frozen_models/mobilenet_embedding_frozen.ckpt-2100
################################################################################

import sys
import CONSTANTS

# We add this path so we can import the speech processing modules.
sys.path.append(f"{CONSTANTS.REPO_FILEPATH}/tensorflow/tensorflow/tensorflow/examples/speech_commands/")
import input_data
import models

import numpy as np
import tensorflow as tf
import argparse
import os


def convert_to_tflite(data_dir: str, saved_model_dir: str, embedding_size: int):
  DATA_URL = ""
  model_settings = models.prepare_model_settings(
      len(CONSTANTS.LABELS) + 2,
      CONSTANTS.SAMPLE_RATE, CONSTANTS.CLIP_DURATION_MS, CONSTANTS.WINDOW_SIZE_MS,
      CONSTANTS.WINDOW_STRIDE, CONSTANTS.FEATURE_BIN_COUNT, CONSTANTS.PREPROCESS, embedding_size)
  audio_processor = input_data.AudioProcessor(
      DATA_URL, data_dir,
      CONSTANTS.SILENT_PERCENTAGE, CONSTANTS.UNKNOWN_PERCENTAGE,
      CONSTANTS.LABELS, CONSTANTS.VALIDATION_PERCENTAGE,
      CONSTANTS.TESTING_PERCENTAGE, model_settings, CONSTANTS.LOGS_DIR)

  # MODEL_TF = os.path.join(saved_model_dir, 'model.pb')
  MODEL_TFLITE = os.path.join(saved_model_dir, 'model.tflite')
  FLOAT_MODEL_TFLITE = os.path.join(saved_model_dir, 'float_model.tflite')
  # MODEL_TFLITE_MICRO = os.path.join(saved_model_dir, 'model.cc')
  SAVED_MODEL = saved_model_dir #os.path.join(, 'saved_model')

  with tf.compat.v1.Session() as sess:
    float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    float_tflite_model = float_converter.convert()
    float_tflite_model_size = open(FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
    print("Float model is %d bytes" % float_tflite_model_size)

    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.lite.constants.INT8
    converter.inference_output_type = tf.lite.constants.INT8

    def representative_dataset_gen():
      for i in range(100):
        data = audio_processor.get_data(1, i*1, model_settings,
                                          CONSTANTS.BACKGROUND_FREQUENCY,
                                          CONSTANTS.BACKGROUND_VOLUME_RANGE,
                                          CONSTANTS.TIME_SHIFT_MS,
                                          'testing',
                                          sess)
        flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, 1960)
        yield [flattened_data]

    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()
    tflite_model_size = open(MODEL_TFLITE, "wb").write(tflite_model)
    print("Quantized model is %d bytes" % tflite_model_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='Where the dataset is saved to. Used to calculate representative dataset.')
    parser.add_argument(
      '--input_model_dir',
      help='Path to input (frozen) model directory.')
    parser.add_argument(
      '--embedding_size',
      help='Size of embeddings used for this training run.')
    FLAGS, unparsed = parser.parse_known_args()

    convert_to_tflite(FLAGS.data_dir, FLAGS.input_model_dir, FLAGS.embedding_size)
