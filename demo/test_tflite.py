################################################################################
#          Verify that the model we've exported is still accurate.
################################################################################
# Modify CONSTANTS.MODEL_CHECKPOINT_PATH if needed, then
# Call:
#   python demo/test_tflite.py --data_dir=PATH --saved_model_dir=PATH --embedding_size=N --run_quantized
#
#   e.g.
#     python demo/test_tflite.py --data_dir=/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/cs329e/edna_mode/dataset --saved_model_dir=/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/cs329e/edna_mode/demo/frozen_models/mobilenet_embedding_frozen.ckpt-200 --embedding_size=50 --run_quantized
################################################################################

import CONSTANTS
import sys
import argparse
import os
import numpy as np
import tensorflow as tf

# We add this path so we can import the speech processing modules.
sys.path.append(f"{CONSTANTS.REPO_FILEPATH}/tensorflow/tensorflow/tensorflow/examples/speech_commands/")
import input_data
import models

def get_test_data(data_dir, embedding_size):
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

  # Load test data
  np.random.seed(0) # set random seed for reproducible test results.
  with tf.compat.v1.Session() as sess:
    test_data, test_labels = audio_processor.get_data(
        -1, 0, model_settings, CONSTANTS.BACKGROUND_FREQUENCY, CONSTANTS.BACKGROUND_VOLUME_RANGE,
        CONSTANTS.TIME_SHIFT_MS, 'testing', sess)
  
  return test_data, test_labels

# Runs inference.
def run_tflite_inference(test_data, test_labels, tflite_model_path,  model_type="Float"):

  test_data = np.expand_dims(test_data, axis=1).astype(np.float32)

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(tflite_model_path)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  # For quantized models, manually quantize the input data from float to integer
  if model_type == "Quantized":
    input_scale, input_zero_point = input_details["quantization"]
    test_data = test_data / input_scale + input_zero_point
    test_data = test_data.astype(input_details["dtype"])

  correct_predictions = 0
  for i in range(len(test_data)):
    interpreter.set_tensor(input_details["index"], test_data[i])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    top_prediction = output.argmax()
    correct_predictions += (top_prediction == test_labels[i])

  print('%s model accuracy is %f%% (Number of test samples=%d)' % (
      model_type, (correct_predictions * 100) / len(test_data), len(test_data)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='Where the dataset is saved to. Used to calculate representative dataset.')
    parser.add_argument(
      '--saved_model_dir',
      help='Path to input (frozen) model directory.')
    parser.add_argument(
      '--embedding_size',
      help='Size of embeddings used for this training run.')
    parser.add_argument(
      '--run_quantized',
      action="store_true",
      help='Run quantized model.')
    FLAGS, unparsed = parser.parse_known_args()

    MODEL_TFLITE = os.path.join(FLAGS.saved_model_dir, 'model.tflite')
    FLOAT_MODEL_TFLITE = os.path.join(FLAGS.saved_model_dir, 'float_model.tflite')

    test_data, test_labels = get_test_data(FLAGS.data_dir, FLAGS.embedding_size)

    if FLAGS.run_quantized:
      # Compute quantized model accuracy
      run_tflite_inference(test_data, test_labels, MODEL_TFLITE, model_type='Quantized')
    else:
      # Compute float model accuracy
      run_tflite_inference(test_data, test_labels, FLOAT_MODEL_TFLITE)
    