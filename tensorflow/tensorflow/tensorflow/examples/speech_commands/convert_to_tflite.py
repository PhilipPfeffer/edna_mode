import os
import argparse
import pathlib

import numpy as np
import tensorflow as tf

import models
import input_data

def main(args):
    with tf.Session() as sess:
        MODELS_DIR = pathlib.Path(args.pb_dir).parent.absolute()

        MODEL_TF = os.path.join(MODELS_DIR, 'model.pb')
        MODEL_TFLITE = os.path.join(MODELS_DIR, 'model.tflite')
        FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, 'float_model.tflite')
        MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, 'model.cc')
        SAVED_MODEL = os.path.join(MODELS_DIR, 'saved_model')

        #model_dir = pathlib.Path(args.pb_file).parent.absolute()
        float_converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)  # NOTE: originally SAVED_MODEL not MODEL_TF
        float_tflite_model = float_converter.convert()
        float_tflite_model_size = open(FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
        print("Float model is %d bytes" % float_tflite_model_size)

        converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.lite.constants.INT8
        converter.inference_output_type = tf.lite.constants.INT8

        # DEFAULT ARGS
        WANTED_WORDS = ""
        SAMPLE_RATE = 16000
        CLIP_DURATION_MS = 1000
        WINDOW_SIZE_MS = 30.0
        WINDOW_STRIDE = 10.0
        FEATURE_BIN_COUNT = 40
        PREPROCESS = 'mfcc'
        EMBEDDING_SIZE = 50

        DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
        DATASET_DIR = args.data_dir
        SILENT_PERCENTAGE = 10.0
        UNKNOWN_PERCENTAGE = 10.0
        VALIDATION_PERCENTAGE = 10
        TESTING_PERCENTAGE = 10
        LOGS_DIR = '/tmp/retrain_logs'

        BACKGROUND_FREQUENCY = 0.0
        BACKGROUND_VOLUME_RANGE = 0.0
        TIME_SHIFT_MS = 0

        model_settings = models.prepare_model_settings(
            len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
            SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
            WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS, EMBEDDING_SIZE)
        audio_processor = input_data.AudioProcessor(
            DATA_URL, DATASET_DIR,
            SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE,
            WANTED_WORDS.split(','), VALIDATION_PERCENTAGE,
            TESTING_PERCENTAGE, model_settings, LOGS_DIR)

        def representative_dataset_gen():
            for i in range(100):
                data = audio_processor.get_data(1, i*1, model_settings,
                                                    BACKGROUND_FREQUENCY, 
                                                    BACKGROUND_VOLUME_RANGE,
                                                    TIME_SHIFT_MS,
                                                    "testing",
                                                    sess)
                flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, 3920)
                yield [flattened_data]
        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()
        tflite_model_size = open(MODEL_TFLITE, "wb").write(tflite_model)
        print("Quantized model is %d bytes" % tflite_model_size)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_dir', default='./models/model.pb', type=str)
    parser.add_argument('--data_dir', type=str)
    main(parser.parse_args())