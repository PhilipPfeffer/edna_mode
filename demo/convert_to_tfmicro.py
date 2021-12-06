################################################################################
#                   Convert a frozen model to TF Lite.
################################################################################
#   Call:
#       python demo/convert_to_tflite.py --saved_model_dir=PATH
#
#   e.g.
#     python demo/convert_to_tfmicro.py --saved_model_dir=/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/cs329e/edna_mode/demo/frozen_models/tiny_embedding_conv.ckpt-200
#
################################################################################

import subprocess
import os
import argparse

import CONSTANTS

def convert_model(MODEL_TFLITE: str, MODEL_TFLITE_MICRO: str):
    # Convert to a C source file
    os.system(f"xxd -i {MODEL_TFLITE} > {MODEL_TFLITE_MICRO}")

    # Update variable names
    REPLACE_TEXT = MODEL_TFLITE.replace('/', '_').replace('.', '_')
    os.system(f"sed -i 's/'{REPLACE_TEXT}'/g_model/g' {MODEL_TFLITE_MICRO}")
    

def print_model(MODEL_TFLITE_MICRO: str):
    os.system(f"cat {MODEL_TFLITE_MICRO}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args()
    
    MODEL_TFLITE_MICRO = os.path.join(CONSTANTS.TFLITE_MODEL_SAVE_PATH, 'model.cc')
    MODEL_TFLITE = os.path.join(CONSTANTS.TFLITE_MODEL_SAVE_PATH, 'model.tflite')
    
    convert_model(MODEL_TFLITE, MODEL_TFLITE_MICRO)
    print_model(MODEL_TFLITE_MICRO)