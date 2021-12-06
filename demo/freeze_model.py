################################################################################
#                   Freeze the model for quantization.
################################################################################
#   Call:
#       python demo/freeze_model.py --save_path=PATH --embedding_size=N
#
#   Please save into demo/frozen_models/ directory!
#   e.g. --save_path=/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/cs329e/edna_mode/demo/frozen_models/mobilenet_embedding_frozen.ckpt-2100
################################################################################


import CONSTANTS
import subprocess
import argparse

def freeze_model(embedding_size: str):
    command = f"\
        python {CONSTANTS.REPO_FILEPATH}tensorflow/tensorflow/tensorflow/examples/speech_commands/freeze.py \
            --wanted_words={CONSTANTS.LABELS} \
            --window_stride_ms={CONSTANTS.WINDOW_STRIDE} \
            --preprocess={CONSTANTS.PREPROCESS} \
            --model_architecture={CONSTANTS.MODEL_ARCHITECTURE} \
            --start_checkpoint={CONSTANTS.MODEL_CHECKPOINT_PATH} \
            --feature_bin_count={CONSTANTS.FEATURE_BIN_COUNT} \
            --save_format=saved_model \
            --output_file={CONSTANTS.TFLITE_MODEL_SAVE_PATH} \
            --embedding_size={embedding_size}"

    result = subprocess.check_output(command, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--embedding_size',
      type=int,
      default=100,
      help='Embedding dimensionality.')
    FLAGS, unparsed = parser.parse_known_args()
    
    freeze_model(FLAGS.embedding_size)

