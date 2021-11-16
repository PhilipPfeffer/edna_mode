# edna_mode

# Installation / Setup

1. create venv: `python3 -m venv env` and activate it with `source env/bin/activate`.
2. download requirements: `pip install requirements.txt` and then install tf_slim by first `cd slim` and then `pip install -e .`.
    - you might need to `pip install -U pip` and `pip install wheel`

# Training
1. Navigate to the speech commands directory e.g. `tensorflow/tensorflow/tensorflow/examples/speech_commands`
2. Run training script with the following command `train.py --model_architecture mobilenet_embedding --embedding_size 100`
    - Arden: currently using `python ./tensorflow/tensorflow/tensorflow/examples/speech_commands/train.py --model_architecture mobilenet_embedding --data_dir ABSOLUTE_PATH_TO_DATASET --optimizer momentum  --background_frequency 0 --embedding_size 50 --batch_size 5`
    - Arden: use `--verbosity debug` for more debug output
    - Arden: use `tensorboard --log_dir /tmp/retrain_logs/train`
    - If we want to quantize, use `--quantize True` flag.

# Conversion
1. Create .pb file from checkpoint:`python tensorflow/tensorflow/tensorflow/examples/speech_commands/freeze.py --model_architecture mobilenet_embedding --data_dir  ~/Classes/EE292D/edna_mode/dataset --window_stride_ms=30 --save_format=saved_model --embedding_size 50 --start_checkpoint "/tmp/speech_commands_train/mobilenet_embedding.ckpt-500" --output_file=frozen_mobilenet_emb.pb --convert_tflite=True`
2. Create tflite model: `python tensorflow/tensorflow/tensorflow/examples/speech_commands/convert_to_tflite.py --pb_dir ~/Classes/EE292D/edna_mode/models/model.pb --data_dir ~/Classes/EE292D/edna_mode/dataset`
# Inference
1. `python tensorflow/tensorflow/tensorflow/examples/speech_commands/train.py --model_architecture mobilenet_embedding --data_dir ~/Classes/EE292D/edna_mode/dataset --optimizer momentum  --background_frequency 0 --embedding_size 50 --batch_size 3 --inference True --inference_checkpoint_path /tmp/speech_commands_train/mobilenet_embedding.ckpt-100 --query_file /Users/arden/Documents/Classes/EE292D/edna_mode/dataset/arden/arden0001.wav`
    - Need to supply `--inference True`, `--inference_checkpoint_path MODEL_CHECKPOINT`, and `--query_file WAV_FILE`