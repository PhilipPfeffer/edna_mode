# edna_mode

# Installation / Setup

1. create venv: `python3 -m venv env` and activate it with `source env/bin/activate`.
2. download requirements: `pip install requirements.txt` and then install tf_slim by first `cd slim` and then `pip install -e .`.
    - you might need to `pip install -U pip` and `pip install wheel`

# Training
1. Navigate to the speech commands directory e.g. `tensorflow/tensorflow/tensorflow/examples/speech_commands`
2. Run training script with the following command `train.py --model_architecture mobilenet_embedding --embedding_size 100`
    - Arden: currently using `python ./tensorflow/tensorflow/tensorflow/examples/speech_commands/train.py --model_architecture mobilenet_embedding --data_dir /home/arden/Classes/EE292D/dataset/vox1_dev_wav_flat/ --optimizer adam  --background_frequency 0 --embedding_size 100 --batch_size 10 --how_many_training_steps "100000,5000" --learning_rate "0.001,0.0001"` 
    - Phil: currently using `python ./tensorflow/tensorflow/tensorflow/examples/speech_commands/train.py --model_architecture mobilenet_embedding --data_dir /Users/philipmateopfeffer/Desktop/stanford/Y5Q1/cs329e/edna_mode/dataset --optimizer adam  --background_frequency 0 --embedding_size 50 --batch_size 5 --how_many_training_steps=100,100`
    - Arden: use `--verbosity debug` for more debug output
    - Arden: use `tensorboard --logdir /tmp/retrain_logs/train`
    - If we want to quantize, use `--quantize True` flag.

<!-- # Conversion
1. `python tensorflow/tensorflow/tensorflow/examples/speech_commands/freeze.py --model_architecture mobilenet_embedding --data_dir  ~/Classes/EE292D/edna_mode/dataset --window_stride_ms=20 --save_format=saved_model --embedding_size 50 --start_checkpoint "/tmp/speech_commands_train/mobilenet_embedding.ckpt-100 " --output_file=frozen_mobilenet_emb.pb --convert_tflite=True` -->

# Inference
1. Calculate new mean embeddings:
    - Modify `CONSTANTS.MODEL_CHECKPOINT_PATH` if model retrained, then
    - Call: `python demo/create_mean_embeddings.py --embedding_size=N`

2. Run inference on a new example:
    - Call `python demo/inference.py --input_path=PATH --embedding_size=N`

# Conversion
1. Freeze model: please save into demo/frozen_models/ directory!
    - Check that `demo/CONSTANTS.py` is correct
    - Call: `python demo/freeze_model.py --save_path=PATH --embedding_size=N`
    - e.g. `python demo/freeze_model.py --save_path=/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/cs329e/edna_mode/demo/frozen_models/mobilenet_embedding_frozen.ckpt-200 --embedding_size=50`

2. Convert to tflite
    - Call: `python demo/convert_to_tflite.py --data_dir=PATH --input_model_dir=PATH --embedding_size=N`
    - e.g. `python demo/convert_to_tflite.py --data_dir=/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/cs329e/edna_mode/dataset --input_model_dir=/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/cs329e/edna_mode/demo/frozen_models/mobilenet_embedding_frozen.ckpt-200 --embedding_size=50`