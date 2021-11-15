# edna_mode

# Installation / Setup

1. create venv: `python3 -m venv env` and activate it with `source env/bin/activate`.
2. download requirements: `pip install requirements.txt` and then install tf_slim by first `cd slim` and then `pip install -e .`.
    - you might need to `pip install -U pip` and `pip install wheel`

# Training
1. Navigate to the speech commands directory e.g. `tensorflow/tensorflow/tensorflow/examples/speech_commands`
2. Run training script with the following command `train.py --model_architecture mobilenet_embedding --embedding_size 100`
    - Arden: currently using `python ./tensorflow/tensorflow/tensorflow/examples/speech_commands/train.py --model_architecture mobilenet_embedding --embedding_size 2 --data_dir ./dataset --batch_size 5`