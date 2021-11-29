import os

# Remember to update!
MODEL_CHECKPOINT_PATH = "/tmp/speech_commands_train/tiny_embedding_conv.ckpt-200"


# General
CONSTANTS_FILEPATH = os.path.dirname(__file__)
REPO_FILEPATH = os.path.join(CONSTANTS_FILEPATH, "../")
DATASET_FILEPATH = os.path.join(REPO_FILEPATH, "dataset")
ADMIN = "phil"
LABELS = ["phil", "greg", "arden"]
DEMO_WAV_PATH = os.path.join(CONSTANTS_FILEPATH, "./demo_dataset/phil/phil_demo.wav")
MEAN_EMBEDDINGS_PATH = os.path.join(REPO_FILEPATH, "demo/mean_embeddings.csv")


# Used in demo/freeze.py
PREPROCESS = 'micro'
WINDOW_STRIDE = 20
MODEL_ARCHITECTURE = "tiny_embedding_conv"

# Used in convert_to_tf_lite.py
SILENT_PERCENTAGE = 10.0
UNKNOWN_PERCENTAGE = 10.0
LOGS_DIR = '/tmp/retrain_logs'
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.0
BACKGROUND_VOLUME_RANGE = 0.0
TIME_SHIFT_MS = 0.0
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10
