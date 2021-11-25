import os

CONSTANTS_FILEPATH = os.path.dirname(__file__)
REPO_FILEPATH = os.path.join(CONSTANTS_FILEPATH, "../")
DATASET_FILEPATH = os.path.join(REPO_FILEPATH, "dataset")
ADMIN = "phil"
LABELS = ["phil", "greg", "arden"]
DEMO_WAV_PATH = os.path.join(CONSTANTS_FILEPATH, "./phil_demo.wav")
MEAN_EMBEDDINGS_PATH = os.path.join(REPO_FILEPATH, "demo/mean_embeddings.csv")
MODEL_CHECKPOINT_PATH = "/tmp/speech_commands_train/mobilenet_embedding.ckpt-2100"
PREPROCESS = 'micro'
WINDOW_STRIDE = 20
MODEL_ARCHITECTURE = "mobilenet_embedding"