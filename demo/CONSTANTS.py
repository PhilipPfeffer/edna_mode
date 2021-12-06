import os

# Remember to update!
#MODEL_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "../models/speech_commands_train/tiny_embedding_conv.ckpt-200")
#MODEL_CHECKPOINT_PATH = "/home/arden/Classes/EE292D/edna_mode/models/tiny_embedding_conv_vox_150k_emb50/tiny_embedding_conv.ckpt-150000"
#MODEL_CHECKPOINT_PATH = "/home/arden/Classes/EE292D/edna_mode/models/tiny_embedding_conv_vox_150k_emb50_bad/tiny_embedding_conv.ckpt-150000"
#MODEL_CHECKPOINT_PATH = "/home/arden/Classes/EE292D/edna_mode/models/tiny_embedding_conv_vox_150k_emb100/tiny_embedding_conv.ckpt-150000"
#MODEL_CHECKPOINT_PATH = "/home/arden/Classes/EE292D/edna_mode/models/tiny_embedding_conv_vox_150k_emb25/tiny_embedding_conv.ckpt-150000"
#MODEL_CHECKPOINT_PATH = "/home/arden/Classes/EE292D/edna_mode/models/mobilenet_embedding_vox_150k_emb50/mobilenet_embedding.ckpt-150000"
#MODEL_CHECKPOINT_PATH = "/home/arden/Classes/EE292D/edna_mode/models/mobilenet_embedding_vox_150k_emb100/mobilenet_embedding.ckpt-148200"
MODEL_CHECKPOINT_PATH = "/tmp/speech_commands_train/tiny_embedding_conv.ckpt-200"
TFLITE_MODEL_SAVE_PATH = "/Users/philipmateopfeffer/Desktop/stanford/Y5Q1/cs329e/edna_mode/demo/frozen_models/tiny_embedding_conv.ckpt-200"

# General
CONSTANTS_FILEPATH = os.path.dirname(__file__)
REPO_FILEPATH = os.path.join(CONSTANTS_FILEPATH, "../")
DATASET_FILEPATH = os.path.join(REPO_FILEPATH, "dataset/train")
ADMIN = "phil"
LABELS = ["phil", "greg", "arden"]
DEMO_WAV_PATH = os.path.join(CONSTANTS_FILEPATH, "./demo_dataset/phil/phil_demo.wav")
MEAN_EMBEDDINGS_FLOAT_PATH = os.path.join(REPO_FILEPATH, "demo/mean_embeddings_float.csv")
MEAN_EMBEDDINGS_QUANT_PATH = os.path.join(REPO_FILEPATH, "demo/mean_embeddings_quant.csv")
THRESHOLD_EMBEDDINGS_FLOAT_PATH = os.path.join(REPO_FILEPATH, "demo/thresholds_float.csv")
THRESHOLD_EMBEDDINGS_QUANT_PATH = os.path.join(REPO_FILEPATH, "demo/thresholds_quant.csv")


# Used in demo/freeze.py
PREPROCESS = 'micro'
WINDOW_STRIDE = 20
MODEL_ARCHITECTURE = "tiny_embedding_conv"
# MODEL_ARCHITECTURE = "mobilenet_embedding"

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
