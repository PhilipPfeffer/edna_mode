#FROZEN_MODEL_SAVE_PATH="/home/arden/Classes/EE292D/edna_mode/demo/frozen_models/tiny_embedding_conv.ckpt-200/"
FROZEN_MODEL_SAVE_PATH="/home/arden/Classes/EE292D/edna_mode/demo/frozen_models/tiny_embedding_conv.ckpt-150000/"
DATA_DIR="/home/arden/Classes/EE292D/edna_mode/dataset/test"
EMBEDDING_SIZE=50

# MAKE SURE CONSTANTS.py HAS THE CORRESPONDING MODEL SET
python demo/create_mean_embeddings.py \
    --embedding_size=${EMBEDDING_SIZE}

python demo/test_tflite.py \
    --data_dir=${DATA_DIR} \
    --saved_model_dir=${FROZEN_MODEL_SAVE_PATH} \
    --embedding_size=${EMBEDDING_SIZE}

python demo/test_tflite.py \
    --data_dir=${DATA_DIR} \
    --saved_model_dir=${FROZEN_MODEL_SAVE_PATH} \
    --embedding_size=${EMBEDDING_SIZE} \
    --run_quantized 