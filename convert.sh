MODEL_SAVE_PATH="/home/arden/Classes/EE292D/edna_mode/demo/frozen_models/tiny_embedding_conv.ckpt-150000/"
DATA_DIR="/home/arden/Classes/EE292D/edna_mode/dataset"
EMBEDDING_SIZE=50

[ -d ${MODEL_SAVE_PATH} ] && echo "Directory ${MODEL_SAVE_PATH} exists. Please remove it or select a new save directory." && exit 1

python demo/freeze_model.py \
    --save_path=${MODEL_SAVE_PATH} \
    --embedding_size=${EMBEDDING_SIZE}

python demo/convert_to_tflite.py \
    --data_dir=${DATA_DIR} \
    --saved_model_dir=${MODEL_SAVE_PATH} \
    --embedding_size=${EMBEDDING_SIZE}

python demo/test_tflite.py \
    --data_dir=${DATA_DIR} \
    --saved_model_dir=${MODEL_SAVE_PATH} \
    --embedding_size=${EMBEDDING_SIZE} \
    --run_quantized 

python demo/convert_to_tfmicro.py \
    --saved_model_dir=${MODEL_SAVE_PATH}