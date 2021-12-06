DATA_DIR="/home/arden/Classes/EE292D/edna_mode/dataset/test"
EMBEDDING_SIZE=50

python demo/freeze_model.py \
    --embedding_size=${EMBEDDING_SIZE}

python demo/convert_to_tflite.py \
    --data_dir=${DATA_DIR} \
    --embedding_size=${EMBEDDING_SIZE}

python demo/convert_to_tfmicro.py \
    --saved_model_dir=${MODEL_SAVE_PATH} > quantized_model.txt