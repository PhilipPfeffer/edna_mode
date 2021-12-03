if [[ $1 == "custom" ]]; then
  DATASET="/home/arden/Classes/EE292D/edna_mode/dataset"
elif [[ $1 == "vox" ]]; then
  DATASET="/home/arden/Classes/EE292D/dataset/vox1_dev_wav_flat/"
fi

EMBEDDING_SIZE=$2

python speech_commands/train.py \
    --model_architecture "tiny_embedding_conv" \
    --data_dir $DATASET \
    --optimizer adam \
    --background_frequency 0 \
    --embedding_size ${EMBEDDING_SIZE} \
    --batch_size 10 \
    --num_samples 3 \
    --how_many_training_steps "1000,500" \
    --learning_rate "0.01,0.0001" \
    --loss 'triplet'

python demo/create_mean_embeddings.py --embedding_size=${EMBEDDING_SIZE}

python demo/inference.py --input_path="/home/arden/Classes/EE292D/edna_mode/dataset/arden/arden0000.wav" --embedding_size=${EMBEDDING_SIZE}
python demo/inference.py --input_path="/home/arden/Classes/EE292D/edna_mode/dataset/arden/arden0003.wav" --embedding_size=${EMBEDDING_SIZE}
python demo/inference.py --input_path="/home/arden/Classes/EE292D/edna_mode/dataset/greg/greg0000.wav" --embedding_size=${EMBEDDING_SIZE}
python demo/inference.py --input_path="/home/arden/Classes/EE292D/edna_mode/dataset/greg/greg0003.wav" --embedding_size=${EMBEDDING_SIZE}
python demo/inference.py --input_path="/home/arden/Classes/EE292D/edna_mode/dataset/phil/phil0000.wav" --embedding_size=${EMBEDDING_SIZE}
python demo/inference.py --input_path="/home/arden/Classes/EE292D/edna_mode/dataset/phil/phil0003.wav" --embedding_size=${EMBEDDING_SIZE}