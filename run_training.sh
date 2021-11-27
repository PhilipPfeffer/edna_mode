if [[ $1 == "custom" ]]; then
  DATASET="/home/arden/Classes/EE292D/edna_mode/dataset"
elif [[ $1 == "vox" ]]; then
  DATASET="/home/arden/Classes/EE292D/dataset/vox1_dev_wav_flat/"
fi


python speech_commands/train.py \
    --model_architecture mobilenet_embedding \
    --data_dir $DATASET \
    --optimizer adam \
    --background_frequency 0 \
    --embedding_size 100 \
    --batch_size 100 \
    --num_samples 3 \
    --how_many_training_steps "100000,50000" \
    --learning_rate "0.001,0.0001" \
    --loss 'triplet' 
