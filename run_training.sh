if [[ $1 == "custom" ]]; then
  DATASET="/home/arden/Classes/EE292D/edna_mode/dataset"
elif [[ $1 == "vox" ]]; then
  DATASET="/home/arden/Classes/EE292D/dataset/vox1_dev_wav_flat/"
fi

rm -rf "/tmp/retrain_logs/train/"
python speech_commands/train.py \
    --model_architecture "tiny_conv_embedding" \
    --data_dir $DATASET \
    --optimizer adam \
    --background_frequency 0 \
    --embedding_size 50 \
    --batch_size 10 \
    --num_samples 3 \
    --how_many_training_steps "100000,50000" \
    --learning_rate "0.001,0.0001" \
    --loss 'triplet' \
    --feature_bin_count 40
DIRNAME="tiny_embedding_conv_vox_150k_emb50_1"
mkdir "/home/arden/Classes/EE292D/edna_mode/models/${DIRNAME}"
cp -v "/tmp/speech_commands_train/tiny_embedding_conv.ckpt-150000.*" "/home/arden/Classes/EE292D/edna_mode/models/${DIRNAME}"
cp -vr "/tmp/retrain_logs/train/" "/home/arden/Classes/EE292D/edna_mode/models/${DIRNAME}"

rm -rf "/tmp/retrain_logs/train/"
python speech_commands/train.py \
    --model_architecture "tiny_conv_embedding" \
    --data_dir $DATASET \
    --optimizer adam \
    --background_frequency 0 \
    --embedding_size 50 \
    --batch_size 10 \
    --num_samples 3 \
    --how_many_training_steps "100000,50000" \
    --learning_rate "0.001,0.0001" \
    --loss 'triplet' \
    --feature_bin_count 40
DIRNAME="tiny_embedding_conv_vox_150k_emb50_2"
mkdir "/home/arden/Classes/EE292D/edna_mode/models/${DIRNAME}"
cp -v "/tmp/speech_commands_train/tiny_embedding_conv.ckpt-150000.*" "/home/arden/Classes/EE292D/edna_mode/models/${DIRNAME}"
cp -vr "/tmp/retrain_logs/train/" "/home/arden/Classes/EE292D/edna_mode/models/${DIRNAME}"

rm -rf "/tmp/retrain_logs/train/"
python speech_commands/train.py \
    --model_architecture "tiny_conv_embedding" \
    --data_dir $DATASET \
    --optimizer adam \
    --background_frequency 0 \
    --embedding_size 50 \
    --batch_size 10 \
    --num_samples 3 \
    --how_many_training_steps "100000,50000" \
    --learning_rate "0.001,0.0001" \
    --loss 'triplet' \
    --feature_bin_count 40
DIRNAME="tiny_embedding_conv_vox_150k_emb50_3"
mkdir "/home/arden/Classes/EE292D/edna_mode/models/${DIRNAME}"
cp -v "/tmp/speech_commands_train/tiny_embedding_conv.ckpt-150000.*" "/home/arden/Classes/EE292D/edna_mode/models/${DIRNAME}"
cp -vr "/tmp/retrain_logs/train/" "/home/arden/Classes/EE292D/edna_mode/models/${DIRNAME}"