# python demo/create_mean_embeddings.py --embedding_size=100
EMBEDDING_SIZE=50

python demo/inference.py --input_path="/home/arden/Classes/EE292D/edna_mode/dataset/arden/arden0000.wav" --embedding_size=${EMBEDDING_SIZE}
python demo/inference.py --input_path="/home/arden/Classes/EE292D/edna_mode/dataset/arden/arden0003.wav" --embedding_size=${EMBEDDING_SIZE}
python demo/inference.py --input_path="/home/arden/Classes/EE292D/edna_mode/dataset/greg/greg0000.wav" --embedding_size=${EMBEDDING_SIZE}
python demo/inference.py --input_path="/home/arden/Classes/EE292D/edna_mode/dataset/greg/greg0003.wav" --embedding_size=${EMBEDDING_SIZE}
python demo/inference.py --input_path="/home/arden/Classes/EE292D/edna_mode/dataset/phil/phil0000.wav" --embedding_size=${EMBEDDING_SIZE}
python demo/inference.py --input_path="/home/arden/Classes/EE292D/edna_mode/dataset/phil/phil0003.wav" --embedding_size=${EMBEDDING_SIZE}
