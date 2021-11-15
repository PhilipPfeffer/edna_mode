import os
import glob
import argparse
from shutil import copyfile

import librosa
import soundfile as sf

def main(args):
    for person_dir in os.listdir(args.data_dir):
        if person_dir == '_background_noise_':
            continue
        for wav_path in glob.glob(os.path.join(args.data_dir, person_dir, '*.wav')):
            if not os.path.exists(wav_path + '.old'):
                print(f"copying: {wav_path} -> {wav_path + '.old'}")
                copyfile(wav_path, wav_path + '.old')
            y, s = librosa.load(wav_path, sr=16000) # Downsample to 16KHz
            print(f"overwriting: {wav_path} with new 16KHz version")
            sf.write(wav_path, y, s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    main(parser.parse_args())