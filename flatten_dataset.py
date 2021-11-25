
'''script to flatten vox celebs dataset into dataset/person/.wav structure'''
import os
import shutil
import glob

from joblib import Parallel, delayed
from tqdm import tqdm

DATASET_DIR = '/home/arden/Classes/EE292D/dataset/vox1_dev_wav'
NEW_DATASET_DIR = '/home/arden/Classes/EE292D/dataset/vox1_dev_wav_flat'
DRY_RUN = False
OVERWRITE = True

def copy_wav(person_dir):
    old_person_dir_path = os.path.join(DATASET_DIR, person_dir)
    new_person_dir_path = os.path.join(NEW_DATASET_DIR, person_dir)
    if not os.path.exists(new_person_dir_path):
        os.makedirs(new_person_dir_path)
    for i, wav_path in enumerate(glob.glob(os.path.join(old_person_dir_path, '*', '*.wav'))):
        if DRY_RUN:
            print(f'moving {wav_path} -> {os.path.join(new_person_dir_path, f"{i:05d}.wav")}')
        else:
            shutil.copy(wav_path, os.path.join(new_person_dir_path, f"{i:05d}.wav"))

def main():
    if not os.path.exists(NEW_DATASET_DIR):
        os.makedirs(NEW_DATASET_DIR)
    elif OVERWRITE:
        print(f"OVERWRITING {NEW_DATASET_DIR}")
        shutil.rmtree(NEW_DATASET_DIR)
        os.makedirs(NEW_DATASET_DIR)
    else:
        raise Exception(f'{NEW_DATASET_DIR} already exists, please either set OVERWRITE=True or delete and retry')

    Parallel(n_jobs=-1, verbose=0)(delayed(copy_wav)(person_dir) for person_dir in tqdm(os.listdir(DATASET_DIR)))

if __name__=='__main__':
    main()

# Example output of dry run: moving /home/arden/Classes/EE292D/dataset/vox1_dev_wav/id10571/0t2eQfsJ148/00005.wav -> /home/arden/Classes/EE292D/dataset/vox1_dev_flat/id10571/00005.wav