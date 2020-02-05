from audio_feature_generator import convert_to_spectrogram
import concurrent.futures
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import os

mixed_files = ["../../data/train/mixed/" + i for i in os.listdir("../../data/train/mixed")]
audio_files = ["../../data/train/audio" + i for i in os.listdir("../../data/train/audio")]

total_files = mixed_files + audio_files


#for f in tqdm(total_files, total=len(total_files)):
def convert(f):
    spec_path = "../../data/train/spec/"
    file_name = f.split('/')[-1]
    file_name = spec_path + file_name
    file_name = file_name.replace(".wav", "")
    spec = convert_to_spectrogram(librosa.load(f, sr=16_000)[0])
    np.save(file_name, spec)

def main():

    df = pd.read_csv("../filtered_val.csv")
    mixed_files = list(df.iloc[:, -1])
    audio_files = list(set(list(df.iloc[:, 2]) + list(df.iloc[:, 3])))
    total_files = mixed_files + audio_files
    total_files = ["../" + i for i in total_files]
    convert(total_files[0])
    with concurrent.futures.ProcessPoolExecutor(4) as executor:
        futures = [executor.submit(convert, f) for f in tqdm(total_files, total=len(total_files))]

        #for f in tqdm(concurrent.futures.as_completed(futures), total=len(total_files)):
        #    pass

main()
