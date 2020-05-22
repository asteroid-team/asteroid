from src.loader import convert_to_spectrogram
import concurrent.futures
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import os


def convert(f):
    spec_path = "../../data/train/spec/"
    file_name = f.split('/')[-1]
    file_name = spec_path + file_name
    file_name = file_name.replace(".wav", "")
    spec = convert_to_spectrogram(librosa.load(f, sr=16_000)[0])
    np.save(file_name, spec)

def get_list(df):
    mixed_files = list(df.iloc[:, -1])
    audio_files = list(set(list(df.iloc[:, 2]) + list(df.iloc[:, 3])))
    total_files = mixed_files + audio_files
    total_files = ["../" + i for i in total_files]
    return total_files

def main():
    total_files = []
    total_files += get_list(pd.read_csv("../train.csv"))
    total_files += get_list(pd.read_csv("../val.csv"))
    convert(total_files[0])

    number_files = len(total_files)

    if number_files * 900_000 > 50_000_000_000:
        print("Total space usage is more than 50GB")
        yn = input("Continue(y/n)?")
        if len(yn) == 0:
            yn = 'n'

        if yn.lower()[0] != 'y':
            return

    with concurrent.futures.ProcessPoolExecutor(4) as executor:
        futures = [executor.submit(convert, f) for f in tqdm(total_files, total=len(total_files))]

        #for f in tqdm(concurrent.futures.as_completed(futures), total=len(total_files)):
        #    pass

if __name__ == "__main__":
    main()
