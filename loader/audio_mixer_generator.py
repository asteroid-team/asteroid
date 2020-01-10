'''
    This files samples (n=2) audio files and creates another audio file with mixed audio.
    It stores the mixed audio in data/train/mixed directory.
'''
import os
import glob
import random
import itertools
import subprocess
import pandas as pd
from pathlib import Path


AUDIO_MIX_COMMAND_PREFIX = "ffmpeg -y -t 00:00:03 -ac 1 "
AUDIO_DIR = "../../data/train/audio"
MIXED_AUDIO_DIR = "../../data/train/mixed"
REL_AUDIO_DIR = "../data/train/mixed"
VIDEO_DIR = "../../data/train"
REL_VIDEO_DIR = "../data/train"
AUDIO_SET_DIR = "./../../data/audio_set/low_volume"

def sample_audio_set():
    """
        sample random audio files as a noise from audio set dataset
    """
    audio_files = glob.glob(os.path.join(AUDIO_SET_DIR, "*"))
    total_files = len(audio_files)

    total_choices = int(random.gauss(mu=1, sigma=1))
    choices = list(range(total_files))
    random.shuffle(choices)

    return [audio_files[i] for i in choices[:total_choices]]


def audio_mixer(dataset_size: int, input_audio_size=2, video_ext=".mp4", audio_ext=".wav", file_name="temp.csv", audio_set=False, validation_size=0.3) -> None:
    """
        generate the combination dataframe used in data_loader.py

        Args:
            dataset_size: restrict total possible combinations
            input_audio_size: input size
            video_ext: extension of video
            audio_ext: extension of audio
            file_name: file name of combination dataframe to save
            audio_set: use audio set dataset
    """
    audio_mix_command_suffix = "-filter_complex amix=inputs={}:duration=longest "
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*"))
    total_audio_files = len(audio_files)
    
    total_val_files = int(total_audio_files * validation_size)
    total_train_files = total_audio_files - total_val_files

    train_files = audio_files[:total_train_files]
    val_files = audio_files[-total_val_files:]

    print(train_files)
    print(val_files)
    
    def mix(audio_filtered_files, file_name_df, offset):
        #Generate all combinations and trim total possibilities
        audio_combinations = itertools.combinations(audio_filtered_files, input_audio_size)
        audio_combinations = itertools.islice(audio_combinations, dataset_size)

        #Store list of tuples, consisting of `input_audio_size`
        #Audio and their corresponding video path
        video_inputs = []
        audio_inputs = []
        mixed_audio = []
        
        for indx, audio_comb in enumerate(audio_combinations):
            if audio_set:
                noise_input = sample_audio_set()
                audio_comb = (*audio_comb, *noise_input)
            audio_inputs.append(audio_comb)
            #Convert audio file path to corresponding video path
            video_inputs.append(tuple(os.path.join(VIDEO_DIR, os.path.splitext(os.path.basename(f))[0]+video_ext)
                                        for f in audio_comb))

            audio_mix_input = ""
            for audio in audio_comb:
                audio_mix_input += f"-i {audio} "

            
            mixed_audio_name = os.path.join(MIXED_AUDIO_DIR, f"{indx+offset}{audio_ext}")
            audio_command = AUDIO_MIX_COMMAND_PREFIX + audio_mix_input + audio_mix_command_suffix.format(len(audio_comb)) + mixed_audio_name
            print(audio_command)
            process = subprocess.Popen(audio_command, shell=True, stdout=subprocess.PIPE)#.communicate()
            mixed_audio.append(mixed_audio_name)
        
        combinations = {}
        for i in range(input_audio_size):
            combinations[f"video_{i+1}"] = []
            combinations[f"audio_{i+1}"] = []
        combinations["mixed_audio"] = []

        assert len(video_inputs) == len(audio_inputs)

        for videos, audios, mixed in zip(video_inputs, audio_inputs, mixed_audio):
            for i in range(input_audio_size):
                combinations[f"video_{i+1}"].append(videos[i])
                combinations[f"audio_{i+1}"].append(audios[i])
            combinations["mixed_audio"].append(mixed)

        columns = [f"video_{i+1}" for i in range(input_audio_size)] + [f"audio_{i+1}" for i in range(input_audio_size)] + ["mixed_audio"]
        df = pd.DataFrame(combinations).reindex(columns=columns)
        df.to_csv(file_name_df, index=False)
        return df.shape[0]

    offset = mix(train_files, "train.csv", 0)
    mix(val_files, "val.csv", offset)


if __name__ == "__main__":
    audio_mixer(100_000_000, audio_set=False)

