import os
import re
import math
import glob
import random
import itertools
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from constants import (
    AUDIO_MIX_COMMAND_PREFIX,
    AUDIO_DIR,
    MIXED_AUDIO_DIR,
    VIDEO_DIR,
    AUDIO_SET_DIR,
    STORAGE_LIMIT,
)


def sample_audio_set():
    """
    sample random audio files as a noise from audio set dataset
    """
    audio_files = glob.glob(os.path.join(AUDIO_SET_DIR, "*"))
    total_files = len(audio_files)

    total_choices = max(1, int(random.gauss(mu=1, sigma=1)))
    choices = list(range(total_files))
    random.shuffle(choices)

    return [audio_files[i] for i in choices[:total_choices]]


def requires_excess_storage_space(n, r):
    # r will be very small anyway
    total = n**r / math.factorial(r)
    # total bytes
    storage_space = (
        total * 96
    )  # approximate storage requirement is (600K for spec and 90K for audio)

    if storage_space > STORAGE_LIMIT:
        return storage_space, True

    return storage_space, False


def nCr(n, r):
    return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))


def audio_mixer(
    dataset_size: int,
    n_src=2,
    video_ext=".mp4",
    audio_ext=".wav",
    file_name="temp.csv",
    audio_set=False,
    validation_size=0.3,
    remove_random_chance=0.9,
) -> None:
    """
    generate the combination dataframe used in data_loader.py

    Args:
        dataset_size: restrict total possible combinations
        n_src: input size
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

    storage_space_train, excess_storage = requires_excess_storage_space(len(train_files), n_src)

    storage_space_val, _ = requires_excess_storage_space(len(val_files), n_src)

    storage_space = storage_space_train + storage_space_val
    if excess_storage:
        storage_space = (1 - remove_random_chance) * storage_space
        print(f"Removing {remove_random_chance * 100} percent of combinations")
        print(
            f"Saving total space: {storage_space - storage_space * remove_random_chance:,} Kbytes"
        )

    print(f"Occupying space: {storage_space:,} Kbytes")

    def retrieve_name(f):
        f = os.path.splitext(os.path.basename(f))[0]
        return re.sub(r"_part\d", "", f)

    def mix(audio_filtered_files, file_name_df, offset, excess_storage):
        # Generate all combinations and trim total possibilities
        audio_combinations = itertools.combinations(audio_filtered_files, n_src)
        audio_combinations = itertools.islice(audio_combinations, dataset_size)

        # Store list of tuples, consisting of `n_src`
        # Audio and their corresponding video path
        video_inputs = []
        audio_inputs = []
        mixed_audio = []
        noises = []

        total_comb_size = nCr(len(audio_filtered_files), n_src)
        for indx, audio_comb in tqdm(enumerate(audio_combinations), total=total_comb_size):
            # skip few combinations if required storage is very high
            try:
                if excess_storage and random.random() < remove_random_chance:
                    continue

                base_names = [os.path.basename(fname)[:11] for fname in audio_comb]
                if len(base_names) != len(set(base_names)):
                    # if audio from the same video, assume same speaker and ignore it.
                    continue
                if audio_set:
                    noise_input = sample_audio_set()
                    noises.append(":".join(noise_input))
                    audio_comb = (*audio_comb, *noise_input)

                audio_inputs.append(audio_comb)
                # Convert audio file path to corresponding video path
                video_inputs.append(
                    tuple(os.path.join(VIDEO_DIR, retrieve_name(f) + video_ext) for f in audio_comb)
                )

                audio_mix_input = ""
                for audio in audio_comb:
                    audio_mix_input += f"-i {audio} "

                mixed_audio_name = os.path.join(MIXED_AUDIO_DIR, f"{indx+offset}{audio_ext}")
                audio_command = (
                    AUDIO_MIX_COMMAND_PREFIX
                    + audio_mix_input
                    + audio_mix_command_suffix.format(len(audio_comb))
                    + mixed_audio_name
                )

                process = subprocess.Popen(
                    audio_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )  # .communicate()
                mixed_audio.append(mixed_audio_name)
                # print(video_inputs, audio_inputs, mixed_audio, noises)
            except KeyboardInterrupt as e:
                print("Caught Interrupt!")
                break

        combinations = {}
        for i in range(n_src):
            combinations[f"video_{i+1}"] = []
            combinations[f"audio_{i+1}"] = []
        combinations["mixed_audio"] = []

        min_length = min(min(len(video_inputs), len(audio_inputs)), len(mixed_audio))
        print(f"Total combinations: {min_length}")

        video_inputs = video_inputs[:min_length]
        audio_inputs = audio_inputs[:min_length]
        mixed_audio = mixed_audio[:min_length]

        assert len(video_inputs) == len(audio_inputs) == len(mixed_audio)

        for videos, audios, mixed in zip(video_inputs, audio_inputs, mixed_audio):
            # fix proper path issue
            mixed = re.sub(r"../../", "", mixed)
            for i in range(n_src):
                v = re.sub(r"../../", "", videos[i])
                a = re.sub(r"../../", "", audios[i])

                combinations[f"video_{i+1}"].append(v)
                combinations[f"audio_{i+1}"].append(a)
            combinations["mixed_audio"].append(mixed)

        columns = (
            [f"video_{i+1}" for i in range(n_src)]
            + [f"audio_{i+1}" for i in range(n_src)]
            + ["mixed_audio"]
        )
        df = pd.DataFrame(combinations).reindex(columns=columns)
        df.to_csv(file_name_df, index=False)

        if audio_set:
            pd.Series(noises).to_csv("../../data/noise_only.csv", index=False, header=False)
        return df.shape[0]

    offset = mix(train_files, "../../data/train.csv", 0, excess_storage)
    mix(val_files, "../../data/val.csv", offset, excess_storage)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--remove-random",
        "-r",
        default=0.9,
        type=float,
        help="ratio of combination to remove",
    )
    parser.add_argument("--use-audio-set", "-u", dest="use_audio_set", action="store_true")
    parser.add_argument(
        "--file-limit",
        "-l",
        default=100_000_000,
        type=int,
        help="restrict total number of files generated",
    )
    parser.add_argument(
        "--validation-size",
        "-v",
        default=0.3,
        type=float,
        help="ratio of files to use in validation data",
    )

    args = parser.parse_args()

    audio_mixer(
        args.file_limit,
        audio_set=args.use_audio_set,
        validation_size=args.validation_size,
        remove_random_chance=args.remove_random,
    )
