import argparse
import json
import os
import soundfile as sf


def preprocess_one_dir(in_dir):
    """Create list of list for one condition, each list contains
    [path, wav_length]."""
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    wav_list.sort()
    for wav_file in wav_list:
        if not wav_file.endswith(".wav"):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples = sf.SoundFile(wav_path)
        file_infos.append((wav_path, len(samples)))
    return file_infos


def preprocess(inp_args):
    """Create .json files for all conditions."""
    # The filenames are shared between directories (and lengths as well) so
    # we can just search once and replace directory name after.
    speaker_list = [
        "s1_anechoic",
        "s2_anechoic",
        "s1_reverb",
        "s2_reverb",
        "mix_single_anechoic",
        "mix_clean_anechoic",
        "mix_both_anechoic",
        "mix_single_reverb",
        "mix_clean_reverb",
        "mix_both_reverb",
    ]
    for data_type in ["tr", "cv", "tt"]:
        out_dir = os.path.join(inp_args.out_dir, data_type)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # Create the first list of wavs
        spk_0 = speaker_list[0]
        to_json = preprocess_one_dir(os.path.join(inp_args.in_dir, data_type, spk_0))
        # Replace directory names to match all conditions
        for spk in speaker_list:
            local_to_json = []
            for wav_info in to_json:
                name, wav_len = wav_info
                local_to_json.append([name.replace(spk_0, spk), wav_len])
            with open(os.path.join(out_dir, spk + ".json"), "w") as f:
                json.dump(local_to_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WHAM data preprocessing")
    parser.add_argument(
        "--in_dir", type=str, default=None, help="Directory path of wham including tr, cv and tt"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None, help="Directory path to put output files"
    )
    args = parser.parse_args()
    print(args)
    preprocess(args)
