import argparse
import glob
import json
import os
import soundfile as sf


def preprocess_dns(in_dir, out_dir="./data"):
    """Create json file from dataset folder.

    Args:
        in_dir (str): Location of the DNS data
        out_dir (str): Where to save the json files.
    """
    # Get all file ids
    clean_wavs = glob.glob(os.path.join(in_dir, "clean/*.wav"))
    clean_dic = make_wav_id_dict(clean_wavs)

    mix_wavs = glob.glob(os.path.join(in_dir, "noisy/*.wav"))
    mix_dic = make_wav_id_dict(mix_wavs)

    noise_wavs = glob.glob(os.path.join(in_dir, "noise/*.wav"))
    noise_dic = make_wav_id_dict(noise_wavs)
    assert clean_dic.keys() == mix_dic.keys() == noise_dic.keys()
    file_infos = {
        k: dict(
            mix=mix_dic[k],
            clean=clean_dic[k],
            noise=noise_dic[k],
            snr=get_snr_from_mix_path(mix_dic[k]),
            file_len=len(sf.SoundFile(mix_dic[k])),
        )
        for k in clean_dic.keys()
    }

    # Save to JSON
    with open(os.path.join(out_dir, "file_infos.json"), "w") as f:
        json.dump(file_infos, f, indent=2)


def make_wav_id_dict(file_list):
    """
    Args:
        file_list(List[str]): List of DNS challenge filenames.

    Returns:
        dict: Look like {file_id: filename, ...}
    """
    return {get_file_id(fp): fp for fp in file_list}


def get_file_id(fp):
    """Split string to get wave id in DNS challenge dataset."""
    return fp.split("_")[-1].split(".")[0]


def get_snr_from_mix_path(mix_path):
    """ Retrieves mixing SNR from mixture filename.

    Args:
        mix_path (str): Path to the mixture. Something like :
        book_11346_chp_0012_reader_08537_8_kFu2mH7D77k-5YOmLILWHyg-\
        gWMWteRIgiw_snr6_tl-35_fileid_3614.wav

    Returns:
        int or None: the SNR value if we could parse it.
    """
    snr_str = mix_path.split("snr")[-1].split("_")[0]
    try:
        snr = int(snr_str)
    except ValueError:
        snr = None
    return snr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Location of data")
    parser.add_argument("--json_dir", default="./data", help="Where to save the json file")

    args = parser.parse_args()

    os.makedirs(args.json_dir, exist_ok=True)
    preprocess_dns(args.data_dir, args.json_dir)
