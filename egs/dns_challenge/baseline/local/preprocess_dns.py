import argparse
import glob
import json
import os
import soundfile as sf


def preprocess_dns(in_dir, out_dir='./data'):
    """ Create json file from dataset folder

    Args:
        in_dir (str): Location of the DNS data
        out_dir (str): Where to save the json files.

    """
    # Get all file ids
    clean_files = os.listdir(os.path.join(in_dir, 'clean'))
    all_ids = [f.split('_')[-1].split('.')[0] for f in clean_files]
    all_ids.sort()

    # Create dict for each utterance
    file_infos = {}
    for file_id in all_ids:
        utt_dict = dict(
            mix=file_from_dir(file_id, os.path.join(in_dir, 'noisy')),
            clean=file_from_dir(file_id, os.path.join(in_dir, 'clean')),
            noise=file_from_dir(file_id, os.path.join(in_dir, 'noise')),
        )
        # Get SNR
        utt_dict['snr'] = get_snr_from_mix_path(utt_dict['mix'])
        # Get utterance length
        utt_dict['file_len'] = len(sf.SoundFile(utt_dict['mix']))
        file_infos[file_id] = utt_dict

    # Save to JSON
    with open(os.path.join(out_dir, 'file_infos.json'), 'w') as f:
        json.dump(file_infos, f, indent=2)


def file_from_dir(file_id, file_dir):
    """ Retrieve file path from file id.

    Args:
        file_id (str): file id.
        file_dir (str): Where to look for the file.

    Returns:
        The file path with the give id.
    """
    file_list = glob.glob(os.path.join(file_dir, '*fileid_' + file_id + '.*'))
    assert len(file_list) == 1, "Found more than one file."
    return file_list[0]


def get_snr_from_mix_path(mix_path):
    """ Retrieves mixing SNR from mixture filename.

    Args:
        mix_path (str): Path to the mixture. Something like :
        book_11346_chp_0012_reader_08537_8_kFu2mH7D77k-5YOmLILWHyg-\
        gWMWteRIgiw_snr6_tl-35_fileid_3614.wav

    Returns:
        int or None: the SNR value if we could parse it.
    """
    snr_str = mix_path.split('snr')[-1].split('_')[0]
    try:
        snr = int(snr_str)
    except ValueError:
        snr = None
    return snr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Location of data')
    parser.add_argument('--json_dir', default='./data',
                        help='Where to save the json file')

    args = parser.parse_args()

    os.makedirs(args.json_dir, exist_ok=True)
    preprocess_dns(args.data_dir, args.json_dir)
