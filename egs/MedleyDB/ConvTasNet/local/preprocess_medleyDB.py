import argparse
import json
import yaml
import os
import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt

import soundfile as sf


def make_processed_filelist(track_list, out_dir, out_filename):
    file_infos = []
    counter = 0
    for track in track_list:
        print("Processing file", track, counter)
        conf, rate = compute_activation_confidence(track)
        file_infos.append((track, conf.tolist(), rate))
        counter += 1
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("writing to json file", (os.path.join(out_dir, out_filename + ".json")))
    with open(os.path.join(out_dir, out_filename + ".json"), "w") as f:
        json.dump(file_infos, f, indent=4)
    print("json file writing complete")
    return


def preprocess_metadata(
    metadata_path,
    inst_list,
    v1_path=None,
    v2_path=None,
    bach10_path=None,
    extra_path=None,
    is_stem=False,
):
    """ Create .json file containing tracks containing given inst_list"""
    # meta_infos = []
    counter = 0
    meta_dir = metadata_path + "/medleydb/data/Metadata"
    resource_path = metadata_path + "/medleydb/resources"
    tracklist_path = {
        "v1": os.path.join(resource_path, "tracklist_v1.txt"),
        "v2": os.path.join(resource_path, "tracklist_v2.txt"),
        # "bach10": os.path.join(resource_path, "tracklist_bach10.txt"),
        # "extra": os.path.join(resource_path, "tracklist_extra.txt"),
    }
    data_path = {"v1": v1_path, "v2": v2_path, "bach10": bach10_path, "extra": extra_path}
    inst_tracks = []
    for ver, path in tracklist_path.items():
        print(path)
        track_list = read_tracklist(path)
        for meta_file in track_list:
            meta_path = os.path.join(meta_dir, meta_file + "_METADATA.yaml")
            # meta_path = meta_dir + meta_file + '.yaml'
            if not os.path.isfile(meta_path):
                print(meta_path, " not found")
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                trim = yaml.safe_load(f)
                if "yes" in trim["has_bleed"]:
                    print(meta_path, " has bleed")
                    continue
                for stem in trim["stems"]:
                    for instrument in inst_list:
                        if is_stem:
                            if instrument in trim["stems"][stem]["instrument"]:
                                print(
                                    os.path.join(
                                        data_path[ver],
                                        meta_file,
                                        trim["stem_dir"],
                                        trim["stems"][stem]["filename"],
                                    ),
                                    counter,
                                    trim["stems"][stem]["instrument"],
                                )
                                # print(trim["stems"][stem]["instrument"])
                                inst_tracks.append(
                                    os.path.join(
                                        data_path[ver],
                                        meta_file,
                                        trim["stem_dir"],
                                        trim["stems"][stem]["filename"],
                                    )
                                )
                                counter += 1
                        else:
                            for raw in trim["stems"][stem]["raw"]:
                                if instrument in trim["stems"][stem]["raw"][raw]["instrument"]:
                                    print(
                                        os.path.join(
                                            data_path[ver],
                                            meta_file,
                                            trim["raw_dir"],
                                            trim["stems"][stem]["raw"][raw]["filename"],
                                        ),
                                        counter,
                                        trim["stems"][stem]["instrument"],
                                    )
                                    # print(trim["stems"][stem]["instrument"])
                                    inst_tracks.append(
                                        os.path.join(
                                            data_path[ver],
                                            meta_file,
                                            trim["raw_dir"],
                                            trim["stems"][stem]["raw"][raw]["filename"],
                                        )
                                    )
                                    counter += 1
    return inst_tracks


def read_tracklist(filename):
    tracklist = []
    with open(filename, "r") as fhand:
        for line in fhand.readlines():
            tracklist.append(line.strip("\n"))
    return tracklist


def compute_activation_confidence(
    track, win_len=4096, lpf_cutoff=0.075, theta=0.15, var_lambda=20.0, amplitude_threshold=0.01
):
    """Create the activation confidence annotation for a multitrack. The final
    activation matrix is computed as:
        `C[i, t] = 1 - (1 / (1 + e**(var_lambda * (H[i, t] - theta))))`
    where H[i, t] is the energy of stem `i` at time `t`

    Parameters
    ----------
    mtrack : MultiTrack
        Multitrack object
    win_len : int, default=4096
        Number of samples in each window
    lpf_cutoff : float, default=0.075
        Lowpass frequency cutoff fraction
    theta : float
        Controls the threshold of activation.
    var_labmda : float
        Controls the slope of the threshold function.
    amplitude_threshold : float
        Energies below this value are set to 0.0

    Returns
    -------
    C : np.array
        Array of activation confidence values shape (n_conf, n_stems)
    stem_index_list : list
        List of stem indices in the order they appear in C

    """
    H = []
    # MATLAB equivalent to @hanning(win_len)
    win = scipy.signal.windows.hann(win_len + 2)[1:-1]

    # audio, rate = librosa.load(track, mono=True)
    audio, rate = sf.read(track)
    H.append(track_energy(audio.T, win_len, win))

    # list to numpy array
    H = np.array(H)

    # normalization (to overall energy and # of sources)
    E0 = np.sum(H, axis=0)

    H = H / np.max(E0)
    # binary thresholding for low overall energy events
    H[:, E0 < amplitude_threshold] = 0.0

    # LP filter
    b, a = scipy.signal.butter(2, lpf_cutoff, "low")
    H = scipy.signal.filtfilt(b, a, H, axis=1)

    # logistic function to semi-binarize the output; confidence value
    C = 1.0 - (1.0 / (1.0 + np.exp(np.dot(var_lambda, (H - theta)))))

    # add time column
    time = librosa.core.frames_to_time(np.arange(C.shape[1]), sr=rate, hop_length=win_len // 2)

    # stack time column to matrix
    C_out = np.vstack((time, C))
    # print(C_out.T)
    return C_out.T, rate


def track_energy(wave, win_len, win):
    """Compute the energy of an audio signal

    Parameters
    ----------
    wave : np.array
        The signal from which to compute energy
    win_len: int
        The number of samples to use in energy computation
    win : np.array
        The windowing function to use in energy computation

    Returns
    -------
    energy : np.array
        Array of track energy

    """
    hop_len = win_len // 2

    wave = np.lib.pad(wave, pad_width=(win_len - hop_len, 0), mode="constant", constant_values=0)

    # post padding
    wave = librosa.util.fix_length(wave, int(win_len * np.ceil(len(wave) / win_len)))

    # cut into frames
    wavmat = librosa.util.frame(wave, frame_length=win_len, hop_length=hop_len)

    # Envelope follower
    wavmat = hwr(wavmat) ** 0.5  # half-wave rectification + compression

    return np.mean((wavmat.T * win), axis=1)


def hwr(x):
    """ Half-wave rectification.

    Parameters
    ----------
    x : array-like
        Array to half-wave rectify

    Returns
    -------
    x_hwr : array-like
        Half-wave rectified array

    """
    return (x + np.abs(x)) / 2


"""
def preprocess_one_dir(in_dir, out_dir, out_filename):
    """ """ Create .json file for one condition.""" """
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    wav_list.sort()
    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples = sf.SoundFile(wav_path)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)


def preprocess(inp_args):
    """ """Create .json files for all conditions.""" """
    speaker_list = ['mix'] + [f"s{n+1}" for n in range(inp_args.n_src)]
    for data_type in ['tr', 'cv', 'tt']:
        for spk in speaker_list:
            preprocess_one_dir(os.path.join(inp_args.in_dir, data_type, spk),
                               os.path.join(inp_args.out_dir, data_type),
                               spk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WSJ0-MIX data preprocessing")
    parser.add_argument('--in_dir', type=str, default=None,
                        help='Directory path of wham including tr, cv and tt')
    parser.add_argument('--n_src', type=int, default=2,
                        help='Number of sources in wsj0-mix')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory path to put output files')
    args = parser.parse_args()
    print(args)
    preprocess(args)
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser("MedleyDB data preprocessing")
    parser.add_argument(
        "--metadata_path", type=str, default=None, help="Directory path of MedleyDB git repo"
    )
    # parser.add_argument('--inst_list', type=str, default=2,
    #                    help='Number of sources in wsj0-mix')
    parser.add_argument(
        "--inst_list",
        nargs="+",
        help="list of instruments",
        default=["male singer", "female singer"],
    )
    parser.add_argument(
        "--if_stem", type=bool, default=False, help="If instrument tracks are stem or raw"
    )
    parser.add_argument(
        "--json_dir", type=str, default=None, help="Directory path for output json files"
    )
    parser.add_argument(
        "--v1_path", type=str, default=None, help="Directory path for output v1 files"
    )
    parser.add_argument(
        "--v2_path", type=str, default=None, help="Directory path for output v2 files"
    )
    parser.add_argument(
        "--bach10_path", type=str, default=None, help="Directory path for output bach10 files"
    )
    parser.add_argument(
        "--extra_path", type=str, default=None, help="Directory path for output others files"
    )

    args = parser.parse_args()
    print(args)
    tracklist = preprocess_metadata(
        args.metadata_path,
        args.inst_list,
        args.v1_path,
        args.v2_path,
        args.bach10_path,
        args.extra_path,
        args.if_stem,
    )
    make_processed_filelist(
        tracklist, args.json_dir, "inst1",
    )
    # print(A[:,1])
    # plt.plot(A[:,0],A[:,1])
    # plt.show()
