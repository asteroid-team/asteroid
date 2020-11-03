import numpy as np
from scipy import signal
import os
import soundfile as sf
import pickle
import argparse
import gpuRIR
from pathlib import Path

# CODE from Yi Luo: https://github.com/yluo42/TAC

parser = argparse.ArgumentParser(description="Generate multi-channel Librispeech data")
parser.add_argument(
    "--output-path",
    metavar="absolute path",
    required=True,
    help="The path to the output directory. Default is the current directory.",
)
parser.add_argument(
    "--dataset",
    metavar="dataset type",
    required=True,
    help="The type of dataset to generate. Can only be 'adhoc' or 'fixed'.",
)
parser.add_argument(
    "--libri-path",
    metavar="absolute path",
    required=True,
    help="Absolute path for Librispeech folder containing train-clean-100, dev-clean and test-clean folders.",
)
parser.add_argument(
    "--noise-path",
    metavar="absolute path",
    required=True,
    help="Absolute path for the 100 Nonspeech sound folder.",
)


# generate audio files
def generate_data(output_path, dataset, libri_path, noise_path):

    assert dataset in ["adhoc", "fixed"], "dataset can only be adhoc or fixed."

    if output_path == "":
        output_path = os.getcwd()

    data_type = ["train", "validation", "test"]
    for i in range(len(data_type)):
        # path for config
        config_path = os.path.join(
            Path(os.path.realpath(__file__)).parent,
            "data_configs",
            "MC_Libri_" + dataset + "_" + data_type[i] + ".pkl",
        )

        # load pickle file
        with open(config_path, "rb") as f:
            configs = pickle.load(f)

        # sample rate is 16k Hz
        sr = 16000
        # signal length is 4 sec
        sig_len = 4

        for utt in range(len(configs)):
            this_config = configs[utt]

            # load audio files
            speakers = this_config["speech"]
            noise = this_config["noise"]
            spk1, _ = sf.read(os.path.join(libri_path, speakers[0]))
            spk2, _ = sf.read(os.path.join(libri_path, speakers[1]))
            noise, _ = sf.read(os.path.join(noise_path, noise))

            # calculate signal length according to overlap ratio
            overlap_ratio = this_config["overlap_ratio"]
            actual_len = int(sig_len / (2 - overlap_ratio) * sr)
            overlap = int(actual_len * overlap_ratio)

            # truncate speech according to start and end indexes
            start_idx = this_config["start_idx"]
            end_idx = start_idx + actual_len
            spk1 = spk1[start_idx:end_idx]
            spk2 = spk2[start_idx:end_idx]

            # rescaling speaker and noise energy according to relative SNR
            spk1 = spk1 / np.sqrt(np.sum(spk1 ** 2) + 1e-8) * 1e2
            spk2 = spk2 / np.sqrt(np.sum(spk2 ** 2) + 1e-8) * 1e2
            spk2 = spk2 * np.power(10, this_config["spk_snr"] / 20.0)
            # repeat noise if necessary
            noise = noise[: int(sig_len * sr)]
            if len(noise) < int(sig_len * sr):
                num_repeat = int(sig_len * sr) // len(noise)
                res = int(sig_len * sr) - num_repeat * len(noise)
                noise = np.concatenate([np.concatenate([noise] * num_repeat), noise[:res]])
            # rescale noise energy w.r.t mixture energy
            noise = (
                noise
                / np.sqrt(np.sum(noise ** 2) + 1e-8)
                * np.sqrt(np.sum((spk1 + spk2) ** 2) + 1e-8)
            )
            noise = noise / np.power(10, this_config["noise_snr"] / 20.0)

            # load locations and room configs
            mic_pos = np.asarray(this_config["mic_pos"])
            spk_pos = np.asarray(this_config["spk_pos"])
            noise_pos = np.asarray(this_config["noise_pos"])
            room_size = np.asarray(this_config["room_size"])
            rt60 = this_config["RT60"]
            num_mic = len(mic_pos)

            # generate RIR
            beta = gpuRIR.beta_SabineEstimation(room_size, rt60)
            nb_img = gpuRIR.t2n(rt60, room_size)
            spk_rir = gpuRIR.simulateRIR(room_size, beta, spk_pos, mic_pos, nb_img, rt60, sr)
            noise_rir = gpuRIR.simulateRIR(room_size, beta, noise_pos, mic_pos, nb_img, rt60, sr)

            # convolve with RIR at different mic
            echoic_spk1 = []
            echoic_spk2 = []
            echoic_mixture = []

            if dataset == "adhoc":
                nmic = this_config["num_mic"]
            else:
                nmic = 6
            for mic in range(nmic):
                spk1_echoic_sig = signal.fftconvolve(spk1, spk_rir[0][mic])
                spk2_echoic_sig = signal.fftconvolve(spk2, spk_rir[1][mic])
                noise_echoic_sig = signal.fftconvolve(noise, noise_rir[0][mic])

                # align the speakers according to overlap ratio
                pad_length = int((1 - overlap_ratio) * actual_len)
                padding = np.zeros(pad_length)
                spk1_echoic_sig = np.concatenate([spk1_echoic_sig, padding])
                spk2_echoic_sig = np.concatenate([padding, spk2_echoic_sig])
                # pad or truncate length to 4s if necessary
                def pad_sig(x):
                    if len(x) < sig_len * sr:
                        zeros = np.zeros(sig_len * sr - len(x))
                        return np.concatenate([x, zeros])
                    else:
                        return x[: sig_len * sr]

                spk1_echoic_sig = pad_sig(spk1_echoic_sig)
                spk2_echoic_sig = pad_sig(spk2_echoic_sig)
                noise_echoic_sig = pad_sig(noise_echoic_sig)

                # sum up for mixture
                mixture = spk1_echoic_sig + spk2_echoic_sig + noise_echoic_sig

                # save waveforms
                this_save_dir = os.path.join(
                    output_path,
                    "MC_Libri_" + dataset,
                    data_type[i],
                    str(num_mic) + "mic",
                    "sample" + str(utt + 1),
                )
                if not os.path.exists(this_save_dir):
                    os.makedirs(this_save_dir)
                sf.write(
                    os.path.join(this_save_dir, "spk1_micf" + str(mic + 1) + ".wav"),
                    spk1_echoic_sig,
                    sr,
                )
                sf.write(
                    os.path.join(this_save_dir, "spk2_mic" + str(mic + 1) + ".wav"),
                    spk2_echoic_sig,
                    sr,
                )
                sf.write(
                    os.path.join(this_save_dir, "mixture_mic" + str(mic + 1) + ".wav"), mixture, sr
                )

            # print progress
            if (utt + 1) % (len(configs) // 5) == 0:
                print(
                    "{} configuration, {} set, {:d} out of {:d} utterances generated.".format(
                        dataset, data_type[i], utt + 1, len(configs)
                    )
                )


if __name__ == "__main__":
    args = parser.parse_args()
    generate_data(
        output_path=args.output_path,
        dataset=args.dataset,
        libri_path=args.libri_path,
        noise_path=args.noise_path,
    )
