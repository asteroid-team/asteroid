import torch
import numpy as np
import argparse
import soundfile as sf
import musdb
import museval
import norbert
from pathlib import Path
import scipy.signal
import resampy
from asteroid.models import XUMX
from asteroid.complex_nn import torch_complex_from_magphase
import os
import warnings
import sys


def load_model(model_name, device="cpu"):
    print("Loading model from: {}".format(model_name), file=sys.stderr)
    model = XUMX.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, model.sources


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2), rate, nperseg=n_fft, noverlap=n_fft - n_hopsize, boundary=True
    )
    return audio


def separate(
    audio,
    x_umx_target,
    instruments,
    niter=1,
    softmask=False,
    alpha=1.0,
    residual_model=False,
    device="cpu",
):
    """
    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    x_umx_target: asteroid.models
        X-UMX model used for separating

    instruments: list
        The list of instruments, e.g., ["bass", "drums", "vocals"]

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary with all estimates obtained by the separation model.
    """

    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)

    source_names = []
    V = []

    masked_tf_rep, _ = x_umx_target(audio_torch)
    # shape: (Sources, frames, batch, channels, fbin)

    for j, target in enumerate(instruments):
        Vj = masked_tf_rep[j, Ellipsis].cpu().detach().numpy()
        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj**alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, Ellipsis])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    # convert to complex numpy type
    tmp = x_umx_target.encoder(audio_torch)
    X = torch_complex_from_magphase(tmp[0].permute(1, 2, 3, 0), tmp[1])
    X = X.detach().cpu().numpy()
    X = X[0].transpose(2, 1, 0)

    if residual_model or len(instruments) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += ["residual"] if len(instruments) > 1 else ["accompaniment"]

    Y = norbert.wiener(V, X.astype(np.complex128), niter, use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            rate=x_umx_target.sample_rate,
            n_fft=x_umx_target.in_chan,
            n_hopsize=x_umx_target.n_hop,
        )
        estimates[name] = audio_hat.T

    return estimates


def inference_args(parser, remaining_args):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inf_parser.add_argument(
        "--softmask",
        dest="softmask",
        action="store_true",
        help=(
            "if enabled, will initialize separation with softmask."
            "otherwise, will use mixture phase with spectrogram"
        ),
    )

    inf_parser.add_argument(
        "--niter", type=int, default=1, help="number of iterations for refining results."
    )

    inf_parser.add_argument(
        "--alpha", type=float, default=1.0, help="exponent in case of softmask separation"
    )

    inf_parser.add_argument("--samplerate", type=int, default=44100, help="model samplerate")

    inf_parser.add_argument(
        "--residual-model", action="store_true", help="create a model for the residual"
    )
    return inf_parser.parse_args()


def eval_main(
    root,
    samplerate=44100,
    niter=1,
    alpha=1.0,
    softmask=False,
    residual_model=False,
    model_name="xumx",
    large=False,
    outdir=None,
    start=0.0,
    duration=-1.0,
    no_cuda=False,
):

    model_name = os.path.abspath(model_name)
    if not (os.path.exists(model_name)):
        outdir = os.path.abspath("./results_using_pre-trained")
        model_name = (
            "r-sawata/XUMXL_MUSDB18_music_separation"
            if large
            else "r-sawata/XUMX_MUSDB18_music_separation"
        )
    else:
        outdir = os.path.join(
            os.path.abspath(outdir),
            "EvaluateResults_musdb18_testdata",
        )
    Path(outdir).mkdir(exist_ok=True, parents=True)
    print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model, instruments = load_model(model_name, device)

    test_dataset = musdb.DB(root=root, subsets="test", is_wav=True)
    results = museval.EvalStore()
    Path(outdir).mkdir(exist_ok=True, parents=True)
    txtout = os.path.join(outdir, "results.txt")
    fp = open(txtout, "w")
    for track in test_dataset:
        input_file = os.path.join(root, "test", track.name, "mixture.wav")

        # handling an input audio path
        info = sf.info(input_file)
        start = int(start * info.samplerate)
        # check if dur is none
        if duration > 0:
            # stop in soundfile is calc in samples, not seconds
            stop = start + int(duration * info.samplerate)
        else:
            # set to None for reading complete file
            stop = None

        audio, rate = sf.read(input_file, always_2d=True, start=start, stop=stop)

        if audio.shape[1] > 2:
            warnings.warn("Channel count > 2! " "Only the first two channels will be processed!")
            audio = audio[:, :2]

        if rate != samplerate:
            # resample to model samplerate if needed
            audio = resampy.resample(audio, rate, samplerate, axis=0)

        if audio.shape[1] == 1:
            # if we have mono, let's duplicate it
            # as the input of OpenUnmix is always stereo
            audio = np.repeat(audio, 2, axis=1)

        estimates = separate(
            audio,
            model,
            instruments,
            niter=niter,
            alpha=alpha,
            softmask=softmask,
            residual_model=residual_model,
            device=device,
        )

        output_path = Path(os.path.join(outdir, track.name))
        output_path.mkdir(exist_ok=True, parents=True)

        print("Processing... {}".format(track.name), file=sys.stderr)
        print(track.name, file=fp)
        for target, estimate in estimates.items():
            sf.write(str(output_path / Path(target).with_suffix(".wav")), estimate, samplerate)
        track_scores = museval.eval_mus_track(track, estimates)
        results.add_track(track_scores.df)
        print(track_scores, file=sys.stderr)
        print(track_scores, file=fp)
    print(results, file=sys.stderr)
    print(results, file=fp)
    results.save(os.path.join(outdir, "results.pandas"))
    results.frames_agg = "mean"
    print(results, file=sys.stderr)
    print(results, file=fp)
    fp.close()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="OSU Inference", add_help=False)

    parser.add_argument("--root", type=str, help="The path to the MUSDB18 dataset")

    parser.add_argument(
        "--outdir",
        type=str,
        default="./results_using_pre-trained",
        help="Results path where " "best_model.pth" " is stored",
    )

    parser.add_argument("--start", type=float, default=0.0, help="Audio chunk start in seconds")

    parser.add_argument(
        "--duration",
        type=float,
        default=-1.0,
        help="Audio chunk duration in seconds, negative values load full track",
    )

    parser.add_argument(
        "--large", action="store_true", default=False, help="Download and use X-UMX Large (X-UMXL)"
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA inference"
    )

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)

    model = os.path.join(args.outdir, "best_model.pth")
    eval_main(
        root=args.root,
        samplerate=args.samplerate,
        alpha=args.alpha,
        softmask=args.softmask,
        niter=args.niter,
        residual_model=args.residual_model,
        model_name=model,
        large=args.large,
        outdir=args.outdir,
        start=args.start,
        duration=args.duration,
        no_cuda=args.no_cuda,
    )
