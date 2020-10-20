import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf
import random
import torchaudio


class MedleydbDataset(data.Dataset):
    """MedleyDB: a dataset of annotated, royaltyfree multitrack recordings.

    It provides a stereo mix and both dry and processed multitrack stems for 
    each song in the dataset. The dataset covers a wide distribution of genres 
    and primarily consists of full length songs with professional 
    or nearprofessional audio quality.
    
    The dataset consists of 254 full lengths music tracks of
    different genres. It provides a stereo mix and both dry and processed
    multitrack stems for each song. 

    This dataset contains a .yml object for each multitrack containining
    metadata for each audio file in the multitrack. This dataloader utilises 
    this metadata to select audio files and downmix them to create mixtures
    of instruments belonging to a list of instrument tags with specified 
    polyphony.


    Args:
        json_dir (str): Path containing .json file with list of audio files
        n_src (int): Number of separate instrument classes (yet to be implemented)
        n_poly (int): Number of instances of each class to be present in mixture.
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        sample_rate (int, optional): Samplerate of files in dataset.
        threshold (float): activity confidence threshold to exclude segments with
            less than given fraction of time of activity in segment.

    References:
        "MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research", 
            R. Bittner et. al. ISMIR 2014
    
    """

    dataset_name = "MedleyDB"

    def __init__(self, json_dir, n_src=1, n_poly=2, sample_rate=44100, segment=5.0, threshold=0.1):
        super(MedleydbDataset, self).__init__()
        
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.n_poly = n_poly
        self.threshold = threshold
        self.seg_len = segment
        self.n_src = n_src
        # Load json files
        sources_json = [
            os.path.join(json_dir, source + ".json")
            for source in [f"inst{n+1}" for n in range(n_src)]
        ]
        sources_conf = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_conf = np.array(json.load(f))

        # Filter out utterances with activity less than threshold
        drop_utt, drop_len, orig_len = 0, 0, 0
        sources_infos = []
        index_array = []

        for i in range(len(sources_conf)):
            conf = sources_conf[i][1]
            duration = sources_conf[i][1][-1][0]
            index_array.append(np.zeros(int(duration // segment) + 1))
            for timestamp, confidence in conf:
                j = int(timestamp // segment)
                index_array[i][j] = index_array[i][j] + confidence
            orig_len = orig_len + duration
            seg_dur = duration / len(index_array[i])
            #save list of segments with sufficient activity
            for k in range(len(index_array[i])):
                conf_thresh = threshold * float(len(sources_conf[i][0]))
                if index_array[i][k] < conf_thresh:
                    drop_utt += 1
                    drop_len += seg_dur
                    continue
                else:
                    sources_infos.append((sources_conf[i][0], k, sources_conf[i][2]))

        print(
            "Drop {} utts ({:.2f} h) from ({:.2f} h) with less than {} percent activity".format(
                drop_utt, drop_len / 3600, orig_len / 3600, threshold
            )
        )
        self.sources = sources_infos

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Load sources
        source_arrays = []

        for i in range(self.n_poly):
            if i:
                idx = random.choice(range(len(self.sources)))

            start = self.sources[idx][1] * self.sources[idx][2]
            if self.like_test:
                stop = None
            else:
                stop = start + (self.seg_len * self.sources[idx][2])

            if self.sources[idx] is None:
                # Target is filled with zeros if n_src > default_nsrc
                s = np.zeros((self.seg_len * self.sources[idx][2],))
            else:
                s, sr = sf.read(self.sources[idx][0], start=start, stop=stop, dtype="float32")
            source_arrays.append(s)
        source = torch.from_numpy(np.vstack(source_arrays))
        if sr is not self.sample_rate:
            source = torchaudio.transforms.Resample(sr, self.sample_rate)(source)
        mix = torch.stack(list(source)).sum(0)
        return mix, source

    def get_infos(self):
        """ Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "harmony_sep"
        infos["licenses"] = [mdb_license]
        return infos


class SourceFolderDataset(data.Dataset):
    dataset_name = "SourceFolder"

    def __init__(self, json_dir, wav_dir, n_src=1, sample_rate=44100, batch_size=1):
        super(SourceFolderDataset, self).__init__()
        # Task setting
        self.json_dir = json_dir
        self.wav_dir = wav_dir
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.like_test = True
        self.batch_size = batch_size
        sources_infos = []
        # Make and load json files
        speaker_list = ["mix"] + [f"s{n+1}" for n in range(n_src)]
        for speaker in speaker_list:
            preprocess_one_dir(
                os.path.join(wav_dir, speaker), json_dir, speaker, sample_rate=sample_rate
            )

        sources_json = [
            os.path.join(json_dir, source + ".json") for source in [f"s{n+1}" for n in range(n_src)]
        ]
        
        mix_json = os.path.join(json_dir, "mix.json")
        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))
        sources_infos = np.array(sources_infos)
        #sources_infos = np.swapaxes(sources_infos, 0,1)    
        self.mix = mix_infos
        self.sources = sources_infos

    def __len__(self):
        return len(self.sources[0])

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Load mixture
        x, sr = sf.read(self.mix[idx][0], dtype="float32")
        seg_len = torch.as_tensor([len(x)])
        # Load sources
        source_arrays = []
        for src in self.sources:
            if src[idx] is None:
                # Target is filled with zeros if n_src > default_nsrc
                s = np.zeros((seg_len,))
            else:
                s, _ = sf.read(src[idx][0], dtype="float32")
            source_arrays.append(s)
        source = torch.from_numpy(np.vstack(source_arrays))
        
        mix = torch.from_numpy(x)
        mix = mix.unsqueeze(0)
        if sr is not self.sample_rate:
            source = torchaudio.transforms.Resample(sr, self.sample_rate)(source)
            mix = torchaudio.transforms.Resample(sr, self.sample_rate)(mix)
        source = source.unsqueeze(0)
        return mix, source

    def get_infos(self):
        """ Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "harmony_sep_eval"
        infos["licenses"] = [mdb_license]
        return infos


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=44100):
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith(".wav"):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = sf.read(wav_path)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + ".json"), "w") as f:
        json.dump(file_infos, f, indent=4)


mdb_license = dict(
    title="MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research",
    title_link="https://medleydb.weebly.com/",
    author="R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam and J. P. Bello",
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License",
    license_link="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    non_commercial=True,
)
