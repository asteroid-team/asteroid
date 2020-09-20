import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf
import random
import torchaudio

class MedleydbDataset(data.Dataset):
    dataset_name = "MedleyDB"

    def __init__(self, json_dir, n_src=1, n_poly=2, sample_rate=44100, segment=5.0, threshold=0.1):
        super(MedleydbDataset,self).__init__()
        # Task setting
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.n_poly = n_poly
        self.threshold = threshold
        if segment is None:
            self.seg_len = None
        else:
            self.seg_len = int(segment)
        self.n_src = n_src
        self.like_test = self.seg_len is None
        # Load json files
        sources_json = [
            os.path.join(json_dir, source + ".json")
            for source in [f"inst{n+1}" for n in range(n_src)]
        ]
        sources_conf = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_conf = np.array(json.load(f))

        # Filter out short utterances only when segment is specified
        # orig_len = len(mix_infos)
        drop_utt, drop_len, orig_len = 0, 0, 0
        sources_infos = []
        index_array = []
        
        if not self.like_test:
            for i in range(len(sources_conf)):
                conf = sources_conf[i][1]
                #print(sources_conf[i][0])
                #index_array = []
                duration = sources_conf[i][1][-1][0]
                index_array.append(np.zeros(int(duration//segment) + 1))
                for timestamp, confidence in conf:
                    j = int(timestamp // segment)
                    #print(j)
                    index_array[i][j] = index_array[i][j] + confidence
                orig_len = orig_len + duration
                seg_dur = duration / len(index_array[i])

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
        # self.mix = mix_infos
        #print(sources_infos[0])
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
        if sr is not self.sample_rate :
            source = torchaudio.transforms.Resample(sr,self.sample_rate)(source) 
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
        super(SourceFolderDataset,self).__init__()
        # Task setting
        self.json_dir = json_dir
        self.wav_dir = wav_dir
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.like_test = True
        self.batch_size = batch_size
        # Make and load json files
        for speaker in ['mix', 's1', 's2']:
            preprocess_one_dir(os.path.join(wav_dir, speaker), json_dir, speaker,
                                sample_rate=sample_rate)
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        sorted_s1_infos = sort(s1_infos)
        sorted_s2_infos = sort(s2_infos)
        # Filter out short utterances only when segment is specified
        # orig_len = len(mix_infos)
        self.mix_infos = sorted_mix_infos
        minibatch = []
        start = 0
        max_dur = 6
        for i in range(len(sorted_mix_infos)):
            if int(sorted_mix_infos[i][1]) > max_dur * sample_rate:
                start = end
                continue
            minibatch.append([sorted_mix_infos[i][0],
                                sorted_s1_infos[i][0],
                                sorted_s2_infos[i][0],
                                #sorted_s3_infos[start],
                                ])
        self.sources = minibatch

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Load sources
        source_arrays = []
        for i in range(self.n_src):   
            s, sr = sf.read(self.sources[idx][i+1], dtype="float32")
            source_arrays.append(s)
        x, sr = sf.read(self.sources[idx][0], dtype="float32")
        source = torch.from_numpy(np.vstack(source_arrays))
        mix = torch.from_numpy(x)
        mix = mix.unsqueeze(0)
        if sr is not self.sample_rate :
            source = torchaudio.transforms.Resample(sr,self.sample_rate)(source) 
            mix = torchaudio.transforms.Resample(sr,self.sample_rate)(mix) 
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
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = sf.read(wav_path)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)


mdb_license = dict(
    title="MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research",
    title_link="https://medleydb.weebly.com/",
    author="R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam and J. P. Bello",
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License",
    license_link="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    non_commercial=True,
)
