import random
import glob
import os
import warnings
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

class OnlineMixDataset(Dataset):
    """Dataset class that creates mixtures on-the-fly from clean single-speaker utterances.

    Args:
        source_dir (str): Root dir containing speaker subdirs.
        n_src (int): Max number of output sources (always pad to this).
        sample_rate (int): Sample rate of the output.
        segment (float): Segment length in seconds.
        min_speakers (int): Minimum number of speakers in the mixture.
        max_speakers (int): Maximum number of speakers in the mixture.
        speaker_count_weights (list, optional): Probability of each speaker count.
        gain_range_db (tuple): Random gain augmentation per source in dB.
        num_examples (int): Virtual epoch size.
        seed (int, optional): Random seed for reproducibility.
    """

    def __init__(
        self,
        source_dir: str,
        n_src: int = 5,
        sample_rate: int = 16000,
        segment: float = 3.0,
        min_speakers: int = 1,
        max_speakers: int = 5,
        speaker_count_weights: list = None,
        gain_range_db: tuple = (-5.0, 5.0),
        num_examples: int = 20000,
        seed: int = None,
    ):
        self.source_dir = os.path.expanduser(source_dir)
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.speaker_count_weights = speaker_count_weights
        self.gain_range_db = gain_range_db
        self.num_examples = num_examples
        self.seed = seed
        self.seg_len = int(segment * sample_rate)

        # Validate inputs
        if max_speakers > n_src:
            raise ValueError(f"max_speakers ({max_speakers}) cannot be greater than n_src ({n_src})")
        if min_speakers < 1:
            raise ValueError(f"min_speakers ({min_speakers}) must be >= 1")

        # Scan for speakers and files
        self.speakers = {}
        speaker_dirs = [
            d for d in os.listdir(self.source_dir) 
            if os.path.isdir(os.path.join(self.source_dir, d))
        ]

        for speaker_id in speaker_dirs:
            speaker_path = os.path.join(self.source_dir, speaker_id)
            # Find all flac and wav files recursively
            files = (glob.glob(os.path.join(speaker_path, "**/*.flac"), recursive=True) +
                     glob.glob(os.path.join(speaker_path, "**/*.wav"), recursive=True))
            
            if len(files) > 0:
                self.speakers[speaker_id] = files
            else:
                warnings.warn(f"Speaker {speaker_id} has no valid audio files and will be skipped.")

        if not self.speakers:
            raise RuntimeError(f"No speakers found in {source_dir}")

        self.speaker_ids = list(self.speakers.keys())

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # 1. Seed the RNG
        if self.seed is not None:
            # Use a seed that depends on the global seed and the index
            # This ensures reproducibility for a given index in a given epoch
            local_seed = self.seed + idx
            rng = random.Random(local_seed)
            np_rng = np.random.RandomState(local_seed)
        else:
            rng = random
            np_rng = np.random

        # 2. Sample speaker count N
        possible_counts = list(range(self.min_speakers, self.max_speakers + 1))
        if self.speaker_count_weights:
            n_current = rng.choices(possible_counts, weights=self.speaker_count_weights, k=1)[0]
        else:
            n_current = rng.choice(possible_counts)

        # 3. Select N distinct speakers
        if len(self.speaker_ids) < n_current:
             # Fallback if we don't have enough speakers
             # Allow replacement if total speakers available is less than requested
             selected_speakers = rng.choices(self.speaker_ids, k=n_current)
        else:
            selected_speakers = rng.sample(self.speaker_ids, n_current)

        sources = []
        
        # 4. For each speaker, pick utterance, load, process
        for speaker_id in selected_speakers:
            audio_files = self.speakers[speaker_id]
            file_path = rng.choice(audio_files)
            
            # Load audio
            info = sf.info(file_path)
            orig_sr = info.samplerate
            
            # We need to read enough audio to cover self.seg_len after resampling
            # If we need L samples at target_sr, we need L * orig_sr / target_sr samples
            # But we just read the whole file for simplicity usually, unless very long.
            # To be efficient, let's just read the whole file. Most LibriSpeech files are short (< 15s).
            audio, _ = sf.read(file_path, dtype="float32")
            audio_t = torch.from_numpy(audio) # shape (time,) or (time, channels)
            
            if audio_t.ndim > 1:
                audio_t = audio_t[:, 0] # Take first channel if stereo

            # Resample if needed
            if orig_sr != self.sample_rate:
                audio_t = torchaudio.functional.resample(audio_t, orig_sr, self.sample_rate)
            
            # Crop or pad
            if audio_t.shape[0] > self.seg_len:
                start = rng.randint(0, audio_t.shape[0] - self.seg_len)
                audio_t = audio_t[start : start + self.seg_len]
            else:
                padding = self.seg_len - audio_t.shape[0]
                audio_t = torch.nn.functional.pad(audio_t, (0, padding))
            
            # 5. Apply random gain
            gain_db = rng.uniform(self.gain_range_db[0], self.gain_range_db[1])
            gain_lin = 10 ** (gain_db / 20.0)
            audio_t = audio_t * gain_lin
            
            sources.append(audio_t)

        # Pad sources list with silence if n_current < self.n_src
        for _ in range(self.n_src - len(sources)):
            sources.append(torch.zeros(self.seg_len, dtype=torch.float32))

        sources_tensor = torch.stack(sources) # Shape (n_src, seg_len)

        # 6. Create mixture
        mixture = torch.sum(sources_tensor, dim=0) # Shape (seg_len,)

        # 7. Normalization
        # Peak normalize mixture to [-1, 1]
        max_amp = torch.max(torch.abs(mixture))
        if max_amp > 0:
            scale_factor = 1.0 / max_amp
            mixture = mixture * scale_factor
            sources_tensor = sources_tensor * scale_factor
        
        return mixture, sources_tensor

    def get_infos(self):
        return {
            "dataset": "OnlineMixDataset",
            "task": "sep_clean",
            "sample_rate": self.sample_rate,
            "n_src": self.n_src,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
        }

if __name__ == "__main__":
    # Sanity check
    import sys
    import shutil
    
    # Use a dummy dir if user didn't provide one, or use a hardcoded path for testing if available
    # For now, we will create a dummy structure to test logic if no path is provided
    # But ideally we run this where data exists. 
    # Let's try to look for some data or just create a temporary test dir.
    
    print("Running sanity check...")
    
    # Create temp dummy data
    temp_dir = "temp_sanity_check_data"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create 6 fake speakers
    sr = 8000
    for i in range(1, 7):
        spk_dir = os.path.join(temp_dir, f"spk{i}")
        os.makedirs(spk_dir, exist_ok=True)
        # Create a fake wav file
        # Make a simple sine wave
        t = np.linspace(0, 5, 5*sr)
        sine = np.sin(2 * np.pi * 440 * i * t).astype(np.float32)
        sf.write(os.path.join(spk_dir, f"audio_{i}.wav"), sine, sr)

    try:
        dataset = OnlineMixDataset(
            source_dir=temp_dir,
            n_src=5,
            sample_rate=sr,
            segment=2.0,
            min_speakers=1,
            max_speakers=5,
            num_examples=5,
            seed=42
        )
        
        print(f"Dataset created with {len(dataset)} examples.")
        print(f"Infos: {dataset.get_infos()}")

        for i in range(5):
            mix, srcs = dataset[i]
            print(f"\nExample {i}:")
            print(f"  Mixture shape: {mix.shape}")
            print(f"  Sources shape: {srcs.shape}")
            
            # Check for non-silent sources
            energies = (srcs**2).mean(dim=1).sqrt()
            active_srcs = (energies > 1e-6).sum().item()
            print(f"  Active sources: {active_srcs}")
            print(f"  Energies: {energies}")
            
            # Save for inspection
            os.makedirs("debug_audio", exist_ok=True)
            sf.write(f"debug_audio/ex{i}_mix.wav", mix.numpy(), sr)
            for j in range(srcs.shape[0]):
                 sf.write(f"debug_audio/ex{i}_src{j}.wav", srcs[j].numpy(), sr)
            print("  Saved audio to debug_audio/")
            
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        # shutil.rmtree("debug_audio") # Keep debug audio for manual inspection
        print("Done.")
