import os
import json
import soundfile as sf
import argparse
import glob
import re
from pathlib import Path

parser = argparse.ArgumentParser("parsing tac dataset")
parser.add_argument("--in_dir", type=str)
parser.add_argument("--out_json", type=str)


def parse_dataset(in_dir, out_json):

    examples = []
    for n_mic_f in glob.glob(os.path.join(in_dir, "*")):
        for sample_dir in glob.glob(os.path.join(n_mic_f, "*")):
            c_ex = {}
            for wav in glob.glob(os.path.join(sample_dir, "*.wav")):

                source_or_mix = Path(wav).stem.split("_")[0]
                n_mic = int(re.findall("\d+", Path(wav).stem.split("_")[-1])[0])
                length = len(sf.SoundFile(wav))

                if n_mic not in c_ex.keys():
                    c_ex[n_mic] = {source_or_mix: wav, "length": length}
                else:
                    assert c_ex[n_mic]["length"] == length
                    c_ex[n_mic][source_or_mix] = wav
            examples.append(c_ex)

    os.makedirs(Path(out_json).parent, exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(examples, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    parse_dataset(args.in_dir, args.out_json)
