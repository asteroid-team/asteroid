import glob
import os
import json
from pathlib import Path
import soundfile as sf
import argparse

parser = argparse.ArgumentParser("parsing wham for mixit")
parser.add_argument("--in_dir", default="/media/sam/cb915f0e-e440-414c-bb74-df66b311d09d/2speakers_wham/wav8k/min/tt/")
parser.add_argument("--out_json", default="/tmp/parse_wham")


def parse_mixtures(wham_folder):

    w = glob.glob(os.path.join(wham_folder, "mix_both", "*.wav"))
    names = [Path(x).stem for x in w]

    examples = {}
    for n in names:
        examples[n] = {"mix_both": os.path.join(wham_folder, "mix_both", n) + ".wav",
                       "mix_clean": os.path.join(wham_folder, "mix_clean", n) + ".wav",
                       "s1": os.path.join(wham_folder, "s1", n)+ ".wav",
                       "s2": os.path.join(wham_folder, "s2", n)+ ".wav",
                       "noise": os.path.join(wham_folder, "noise", n)+ ".wav",
                       "mix_single": os.path.join(wham_folder, "mix_single", n)+ ".wav",
                       "length": len(sf.SoundFile(os.path.join(wham_folder, "mix_both", n) + ".wav"))
                       }

    return examples


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(Path(args.out_json).parent, exist_ok=True)

    examples = parse_mixtures(args.in_dir)

    with open(args.out_json, "w") as f:
        json.dump(examples, f, indent=4)

