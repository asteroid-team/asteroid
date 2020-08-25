import json
import os
import soundfile as sf
from glob import glob
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    "Script to parse data to json in order to avoid parsing each time at beginning of each experiment"
)
parser.add_argument("--input_dir", type=str)
parser.add_argument("--output_json", type=str)
parser.add_argument("--regex", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    assert os.path.exists(args.input_dir), "Input dir does not exist"
    files = glob(os.path.join(args.input_dir, args.regex), recursive=True)
    to_json = []
    for f in files:
        meta = sf.SoundFile(f)
        samples = len(meta)
        to_json.append({"file": f, "length": samples})

    os.makedirs(Path(args.output_json).parent, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(to_json, f)
