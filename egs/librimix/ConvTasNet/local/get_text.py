import argparse
import glob
import os
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--libridir", required=True, type=str)
parser.add_argument("--outfile", required=True, type=str)
parser.add_argument("--split", type=str, default="train-360")

args = parser.parse_args()

libridir = os.path.join(args.libridir, args.split)
trans_txt_list = glob.glob(os.path.join(libridir, "**/*.txt"), recursive=True)
row_list = []
for name in trans_txt_list:
    f = open(name, "r")
    for line in f:
        dict1 = {}
        split_line = line.split(" ", maxsplit=1)
        dict1["utt_id"] = split_line[0]
        dict1["text"] = split_line[1].replace("\n", "").replace("\r", "")
        row_list.append(dict1)

df = pd.DataFrame(row_list)
df.to_csv(args.outfile, index=False)
