import os
import pandas as pd

df_train = pd.read_csv("../../data/train.csv")
df_val = pd.read_csv("../../data/val.csv")

print(df_train.shape)
print(df_val.shape)

corrupt_files = []

with open("../../data/corrupt_frames_list.txt") as f:
    corrupt_files = f.readlines()

corrupt_files = set(corrupt_files)
print(len(corrupt_files))
corrupt_files = [c[:-1] for c in corrupt_files]
print(corrupt_files)

df_train = df_train[~df_train["video_1"].isin(corrupt_files)]
df_val = df_val[~df_val["video_1"].isin(corrupt_files)]

df_train = df_train[~df_train["video_2"].isin(corrupt_files)]
df_val = df_val[~df_val["video_2"].isin(corrupt_files)]

print(df_train.shape)
print(df_val.shape)

df_train.to_csv("../../data/train.csv", index=False)

df_val.to_csv("../../data/val.csv", index=False)
