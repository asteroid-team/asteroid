import re
import pandas as pd

df = pd.read_csv("../filtered_train.csv")
df_val = pd.read_csv("../filtered_val.csv")

change = lambda x: re.sub(r'\.\./', '../../', x)

for i in list(df):
    df[i] = df[i].apply(change)

for i in list(df_val):
    df_val[i] = df_val[i].apply(change)

df.to_csv("train.csv", index=False)
df_val.to_csv("val.csv", index=False)
