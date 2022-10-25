import pandas as pd
import math
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
import seaborn as sns
import argparse

global args
parser = argparse.ArgumentParser(description='plotter')
parser.add_argument('--input', default="", required=True, help='Input feather')
parser.add_argument('--output', default="", required=True, help='Output pdf')
parser.add_argument('--logscale', default=False, action="store_true", required=False, help='')
parser.add_argument('--split', default=5, required=False, type=int, help='Split at %')
args = parser.parse_args()


sns.set_theme()
sns.set_style("whitegrid")

df = pd.read_feather(args.input)
if args.logscale:
    df["Loss"] = np.log10(df["Loss"])
print(df)

if args.split:
    df_full = df.assign(Dataset=["All"]*len(df))
    for i in range(df["Epoch"].min(), df["Epoch"].max() + 1):
        df2 = df[df["Epoch"] == i]
        df2 = df2.sort_values(by=["Loss"])
        df2top = df2.tail(int(len(df2) * 0.01 * args.split))
        df2top = df2top.assign(Dataset=["Top{}%".format(args.split)]*len(df2top))
        df2bottom = df2.head(int(len(df2) * (1 - (0.01 * args.split))))
        df2bottom = df2bottom.assign(Dataset=["Bottom{}%".format(100 - args.split)]*len(df2bottom))
        df_full = pd.concat([df_full, df2top], ignore_index=True)
        df_full = pd.concat([df_full, df2bottom], ignore_index=True)
    df = df_full


if args.split:
    plt.figure(figsize=(60,7.5))
    ax = sns.violinplot(data=df, x="Epoch", y="Loss", hue="Dataset")
else:
    plt.figure(figsize=(20,7.5))
    ax = sns.violinplot(data=df, x="Epoch", y="Loss")
ax.set_ylabel('Loss distribution', size=15, fontdict=dict(weight='bold'))
ax.set_xlabel('Epoch', size=15, fontdict=dict(weight='bold'))

if args.logscale:
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ymin, ymax = ax.get_ylim()
    tick_range = np.arange(ymin, ymax)
    ax.yaxis.set_ticks(tick_range)
    ax.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)


plt.tight_layout()
plt.savefig(args.output)



