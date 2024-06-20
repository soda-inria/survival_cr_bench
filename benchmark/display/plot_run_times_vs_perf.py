# %%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style(style="white")
sns.set_context("paper")
sns.set_palette("colorblind")

USER = "vincentmaladiere"



df = pd.DataFrame([
    {
        "Model": "MultiIncidence",
        "Dataset": "Support",
        "fit time": 7,
        "IBS": 0.191,
        "$S_{Cen-log-simple}$": 1.740,
    },
    {
        "Model": "MultiIncidence",
        "Dataset": "Metabric",
        "fit time": 3.9,
        "IBS": 0.168,
        "$S_{Cen-log-simple}$": 2.169,
    }, 
    {
        "Model": "SurvTRACE",
        "Dataset": "Support",
        "fit time": 29.2,
        "IBS": 0.194,
        "$S_{Cen-log-simple}$": 1.870,
    },
    {
        "Model": "SurvTRACE",
        "Dataset": "Metabric",
        "fit time": 22.6,
        "IBS": 0.168,
        "$S_{Cen-log-simple}$": 2.270,
    }, 
    {
        "Model": "DeepHit",
        "Dataset": "Support",
        "fit time": 29.2,
        "IBS": 0.217,
        "$S_{Cen-log-simple}$": 2.249,
    },
    {
        "Model": "DeepHit",
        "Dataset": "Metabric",
        "fit time": 8,
        "IBS": 0.180,
        "$S_{Cen-log-simple}$": 2.271,
    }, 
    {
        "Model": "SumoNet",
        "Dataset": "Support",
        "fit time": 45.9,
        "IBS": 0.194,
        "$S_{Cen-log-simple}$": 1.721,
    },
    {
        "Model": "SumoNet",
        "Dataset": "Metabric",
        "fit time": 4.9,
        "IBS": 0.169,
        "$S_{Cen-log-simple}$": 2.302,
    }, 
    {
        "Model": "DQS",
        "Dataset": "Support",
        "fit time": 1.2,
        "IBS": 0.202,
        "$S_{Cen-log-simple}$": 1.987,
    },
    {
        "Model": "DQS",
        "Dataset": "Metabric",
        "fit time": .8,
        "IBS": 0.180,
        "$S_{Cen-log-simple}$": 2.205,
    }, 
    {
        "Model": "Han et al.",
        "Dataset": "Support",
        "fit time": 344.8,
        "IBS": 0.260,
        "$S_{Cen-log-simple}$": 3.483,
    },
    {
        "Model": "Han et al.",
        "Dataset": "Metabric",
        "fit time": 135.5,
        "IBS": 0.191,
        "$S_{Cen-log-simple}$": 2.420,
    }, 
    {
        "Model": "Random Survival Forests",
        "Dataset": "Support",
        "fit time": 20.5,
        "IBS": 0.225,
        "$S_{Cen-log-simple}$": 1.987,
    },
    {
        "Model": "Random Survival Forests",
        "Dataset": "Metabric",
        "fit time": 5.5,
        "IBS": 0.197,
        "$S_{Cen-log-simple}$": 2.442,
    }, 
    {
        "Model": "PCHazard",
        "Dataset": "Support",
        "fit time": 0.70,
        "IBS": 0.210,
        "$S_{Cen-log-simple}$": 2.192,
    },
    {
        "Model": "PCHazard",
        "Dataset": "Metabric",
        "fit time": 1,
        "IBS": 0.176,
        "$S_{Cen-log-simple}$": 2.246,
    }
])

y = "$S_{Cen-log-simple}$" # "IBS"


fig, ax = plt.subplots(
    figsize=(5*.9, 3*.9),
    dpi=300,
)

unique = df["Model"].unique()

palette = dict(zip(unique, sns.color_palette('colorblind', n_colors=len(unique))))
blue = palette["MultiIncidence"]
yellow = palette["SurvTRACE"]
green = palette["DeepHit"]
purple = palette["DQS"]
red = palette["SumoNet"]
pink = palette["Random Survival Forests"]

palette["MultiIncidence"] = red
palette["SurvTRACE"] = green
palette["DeepHit"] = yellow 
palette["Random Survival Forests"] = purple
palette["SumoNet"] = "grey"
palette["DQS"] = pink
palette["Han et al."] = blue
palette["PCHazard"] = "gold"

hue_order = [
    "MultiIncidence",
    "SurvTRACE",
    "DeepHit",
    "SumoNet",
    "DQS", 
    "Han et al.",
    "Random Survival Forests",
    "PCHazard",
]

ax = sns.scatterplot(
    df.query("Model != 'MultiIncidence'"),
    x="fit time",
    y=y,
    hue="Model",
    hue_order=hue_order,
    style="Dataset",
    #markers=markers,
    ax=ax,
    s=100,
    palette=palette,
)
ax = sns.scatterplot(
    df.query("Model == 'MultiIncidence'"),
    x="fit time",
    y=y,
    hue="Model",
    style="Dataset",
    #markers=markers,
    ax=ax,
    s=100,
    palette=palette,
    legend=False,
)
ch = ax.get_children()
ch[2].set_zorder(100)

ticks = [0, 10, 60, 120, 300]
labels = ["", "10s", "1min", "2min", "5min"]
ax.set_xticks(ticks, labels=labels, fontsize=12)
plt.yticks(fontsize=12)

ax.set_xlabel("Fit time", fontsize=13)
ax.set_ylabel(y, fontsize=13)

ax.grid(axis="x")

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

file_path = f"/Users/{USER}/Desktop/run_fit_tradeoff_{y}.png"
fig.savefig(file_path, format="png", dpi=300, bbox_inches="tight")


# %%
