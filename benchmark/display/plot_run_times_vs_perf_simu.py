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
        "fit time": 25,
        "IBS": 0.1329,
    },
    {
        "Model": "SurvTRACE",
        "fit time": 125,
        "IBS": 0.1526,
    }, 
    {
        "Model": "DeepHit",
        "fit time": 303,
        "IBS": 0.155,
    },
    {
        "Model": "DSM",
        "fit time": 28,
        "IBS": 0.152,
    },
    {
        "Model": "DeSurv",
        "fit time": 84.2,
        "IBS": 0.164,
    }, 
    {
        "Model": "Random Survival Forests",
        "fit time": 25691,
        "IBS": 0.136,
    },
    {
        "Model": "Fine & Gray",
        "fit time": 25691,
        "IBS": 0.1476,
    }, 
    {
        "Model": "Aalen-Johansen",
        "fit time": 5,
        "IBS": 0.1711,
    },
])

y = "IBS"


fig, ax = plt.subplots(
    figsize=(5*.9, 3*.9),
    dpi=300,
)

unique = df["Model"].unique()

palette = dict(zip(unique, sns.color_palette('colorblind', n_colors=len(unique))))
blue = palette["MultiIncidence"]
yellow = palette["SurvTRACE"]
green = palette["DeepHit"]
red = palette["DSM"]
purple = palette["DeSurv"]
brown = palette["Random Survival Forests"]
pink = palette["Fine & Gray"]
grey = palette["Aalen-Johansen"]

palette["MultiIncidence"] = red
palette["SurvTRACE"] = green
palette["DeepHit"] = yellow 
palette["DSM"] = pink
palette["DeSurv"] = grey
palette["Random Survival Forests"] = purple
palette["Fine & Gray"] = "black"
palette["Aalen-Johansen"] = blue

hue_order = [
    "MultiIncidence",
    "SurvTRACE",
    "DeepHit",
    "DSM",
    "DeSurv", 
    "Random Survival Forests",
    "Fine & Gray",
    "Aalen-Johansen",
]


markers=["D", "P", (4, 1, 0), "^", (4, 1, 45), "s", "X", "o"]

ax = sns.scatterplot(
    df, #.sort_values("Model"),
    x="fit time",
    y=y,
    hue="Model",
    hue_order=hue_order,
    markers=markers,
    style="Model",
    sizes=[80, 120, 140, 120, 140, 80, 80, 80],
    size="Model",
    ax=ax,
    s=100,
    palette=palette,
)
ax.set_xscale('log')


ch = ax.get_children()

ticks = [1, 5, 30, 120, 600, 26_000]
labels = ["", "5s", "30s", "2min", "10min", "7h"]
ax.set_xticks(ticks, labels=labels, fontsize=12)
plt.yticks(fontsize=12)

ax.set_xlabel("Fit time on cpu", fontsize=13)
ax.set_ylabel(y, fontsize=13)

ax.grid(axis="x")

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

file_path = f"/Users/{USER}/Desktop/run_fit_tradeoff_{y}.png"
fig.savefig(file_path, format="png", dpi=300, bbox_inches="tight")


# %%
