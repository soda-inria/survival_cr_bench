# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


model_remaming = {
    "gbmi": "MultiIncidence",
    "survtrace": "SurvTRACE",
    "deephit": "DeepHit",
    "DSM": "DSM",
    "DeSurv": "DeSurv",
    "random_survival_forest": "RandomSurvivalForest",
    "fine_and_gray": "Fine & Gray",
    "aalen_johansen": "Aalen Johansen",
}
include_datasets = ["seer"]
filename = "figure_02_seer_fit_time_vs_ibs.png"

path_scores = Path("../scores/agg/")
results = []

for path_model in path_scores.glob("*"):
    model_name = path_model.name
    if not model_name in list(model_remaming):
        continue
    for path_dataset in path_model.glob("*"):
        dataset_name = path_dataset.name.split(".")[0]
        if not dataset_name in include_datasets:
            continue
        agg_result = json.load(open(path_dataset))
        global_mean = np.mean(
            [row["mean_ibs"] for row in agg_result["event_specific_ibs"]]
        )
        global_std = np.sum(
            [row["std_ibs"] for row in agg_result["event_specific_ibs"]]
        )
        results.append(
            dict(
                mean_fit_time=agg_result["mean_fit_time"],
                std_fit_time=agg_result["std_fit_time"],
                mean_ibs=global_mean,
                std_ibs=global_std,
                model_name=model_name,
                dataset_name=dataset_name,
            )
        )

df = pd.DataFrame(results)
df["model_name"] = df["model_name"].map(model_remaming)
model_names = df["model_name"].unique()

hue_order = [
    "MultiIncidence",
    "SurvTRACE",
    #"DeepHit",
    "DSM",
    "DeSurv", 
    #"Random Survival Forests",
    "Fine & Gray",
    "Aalen Johansen",
]

palette = dict(
    zip(
        hue_order,
        sns.color_palette('colorblind', n_colors=len(hue_order))
    )
)
# blue = palette["MultiIncidence"]
# yellow = palette["SurvTRACE"]
# green = palette["DSM"]
# red = palette["DeSurv"]
# pink = palette["Fine & Gray"]
# brown = palette["Aalen Johansen"]

# palette["MultiIncidence"] = red
# palette["SurvTRACE"] = green
# palette["DSM"] = pink
# palette["DeSurv"] = yellow
# palette["Fine & Gray"] = "black"
# palette["Aalen Johansen"] = blue

c = "black"
plt.errorbar(
    x=df["mean_fit_time"],
    y=df["mean_ibs"],
    yerr=df['std_ibs'],
    fmt='none',
    c=c,
    capsize = 2,
)
plt.errorbar(
    x=df["mean_fit_time"],
    xerr=df['std_fit_time'],
    y=df["mean_ibs"],
    fmt='none',
    c=c,
    capsize = 2,
)
ax = sns.scatterplot(
    df,
    x="mean_fit_time",
    y="mean_ibs",
    hue="model_name",
    hue_order=hue_order,
    style="model_name",
    s=200,
    palette=palette,
    zorder=10,
    alpha=1
)

ch = ax.get_children()

ticks = [0, 5 * 60, 20 * 60, 40 * 60, 60 * 60]
labels = ["", "5min", "20min", "40min", "1h"]
ax.set_xticks(ticks, labels=labels, fontsize=12)
plt.yticks(fontsize=12)

ax.set_xlabel("Fit time", fontsize=13)
ax.set_ylabel("Mean IBS", fontsize=13)

ax.grid(axis="x")

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.savefig(filename)

# %%
