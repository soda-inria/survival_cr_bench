# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


model_remaming = {
    "gbmi": "MultiIncidence (CPU)",
    "survtrace": "SurvTRACE (GPU)",
    "deephit": "DeepHit (GPU)",
    "sumonet": "SumoNet (GPU)",
    "dqs": "DQS (GPU)",
    "han-bs_game": "Han et al. (GPU)",
    "sksurv_boosting": "GBS (CPU)",
    "random_survival_forest": "RSF (CPU)",
    "pchazard": "PCHazard (GPU)",
}
include_datasets = ["kkbox_100k", "kkbox_1M", "kkbox_2M"]
metric = "ibs"

filename = "figure_rebutal_kkbox_ibs.png"

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
        
        results.append(
            dict(
                mean_fit_time=agg_result["mean_fit_time"],
                std_fit_time=agg_result["std_fit_time"],
                mean_ibs=agg_result["event_specific_ibs"][0]["mean_ibs"],
                std_ibs=agg_result["event_specific_ibs"][0]["std_ibs"],
                model_name=model_name,
                dataset_name=dataset_name,
            )
        )

df = pd.DataFrame(results)
df["model_name"] = df["model_name"].map(model_remaming)

order = {
    "DeepHit (GPU)": 0,
    "PCHazard (GPU)": 1,
#    "Han et al. (nll)": 2,
    "Han et al. (GPU)": 3,
    "DQS (GPU)": 4,
#    "SumoNet": 5,
    "SurvTRACE (GPU)": 6,
    "RSF (CPU)": 7,
    "GBS (CPU)": 8, 
    "MultiIncidence (CPU)": 9,
}

df["order"] = df["model_name"].map(order)
df = df.sort_values(["order", "dataset_name"]).drop("order", axis=1)

palette = dict(
    zip(
        list(order),
        sns.color_palette("colorblind", n_colors=len(order))
    )
)
red = palette["DQS (GPU)"]
green = palette["Han et al. (GPU)"]
grey = palette["MultiIncidence (CPU)"]
purple = palette["SurvTRACE (GPU)"]

palette["MultiIncidence (CPU)"] = red
palette["SurvTRACE (GPU)"] = green
palette["Han et al. (GPU)"] = purple
palette["DQS (GPU)"] = grey

fig, ax = plt.subplots(figsize=(6.5, 3), dpi=300)
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
    #hue_order=hue_order,
    style="dataset_name",
    s=200,
    palette=palette,
    zorder=10,
    alpha=1
)

ch = ax.get_children()

ax.set_xscale("log")
ticks = [10, 2 * 60, 10 * 60, 30 * 60, 2 * 60 * 60, 4 * 60 * 60, 8 * 60 * 60]
labels = ["10s", "2min", "10min", "30min", "2h", "4h", "8h"]
ax.set_xticks(ticks)
ax.set_xticklabels(labels, rotation=45, fontsize=12)
plt.yticks(fontsize=12)

ax.set_xlabel("Fit time", fontsize=13)
ax.set_ylabel("Mean IBS", fontsize=13)

ax.grid()

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(filename)

# %%
