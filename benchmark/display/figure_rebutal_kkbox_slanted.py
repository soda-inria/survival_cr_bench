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

filename = "figure_rebutal_kkbox.png"

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

#fig, ax = plt.subplots(figsize=(6.5, 3), dpi=300)

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(4, 2),
    dpi=300,
    sharex=True,
    height_ratios=[1, 3]
)
fig.subplots_adjust(hspace=0.1)

ax2.set_ylim([0.085, 0.171])
ax1.set_ylim([0.245, 0.255])

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
sns.scatterplot(
    df,
    x="mean_fit_time",
    y="mean_ibs",
    hue="model_name",
    #hue_order=hue_order,
    style="dataset_name",
    s=200,
    palette=palette,
    zorder=10,
    alpha=1,
    ax=ax2,
)
sns.scatterplot(
    df.query("model_name == 'Han et al. (GPU)'"),
    x="mean_fit_time",
    y="mean_ibs",
    hue="model_name",
    #hue_order=hue_order,
    style="dataset_name",
    s=200,
    palette=palette,
    zorder=10,
    alpha=1,
    ax=ax1,
)

ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax2.set_xscale("log")
ticks = [10, 2 * 60, 10 * 60, 30 * 60, 2 * 60 * 60, 4 * 60 * 60, 8 * 60 * 60]
labels = ["10s", "2min", "10min", "30min", "2h", "4h", "8h"]
ax2.set_xticks(ticks)
ax2.set_xticklabels(labels, rotation=45, fontsize=12)

ticks = [0.09, 0.11, 0.13, 0.15]
labels = ["0.09", "0.11", "0.13", "0.15"]
ax2.set_yticks(ticks)
ax2.set_yticklabels(labels, fontsize=12)

ax1.set_ylabel("")
ax1.tick_params(bottom=False)

labels = ax1.get_yticklabels()[1:-1]
for label in labels:
    label.set_text("0.25") 
ax1.set_yticks(
    ax1.get_yticks()[1:-1],
    labels,
    fontsize=12,
)

ax1.minorticks_off()
ax2.set_xlabel("Fit time", fontsize=13)
ax2.set_ylabel("Mean IBS", fontsize=13)

ax1.grid()
ax2.grid()

ax1.get_legend().remove()
sns.move_legend(ax2, "upper left", bbox_to_anchor=(1.01, 1.8), fontsize=9)
plt.savefig(filename)

# %%
