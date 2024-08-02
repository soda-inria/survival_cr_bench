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
    "sumonet": "SumoNet",
    "dqs": "DQS",
    "han-nll": "Han et al. (nll)",
    "han-bll_game": "Han et al. (game)",
    "sksurv_boosting": "Gradient Boosting Survival",
    "random_survival_forest": "RandomSurvivalForest",
    "fine_and_gray": "Fine & Gray",
    "aalen_johansen": "Aalen Johansen",
    "pchazard": "PCHazard",
}
include_datasets = ["support", "metabric"]
filename = "figure_s8_fit_time_vs_scenlog.png"

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
                mean_cenlog=agg_result["mean_censlog"],
                std_cenlog=agg_result["std_censlog"],
                model_name=model_name,
                dataset_name=dataset_name,
            )
        )

df = pd.DataFrame(results)

# hue_order = [
#     "MultiIncidence",
#     "SurvTRACE",
#     # "DeepHit",
#     "DSM",
#     "DeSurv",
#     "Random Survival Forests",
#     "Fine & Gray",
#     "Aalen Johansen",
# ]

# palette = dict(
#     zip(
#         hue_order,
#         sns.color_palette("colorblind", n_colors=len(hue_order))
#     )
# )

c = "black"
plt.errorbar(
    x=df["mean_fit_time"],
    y=df["mean_cenlog"],
    yerr=df['std_cenlog'],
    fmt='none',
    c=c,
    capsize = 2,
)
plt.errorbar(
    x=df["mean_fit_time"],
    xerr=df['std_fit_time'],
    y=df["mean_cenlog"],
    fmt='none',
    c=c,
    capsize = 2,
)
ax = sns.scatterplot(
    df,
    x="mean_fit_time",
    y="mean_cenlog",
    hue="model_name",
    #hue_order=hue_order,
    style="dataset_name",
    s=200,
    #palette=palette,
    zorder=10,
    alpha=1
)

ch = ax.get_children()

ticks = [0, 10, 1 * 60, 2 * 60, 5 * 60]
labels = ["", "10s", "1min", "2min", "5min"]
ax.set_xticks(ticks, labels=labels, fontsize=12)
plt.yticks(fontsize=12)

ax.set_xlabel("Fit time", fontsize=13)
ax.set_ylabel("Mean Cenlog", fontsize=13)

ax.grid(axis="x")

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(filename)

# %%
