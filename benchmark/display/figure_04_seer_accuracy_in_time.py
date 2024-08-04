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
    "random_survival_forest": "Random Survival Forests",
    "fine_and_gray": "Fine & Gray",
    "aalen_johansen": "Aalen Johansen",
}
include_datasets = ["seer"]
filename = "figure_04_seer_accuracy_in_time.png"

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

        quantiles = agg_result["accuracy_in_time"]["time_quantiles"]
        mean_accuracy = agg_result["accuracy_in_time"]["mean_accuracy"]
        std_accuracy = agg_result["accuracy_in_time"]["std_accuracy"]

        for q, mean_acc, std_acc in zip(quantiles, mean_accuracy, std_accuracy):
            results.append(
                dict(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    q=q,
                    mean_acc=mean_acc,
                    std_acc=std_acc,
                )
            )

df = pd.DataFrame(results)
df["model_name"] = df["model_name"].map(model_remaming)

hue_order = [
    "MultiIncidence",
    "SurvTRACE",
    # "DeepHit",
    "DSM",
    "DeSurv",
    "Random Survival Forests",
    "Fine & Gray",
    "Aalen Johansen",
]
order = dict(zip(hue_order, range(len(hue_order))))
df["order"] = df["model_name"].map(order)
df = df.sort_values("order", ascending=False).drop("order", axis=1)

palette = dict(
    zip(
        hue_order,
        sns.color_palette("colorblind", n_colors=len(hue_order))
    )
)

fig, ax = plt.subplots(figsize=(6, 3), dpi=300)

sns.lineplot(
    df,
    x="q",
    y="mean_acc",
    hue="model_name",
    ax=ax,
    legend=False,
    hue_order=hue_order,
    #palette=palette,
)
# for model_name in df["model_name"].unique():
#     df_model = df.query("model_name == @model_name")

#     plt.fill_between(
#         x=df_model["q"],
#         y1=df_model["mean_acc"] - df_model["std_acc"],
#         y2=df_model["mean_acc"] + df_model["std_acc"],
#         alpha=.4,
#     )

sns.scatterplot(
    df,
    x="q",
    y="mean_acc",
    hue="model_name",
    ax=ax,
    s=50,
    zorder=100,
    style="model_name",
    hue_order=hue_order,
    # markers=["P", (4, 1, 0), "^", (4, 1, 45), "s", "X", ],
    #palette=palette,
)

quantiles = np.arange(0.125, 1, 0.125)
ax.set_xticks(
    quantiles,
    labels=[f"{q:.3f}" for q in quantiles],
    fontsize=10,
);
# ax.set_yticks(
#     ax.get_yticks(),
#     ax.get_yticklabels(),
#     fontsize=10,
# )
ax.set_xlim([.125, .900])
ax.grid()
ax.set_xlabel("Time quantiles", fontsize=10)
ax.set_ylabel("Accuracy in time", fontsize=10)

# h, l = ax.get_legend_handles_labels()
# for h_ in h:
#     #     if h_.get_label() == "MultiIncidence":
#     #         h_.set_marker("D")
#     #         h_.set_markersize(5)
#     #     elif h_.get_label() == "DeSurv":
#     #         h_.set_marker((4, 1, 45))
#     #         h_.set_markersize(7)
#     #     elif h_.get_label() == "DeepHit":
#     #         h_.set_marker((4, 1, 0))
#     #         h_.set_markersize(7)
#     #     elif h_.get_label() == "DSM":
#     #         h_.set_marker("^")
#     #         h_.set_markersize(7)
#     #     else:
#     h_.set_markersize(8)
# ax.legend(h, l)

sns.move_legend(ax, "lower left", bbox_to_anchor=(1, 0))

sns.despine()
plt.tight_layout()
plt.savefig(filename)


# %%
