# %%
import pickle
import pandas as pd
import numpy as np

preds = pickle.load(open("../../../Downloads/preds.pkl", "rb"))
y_test = preds["y_test"]
y_pred = preds["y_pred"]
time_grid = preds["time_grid"]

truncation_quantiles = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
times = np.quantile(time_grid, truncation_quantiles)

y_pred = np.concatenate([y_[None, :, :] for y_ in y_pred], axis=0)

results = []
for time_idx in range(len(times)):
    y_pred_time = y_pred[:, :, time_idx]
    mask = (y_test["event"] == 0) & (y_test["duration"] < times[time_idx])
    y_pred_time = y_pred_time[:, ~mask]
    
    y_pred_class = y_pred_time.argmax(axis=0)
    y_test_class = y_test["event"] * (y_test["duration"] < times[time_idx])
    y_test_class = y_test_class.loc[~mask]

    score = (y_test_class.values == y_pred_class).mean()

    results.append(
        dict(
            time=times[time_idx],
            quantile=truncation_quantiles[time_idx],
            score=score,
        )
    )

results = pd.DataFrame(results)
results


# %%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

USER = "vincentmaladiere"



df = pd.DataFrame(
    [
        {
            "Model": "Aalen-Johansen",
            "acc": .929,
            "q": .125,
        },
        {
            "Model": "Aalen-Johansen",
            "acc": .85,
            "q": .250,
        },
        {
            "Model": "Aalen-Johansen",
            "acc": .79,
            "q": .375,
        },
        {
            "Model": "Aalen-Johansen",
            "acc": .705,
            "q": .500,
        },
        {
            "Model": "Aalen-Johansen",
            "acc": .59,
            "q": .625,
        },
        {
            "Model": "Aalen-Johansen",
            "acc": .45,
            "q": .750,
        },
        {
            "Model": "Fine-Gray",
            "acc": .929,
            "q": .125,
        },
        {
            "Model": "Fine-Gray",
            "acc": .851,
            "q": .250,
        },
        {
            "Model": "Fine-Gray",
            "acc": .8,
            "q": .375,
        },
        {
            "Model": "Fine-Gray",
            "acc": .72,
            "q": .500,
        },
        {
            "Model": "Fine-Gray",
            "acc": .62,
            "q": .625,
        },
        {
            "Model": "Fine-Gray",
            "acc": .52,
            "q": .750,
        },
        {
            "Model": "Random Survival Forests",
            "acc": 0.932,
            "q": .125,
        },
        {
            "Model": "Random Survival Forests",
            "acc": 0.872,
            "q": .250,
        },
        {
            "Model": "Random Survival Forests",
            "acc": 0.824,
            "q": .375,
        },
        {
            "Model": "Random Survival Forests",
            "acc": 0.759,
            "q": .500,
        },
        {
            "Model": "Random Survival Forests",
            "acc": 0.677,
            "q": .625,
        },
        {
            "Model": "Random Survival Forests",
            "acc": 0.594,
            "q": .750,
        },
        {
            "Model": "SurvTRACE",
            "acc": .929,
            "q": .125,
        },
        {
            "Model": "SurvTRACE",
            "acc": .865,
            "q": .250,
        },
        {
            "Model": "SurvTRACE",
            "acc": .82,
            "q": .375,
        },
        {
            "Model": "SurvTRACE",
            "acc": .755,
            "q": .500,
        },
        {
            "Model": "SurvTRACE",
            "acc": .68,
            "q": .625,
        },
        {
            "Model": "SurvTRACE",
            "acc": .605,
            "q": .750,
        },
        {
            "Model": "MultiIncidence",
            "acc": .929,
            "q": .125,
        },
        {
            "Model": "MultiIncidence",
            "acc": .865,
            "q": .250,
        },
        {
            "Model": "MultiIncidence",
            "acc": .823,
            "q": .375,
        },
        {
            "Model": "MultiIncidence",
            "acc": .765,
            "q": .500,
        },
        {
            "Model": "MultiIncidence",
            "acc": .705,
            "q": .625,
        },
        {
            "Model": "MultiIncidence",
            "acc": .65,
            "q": .750,
        },
        {
            "Model": "DeepHit",
            "acc": 0.929014,
            "q": .125,
        },
        {
            "Model": "DeepHit",
            "acc": 0.859469,
            "q": .250,
        },
        {
            "Model": "DeepHit",
            "acc": 0.794160,
            "q": .375,
        },
        {
            "Model": "DeepHit",
            "acc": 0.709362,
            "q": .500,
        },
        {
            "Model": "DeepHit",
            "acc": 0.629618,
            "q": .625,
        },
        {
            "Model": "DeepHit",
            "acc": 0.564610,
            "q": .750,
        },
        {
            "Model": "DSM",
            "acc": 0.9306,
            "q": .125,
        },
        {
            "Model": "DSM",
            "acc": 0.8674,
            "q": .250,
        },
        {
            "Model": "DSM",
            "acc": 0.8188,
            "q": .375,
        },
        {
            "Model": "DSM",
            "acc":  0.7487,
            "q": .500,
        },
        {
            "Model": "DSM",
            "acc":  0.6628,
            "q": .625,
        },
        {
            "Model": "DSM",
            "acc": 0.5944,
            "q": .750,
        },
        {
            "Model": "DeSurv",
            "acc": 0.9304,
            "q": .125,
        },
        {
            "Model": "DeSurv",
            "acc": 0.8637,
            "q": .250,
        },
        {
            "Model": "DeSurv",
            "acc": 0.8068,
            "q": .375,
        },
        {
            "Model": "DeSurv",
            "acc":  0.7344,
            "q": .500,
        },
        {
            "Model": "DeSurv",
            "acc":  0.6526,
            "q": .625,
        },
        {
            "Model": "DeSurv",
            "acc": 0.5638,
            "q": .750,
        },
    ]
)


sns.set_style(style="white")
sns.set_context("paper")
sns.set_palette("colorblind")

unique = df["Model"].unique()

palette = dict(zip(unique, sns.color_palette('colorblind', n_colors=len(unique))))
pink = palette["MultiIncidence"]
orange = palette["SurvTRACE"]
green = palette["Random Survival Forests"]
brown = palette["DeepHit"]
yellow = palette["Fine-Gray"]
blue = palette["Aalen-Johansen"]

palette["MultiIncidence"] = orange
palette["SurvTRACE"] = green
palette["Random Survival Forests"] = pink
palette["Fine-Gray"] = "black"
palette["DeepHit"] = brown 
palette["Aalen-Johansen"] = blue

hue_order = [
    "MultiIncidence",
    "SurvTRACE",
    "DeepHit",
    "DSM",
    "DeSurv",
    "Random Survival Forests",    
    "Fine-Gray",
    "Aalen-Johansen",
]

fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
sns.lineplot(
    df.query("Model != 'MultiIncidence'"), x="q", y="acc", hue="Model", ax=ax, legend=False,
    hue_order=hue_order, palette=palette   
)
sns.lineplot(
    df.query("Model == 'MultiIncidence'"), x="q", y="acc", hue="Model", ax=ax, legend=False,
    hue_order=hue_order, palette=palette   
)
sns.scatterplot(
    df, x="q", y="acc", hue="Model", ax=ax, s=40, zorder=100,
    style="Model",
    #markers=["P", (4, 1, 0), "^", (4, 1, 45), "s", "X", ],
    hue_order=hue_order,
    palette=palette,
)

quantiles = np.arange(.125, 1 - .125, .125)
ax.set_xticks(
    quantiles,
    labels=[f"{q:.3f}" for q in quantiles],
    fontsize=10,
)
ax.set_yticks(
    ax.get_yticks(),
    ax.get_yticklabels(),
    fontsize=10,
)
ax.grid()
ax.set_xlabel("Time quantiles", fontsize=10)
ax.set_ylabel("Accuracy in time", fontsize=10)

h, l = ax.get_legend_handles_labels()
for h_ in h:
#     if h_.get_label() == "MultiIncidence":
#         h_.set_marker("D")
#         h_.set_markersize(5)
#     elif h_.get_label() == "DeSurv":
#         h_.set_marker((4, 1, 45))
#         h_.set_markersize(7)
#     elif h_.get_label() == "DeepHit":
#         h_.set_marker((4, 1, 0))
#         h_.set_markersize(7)
#     elif h_.get_label() == "DSM":
#         h_.set_marker("^")
#         h_.set_markersize(7)
#     else:
    h_.set_markersize(8)
ax.legend(h, l)

sns.move_legend(ax, "lower left", bbox_to_anchor=(1, 0))

sns.despine()

file_path = f"/Users/{USER}/Desktop/acc_in_time.png"
fig.savefig(file_path, format="png", dpi=300, bbox_inches="tight")

# %%
