# %%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import json
import itertools

sns.set_style(style="white")
sns.set_context("paper")
sns.set_palette("colorblind")

dataset_name = "weibull"

#%%
def ave_ibs(x):
    ibs = [event["ibs"] for event in x]
    return np.mean(ibs)
#%%
raw_scores = json.load(open("../scores/raw_scores.json", "r"))
# agg_scores = json.load(open("../scores/agg_scores.json", "r"))

key_scores = [key for key in raw_scores.keys() if dataset_name in key]
# %%
results = [
    [ {
        "Model": model['model_name'],
        "fit time": model["fit_time"],
        "event_specific IBS": model["event_specific_ibs"],
        "y_test": model["y_test"],
        "y_pred": model["y_pred"],
        "time_grid": model["time_grid"],
        "n_rows": model["n_rows"],
        "random_state": model["random_state"],
    }

    for model in  raw_scores[model_dataset]] 
    for model_dataset in raw_scores
]

results_1d = list(itertools.chain(*results))

df = pd.DataFrame(results_1d)
df["Average IBS"] = df["event_specific IBS"].apply(ave_ibs)
#%%
names_plotted = {"random_survival_forest": "Random Survival Forests"}
df["Model"] = df["Model"].replace(names_plotted)
# %%
trade_off_mean = df.groupby("Model")[["fit time", "Average IBS"]].mean().reset_index()
trade_off_std = df.groupby("Model")[["fit time", "Average IBS"]].std().reset_index()
# %%
fig, ax = plt.subplots(
    figsize=(5*.9, 3*.9),
    dpi=300,
)

unique = trade_off_mean["Model"].unique()

palette = dict(zip(unique, sns.color_palette('colorblind', n_colors=len(unique))))
#blue = palette["MultiIncidence"]
#orange = palette["SurvTRACE"]
green = palette["Random Survival Forests"]

#palette["MultiIncidence"] = orange
#palette["SurvTRACE"] = green
palette["Random Survival Forests"] = green
#palette["Fine & Gray"] = "black"

hue_order = [
    #"MultiIncidence",
    #"DeepHit",
    "Random Survival Forests",    
    #"SurvTRACE",
    #"Fine & Gray",
    #"Aalen-Johansen",
]

ax = sns.scatterplot(
    trade_off_mean,
    x="fit time",
    y="Average IBS",
    hue="Model",
    hue_order=hue_order,
    #style="Dataset",
    #markers=markers,
    ax=ax,
    s=100,
    palette=palette,
)

ticks = [1, 5 * 60, 20 * 60, 40 * 60, 60 * 60]
labels = ["", "5min", "20min", "40min", "1h"]
ax.set_xticks(ticks, labels=labels, fontsize=12)
plt.yticks(fontsize=12)

ax.set_xlabel("Fit time", fontsize=12)
ax.set_ylabel("Average IBS", fontsize=12)

ax.grid(axis="x")
# %%
