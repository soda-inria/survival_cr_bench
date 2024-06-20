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
    # {
    #     "Model": "Aalen-Johansen",
    #     "fit time": 1.,
    #     "Average IBS": np.mean([0.1209, 0.2832, 0.0834]),
    #     "$S_{Cen-log-simple}$": 2.249,
    # },
    {
        "Model": "Fine & Gray",
        "fit time": 4000,
        "Average IBS": np.mean([0.1055, 0.0281, 0.0822]),
        "$S_{Cen-log-simple}$": 2.249,
    },
    {
        "Model": "DeepHit",
        "fit time": 3000,
        "Average IBS": np.mean([0.0931, 0.0330, 0.0831]),
        "$S_{Cen-log-simple}$": 2.249,
    },
    {
        "Model": "Random Survival Forests",
        "fit time": 42.6 * 60,
        "Average IBS": np.mean([0.0849, 0.0300, 0.0823]),
        "$S_{Cen-log-simple}$": 1.987,
    },
    {
        "Model": "MultiIncidence",
        "fit time": 238,
        "Average IBS": np.mean([0.0832, 0.0273, 0.0757]),
        "$S_{Cen-log-simple}$": 1.740,
    },
    {
        "Model": "SurvTRACE",
        "fit time": 1394,
        "Average IBS": np.mean([0.0871, 0.0287, 0.0800]),
        "$S_{Cen-log-simple}$": 2.270,
    }, 
])

y = "Average IBS" # "$S_{Cen-log-simple}$"


fig, ax = plt.subplots(
    figsize=(5*.9, 3*.9),
    dpi=300,
)

unique = df["Model"].unique()

palette = dict(zip(unique, sns.color_palette('colorblind', n_colors=len(unique))))
blue = palette["MultiIncidence"]
orange = palette["SurvTRACE"]
green = palette["Random Survival Forests"]

palette["MultiIncidence"] = orange
palette["SurvTRACE"] = green
palette["Random Survival Forests"] = blue
palette["Fine & Gray"] = "black"

hue_order = [
    "MultiIncidence",
    "DeepHit",
    "Random Survival Forests",    
    "SurvTRACE",
    "Fine & Gray",
    #"Aalen-Johansen",
]

ax = sns.scatterplot(
    df,
    x="fit time",
    y=y,
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
ax.set_ylabel(y, fontsize=12)

ax.grid(axis="x")

# h, l = ax.get_legend_handles_labels()
# h = h[:7]+h[11:]
# l = l[:7]+l[11:]

# h.insert(1, h.pop(-4))
# l.insert(1, l.pop(-4))

# ax.legend(h, l)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))



file_path = f"/Users/{USER}/Desktop/run_fit_tradeoff_{y}.png"
fig.savefig(file_path, format="png", dpi=300, bbox_inches="tight")


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
        "Model": "Aalen-Johansen",
        "fit time": 1.,
        "Average IBS": np.mean([0.1209, 0.2832, 0.0834]),
        "$S_{Cen-log-simple}$": 2.249,
    },
    {
        "Model": "Fine & Gray",
        "fit time": 4000,
        "Average IBS": np.mean([0.1055, 0.0281, 0.0822]),
        "$S_{Cen-log-simple}$": 2.249,
    },
    {
        "Model": "Random Survival Forests",
        "fit time": 42.6 * 60,
        "Average IBS": np.mean([0.0849, 0.0300, 0.0823]),
        "$S_{Cen-log-simple}$": 1.987,
    },
    {
        "Model": "SurvTRACE",
        "fit time": 1394,
        "Average IBS": np.mean([0.0871, 0.0287, 0.0800]),
        "$S_{Cen-log-simple}$": 2.270,
    }, 
    {
        "Model": "MultiIncidence",
        "fit time": 238,
        "Average IBS": np.mean([0.0832, 0.0273, 0.0757]),
        "$S_{Cen-log-simple}$": 1.740,
    },
    {
        "Model": "DeepHit",
        "fit time": 56 * 60 + 12,
        "Average IBS": np.mean([0.0931, 0.0330, 0.0831]),
        "$S_{Cen-log-simple}$": 2.249,
    },
    {
        "Model": "DSM",
        "fit time": 397.79,
        "Average IBS": np.mean([0.0875, 0.0310, 0.0869]),
        "$S_{Cen-log-simple}$": None,
    },
    {
        "Model": "DeSurv",
        "fit time": 2644.98,
        "Average IBS": np.mean([0.0975, 0.0327, 0.0869]),
        "$S_{Cen-log-simple}$": None,
    },
])

y = "Average IBS" # "$S_{Cen-log-simple}$"


fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(5*.7, 3*.8),
    dpi=300,
    sharex=True,
)
fig.subplots_adjust(hspace=0.05)

ax2.set_ylim([0.06, 0.075])
ax1.set_ylim([0.16, 0.165])

unique = df["Model"].unique()

palette = dict(zip(unique, sns.color_palette('colorblind', n_colors=len(unique))))
pink = palette["MultiIncidence"]
orange = palette["SurvTRACE"]
green = palette["Random Survival Forests"]
brown = palette["DeepHit"]
yellow = palette["Fine & Gray"]
blue = palette["Aalen-Johansen"]

palette["MultiIncidence"] = orange
palette["SurvTRACE"] = green
palette["Random Survival Forests"] = pink
palette["Fine & Gray"] = "black"
palette["DeepHit"] = brown 
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

sns.scatterplot(
    df,#.query("Model != 'Aalen-Johansen'"),
    x="fit time",
    y=y,
    hue="Model",
    hue_order=hue_order,
    style="Model",
    #markers=["+", "^", "D", "s", "o", "o", "o"],
    ax=ax2,
    s=100,
    palette=palette,
)
sns.scatterplot(
    df.query("Model == 'Aalen-Johansen'"),
    x="fit time",
    y=y,
    hue="Model",
    # hue_order=hue_order,
    markers="o",
    s=100,
    ax=ax1
    # palette=palette,
)

ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)


d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)


ticks = [1, 5 * 60, 20 * 60, 40 * 60, 60 * 60]
labels = ["", "5min", "20min", "40min", "1h"]
ax2.set_xticks(ticks, labels=labels, fontsize=12)

ax2.set_yticks(
    ax2.get_yticks()[:-1],
    ax2.get_yticklabels()[:-1],
    fontsize=12,
)
ax1.set_yticks(
    ax1.get_yticks()[1:-1],
    ax1.get_yticklabels()[1:-1],
    fontsize=12,
)

ax2.set_xlabel("Fit time on cpu", fontsize=12)
ax2.set_ylabel(y, fontsize=12, y=1)
ax1.set_ylabel("")

ax2.grid(axis="x")
ax1.grid(axis="x")

ax2.yaxis.tick_left()
ax1.yaxis.tick_left()

ax1.get_legend().remove()

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

#ax1.get_legend().remove()

h = h2[:-1] + h1
l = l2[:-1] + l1
ax2.legend(h, l)

sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1.8))


file_path = f"/Users/{USER}/Desktop/run_fit_tradeoff_{y}.png"
fig.savefig(file_path, format="png", dpi=300, bbox_inches="tight")
# %%
