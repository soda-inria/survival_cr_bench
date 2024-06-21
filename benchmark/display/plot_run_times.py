# %%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style(style="white")
sns.set_context("paper")
sns.set_palette("colorblind")

USER = "vincentmaladiere"



df = pd.DataFrame({
    "model": ["Random Survival Forests", "DeepHit", "Han", "DQS", "SuMo Net", "SurvTRACE", "MultiIncidence"],
    "1k": [5, 1, 60 + 8, 1, 4, 6, 6],
    "10k": [8 * 60 + 30, 7, 10 * 60 + 30, 1, 31, 27, 11],
    "100k": [0, 60 * 1 + 20, 60 * 60 + 36 * 60, 6, 15 * 60, 4 * 60, 61],
})
df = df.sort_values("10k")
df = df.melt(id_vars="model", var_name="$N_{samples}$", value_name="Time")
df["Time"] = np.log(df["Time"]) / np.log(60) + 1

fig, ax = plt.subplots(
    figsize=(5, 3),
    dpi=300,
)

unique = df["model"].unique()
palette = dict(zip(unique, sns.color_palette('colorblind', n_colors=len(unique))))
green, red = palette["MultiIncidence"], palette["SurvTRACE"]
palette["MultiIncidence"] = red
palette["SurvTRACE"] = green
ax = sns.barplot(
    df,
    x="$N_{samples}$",
    y="Time",
    hue="model",
    ax=ax,
    palette=palette
)
ax.set_ylim([.5, 3.5])
every_nth = 2
dict_tick = {
    "0.0": "0",
    "0.5": "0",
    "1.0": "1 second",
    "1.5": "10 seconds",
    "2.0": "1 minute",
    "2.5": "10 minutes",
    "3.0": "1 hour",
    "3.5": "10 hours"
}
labels = []
for idx, label in enumerate(ax.yaxis.get_ticklabels()):
    # if idx % every_nth != 0:
    #     label.set_visible(False)
    # else:
    label.set_text(dict_tick[label.get_text()])
    labels.append(label)

ax.yaxis.set_ticklabels(labels)
ax.grid(axis="y")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.figtext(.82, .13, "x", size=15, color="red")


file_path = f"/Users/{USER}/Desktop/run_fit.pdf"
fig.savefig(file_path, format="pdf", dpi=300, bbox_inches="tight")
