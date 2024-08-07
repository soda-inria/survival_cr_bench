# %%
import json
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


path = Path("../scores/raw/han-bs_game/kkbox_1024.json")
data = json.load(open(path))
han_1024 = data[0]
han_100k = data[1]

fig, ax = plt.subplots()
palette = sns.color_palette("colorblind")
for idx in range(4, 7):
    sns.lineplot(
        x=han_1024["time_grid"],
        y=np.asarray(han_1024["y_pred"])[1, idx, :],
        ax=ax,
        color=palette[idx],
        label="1024",
    )
    sns.lineplot(
        x=han_100k["time_grid"],
        y=np.asarray(han_100k["y_pred"])[1, idx, :],
        ax=ax,
        linestyle="--",
        color=palette[idx],
        label="100k"
    )

ax.legend()

# %%
fig, ax = plt.subplots()
palette = sns.color_palette("colorblind")
sns.lineplot(
    x=han_1024["time_grid"],
    y=np.asarray(han_1024["y_pred"])[1].mean(axis=0),
    ax=ax,
    color=palette[idx],
    label="1024",
)
sns.lineplot(
    x=han_100k["time_grid"],
    y=np.asarray(han_100k["y_pred"])[1].mean(axis=0),
    ax=ax,
    linestyle="--",
    color=palette[idx],
    label="100k"
)

ax.legend()
# %%
y_pred_100k = han_100k.pop("y_pred")
y_pred_1024 = han_1024.pop("y_pred")
# %%
han_100k["event_specific_c_index"]

# %%
han_1024["event_specific_c_index"]
# %%

han_100k["event_specific_ibs"]

# %%
han_1024["event_specific_ibs"]
# %%

han_100k
# %%
