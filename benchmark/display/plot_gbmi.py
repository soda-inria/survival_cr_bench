# %%
import json
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


path = Path("../scores/raw/gbmi/kkbox.json")
data = json.load(open(path))[0]

# %%

fig, ax = plt.subplots()
palette = sns.color_palette("colorblind")
for idx in range(8, 12):
    sns.lineplot(
        x=data["time_grid"],
        y=np.asarray(data["y_pred"])[1, idx, :],
        ax=ax,
        color=palette[idx],
        label=f"{idx=}"
    )
ax.legend()

# %%
fig, ax = plt.subplots()
palette = sns.color_palette("colorblind")
sns.lineplot(
    x=data["time_grid"],
    y=np.asarray(data["y_pred"])[1].mean(axis=0),
    ax=ax,
    color=palette[idx],
    label="1024",
)

ax.legend()
