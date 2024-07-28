# %%
"""Require y_pred in raw scores.
"""
import numpy as np
import json
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt

from lifelines import AalenJohansenFitter

from hazardous.data._competing_weibull import make_synthetic_competing_weibull

sns.set_style(style="white")
sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")

DATASET_NAME = "weibull_XXX"

data_params = dict(
    n_events=3,
    independent_censoring=True,
    complex_features=True,
    censoring_relative_scale=1.5,
    n_samples=10_000,
)

fig, axes = plt.subplots(
    figsize=(5, 3),
    ncols=data_params["n_events"],
    sharey=True,
    dpi=300,
)

path_raw = Path("../scores/raw/")
for estimator_path in path_raw.glob("*.json"):
    
    data_estimator = json.load(open(estimator_path))
    estimator_name = data_estimator["estimator_name"]

    if DATASET_NAME in data_estimator and "y_pred" in data_estimator[DATASET_NAME][0]:
        data_raw = data_estimator[DATASET_NAME]
        time_grid = data_raw[0]["time_grid"]
        for event_idx, ax in enumerate(axes):
            y_pred_event = np.concat(
                [
                    data_seed["y_pred"][event_idx + 1][None, :, :]
                    for data_seed in data_raw
                ],
                axis=0,
            )
            y_pred_event_mean = y_pred_event.mean(axis=0)
            y_pred_event_std = y_pred_event.std(axis=0)
        
            ax.plot(time_grid, y_pred_event_mean, label=estimator_name)
        ax.legend()


bunch = make_synthetic_competing_weibull(
    return_X_y=False,
    **data_params,
)
X, y, y_uncensored = bunch.X, bunch.y, bunch.y_uncensored

for idx, ax in enumerate(axes):
    aj = AalenJohansenFitter(calculate_variance=False).fit(
        durations=y["duration"],
        event_observed=y["event"],
        event_of_interest=idx + 1,
    )
    aj.plot(ax=ax, label="Aalen Johansen", color="k")

    aj.fit(
        durations=y["duration"],
        event_observed=y_uncensored["event"],
        event_of_interest=idx + 1,
    )
    aj.plot(ax=ax, label="Aalen Johansen uncensored", color="k", linestyle="--")

    ax.set(
        title=f"Event {idx+1}",
        xlabel="",
    )
    ax.get_xaxis().set_visible(False)

    if ax is not axes[-1]:
        ax.get_legend().remove()

sns.move_legend(axes[-1], "upper left", bbox_to_anchor=(1, 1))
sns.despine()
plt.tight_layout()

file_path = "marginal_incidence.pdf"
fig.savefig(file_path, format="pdf")


# %%
