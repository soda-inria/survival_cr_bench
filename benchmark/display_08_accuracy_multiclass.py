# %%
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from hazardous._aalen_johansen import AalenJohansenEstimator
from hazardous.utils import get_n_events
from display_utils import (
    aggregate_result,
    get_estimator,
    load_dataset,
)
from hazardous.utils import make_time_grid

sns.set_style(style="white")
sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")

USER = "jalberge"
DATASET_NAME = "seer"
SEER_PATH = "../hazardous/data/seer_cancer_cardio_raw_data.txt"
SEED = 0


path_session_dataset = Path("2024-01-29") / DATASET_NAME
estimator_names = ["gbmi_log_loss", "survtrace", "aalen_johansen", "fine_and_gray"]
df = aggregate_result(path_session_dataset, estimator_names)
df["n_samples"] = df["estimator"].apply(lambda x: len(x.y_train))


idx_max = df[df["n_samples"] == df["n_samples"].max()].index
idx_max
# %%
df
# %%
df = df.iloc[[1, 2, 5, 8]]
# %%
df
# %%
bunch = load_dataset(DATASET_NAME, data_params={}, random_state=SEED)
X_test, y_test = bunch.X, bunch.y


time_grid = make_time_grid(y_test["duration"])
n_events = get_n_events(y_test["event"])

truncation_quantiles = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
times = np.quantile(time_grid, truncation_quantiles)

aj = AalenJohansenEstimator(seed=SEED).fit(
    X=None,
    y=y_test,
)
y_pred_aj = aj.predict_cumulative_incidence(X_test, times)


all_y_pred = {}
for estimator_name, df_est in df.groupby("estimator_name"):
    print(estimator_name)
    estimator = get_estimator(df_est, estimator_name)
    y_pred = estimator.predict_cumulative_incidence(X_test, times)
    y_train = estimator.y_train
    all_y_pred[estimator_name] = y_pred

# %%
new_metric = {"estimator_name": [], "time": [], "score": [], "quantile": []}
for estimator_name, y_pred_all_times in all_y_pred.items():
    for time_idx in range(len(times)):
        y_pred = y_pred_all_times[:, :, time_idx]
        mask = (y_test["event"] == 0) & (y_test["duration"] < times[time_idx])
        y_pred = y_pred[:, ~mask]
        y_pred_class = y_pred.argmax(axis=0)
        y_test_class = y_test["event"] * (y_test["duration"] < times[time_idx])
        y_test_class = y_test_class.loc[~mask]
        new_metric["estimator_name"].append(estimator_name)
        new_metric["time"].append(times[time_idx])
        new_metric["quantile"].append(truncation_quantiles[time_idx])
        new_metric["score"].append((y_test_class.values == y_pred_class).mean())


new_metric = pd.DataFrame(new_metric)


# %%

new_metric["estimator_name"] = new_metric["estimator_name"].apply(
    lambda x: {
        "aalen_johansen": "Aalen-Johansen",
        "gbmi_log_loss": "MultiIncidence",
        "survtrace": "SurvTRACE",
        "fine_and_gray": "Fine-Gray",
    }[x]
)
new_metric.rename(columns={"estimator_name": "Estimator"}, inplace=True)

# %%

fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
sns.lineplot(
    new_metric,
    x="quantile",
    y="score",
    hue="Estimator",
    marker="o",
    ax=ax,
    hue_order=["Aalen-Johansen", "Fine-Gray", "SurvTRACE", "MultiIncidence"],
)


ax.set_xticks(new_metric["quantile"].values)
ax.grid(axis="y")
sns.despine()
# ax.set_yscale("log")
ax.legend(loc="lower left")
ax.set(
    xlabel="Empirical Quantiles of Observed Censored Times",
    ylabel="Accuracy of the Argmax of the CIFs",
)

file_path = f"/Users/{USER}/Desktop/08_accuracy.pdf"
fig.savefig(file_path, format="pdf", dpi=300, bbox_inches="tight")
# %%
