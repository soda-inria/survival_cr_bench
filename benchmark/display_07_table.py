# %%
from pathlib import Path
from IPython.display import display

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from hazardous._aalen_johansen import AalenJohansenEstimator
from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
)
from hazardous.metrics._concordance import concordance_index_ipcw
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
estimator_names = ["gbmi_log_loss", "survtrace", "fine_and_gray"]
df = aggregate_result(path_session_dataset, estimator_names)
df["n_samples"] = df["estimator"].apply(lambda x: len(x.y_train))
# df = df[df["n_samples"] == df["n_samples"].max()]
# %%
df = df.iloc[[1, 5, 2]]

bunch = load_dataset(DATASET_NAME, data_params={}, random_state=SEED)
X_test, y_test = bunch.X, bunch.y

time_grid = make_time_grid(y_test["duration"])
n_events = get_n_events(y_test["event"])

truncation_quantiles = [0.25, 0.5, 0.75]
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


results = []
y_multiclass_train = y_train.copy()
y_multiclass_test = y_test.copy()

for event_idx in range(n_events):
    print(f"Event {event_idx}")

    y_train["event"] = (y_multiclass_train["event"] == (event_idx + 1)).astype("int32")

    y_test["event"] = (y_multiclass_test["event"] == (event_idx + 1)).astype("int32")

    for estimator_name, y_pred in all_y_pred.items():
        ibs = integrated_brier_score_incidence(
            y_train=y_multiclass_train,
            y_test=y_multiclass_test,
            y_pred=y_pred[event_idx + 1],
            times=times,
            event_of_interest=event_idx + 1,
        )
        for time_idx, (time, quantile) in enumerate(zip(times, truncation_quantiles)):
            y_pred_at_t = y_pred[event_idx + 1][:, time_idx]
            ct_index, _, _, _, _ = concordance_index_ipcw(
                y_train,
                y_test,
                y_pred_at_t,
                tau=time,
            )
            results.append(
                dict(
                    estimator_name=estimator_name,
                    ibs=ibs,
                    event=event_idx + 1,
                    truncation_q=quantile,
                    ct_index=ct_index,
                )
            )

results = pd.DataFrame(results)
results_ibs = (
    results[["estimator_name", "event", "ibs"]]
    .drop_duplicates()
    .pivot(
        index="estimator_name",
        columns="event",
        values="ibs",
    )
)
_ = results.pop("ibs")
results_ct_index = results.sort_values(["truncation_q", "event"]).pivot(
    index="estimator_name",
    columns=["truncation_q", "event"],
    values="ct_index",
)

print("IBS")
display(results_ibs)
print("\n")
print("Ct-index")
display(results_ct_index)

print(results_ibs.to_latex())
print("\n")
print(results_ct_index.to_latex())

# %%
