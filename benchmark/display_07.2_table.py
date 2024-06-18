# %%
from pathlib import Path
from IPython.display import display

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
)
from sksurv.metrics import concordance_index_ipcw

# from hazardous.metrics._concordance import concordance_index_ipcw
from hazardous.utils import get_n_events
from display_utils import (
    aggregate_result,
    get_estimator,
    load_dataset,
)
from hazardous.utils import make_time_grid
from hazardous.metrics._yana import CensoredNegativeLogLikelihoodSimple

sns.set_style(style="white")
sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")

USER = "jalberge"
DATASET_NAME = "metabric"
SEER_PATH = "../hazardous/data/seer_cancer_cardio_raw_data.txt"
SEED = 0

path_session_dataset = Path("final_metabric") / DATASET_NAME
estimator_names = ["gbmi_log_loss", "survtrace", "rsf"]

df = aggregate_result(path_session_dataset, estimator_names)


def get_number_train_samples(x):
    return len(x.y_train)


df["n_samples"] = df["estimator"].apply(get_number_train_samples)
# df = df[df["n_samples"] == df["n_samples"].max()]


# %%
def get_test_data(DATASET_NAME, seed=0):
    bunch = load_dataset(DATASET_NAME, data_params={}, random_state=seed)
    X_test, y_test = bunch.X, bunch.y
    return X_test, y_test


def predict_on_quantiles_for_seed(
    df, X_test, time_grid, truncation_quantiles=[0.25, 0.5, 0.75], random_state=0
):
    df_ = df[df["seed"] == random_state]

    all_y_pred = {}
    times = np.quantile(time_grid, truncation_quantiles)
    for estimator_name, df_est in df_.groupby("estimator_name"):
        print(estimator_name)
        estimator = get_estimator(df_est, estimator_name)
        y_pred = estimator.predict_cumulative_incidence(X_test, time_grid)

        y_train = estimator.y_train
        all_y_pred[estimator_name] = y_pred

    return times, y_train, all_y_pred


def compute_metrics(
    y_train,
    y_test,
    all_y_pred,
    time_grid,
    times,
    truncation_quantiles=[0.25, 0.5, 0.75],
    n_events=3,
):
    results = []
    for estimator_name, y_pred in all_y_pred.items():
        ibs = integrated_brier_score_incidence(
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred[1],
            times=time_grid,
            event_of_interest=1,
        )

        def get_target(df):
            return (df["duration"].values, df["event"].values)

        durations_train, events_train = get_target(y_train)
        et_train = np.array(
            [(events_train[i], durations_train[i]) for i in range(len(events_train))],
            dtype=[("e", bool), ("t", float)],
        )

        durations_test, events_test = get_target(y_test)
        et_test = np.array(
            [(events_test[i], durations_test[i]) for i in range(len(events_test))],
            dtype=[("e", bool), ("t", float)],
        )
        for time_idx, (time, quantile) in enumerate(zip(times, truncation_quantiles)):
            y_pred_at_t = y_pred[1, :, time_idx]
            ct_index, _, _, _, _ = concordance_index_ipcw(
                et_train,
                et_test,
                y_pred_at_t,
                tau=time,
            )
            results.append(
                dict(
                    estimator_name=estimator_name,
                    ibs=ibs,
                    event=1,
                    truncation_q=quantile,
                    ct_index=ct_index,
                )
            )

    results = pd.DataFrame(results)
    return results


# %%
truncation_quantiles = [0.25, 0.5, 0.75]
results_all_seeds = []
for seed in range(5):
    X_test, y_test = get_test_data(DATASET_NAME, 0)
    time_grid = make_time_grid(y_test["duration"], n_steps=100)
    n_events = get_n_events(y_test["event"])
    times, y_train, all_y_pred = predict_on_quantiles_for_seed(
        df, X_test, time_grid, random_state=seed
    )
    results = compute_metrics(
        y_train, y_test, all_y_pred, time_grid, times, truncation_quantiles, n_events
    )

    time_grid_yana = make_time_grid(y_test["duration"], n_steps=32)
    yana_loss = CensoredNegativeLogLikelihoodSimple()

    _, _, all_y_pred_yana = predict_on_quantiles_for_seed(
        df, X_test, time_grid_yana, random_state=seed
    )
    for estimator_name, y_pred in all_y_pred_yana.items():
        yana_l = yana_loss.loss(
            y_pred, y_test["duration"], y_test["event"], time_grid_yana
        )
        results.loc[results["estimator_name"] == estimator_name, "yana_l"] = yana_l

    results["seed"] = seed
    results_all_seeds.append(results)
# %%
results_all_seeds = pd.concat(results_all_seeds)
# %%
results_all_seeds

# %%
results_ibs = (
    (
        results_all_seeds[["estimator_name", "event", "ibs", "seed"]]
        .drop_duplicates()
        .pivot(
            index=["estimator_name", "seed"],
            columns="event",
            values="ibs",
        )
    )
    .reset_index()
    .groupby("estimator_name")
)

# %%
print("MEAN IBS")
display(results_ibs.mean().iloc[:, 1:])
print("\n")
print("STD IBS")
display(results_ibs.std().iloc[:, 1:])

# %%
results_yana = (
    (
        results_all_seeds[["estimator_name", "event", "yana_l", "seed"]]
        .drop_duplicates()
        .pivot(
            index=["estimator_name", "seed"],
            columns="event",
            values="yana_l",
        )
    )
    .reset_index()
    .groupby("estimator_name")
)

# %%
print("MEAN YANA")
display(results_yana.mean().iloc[:, 1:])
print("\n")
print("STD YANA")
display(results_yana.std().iloc[:, 1:])

# %%
results_ct_index = (
    results_all_seeds.sort_values(["truncation_q", "event"])
    .pivot(
        index=["estimator_name", "seed"],
        columns=["truncation_q", "event"],
        values="ct_index",
    )
    .reset_index()
    .groupby("estimator_name")
)
# %%
print("MEAN C INDEX")
display(results_ct_index.mean().iloc[:, 1:])
print("\n")
print("STD C INDEX")
display(results_ct_index.std().iloc[:, 1:])

# %%

print(results_ibs.mean().iloc[:, 1:].to_latex())
print("\n")
print(results_ct_index.mean().iloc[:, 1:].to_latex())
