# %%
import pandas as pd
import json
from pathlib import Path
import numpy as np
from SurvivalEVAL import SurvivalEvaluator
from matplotlib import pyplot as plt
import seaborn as sns

from _dataset import LOAD_DATASET_FUNCS


def plot_metrics(
    model_name,
    dataset_name,
    seed=0,
    plot_km=True,
    plot_d=True,
    plot_one=True,
):

    path_file = Path(f"scores/raw/{model_name}/{dataset_name}.json")
    data = json.load(open(path_file))[seed]
    y_pred = np.asarray(data["y_pred"])

    dataset_params = {"random_state": data["random_state"]}
    bunch = LOAD_DATASET_FUNCS[dataset_name](dataset_params)

    eval = SurvivalEvaluator(
        predicted_survival_curves=y_pred[0, :, :],
        time_coordinates=data["time_grid"],
        test_event_times=bunch.y_test["duration"].to_numpy(),
        test_event_indicators=bunch.y_test["event"].to_numpy(),
        train_event_times=bunch.y_train["duration"].to_numpy(),
        train_event_indicators=bunch.y_train["event"].to_numpy(),
    )

    d_cal, d_bins = eval.d_calibration()

    median_time = np.quantile(data["time_grid"], 0.5)
    one_cal_median, one_observed, one_expected = eval.one_calibration(median_time)

    title = f"{model_name} -- {dataset_name}"
    scores = dict(
        model_name=model_name,
        dataset_name=dataset_name,
        mae=eval.mae(method="Pseudo_obs"),
        mse=eval.mse(method="Pseudo_obs"),
        auc=eval.auc(),
        one_cal_median=one_cal_median,
        d_cal=d_cal,
        km_calibration=eval.km_calibration(draw_figure=False, title=title),
        x_calibration=eval.x_calibration(),
    )

    if plot_km:
        fig, ax = eval.plot_survival_curves(np.arange(50, 100))
        ax.set_title(f"{model_name} -- {dataset_name}")
        ax.legend().remove()
        plt.show()

    if plot_d:

        x_bins = ["[0, 0.1)", "[0.1, 0.2)", "[0.2, 0.3)", "[0.3, 0.4)", "[0.4, 0.5)", "[0.5, 0.6)", "[0.6, 0.7)", "[0.7, 0.8)",
                "[0.8, 0.9)", "[0.9, 1]"]
        plt.clf()
        fig, ax = plt.subplots()
        ax.bar(x_bins, d_bins)
        plt.setp(ax.get_xticklabels(), rotation=30)
        plt.ylabel("Counts in bins")
        ax.set_title(title)
        plt.show()

    if plot_one:
        x_bins = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        fig, ax = plt.subplots()
        bar1 = ax.bar([x - 0.2 for x in x_bins], one_observed, width=0.4, color='r', align='center')
        bar2 = ax.bar([x + 0.2 for x in x_bins], one_expected, width=0.4, color='g', align='center')

        ax.legend((bar1[0], bar2[0]), ('Observed', 'Expected'))
        ax.set_title(title)
        plt.show()

    return scores


dataset_name = "support"
model_names = [
    "gbmi",
    "survtrace",
    "random_survival_forest",
    "sksurv_boosting",
    "pchazard",
    "deephit",
]
seed = 0
results = []
for model_name in model_names:
    results.append(plot_metrics(model_name, dataset_name, seed=seed))

pd.DataFrame(results)

