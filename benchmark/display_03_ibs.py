# %%


from tqdm import tqdm
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
)
from display_utils import (
    aggregate_result,
    load_dataset,
    make_query,
    get_estimator,
)
from hazardous.utils import make_time_grid

sns.set_style(style="white")
sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")

USER = "jalberge"
DATASET_NAME = "weibull"
SEEDS = {
    "weibull": list(range(5)),
    "seer": [0],
}[DATASET_NAME]


def plot_ibs_vs_x(results, data_params, x_col, x_label, filename):
    if DATASET_NAME == "seer":
        data_params = {"n_events": 3}
    else:
        mask = make_query(data_params)
        results = results.query(mask)
        # The same size and censoring relative scale for all folds.
        data_params.update(
            {
                "n_samples": 10_000,
            }
        )

    x_values = results[x_col].unique()

    all_y_pred = []
    for x_val in tqdm(x_values):
        for seed in SEEDS:
            bunch = load_dataset(
                DATASET_NAME, {x_col: x_val, **data_params}, random_state=seed
            )
            X_test, y_test = bunch.X, bunch.y

            time_grid = make_time_grid(y_test["duration"], n_steps=10)

            results_x_col = results.loc[results[x_col] == x_val]
            for estimator_name, results_est in results_x_col.groupby("estimator_name"):
                print(estimator_name)
                estimator = get_estimator(results_est, estimator_name)
                y_pred = estimator.predict_cumulative_incidence(X_test, time_grid)
                y_train = estimator.y_train

                for event_idx in range(data_params["n_events"]):
                    ibs = integrated_brier_score_incidence(
                        y_train,
                        y_test,
                        y_pred[event_idx + 1],
                        time_grid,
                        event_of_interest=event_idx + 1,
                    )

                    if x_col == "censoring_relative_scale":
                        x_val_ = round((y_train["event"] == 0).mean(), 2) * 100
                    else:
                        x_val_ = x_val

                    all_y_pred.append(
                        {
                            x_col: x_val_,
                            "seed": seed,
                            "estimator_name": estimator_name,
                            "event_of_interest": event_idx + 1,
                            "test_score": ibs,
                        }
                    )

    all_y_pred = (
        pd.DataFrame(all_y_pred)
        .groupby([x_col, "seed", "estimator_name"])[
            "test_score"
        ]  # agg event_of_interest
        .mean()
        .reset_index()
        .groupby([x_col, "estimator_name"])["test_score"]  # agg seed
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns=dict(
                mean="mean_test_score",
                std="std_test_score",
            )
        )
    )

    print(all_y_pred)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    sns.lineplot(
        all_y_pred,
        x=x_col,
        y="mean_test_score",
        hue="estimator_name",
        marker="o",
        ax=ax,
    )
    for _, df_est in all_y_pred.groupby("estimator_name"):
        df_est = df_est.sort_values(x_col)
        low = df_est["mean_test_score"] - df_est["std_test_score"]
        high = df_est["mean_test_score"] + df_est["std_test_score"]
        ax.fill_between(df_est[x_col], y1=low, y2=high, alpha=0.3)

    ax.set(
        xlabel=x_label,
        ylabel="Average IBS",
    )
    ax.grid(axis="y")
    sns.despine()
    if x_col == "censoring_relative_scale":
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    file_path = f"/Users/{USER}/Desktop/03_{filename}.pdf"
    fig.savefig(file_path, format="pdf", dpi=300, bbox_inches="tight")


path_session_dataset = Path("2024-01-30") / DATASET_NAME
estimator_names = [
    # "gbmi_competing_loss",
    "gbmi_log_loss",
    "survtrace",
    # "fine_and_gray",
]
results = aggregate_result(path_session_dataset, estimator_names)

# %%
results

idx_drop = results.loc[
    (results["estimator_name"] == "fine_and_gray") & (results["n_samples"] > 10_000)
].index
# %%
results.drop(idx_drop, inplace=True)
# %%
results["n_samples"] = results["estimator"].apply(lambda x: len(x.y_train))
# %%

data_params = dict(
    n_events=3,
    independent_censoring=False,
    complex_features=True,
    censoring_relative_scale=0.8,
)
plot_ibs_vs_x(
    results,
    data_params,
    x_col="n_samples",
    x_label="# training samples",
    filename="ibs_samples",
)
# %%

# %%

# %%
data_params = dict(
    n_events=3,
    independent_censoring=False,
    complex_features=True,
    n_samples=10_000,
)
plot_ibs_vs_x(
    results,
    data_params,
    x_col="censoring_relative_scale",
    x_label="Censoring rate",
    filename="ibs_censoring",
)

# %%
print("hello")
# %%
plot_ibs_vs_x(
    results,
    data_params={},
    x_col="n_samples",
    x_label="Number of samples",
    filename="ibs_censoring",
)
# %%
results
# %%
