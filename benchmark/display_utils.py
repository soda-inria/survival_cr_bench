# %%
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from pycox.datasets import support, metabric

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.data._seer import load_seer

from main import SEER_PATH, SEED

sns.set_style(
    style="white",
)
sns.set_context("paper")
sns.set_palette("colorblind")


def aggregate_result(path_session_dataset, estimator_names):
    data = []

    if not Path(path_session_dataset).exists():
        raise ValueError(f"{path_session_dataset} doesn't exist.")

    for path_profile in Path(path_session_dataset).glob("*"):
        if str(path_profile).endswith(".DS_Store"):
            continue
        results = json.load(open(path_profile / "cv_results.json"))
        estimator_name = results["estimator_name"]
        if estimator_name in estimator_names:
            dataset_params = json.load(open(path_profile / "dataset_params.json"))
            estimator = load(path_profile / "best_estimator.joblib")
            estimator = {"estimator": estimator}
            data.append({**dataset_params, **results, **estimator})

    return pd.DataFrame(data)


def load_dataset(dataset_name, data_params, random_state=0):
    if dataset_name == "weibull":
        return make_synthetic_competing_weibull(
            return_X_y=False,
            random_state=random_state + 100,
            **data_params,
        )

    elif dataset_name == "seer":
        bunch = load_seer(
            input_path=SEER_PATH,
            survtrace_preprocessing=True,
            return_X_y=False,
        )
        _, X_test, _, y_test = train_test_split(
            bunch.X,
            bunch.y,
            test_size=0.3,
            stratify=bunch.y["event"],
            random_state=SEED,
        )
        bunch.X, bunch.y = X_test, y_test
        return bunch
    elif dataset_name in ["metabric", "support"]:
        if dataset_name == "metabric":
            df = metabric.read_df()
            categorical_features = ["x4", "x5", "x6", "x7"]
            numerical_features = ["x0", "x1", "x2", "x3", "x8"]
        else:
            df = support.read_df()
            categorical_features = ["x1", "x2", "x3", "x4", "x5", "x6"]
            numerical_features = ["x0", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]

        X = df.drop(columns=["duration", "event"])
        X[categorical_features] = X[categorical_features].astype("category")
        X[numerical_features] = X[numerical_features].astype("float32")

        y = df[["duration", "event"]]
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=y["event"],
            random_state=random_state,
        )
        return Bunch(X=X_test, y=y_test)
    else:
        raise ValueError(f"Got {dataset_name} instead of ('seer', 'weibull').")


def make_query(data_params):
    query = []
    for k, v in data_params.items():
        if isinstance(v, str):
            v = f"'{v}'"
        query.append(f"({k} == {v})")
    return " & ".join(query)


def get_estimator(df, estimator_name):
    df_est = df.query("estimator_name == @estimator_name")
    if df_est.shape[0] != 1:
        raise ValueError(f"selection should be a single row, got {df_est}.")
    row = df_est.iloc[0]

    return row["estimator"]


def get_kind(data_params):
    if "independent_censoring" in data_params:
        return "independent" if data_params["independent_censoring"] else "dependent"
    return ""


# %%
