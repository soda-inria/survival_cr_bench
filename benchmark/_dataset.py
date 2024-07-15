import os
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from pycox.datasets import support, metabric
from hazardous.data._seer import (
    load_seer,
    CATEGORICAL_COLUMN_NAMES,
    NUMERIC_COLUMN_NAMES,
)
from hazardous.data._competing_weibull import make_synthetic_competing_weibull

#PATH_SEER = os.getenv("PATH_SEER")
PATH_SEER = "../hazardous/data/seer_cancer_cardio_raw_data.txt"
PATH_CHURN = "../hazardous/data/churn.csv"


def get_split_seer(dataset_params):
    bunch = load_seer(
        PATH_SEER,
        survtrace_preprocessing=True,
        return_X_y=False,
    )
    X, y = bunch.X, bunch.y
    column_names = CATEGORICAL_COLUMN_NAMES + NUMERIC_COLUMN_NAMES
    X = X[column_names]

    return split(X, y, dataset_params)


def get_split_synthetic(dataset_params):
    bunch = make_synthetic_competing_weibull(**dataset_params)
    X, y = bunch.X, bunch.y

    return split(
        X,
        y,
        dataset_params,
        shape_censoring=bunch.shape_censoring,
        scale_censoring=bunch.scale_censoring,
    )


def get_split_metabric(dataset_params):
    df = metabric.read_df()
    categorical_features = ["x4", "x5", "x6", "x7"]
    numerical_features = ["x0", "x1", "x2", "x3", "x8"]

    return pycox_preprocessing(
        df, categorical_features, numerical_features, dataset_params
    )


def get_split_support(dataset_params):
    df = support.read_df()
    categorical_features = ["x1", "x2", "x3", "x4", "x5", "x6"]
    numerical_features = ["x0", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]

    return pycox_preprocessing(
        df, categorical_features, numerical_features, dataset_params
    )


def pycox_preprocessing(df, categorical_features, numerical_features, dataset_params):
    X = df.drop(columns=["duration", "event"])
    X[categorical_features] = X[categorical_features].astype("category")
    X[numerical_features] = X[numerical_features].astype("float64")

    y = df[["duration", "event"]]

    return split(X, y, dataset_params)


def split(X, y, dataset_params, **kwargs):
    
    if "random_state" in dataset_params:
        random_state = dataset_params["random_state"]
    else:
        # for the HP search to be faster
        random_state = 0
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y["event"],
        random_state=random_state,
    )

    # Used by models with scaling issues during training.
    if dataset_params["max_samples"] is not None:
        if X_train.shape[0] > dataset_params["max_samples"]:
            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                train_size=dataset_params["max_samples"],
                stratify=y_train["event"],
                random_state=0,
            )

    return Bunch(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        **kwargs,
    )


LOAD_DATASET_FUNCS = {
    "seer": get_split_seer,
    "weibull": get_split_synthetic,
    "support": get_split_support,
    "metabric": get_split_metabric,
}
