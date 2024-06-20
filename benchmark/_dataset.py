from sklearn.model_selection import train_test_split
from pycox.datasets import support, metabric
from hazardous.data._seer import (
    load_seer,
    CATEGORICAL_COLUMN_NAMES,
    NUMERIC_COLUMN_NAMES,
)
from hazardous.data._competing_weibull import make_synthetic_competing_weibull

from config import SEER_PATH

SEED = 0


def get_split_seer(dataset_params):
    data_bunch = load_seer(
        SEER_PATH,
        survtrace_preprocessing=True,
        return_X_y=False,
    )
    X, y = data_bunch.X, data_bunch.y
    column_names = CATEGORICAL_COLUMN_NAMES + NUMERIC_COLUMN_NAMES
    X = X[column_names]

    seed = dataset_params.get("seed", None) or SEED

    return split(X, y, seed)


def get_split_synthetic(dataset_params):
    data_bunch = make_synthetic_competing_weibull(**dataset_params)
    X, y = data_bunch.X, data_bunch.y

    seed = dataset_params.get("seed", None) or SEED

    return split(X, y, seed)


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


def split(X, y, dataset_params):

    seed = dataset_params.get("seed", None) or SEED

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y["event"],
        random_state=seed,
    )

    # Used by models with scaling issues during training.
    if dataset_params["n_samples"] is not None:
        n_samples = min(dataset_params["n_samples"], X_train.shape[0])
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=n_samples,
            stratify=y_train["event"],
            random_state=seed,
        )

    return X_train, X_test, y_train, y_test


LOAD_DATASET_FUNCS = {
    "seer": get_split_seer,
    "synthetic": get_split_synthetic,
    "support": get_split_support,
    "metabric": get_split_metabric,
}
