from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from pycox.datasets import support, metabric, kkbox
from hazardous.data._seer import (
    load_seer,
    CATEGORICAL_COLUMN_NAMES,
    NUMERIC_COLUMN_NAMES,
)
from hazardous.data._competing_weibull import make_synthetic_competing_weibull

PATH_SEER = "../hazardous/data/seer_cancer_cardio_raw_data.txt"
PATH_CHURN = "../hazardous/data/churn.csv"


def get_split_seer(dataset_params, dropna=False):
    bunch = load_seer(
        PATH_SEER,
        survtrace_preprocessing=True,
        return_X_y=False,
    )
    X, y = bunch.X, bunch.y
    if dropna:
        X = X.dropna()
        y = y.iloc[X.index]
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


def get_split_kkbox(dataset_params):
    df = kkbox.read_df()
    target_columns = ["censor_duration", "msno"]
    df = df.drop(target_columns, axis=1)
    categorical_columns = ["city", "gender", "registered_via", "payment_method_id"]
    numerical_columns = [
        "n_prev_churns",
        "log_days_between_subs",
        "log_days_since_reg_init",
        "log_payment_plan_days",
        "log_plan_list_price",
        "log_actual_amount_paid",
        "is_auto_renew",
        "is_cancel",
        "age_at_start",
        "strange_age",
        "nan_days_since_reg_init",
        "no_prev_churns",
    ]

    return pycox_preprocessing(
        df, categorical_columns, numerical_columns, dataset_params
    )


def pycox_preprocessing(df, categorical_features, numerical_features, dataset_params):
    X = df.drop(columns=["duration", "event"])
    X[categorical_features] = X[categorical_features].astype("category")
    X[numerical_features] = X[numerical_features].astype("float64")

    y = df[["event", "duration"]]

    return split(X, y, dataset_params)


def split(X, y, dataset_params, **kwargs):

    random_state = dataset_params.get("random_state", 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y["event"],
        random_state=random_state,
    )

    n_samples = dataset_params.get("n_samples", None)
    if not n_samples is None and n_samples < X_train.shape[0]:
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=n_samples,
            stratify=y_train["event"],
            random_state=random_state,
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
    "kkbox": get_split_kkbox,
}
