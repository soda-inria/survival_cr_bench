# %%
import json
from pathlib import Path
from itertools import product
from datetime import datetime
import pandas as pd

from joblib import dump
from scipy.stats import loguniform, randint
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.utils import Bunch
from pycox.datasets import support, metabric

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.data._seer import (
    load_seer,
    CATEGORICAL_COLUMN_NAMES,
    NUMERIC_COLUMN_NAMES,
)
from hazardous._gb_multi_incidence import GBMultiIncidence

# from hazardous.survtrace._encoder import SurvFeatureEncoder
# from hazardous._deep_hit import _DeepHit
# from hazardous._fine_and_gray import FineGrayEstimator
from hazardous._aalen_johansen import AalenJohansenEstimator
from hazardous.survtrace._model import SurvTRACE
from hazardous.survtrace._encoder import SurvFeatureEncoder
from hazardous.utils import (
    SurvStratifiedSingleSplit,
    SurvStratifiedShuffleSplit,
    CumulativeIncidencePipeline,
)

SEED = 0
# Enable oracle scoring for GridSearchCV
# GBMI.set_score_request(scale=True, shape=True)

gbmi_competing_loss = CumulativeIncidencePipeline(
    [
        ("encoder", SurvFeatureEncoder()),
        ("estimator", GBMultiIncidence(
            loss="competing_risks",
            show_progressbar=True,
            n_iter=10,
            n_iter_before_feedback=20,
        )),
    ]
)
gbmi_log_loss = CumulativeIncidencePipeline(
    [
        ("encoder", SurvFeatureEncoder()),
        (
            "estimator",
            GBMultiIncidence(
                loss="inll",
                show_progressbar=True,
                n_times=1,
                n_iter=10,
                n_iter_before_feedback=20,
            ),
        ),
    ]
)

# deephit = _DeepHit(
#   num_nodes_shared=[64, 64],
#    num_nodes_indiv=[32],
#    verbose=True,
#    num_durations=10,
#    batch_norm=True,
#    dropout=None,
# )

# fine_and_gray = CumulativeIncidencePipeline(
#     [("encoder", SurvFeatureEncoder()), ("estimator", FineGrayEstimator())]
# )
aalen_johansen = AalenJohansenEstimator(calculate_variance=False, seed=SEED)
survtrace = SurvTRACE(batch_size=128, optimizer__weight_decay=0, lr=1e-3, max_epochs=20)

gbmi_param_grid = {
    "estimator__learning_rate": loguniform(0.01, 0.1),
    "estimator__max_depth": randint(2, 10),
    #"estimator__n_iter": randint(5, 50),
    # "estimator__n_times": randint(1, 5),
    #"estimator__n_iter_before_feedback": randint(5, 50),
}

survtrace_grid = {
    "lr": loguniform(1e-5, 1e-3),
    "batch_size": [256, 512, 1024],
}

ESTIMATOR_GRID = {
    "gbmi_competing_loss": {
        "estimator": gbmi_competing_loss,
        "param_grid": gbmi_param_grid,
    },
    "gbmi_log_loss": {
        "estimator": gbmi_log_loss,
        "param_grid": gbmi_param_grid,
    },
    "survtrace": {"estimator": survtrace, "param_grid": survtrace_grid},
    # "fine_and_gray": {
    #     "estimator": fine_and_gray,
    #     "param_grid": {},
    # },
    "aalen_johansen": {
        "estimator": aalen_johansen,
        "param_grid": {},
    },
}


# Parameters of the make_synthetic_competing_weibull function.
# DATASET_GRID = {
#     "weibull": {
#         "n_events": [3],
#         "n_samples": [1_000, 5_000, 10_000, 20_000, 50_000],
#         "censoring_relative_scale": [0.8, 1.5, 2.5],
#         "complex_features": [True],
#         "independent_censoring": [True, False],
#     },
#     "seer": {
#         "n_samples": [50_000, 100_000, 300_000],
#     },
# }

DATASET_GRID = {
    "weibull": {
        "n_events": [3],
        "n_samples": [1_000, 5_000, 10_000, 20_000],
        "censoring_relative_scale": [1.5],
        "complex_features": [True],
        "independent_censoring": [False],
    },
    "seer": {
        "n_samples": [None],
        "seed": list(range(3)),
        "n_samples": [10_000],
    },  # , 100_000, 300_000],
    "metabric": {
        "n_samples": [None],
        "seed": list(range(5)),
    },
    "support": {
        "n_samples": [None],
        "seed": list(range(5)),
    },
}


PATH_DAILY_SESSION = Path(datetime.now().strftime("%Y-%m-%d"))

SEER_PATH = "../hazardous/data/seer_cancer_cardio_raw_data.txt"
CHURN_PATH = "../hazardous/data/churn.csv"
N_JOBS_CV = 1
SEARCH_HP = True
# N_ITER_CV = 10


def run_all_datasets(dataset_name, estimator_name):
    dataset_grid = DATASET_GRID[dataset_name]
    grid_dataset_params = list(product(*dataset_grid.values()))

    run_fn = {
        "seer": run_seer,
        "weibull": run_synthetic_dataset,
        "metabric": run_surv_dataset,
        "support": run_surv_dataset,
    }[dataset_name]

    # deactivate parallelization on dataset params to avoid
    # nested parallelism and threads oversubscription.
    # parallel = Parallel(n_jobs=None)
    # parallel(
    #     delayed(run_fn)(dataset_params, estimator_name)
    #     for dataset_params in grid_dataset_params
    # )
    for dataset_params in grid_dataset_params:
        run_fn(dataset_params, estimator_name, dataset_name=dataset_name)


def run_synthetic_dataset(dataset_params, estimator_name, dataset_name="weibull"):
    del dataset_name
    dataset_grid = DATASET_GRID["weibull"]
    dataset_params = dict(zip(dataset_grid.keys(), dataset_params))

    data_bunch = make_synthetic_competing_weibull(**dataset_params)
    run_estimator(
        estimator_name,
        data_bunch,
        dataset_name="weibull",
        dataset_params=dataset_params,
    )


def run_seer(dataset_params, estimator_name, dataset_name="seer"):
    del dataset_name
    dataset_grid = DATASET_GRID["seer"]
    dataset_params = dict(zip(dataset_grid.keys(), dataset_params))
    print(dataset_params)

    data_bunch = load_seer(
        SEER_PATH,
        survtrace_preprocessing=True,
        return_X_y=False,
    )
    X, y = data_bunch.X, data_bunch.y
    column_names = CATEGORICAL_COLUMN_NAMES + NUMERIC_COLUMN_NAMES
    data_bunch.X = data_bunch.X[column_names]

    seed = dataset_params.get("seed", None) or SEED

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y["event"],
        random_state=seed,
    )

    if dataset_params["n_samples"] is not None:
        n_samples = min(dataset_params["n_samples"], X_train.shape[0])
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=n_samples,
            stratify=y_train["event"],
            random_state=seed,
        )

    data_bunch.X, data_bunch.y = X_train, y_train

    run_estimator(
        estimator_name,
        data_bunch,
        dataset_name="seer",
        dataset_params=dataset_params,
    )


def run_surv_dataset(dataset_params, estimator_name, dataset_name="metabric"):
    dataset_grid = DATASET_GRID[dataset_name]
    dataset_params = dict(zip(dataset_grid.keys(), dataset_params))
    print(dataset_params)

    if dataset_name == "metabric":
        df = metabric.read_df()
        categorical_features = ["x4", "x5", "x6", "x7"]
        numerical_features = ["x0", "x1", "x2", "x3", "x8"]
    elif dataset_name == "support":
        df = support.read_df()
        categorical_features = ["x1", "x2", "x3", "x4", "x5", "x6"]
        numerical_features = ["x0", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    X = df.drop(columns=["duration", "event"])
    X[categorical_features] = X[categorical_features].astype("category")
    X[numerical_features] = X[numerical_features].astype("float64")

    y = df[["duration", "event"]]

    seed = dataset_params.get("seed", None) or SEED

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y["event"],
        random_state=seed,
    )

    if dataset_params["n_samples"] is not None:
        n_samples = min(dataset_params["n_samples"], X_train.shape[0])
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=n_samples,
            stratify=y_train["event"],
            random_state=seed,
        )

    data_bunch = Bunch(X=X_train, y=y_train)

    run_estimator(
        estimator_name,
        data_bunch,
        dataset_name=dataset_name,
        dataset_params=dataset_params,
    )


def run_estimator(estimator_name, data_bunch, dataset_name, dataset_params):
    """Find the best hyper-parameters for a given model and a given dataset."""

    X, y = data_bunch.X, data_bunch.y
    # scale_censoring = data_bunch.scale_censoring
    # shape_censoring = data_bunch.shape_censoring
    estimator = ESTIMATOR_GRID[estimator_name]["estimator"]
    param_grid = ESTIMATOR_GRID[estimator_name]["param_grid"]

    if estimator_name == "survtrace" or not SEARCH_HP:
        cv = SurvStratifiedShuffleSplit()
    else:
        cv = SurvStratifiedShuffleSplit(n_splits=3)

    if not SEARCH_HP:
        param_grid = {}

    hp_search = RandomizedSearchCV(
        estimator,
        param_grid,
        cv=cv,
        return_train_score=False,
        refit=False,
        n_jobs=1,
        n_iter=3,
    )
    hp_search.fit(
        X,
        y,
    )

    best_params = hp_search.best_params_

    # With refit=True, the best estimator is already fitted on X, y.
    #best_estimator = hp_search.best_estimator_

    cols = [
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    ]
    best_results = pd.DataFrame(hp_search.cv_results_)[cols]
    best_results = best_results.iloc[hp_search.best_index_].to_dict()
    best_results["estimator_name"] = estimator_name

    # hack for benchmarks
    #best_estimator.y_train = y

    str_params = [str(v) for v in dataset_params.values()]
    str_params = "_".join([estimator_name, *str_params])
    path_profile = PATH_DAILY_SESSION / dataset_name / str_params
    path_profile.mkdir(parents=True, exist_ok=True)

    json.dump(best_params, open(path_profile / "best_params.json", "w"))
    # dump(best_estimator, path_profile / "best_estimator.joblib")
    # json.dump(best_results, open(path_profile / "cv_results.json", "w"))
    json.dump(dataset_params, open(path_profile / "dataset_params.json", "w"))


# %%
# run_all_datasets("seer", "aalen_johansen")
# %%
# run_all_datasets("support", "gbmi_log_loss")
# %%
# run_all_datasets("support", "survtrace")
# %%
# run_all_datasets("metabric", "fine_and_gray")

# %%
run_all_datasets("seer", "gbmi_log_loss")

# %%
