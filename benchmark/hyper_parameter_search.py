# %%
import json
from pathlib import Path
from scipy.stats import loguniform, randint
from sklearn.model_selection import (
    RandomizedSearchCV,
    ParameterGrid,
)
from hazardous.utils import (
    SurvStratifiedSingleSplit,
    SurvStratifiedShuffleSplit,
)

from _model import INIT_MODEL_FUNCS
from _dataset import LOAD_DATASET_FUNCS


gbmi_param_grid = {
    "estimator__learning_rate": loguniform(0.01, 0.1),
    "estimator__max_depth": randint(2, 10),
    "estimator__n_iter": randint(5, 50),
    "estimator__n_times": randint(1, 5),
    "estimator__n_iter_before_feedback": randint(5, 50),
}

survtrace_grid = {
    "lr": loguniform(1e-5, 1e-3),
    "batch_size": [256, 512, 1024],
}

HYPER_PARAMS_GRID = {
    "gbmi": gbmi_param_grid,
    "survtrace": survtrace_grid,
    "fine_and_gray": {},
    "aalen_johansen": {},
}

DATASET_GRID = {
    "weibull": {
        "n_events": [3],
        "n_samples": [10_000],
        "censoring_relative_scale": [1.5],
        "complex_features": [True],
        "independent_censoring": [False],
        "seed": list(range(5)),
    },
    "seer": {
        "n_samples": [ None],
        "seed": list(range(5))[1:],
    },
    "metabric": {
        "n_samples": [None],
        "seed": list(range(5)),
    },
    "support": {
        "n_samples": [None],
        "seed": list(range(5)),
    },
}

PATH_HP_SEARCH = Path("./best_hyper_parameters")

SEARCH_HP = True
N_ITER_OUTER_LOOP_CV = 10
N_ITER_INNER_LOOP_CV = 5


def search_all_dataset_params(dataset_name, model_name):

    for dataset_params in ParameterGrid(DATASET_GRID[dataset_name]):
        search_hp(dataset_name, dataset_params, model_name)


def search_hp(dataset_name, dataset_params, model_name):
    """Find the best hyper-parameters for a given model and a given dataset."""

    load_data_func = LOAD_DATASET_FUNCS[dataset_name]
    bunch = load_data_func(dataset_params)
    X_train, y_train = bunch.X_train, bunch.y_train

    model_init_func = INIT_MODEL_FUNCS[model_name]
    model = model_init_func()
    param_grid = HYPER_PARAMS_GRID[model_name]

    if model_name == "survtrace" or not SEARCH_HP:
        # Used when nested CV is too expensive.
        # Equivalent of setting N_ITER_INNER_LOOP_CV = 1.
        cv = SurvStratifiedSingleSplit()
    else:
        cv = SurvStratifiedShuffleSplit(n_splits=N_ITER_INNER_LOOP_CV)

    if not SEARCH_HP:
        param_grid = {}

    hp_search = RandomizedSearchCV(
        model,
        param_grid,
        cv=cv,
        return_train_score=False,
        refit=False,
        n_jobs=1,
        n_iter=N_ITER_OUTER_LOOP_CV,
    ).fit(X_train, y_train)

    best_params = hp_search.best_params_
    best_params["model_name"] = model_name

    str_params = [str(v) for v in dataset_params.values()]
    str_params = "_".join([model_name, *str_params])
    path_profile = PATH_HP_SEARCH / dataset_name / str_params
    path_profile.mkdir(parents=True, exist_ok=True)

    json.dump(best_params, open(path_profile / "best_params.json", "w"))
    json.dump(dataset_params, open(path_profile / "dataset_params.json", "w"))
