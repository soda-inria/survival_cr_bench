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


# gbmi_param_grid = {
#     "estimator__learning_rate": loguniform(0.01, 0.1),
#     "estimator__max_depth": randint(2, 10),
#     "estimator__n_iter": randint(5, 50),
#     "estimator__n_times": randint(1, 5),
#     "estimator__n_iter_before_feedback": randint(5, 50),
# }
gbmi_param_grid = {
    "estimator__learning_rate": [0.01],
    "estimator__max_depth": [9],
    "estimator__n_iter": [100],
    "estimator__n_times": [1],
    "estimator__n_iter_before_feedback": [20],
}

# survtrace_grid = {
#     "lr": loguniform(1e-5, 1e-3),
#     "batch_size": [256, 512, 1024],
# }
survtrace_grid = {
    "lr": [0.001],
    "batch_size": [256],
}

fine_and_gray_grid = {
    "estimator__max_samples": [10_000],
}

sksurv_boosting = {
    "estimator__loss": ["coxph"],
    "estimator__learning_rate": loguniform(0.01, 0.1),
    "estimator__n_estimators": randint(50, 100),
    "estimator__max_depth": randint(2, 10),
    "estimator__subsample": [1.0],
    "estimator__criterion": ["friedman_mse"],
    "estimator__verbose": [1],
}

HYPER_PARAMS_GRID = {
    "gbmi": gbmi_param_grid,
    "survtrace": survtrace_grid,
    "fine_and_gray": fine_and_gray_grid,
    "aalen_johansen": {},
    "deephit": {},
    "random_survival_forest": {},
    "sksurv_boosting": sksurv_boosting,
}

DATASET_GRID = {
    "weibull": {
        "n_events": [3],
        "n_samples": [20_000],
        "censoring_relative_scale": [1.5],
        "complex_features": [False],
        "independent_censoring": [False],
        "random_state": range(5),
    },
    "seer": {
        "random_state": range(5),
    },
    "metabric": {
        "random_state": range(5),
    },
    "support": {
        "random_state": range(5),
    },
}

PATH_HP_SEARCH = Path("./best_hyper_parameters")

SEARCH_HP = True
N_ITER_OUTER_LOOP_CV = 10
N_ITER_INNER_LOOP_CV = 1
N_JOBS_CV = None


def search_all_dataset_params(dataset_name, model_name):
    """Find the best hyper-parameters for a given model and all datasets."""
    print(f"{' HP search of ' + model_name + ' on ' + dataset_name + ' ':=^50}")
    for dataset_params in ParameterGrid(DATASET_GRID[dataset_name]):
        search_hp(dataset_name, dataset_params, model_name)


def search_hp(dataset_name, dataset_params, model_name):
    """Find the best hyper-parameters for a given model and a given dataset."""

    load_data_func = LOAD_DATASET_FUNCS[dataset_name]
    bunch = load_data_func(dataset_params)
    X_train, y_train = bunch.X_train, bunch.y_train

    model_init_func = INIT_MODEL_FUNCS[model_name]
    model = model_init_func(
        random_state=dataset_params["random_state"]
    )
    param_grid = HYPER_PARAMS_GRID[model_name]

    print(f"{dataset_params=}")
    print(f"{param_grid=}")

    if N_ITER_INNER_LOOP_CV == 1:
        # Used when nested CV is too expensive.
        cv = SurvStratifiedSingleSplit()
    else:
        cv = SurvStratifiedShuffleSplit(n_splits=N_ITER_INNER_LOOP_CV)

    best_model_params = {
        "model_name": model_name,
        "random_state": dataset_params["random_state"],
    }

    if not SEARCH_HP:
        print("No search for HP")
        sk_param_grid = ParameterGrid(param_grid)
        if len(sk_param_grid) == 1:
            best_model_params.update(sk_param_grid[0])
        print(f"{best_model_params=}")
    
    else:
        hp_search = RandomizedSearchCV(
            model,
            param_grid,
            cv=cv,
            return_train_score=False,
            refit=False,
            n_jobs=N_JOBS_CV,
            n_iter=N_ITER_OUTER_LOOP_CV,
            error_score='raise',
        ).fit(X_train, y_train)

        best_model_params.update(hp_search.best_params_)

    str_params = [f"{k}{v}" for k, v in dataset_params.items()]
    str_params = "_".join(str_params)
    path_profile = PATH_HP_SEARCH / model_name / dataset_name / str_params
    path_profile.mkdir(parents=True, exist_ok=True)

    json.dump(best_model_params, open(path_profile / "best_params.json", "w"))
    json.dump(dataset_params, open(path_profile / "dataset_params.json", "w"))


# %%
if __name__ == "__main__":
    search_all_dataset_params("seer", "gbmi")


# %%
