"""
This file performs the following operations:

1. Fetch the best hyper parameters for the selected models
2. Fit models on the same train set they have been tuned on
3. Evaluate model performance on the matching test set
4. Write performance in a single score file

Here are the schemas of the data we collect.

Raw_score
=========
{
    task_type (survival or competing_risk)
    model_name
    dataset_name
    seed
    n_rows
    n_cols
    censoring_rate
    fit_time
    predict_time
    ibs
    censlog
    accuracy_in_time: {
        time_quantiles: []
        accuracy: []
    },
    event_specific_brier_scores: [
        {
             event
             time: []
             brier_score: []
        }
    ]
    event_specific_ibs: [
        {
            event:
            ibs:
        }
    ]
    c_index: [
        {
            time_quantile
            event: []
            c_index: []
        }
    ]
}


After aggregating on the seed and splitting between competing and survival:

Competing_risk
==============
{
    model_name
    dataset_name
    n_rows
    n_cols
    censoring_rate
    mean_fit_time
    std_fit_time
    mean_predict_time
    std_predict_time
    mean_ibs
    std_ibs
    accuracy_in_time: {
        time_quantiles: []
        accuracy: []
    }
    event_specific_brier_scores: [
        {
            event
            time: []
            mean_brier_score: []
            std_brier_score: []
        }
    ]
    event_specific_ibs: [
        {
            event
            mean_ibs
            std_ibs
        }
    ]
    c_index: [
        {
            time_quantile
            event: []
            mean_c_index: []
            std_c_index: []
        }
    ]
}

Survival
========
{
    model_name
    ...
    mean_ibs
    std_ibs
    mean_cenlog
    std_cenlog
    c_index: {
        time_quantile
        mean_c_index: []
        std_c_index: []
    },
}

"""
import time
import json
from pathlib import Path
import pandas as pd

from ._dataset import LOAD_DATASET_FUNCS
from ._model import INIT_MODEL_FUNCS

PARAMS_PATH = Path("./best_hyper_parameters")


def main():

    all_scores = []

    all_dataset_params, all_model_params = get_params()

    for (dataset_name, dataset_params), (model_name, model_params) in zip(
        all_dataset_params, all_model_params
    ):
        X_train, y_train, X_test, y_test = LOAD_DATASET_FUNCS[dataset_name](
            dataset_params
        )
        model = INIT_MODEL_FUNCS[model_name](model_params)

        tic = time()
        model = model.fit(X_train, y_train)
        toc = time()
        fit_time = toc - tic

        scores = evaluate(
            model, X_test, y_test, dataset_name, dataset_params, model_name
        )
        all_scores.append(scores)
        pd.DataFrame(all_scores).to_csv("./scores/raw_scores.csv", index=False)
    
    agg_scores = aggregate_scores(scores)
    pd.DataFrame(agg_scores).to_csv("agg_scores.csv", index=False)


def get_params():

    all_model_params, all_dataset_params = [], []

    for ds_path in PARAMS_PATH.glob("*/"):
        dataset_name = ds_path.name
        for model_path in ds_path.glob("*"):
            best_model_params = json.load(open(model_path / "best_params.json"))
            dataset_params = json.load(open(model_path / "dataset_params.json"))
            model_name = best_model_params.pop("model_name")
            all_dataset_params.append([dataset_name, dataset_params])
            all_model_params.append([model_name, best_model_params])

    return all_dataset_params, all_model_params


def evaluate(model, X_test, y_test, dataset_params):
    """Evaluate a model against its test set.
    """
    scores = dict()
    # TODO
    return scores


def aggregate_scores(all_scores):
    """
    """
    agg_scores = dict()
    # TODO
    return agg_scores


if __name__ == "__main__":
    main()
