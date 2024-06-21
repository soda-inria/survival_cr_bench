"""
This file performs the following operations:

1. Fetch data_params and model_params from the best_hyper_parameters/ folder
2. Load the train and test set using the dataset params used for hp searching
3. Fit the model using the train set
3. Evaluate the model performance on the test set
4. Write performance in scores/agg_scores.json

Here are the schemas of the data we collect:

raw_scores
==========
dataset_name__model_name: {
    is_competing_risk: bool
    n_events: int
    model_name: str
    dataset_name: str
    seed: int
    n_rows: int
    n_cols: int
    censoring_rate: float
    fit_time: float
    predict_time: float
    censlog: float
    accuracy_in_time: {
        time_quantiles: [float]
        accuracy: [float]
    },
    event_specific_brier_scores: [
        {
            event: int
            time: [float]
            brier_score: [float]
        }
    ]
    event_specific_ibs: [
        {
            event: int
            ibs: float
        }
    ]
    event_specific_c_index: [
        {
            event: int
            time_quantile: [float]
            c_index: [float]
        }
    ]
}


Next, we aggregate across the seeds to compute the different mean and std.
Some values can be None when is_competing_risk is either True or False.

agg_scores
==========
dataset_name: {
    model_name: {
        is_competing_risk: bool
        model_name: str
        dataset_name: str
        n_rows: int
        n_cols: int
        censoring_rate: float
        mean_fit_time: float
        std_fit_time: float
        mean_predict_time: float
        std_predict_time: float
        mean_ibs: float
        std_ibs: float
        mean_cenlog: float (None if is_competing_risk = True)
        std_cenlog: float (None if is_competing_risk = True)
        accuracy_in_time: {
            time_quantiles: [float]
            mean_accuracy: [float]
            std_accuracy: [float]
        }
        event_specific_brier_scores: [
            {
                event: int
                time: [float]
                mean_brier_score: [float]
                std_brier_score: [float]
            }
        ]
        event_specific_ibs: [
            {
                event: int
                mean_ibs: float
                std_ibs: float
            }
        ]
        c_index: [
            {
                time_quantile: float
                event: [int]
                mean_c_index: [int]
                std_c_index: [int]
            }
        ]
    }
}
"""
# %%
from time import time
import json
from collections import defaultdict
import numpy as np
from hazardous.utils import make_time_grid
from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
    integrated_brier_score_incidence_oracle,
    brier_score_incidence,
    brier_score_incidence_oracle,
)
from hazardous.metrics._yana import CensoredNegativeLogLikelihoodSimple
from hazardous.metrics._concordance import concordance_index_ipcw

from _dataset import LOAD_DATASET_FUNCS
from _model import INIT_MODEL_FUNCS
from hyper_parameter_search import PATH_HP_SEARCH

# Setting an integer value will perform subsampling on the test set
# to debug faster. Otherwise, setting it to None will disable this option.
DEBUG_N_SUBSAMPLE = 1000


def evaluate_all_models():

    all_scores = defaultdict(list)

    all_dataset_params, all_model_params = get_params()

    for (dataset_name, dataset_params), (model_name, model_params) in zip(
        all_dataset_params, all_model_params
    ):
        bunch = LOAD_DATASET_FUNCS[dataset_name](dataset_params)
        X_train, y_train = bunch.X_train, bunch.y_train

        model = INIT_MODEL_FUNCS[model_name](**model_params)

        print(f"start fitting {model_name} on {dataset_name}")
        tic = time()
        model = model.fit(X_train, y_train)
        toc = time()
        fit_time = round(toc - tic, 2)

        scores = evaluate(
            model, bunch, dataset_name, dataset_params, model_name
        )
        scores["fit_time"] = fit_time
        
        all_scores[f"{dataset_name}__{model_name}"].append(scores)
        json.dump(all_scores, open("./scores/raw_scores.json", "w"))
    
    agg_scores = aggregate_scores(all_scores)
    json.dump(agg_scores, open("./scores/agg_scores.json", "w"))


def get_params():
    """Fetch and accumulate the data and models params.
    """
    all_model_params, all_dataset_params = [], []

    for ds_path in PATH_HP_SEARCH.glob("*"):
        dataset_name = ds_path.name
        for model_path in ds_path.glob("*"):
            best_model_params = json.load(open(model_path / "best_params.json"))
            dataset_params = json.load(open(model_path / "dataset_params.json"))
            model_name = best_model_params.pop("model_name")
            all_dataset_params.append([dataset_name, dataset_params])
            all_model_params.append([model_name, best_model_params])

    return all_dataset_params, all_model_params


def evaluate(model, bunch, dataset_name, dataset_params, model_name):
    """Evaluate a model against its test set.
    """
    n_events = np.unique(bunch.y_train["event"]).shape[0] - 1
    is_competing_risk = n_events > 1

    scores = {
        "is_competing_risk": is_competing_risk,
        "n_events": n_events,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seed": dataset_params["seed"],
        "n_rows": bunch.X_train.shape[0],
        "n_cols": bunch.X_train.shape[1],
        "censoring_rate": (bunch.y_train["event"] == 0).mean(),
    }

    X_test, y_test, y_train = bunch.X_test, bunch.y_test, bunch.y_train
    if DEBUG_N_SUBSAMPLE is not None:
        X_test, y_test = X_test.iloc[:DEBUG_N_SUBSAMPLE], y_test.iloc[:DEBUG_N_SUBSAMPLE]

    time_grid = make_time_grid(y_test["duration"])

    print("start evaluating")
    tic = time()
    y_pred = model.predict_cumulative_incidence(X_test, time_grid)
    toc = time()
    scores["predict_time"] = round(toc - tic, 2)

    event_specific_ibs, event_specific_brier_scores = [], []
    event_specific_c_index = []
    print("computing brier scores, ibs and c-index")
    for event_id in range(1, n_events+1):

        # Brier score and IBS
        if dataset_name == "synthetic":
            # Use oracle metrics with the synthetic dataset.
            ibs = integrated_brier_score_incidence_oracle(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                shape_censoring=bunch.shape_censoring,
                scale_censoring=bunch.scale_censoring,
                event_of_interest=event_id,
            )
            brier_scores = brier_score_incidence_oracle(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                shape_censoring=bunch.shape_censoring,
                scale_censoring=bunch.scale_censoring,
                event_of_interest=event_id,  
            )
        else:
            ibs = integrated_brier_score_incidence(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                event_of_interest=event_id,
            )
            brier_scores = brier_score_incidence(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                event_of_interest=event_id,
            )   
            
        event_specific_ibs.append({
            "event": event_id,
            "ibs": round(ibs, 4),
        })
        event_specific_brier_scores.append({
            "event": event_id,
            "time": list(time_grid.round(2)),
            "brier_score": list(brier_scores.round(4)),
        })

        # C-index
        truncation_quantiles = [0.25, 0.5, 0.75]
        times = np.quantile(time_grid, truncation_quantiles)
        c_indices = []
        for time_idx, tau in enumerate(times):
            y_pred_at_t = y_pred[event_id][:, time_idx]
            ct_index, _, _, _, _ = concordance_index_ipcw(
                y_train,
                y_test,
                y_pred_at_t,
                tau=tau,
            )
            c_indices.append(round(ct_index, 4))

        event_specific_c_index.append({
            "event": event_id,
            "time_quantile": truncation_quantiles,
            "c_index": c_indices,
        })

    scores.update({
        "event_specific_ibs": event_specific_ibs,
        "event_specific_brier_scores": event_specific_brier_scores,
        "event_specific_c_index": event_specific_c_index,
    })

    if not is_competing_risk:
        # Yana loss
        print("computing censlog")
        censlog = CensoredNegativeLogLikelihoodSimple().loss(
            y_pred, y_test["duration_test"], y_test["event"], time_grid
        )
        scores["censlog"] = round(censlog, 4)
            
    else:
        # Accuracy in time
        print("computing accuracy in time")
        truncation_quantiles = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
        times = np.quantile(time_grid, truncation_quantiles)
        accuracy = []
        
         # TODO: put it into a function in hazardous._metrics
        for time_idx in range(len(times)):
            y_pred_at_t = y_pred[:, :, time_idx]
            mask = (y_test["event"] == 0) & (y_test["duration"] < times[time_idx])
            y_pred_class = y_pred_at_t[:, ~mask].argmax(axis=0)
            y_test_class = y_test["event"] * (y_test["duration"] < times[time_idx])
            y_test_class = y_test_class.loc[~mask]
            accuracy.append(
                round(
                    (y_test_class.values == y_pred_class).mean(),
                    4
                )
            )
        
        scores["accuracy_in_time"] = {
            "time_quantile": truncation_quantiles,
            "accuracy": accuracy,
        }

    return scores


def aggregate_scores(all_scores):
    """Aggregate model seeds
    """
    agg_scores = defaultdict(dict)

    for key, scores in all_scores.items():
        dataset_name, model_name = key.split("__")

        agg_score = _aggregate_scores(scores)

        if scores[0]["is_competing_risk"]:
            agg_score.update(
                _agg_competing_risk(scores)
            )
            agg_score["average_ibs"] = np.mean([
                event_score["mean_ibs"]
                for event_score in agg_score["event_specific_ibs"]
            ]).round(4)
        else:
            agg_score.update(
                _agg_survival(scores)
            )

        for col in ["fit_time", "predict_time"]:
            agg_score.update({
                f"mean_{col}": np.mean([score[col] for score in scores]).round(2),
                f"std_{col}": np.std([score[col] for score in scores]).round(2),
            })

        fields = [
            "is_competing_risk",
            "n_events",
            "n_rows",
            "n_cols",
            "censoring_rate",
        ]
        for k in fields:
            agg_score[k] = scores[0][k]

        agg_scores[dataset_name][model_name] = agg_score

    return agg_scores


def _aggregate_scores(scores):
    agg_score = dict()
    
    # Brier score
    n_event = scores[0]["n_events"]
    event_specific_brier_scores = []
    for event_idx in range(n_event):
        brier_scores = []
        for score in scores:
            brier_scores.append(
                score[f"event_specific_brier_scores"][event_idx]["brier_score"]
            )
        brier_scores = np.vstack(brier_scores)
        event_specific_brier_scores.append({
            "event": event_idx + 1,
            "time": score["event_specific_brier_scores"][0]["time"],
            f"mean_brier_scores": list(brier_scores.mean(axis=0).round(4)),
            f"std_brier_scores": list(brier_scores.std(axis=0).round(4)),
        })
    agg_score[f"event_specific_brier_scores"] = event_specific_brier_scores

    # IBS
    event_specific_ibs = []
    for event_idx in range(n_event):
        ibs = [score["event_specific_ibs"][event_idx]["ibs"] for score in scores]
        event_specific_ibs.append({
            "event": event_idx + 1,
            "mean_ibs": np.mean(ibs).round(4),
            "std_ibs":  np.std(ibs).round(4),
        })
    agg_score["event_specific_ibs"] = event_specific_ibs

    # C-index
    time_quantiles = scores[0]["event_specific_c_index"][0]["time_quantile"]
    q_specific_c_index = []
    for idx, q in enumerate(time_quantiles):
        mean_c_index, std_c_index = [], []
        for event_idx in range(n_event):
            c_indices = [
                score["event_specific_c_index"][event_idx]["c_index"][idx]
                for score in scores
            ]
            mean_c_index.append(np.mean(c_indices).round(4))
            std_c_index.append(np.std(c_indices).round(4))
        q_specific_c_index.append({
            "time_quantile": round(q, 4),
            "event": list(range(1, n_event+1)),
            "mean_c_index": mean_c_index,
            "std_c_index": std_c_index,
        })

    agg_score["c_index"] = q_specific_c_index

    return agg_score


def _agg_competing_risk(scores):
    # Accuracy in time
    agg_score = dict()

    time_quantiles = scores[0]["accuracy_in_time"]["time_quantile"]
    accuracies = np.vstack([
        score["accuracy_in_time"]["accuracy"] for score in scores
    ])
    agg_score["accuracy_in_time"] = {
        "time_quantiles": time_quantiles,
        "mean_accuracy": list(accuracies.mean(axis=0).round(4)),
        "std_accuracy": list(accuracies.std(axis=0).round(4)),
    }
    return agg_score


def _agg_survival(scores):
    # censlog
    censlog = [score["censlog"] for score in scores]
    return {
        "mean_censlog": np.mean(censlog).round(4),
        "std_censlog": np.std(censlog).round(4),
    }


def standalone_aggregate():
    """Run to restart the aggregation from the raw scores checkpoint
    in case it failed."""
    all_scores = json.load(open("./scores/raw_scores.json"))
    agg_scores = aggregate_scores(all_scores)
    json.dump(agg_scores, open("./scores/agg_scores.json", "w"))


# %%
if __name__ == "__main__":
    evaluate_all_models()

# %%
