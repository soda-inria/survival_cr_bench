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
    n_events
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
    event_specific_c_index: [
        {
            event
            time_quantile: []
            c_index: []
        }
    ]
}


Next, we aggregate on the seed and split between competing and survival.
This is the data we will be using for our plots.

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
        mean_accuracy: []
        std_accuracy: []
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

from ._dataset import LOAD_DATASET_FUNCS
from ._model import INIT_MODEL_FUNCS
from .hyper_parameter_search import PATH_HP_SEARCH


def main():

    all_scores = defaultdict(list)

    all_dataset_params, all_model_params = get_params()

    for (dataset_name, dataset_params), (model_name, model_params) in zip(
        all_dataset_params, all_model_params
    ):
        bunch = LOAD_DATASET_FUNCS[dataset_name](dataset_params)
        X_train, y_train = bunch.X_train, bunch.y_train

        model = INIT_MODEL_FUNCS[model_name](model_params)

        print(f"start fitting {model_name} on {dataset_name}")
        tic = time()
        model = model.fit(X_train, y_train)
        toc = time()
        fit_time = toc - tic

        scores = evaluate(
            model, bunch, dataset_name, dataset_params, model_name
        )
        scores["fit_time"] = fit_time
        
        all_scores[(dataset_name, model_name)].append(scores)
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
    n_events = np.unique(bunch.y_train).shape[0] - 1
    is_competing_risk = n_events > 1

    scores = {
        "is_competing_risk": is_competing_risk,
        "n_events": n_events,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seed": dataset_params.get("seed", None) or SEED,
        "n_rows": bunch.X_train.shape[0],
        "n_cols": bunch.X_train.shape[1],
        "censoring_rate": (bunch.y_train["event"] == 0).mean(),
    }

    X_test, y_test, y_train = bunch.X_test, bunch.y_test, bunch.y_train

    time_grid = make_time_grid(y_test["duration"])

    print("start evaluating")
    tic = time()
    y_pred = model.predict_cumulative_incidence(X_test, time_grid)
    toc = time()
    scores["predict_time"] = toc - tic

    event_specific_ibs, event_specific_brier_scores = [], []
    event_specific_c_index = []
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
            "ibs": ibs,
        })
        event_specific_brier_scores.append({
            "event": event_id,
            "time": time_grid,
            "brier_score": brier_scores,
        })

        # C-index
        truncation_quantiles = [0.25, 0.5, 0.75]
        times = np.quantile(time_grid, truncation_quantiles)
        c_indices = []
        for time_idx, time in enumerate(times):
            y_pred_at_t = y_pred[event_id][:, time_idx]
            ct_index, _, _, _, _ = concordance_index_ipcw(
                y_train,
                y_test,
                y_pred_at_t,
                tau=time,
            )
            c_indices.append(ct_index)

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
        scores["censlog"] = CensoredNegativeLogLikelihoodSimple().loss(
            y_pred, y_test["duration_test"], y_test["event"], time_grid
        )
    else:
        # Accuracy in time
        truncation_quantiles = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
        times = np.quantile(time_grid, truncation_quantiles)
        accuracy = []
        
         # TODO: put it into a function in hazardous._metrics
        for time_idx in range(len(times)):
            y_pred_at_t = y_pred[:, :, time_idx]
            mask = (y_test["event"] == 0) & (y_test["duration"] < times[time_idx])
            y_pred = y_pred[:, ~mask]
            y_pred_class = y_pred.argmax(axis=0)
            y_test_class = y_test["event"] * (y_test["duration"] < times[time_idx])
            y_test_class = y_test_class.loc[~mask]
            accuracy.append((y_test_class.values == y_pred_class).mean())
        
        scores["accuracy_in_time"] = {
            "time_quantile": truncation_quantiles,
            "accuracy": accuracy,
        }

    return scores


def aggregate_scores(all_scores):
    """Aggregate model seeds
    """
    agg_scores = defaultdict(dict)

    for (dataset_name, model_name), scores in all_scores.items():

        agg_score = _aggregate_scores(scores)

        if scores[0]["is_competing_risk"]:
            agg_score.update(
                agg_competing_risk(scores)
            )
        else:
            agg_score.update(
                agg_survival(scores)
            )

        for col in ["ibs", "fit_time", "predict_time"]:
            agg_score.update({
                f"mean_{col}": np.mean([score[col] for score in scores]),
                f"std_{col}": np.std([score[col] for score in scores]),
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
    n_event = scores[0]["n_event"]
    event_specific_brier_scores = []
    for event_idx in range(n_event):
        brier_scores = []
        for score in scores:
            brier_scores.append(
                score[f"event_specific_brier_scores"][event_idx]
            )
        brier_scores = np.vstack(brier_scores)
        event_specific_brier_scores.append({
            "event": event_idx + 1,
            "time": score["time"],
            f"mean_brier_scores": brier_scores.mean(axis=0),
            f"std_scores": brier_scores.std(axis=0),
        })
    agg_score[f"event_specific_brier_scores"] = event_specific_brier_scores

    # IBS
    event_specific_ibs = []
    for event_idx in range(n_event):
        ibs = [score["event_specific_ibs"][event_idx] for score in scores]
        event_specific_ibs.append({
            "event": event_idx + 1,
            "mean_ibs": np.mean(ibs),
            "std_ibs":  np.std(ibs),
        })
    agg_score["event_specific_ibs"] = event_specific_ibs

    # C-index
    time_quantiles = scores[0]["time_quantile"]
    q_specific_c_index = []
    for idx, q in enumerate(time_quantiles):
        mean_c_index, std_c_index = [], []
        for event_idx in range(n_event):
            c_indices = [
                score["event_specific_c_index"][event_idx]["c_index"][idx]
                for score in scores
            ]
            mean_c_index.append(np.mean(c_indices))
            std_c_index.append(np.std(c_indices))
        q_specific_c_index.append({
            "time_quantile": q,
            "event": list(range(1, n_event+1)),
            "mean_c_index": mean_c_index,
            "std_c_index": std_c_index,
        })

    agg_score["c_index"] = q_specific_c_index

    return agg_score


def agg_competing_risk(scores):
    # Accuracy in time
    agg_score = dict()

    time_quantiles = scores[0]["accuracy_in_time"]["time_quantiles"]
    accuracies = np.vstack([
        score["accuracy_in_time"]["accuracy"] for score in scores
    ])
    agg_score["accuracy_in_time"] = {
        "time_quantiles": time_quantiles,
        "mean_accuracy": accuracies.mean(axis=0),
        "std_accuracy": accuracies.std(axis=0),
    }
    return agg_score


def agg_survival(scores):
    # censlog
    censlog = [score["censlog"] for score in scores]
    return {
        "mean_censlog": np.mean(censlog),
        "std_censlog": np.std(censlog),
    }


if __name__ == "__main__":
    main()
