# %%
from time import time
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

from sksurv.metrics import concordance_index_ipcw
from hazardous.utils import make_time_grid, make_recarray
from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
    integrated_brier_score_incidence_oracle,
    brier_score_incidence,
    brier_score_incidence_oracle,
)
from hazardous.metrics._yana import CensoredNegativeLogLikelihoodSimple

from _dataset import LOAD_DATASET_FUNCS
from _model import INIT_MODEL_FUNCS
from hyper_parameter_search import PATH_HP_SEARCH

# Setting an integer value will perform subsampling on the test set
# to debug faster. Otherwise, setting it to None will disable this option.
DEBUG_N_SUBSAMPLE = None
PATH_SCORES = Path("scores/")
N_STEPS_TIME_GRID = 20
N_TEST_C_INDEX = 10_000


def evaluate_all_models(include_models=None, include_datasets=None, verbose=True):

    all_params = get_params()

    all_scores = defaultdict(lambda: defaultdict(list))

    if isinstance(include_models, str):
        include_models = [include_models]
    
    if isinstance(include_datasets, str):
        include_datasets = [include_datasets]

    # We iterate over each model, dataset and random_state in best_hyper_parameters/
    for (dataset_name, dataset_params, model_name, model_params) in all_params:

        if (
            include_models is not None and not model_name in include_models
            or include_datasets is not None and not dataset_name in include_datasets
        ):
            continue
        
        bunch = LOAD_DATASET_FUNCS[dataset_name](dataset_params)
        X_train, y_train = bunch.X_train, bunch.y_train

        model = INIT_MODEL_FUNCS[model_name](**model_params)

        if verbose:
            print(
                f"{' Start fitting ' + model_name + ' on ' + dataset_name + ' ':=^50}"
            )
            print(f"dataset_params: {dataset_params}")
            print(f"model_params: {model_params}")
        
        tic = time()
        model = model.fit(X_train, y_train)
        toc = time()
        fit_time = round(toc - tic, 2)

        scores = evaluate(
            model, bunch, dataset_name, dataset_params, model_name, verbose,
        )
        scores["fit_time"] = fit_time

        if verbose:
            print(f"Evaluation done")
        
        all_scores[model_name][dataset_name].append(scores)
        path_dir = PATH_SCORES / "raw" / model_name
        path_dir.mkdir(parents=True, exist_ok=True)
        path_file = path_dir / f"{dataset_name}.json"
        json.dump(
            all_scores[model_name][dataset_name],
            open(path_file, "w")
        )
        print(f"Wrote {path_file}")
        
    for model_name in all_scores.keys():
        for dataset_name in all_scores[model_name].keys():
            agg_scores = aggregate_scores(all_scores[model_name][dataset_name])
            path_dir = PATH_SCORES / "agg" / model_name
            path_dir.mkdir(parents=True, exist_ok=True)
            path_file = path_dir / f"{dataset_name}.json"
            json.dump(agg_scores, open(path_file, "w"))
            print(f"Wrote {path_file}")


def get_params():
    """Fetch and accumulate the data and models params from \
    the hp tuning results.
    """
    all_params = []

    for model_path in PATH_HP_SEARCH.glob("*"):
        model_name = model_path.name
        
        for dataset_path in model_path.glob("*"):
            dataset_name = dataset_path.name    
            
            for run_path in dataset_path.glob("*"):
                
                best_model_params = json.load(open(run_path / "best_params.json"))
                dataset_params = json.load(open(run_path / "dataset_params.json"))
                best_model_params.pop("model_name", None)

                all_params.append(
                    [dataset_name, dataset_params, model_name, best_model_params]
                )

    return all_params


def evaluate(
    model, bunch, dataset_name, dataset_params, model_name, verbose=True
):
    """Evaluate a model against its test set.
    """
    X_train, y_train = bunch.X_train, bunch.y_train
    X_test, y_test = bunch.X_test, bunch.y_test

    n_events = np.unique(y_train["event"]).shape[0] - 1
    is_competing_risk = n_events > 1

    scores = {
        "is_competing_risk": is_competing_risk,
        "n_events": n_events,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_rows": X_train.shape[0],
        "n_cols": X_train.shape[1],
        "censoring_rate": (y_train["event"] == 0).mean(),
        **dataset_params,
    }

    if DEBUG_N_SUBSAMPLE is not None:
        X_test, _, y_test, _ = train_test_split(
            X_test,
            y_test,
            train_size=DEBUG_N_SUBSAMPLE,
            stratify=y_test["event"],
            random_state=0 # Fix seed for evaluation split
        )

    time_grid = make_time_grid(y_test["duration"], n_steps=N_STEPS_TIME_GRID)

    if verbose:
        print("Running prediction")
    tic = time()
    y_pred = model.predict_cumulative_incidence(X_test, time_grid)
    toc = time()

    scores["time_grid"] = time_grid.round(4).tolist()
    scores["y_pred"] = y_pred.round(4).tolist()
    scores["predict_time"] = round(toc - tic, 2)

    event_specific_ibs, event_specific_brier_scores = [], []
    event_specific_c_index = []

    if verbose:
        print("Computing Brier scores, IBS and C-index")

    y_train_binary = y_train.copy()
    y_test_binary = y_test.copy()

    for event_id in range(1, n_events+1):

        # Brier score and IBS
        if dataset_name == "weibull":
            # Use oracle metrics with the synthetic dataset.
            ibs = integrated_brier_score_incidence_oracle(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                shape_censoring=bunch.shape_censoring.loc[y_test.index],
                scale_censoring=bunch.scale_censoring.loc[y_test.index],
                event_of_interest=event_id,
            )
            brier_scores = brier_score_incidence_oracle(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                shape_censoring=bunch.shape_censoring.loc[y_test.index],
                scale_censoring=bunch.scale_censoring.loc[y_test.index],
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
        y_train_binary["event"] = (y_train["event"] == event_id)
        y_test_binary["event"] = (y_test["event"] == event_id)

        y_pred_c_index = y_pred.copy()

        if N_TEST_C_INDEX is not None:
            y_test_binary = y_test_binary.reset_index(drop=True)
            y_test_binary, _ = train_test_split(
                y_test_binary,
                stratify=y_test_binary["event"],
                train_size=N_TEST_C_INDEX,
                shuffle=True,
                random_state=dataset_params["random_state"],
            )
            y_pred_c_index = y_pred_c_index[:, y_test_binary.index, :]

        truncation_quantiles = [0.25, 0.5, 0.75]
        taus = np.quantile(time_grid, truncation_quantiles)
        if verbose and event_id == 1:
            print(f"{taus=}")
        taus = tqdm(
            taus,
            desc=f"c-index at tau for event {event_id}",
            total=len(taus),
        )
        c_indices = []
        for tau in taus:
            tau_idx = np.searchsorted(time_grid, tau)
            y_pred_at_t = y_pred_c_index[event_id][:, tau_idx]
            if model_name == "aalen_johansen":
                ct_index = 0.5
            else:
                ct_index, _, _, _, _ = concordance_index_ipcw(
                    make_recarray(y_train_binary),
                    make_recarray(y_test_binary),
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

    if is_competing_risk:
        # Accuracy in time
        truncation_quantiles = np.arange(0.125, 1, 0.125).tolist()
        taus = np.quantile(time_grid, truncation_quantiles)
        if verbose:
            print("Computing accuracy in time")
            print(f"{taus=}")
        accuracy = []
        
         # TODO: put it into a function in hazardous._metrics
        for tau in taus:
            tau_idx = np.searchsorted(time_grid, tau)
            y_pred_at_t = y_pred[:, :, tau_idx]
            mask = (y_test["event"] == 0) & (y_test["duration"] < tau)
            y_pred_class = y_pred_at_t[:, ~mask].argmax(axis=0)
            y_test_class = y_test["event"] * (y_test["duration"] < tau)
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
        if verbose:
            print(f"{accuracy=}")

    else:
        # Yana loss
        censlog = CensoredNegativeLogLikelihoodSimple().loss(
            y_pred, y_test["duration"], y_test["event"], time_grid
        )
        scores["censlog"] = round(censlog, 4)        
        if verbose:
            print(f"{censlog=}")

    if verbose:
        print(f"{event_specific_ibs=}")
        print(f"{event_specific_c_index}")

    return scores


def aggregate_scores(seed_scores, already_aggregated=False):
    """Aggregate model seeds
    """
    agg_score = _aggregate_scores(seed_scores, already_aggregated)

    if seed_scores[0]["is_competing_risk"]:
        agg_score.update(
            _agg_competing_risk(seed_scores, already_aggregated)
        )
        agg_score["average_ibs"] = np.mean([
            event_score["mean_ibs"]
            for event_score in agg_score["event_specific_ibs"]
        ]).round(4)
    else:
        agg_score.update(
            _agg_survival(seed_scores, already_aggregated)
        )

    if already_aggregated:
        for col in ["mean_fit_time", "mean_predict_time"]:
            agg_col = col.split("mean_")[1]
            agg_score.update({
                f"mean_{agg_col}": np.mean([score[col] or 0 for score in seed_scores]).round(2),
                f"std_{agg_col}": np.std([score[col] or 0 for score in seed_scores]).round(2),
            })
    else:
        for col in ["fit_time", "predict_time"]:
            agg_score.update({
                f"mean_{col}": np.mean([score[col] or 0 for score in seed_scores]).round(2),
                f"std_{col}": np.std([score[col] or 0 for score in seed_scores]).round(2),
            })

    fields = [
        "is_competing_risk",
        "n_events",
        "n_rows",
        "n_cols",
        "censoring_rate",
    ]
    for k in fields:
        agg_score[k] = seed_scores[0][k]

    return agg_score


def _aggregate_scores(scores, already_aggregated=False):
    agg_score = dict()
    
    # Brier score
    n_event = scores[0]["n_events"]
    event_specific_brier_scores = []
    for event_idx in range(n_event):
        brier_scores = []
        for score in scores:
            if already_aggregated:
                key = "mean_brier_scores"
            else:
                key = "brier_scores"
            brier_scores.append(
                score[f"event_specific_brier_scores"][event_idx][key]
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
        if already_aggregated:
            key = "mean_ibs"
        else:
            key = "ibs"
        ibs = [score["event_specific_ibs"][event_idx][key] for score in scores]
        event_specific_ibs.append({
            "event": event_idx + 1,
            "mean_ibs": np.mean(ibs).round(4),
            "std_ibs":  np.std(ibs).round(4),
        })
    agg_score["event_specific_ibs"] = event_specific_ibs

    # C-index
    if already_aggregated:
        time_quantiles = [score["time_quantile"] for score in scores[0]["c_index"]]
        q_specific_c_index = []
        for idx, q in enumerate(time_quantiles):
            mean_c_index, std_c_index = [], []
            for event_idx in range(n_event):
                c_indices = [
                    score["c_index"][idx]["mean_c_index"][event_idx]
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

    else:
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


def _agg_competing_risk(scores, already_aggregated=False):
    # Accuracy in time
    agg_score = dict()

    if already_aggregated:
        quantile_name = "time_quantiles"
        metric_to_aggregate = "mean_accuracy"
    else:
        quantile_name = "time_quantile"  # TODO: fix 
        metric_to_aggregate = "accuracy"

    time_quantiles = scores[0]["accuracy_in_time"][quantile_name]
    accuracies = np.vstack([
        score["accuracy_in_time"][metric_to_aggregate] for score in scores
    ])
    agg_score["accuracy_in_time"] = {
        "time_quantiles": time_quantiles,
        "mean_accuracy": list(accuracies.mean(axis=0).round(4)),
        "std_accuracy": list(accuracies.std(axis=0).round(4)),
    }
    return agg_score


def _agg_survival(scores, already_aggregated=False):
    # censlog
    if already_aggregated:
        metric_to_aggregate = "mean_censlog"
    else:
        metric_to_aggregate = "censlog"
    censlog = [score[metric_to_aggregate] for score in scores]
    return {
        "mean_censlog": np.mean(censlog).round(4),
        "std_censlog": np.std(censlog).round(4),
    }


def standalone_aggregate(model_name, dataset_name, already_aggregated=False):
    """Run to restart the aggregation from the raw scores checkpoint
    in case it failed."""
    path_dir_raw = PATH_SCORES / "raw" / model_name
    model_scores = json.load(open(path_dir_raw / f"{dataset_name}.json"))
    agg_scores = aggregate_scores(model_scores, already_aggregated)
    path_dir_agg = PATH_SCORES / "agg" / model_name
    path_dir_agg.mkdir(parents=True, exist_ok=True) 
    json.dump(agg_scores, open(path_dir_agg / f"{dataset_name}.json", "w"))


# %%

if __name__ == "__main__":
    #evaluate_all_models(include_models=["sksurv_boosting"], include_datasets=["kkbox"])
    standalone_aggregate("deephit", "kkbox_100k", already_aggregated=True)


# %%
