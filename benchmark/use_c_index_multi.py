# %%
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

from _dataset import get_split_seer
from hazardous.metrics._c_index_multi import (
    concordance_index_incidence,
)

N_TEST_C_INDEX = 10_000


def compute_c_index_multi_risk(model_dataset_path):

    data = json.load(open(model_dataset_path))

    model_c_index = defaultdict(list)
    for seed_data in tqdm(data):
        y_pred = np.asarray(seed_data["y_pred"])
        y_pred = np.transpose(y_pred, (1, 0, 2))

        random_state = seed_data["random_state"]
        bunch = get_split_seer(dict(random_state=random_state))

        if (
            N_TEST_C_INDEX is not None
            and N_TEST_C_INDEX < bunch.y_test.shape[0]
        ):
            bunch.y_test = bunch.y_test.reset_index(drop=True)
            bunch.y_test, _ = train_test_split(
                bunch.y_test,
                stratify=bunch.y_test["event"],
                train_size=N_TEST_C_INDEX,
                shuffle=True,
                random_state=random_state,
            )
            y_pred = y_pred[bunch.y_test.index, :, :]

        time_grid = np.asarray(seed_data["time_grid"])
        quantile_horizons = [.25, .5, .75]
        taus = np.quantile(time_grid, quantile_horizons)

        for event_id in range(1, 4):
            c_indices_tau = concordance_index_incidence(
                y_pred=y_pred[:, event_id, :],
                y_test=bunch.y_test,
                y_train=bunch.y_train,
                time_grid=time_grid,
                taus=taus,
                event_of_interest=event_id,
            )
            model_c_index[event_id].append(c_indices_tau)

    agg_c_index = defaultdict(list)
    for event_id, scores in model_c_index.items():
        scores = np.asarray(scores)
        mean, std = scores.mean(axis=0), scores.std(axis=0)
        for idx, tau in enumerate(quantile_horizons):
            agg_c_index[tau].append({"mean": mean[idx], "std": std[idx]})

    agg_results = []
    for tau, result in agg_c_index.items():
        tau_results = {
            "time_quantile": tau,
            "event": list(range(1, 4)),
            "mean_c_index": [round(event["mean"], 4) for event in result],
            "std_c_index": [round(event["std"], 4) for event in result],
        }
        agg_results.append(tau_results)

    return agg_results


def add_c_index_multi_to_agg():
    model_paths = [
       path for path in Path("scores/raw/").glob("*")
        if (path / "seer.json").exists() and not "deephit" in str(path)
    ]
    for model_path in model_paths:
        print(f"--- {model_path} ---")
        c_index_multi = compute_c_index_multi_risk(model_path / "seer.json")

        model = model_path.name
        path_agg = Path("scores/agg") / model / "seer.json"
        agg_results = json.load(open(path_agg))
        agg_results["c_index_multi"] = c_index_multi
        
        json.dump(agg_results, open(path_agg, "w"))
        print(f"Wrote {path_agg}")


add_c_index_multi_to_agg()
# %%
