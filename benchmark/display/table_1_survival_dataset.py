# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd


model_remaming = {
    "gbmi": "MultiIncidence",
    "survtrace": "SurvTRACE",
    "deephit": "DeepHit",
    "sumonet": "SumoNet",
    "dqs": "DQS",
    "han-nll": "Han et al. (nll)",
    "han-bll_game": "Han et al. (game)",
    "sksurv_boosting": "Gradient Boosting Survival",
    "random_survival_forest": "Random Survival Forests",
    "fine_and_gray": "Fine & Gray",
    "aalen_johansen": "Aalen Johansen",
    "pchazard": "PCHazard",
}

include_datasets = ["support", "metabric"]
filename = "table_1_survival_dataset.txt"

path_scores = Path("../scores/agg/")
results = []

for path_model in path_scores.glob("*"):
    model_name = path_model.name
    if not model_name in list(model_remaming):
        continue
    for path_dataset in path_model.glob("*"):
        dataset_name = path_dataset.name.split(".")[0]
        if not dataset_name in include_datasets:
            continue
        agg_result = json.load(open(path_dataset))

        mean_ibs = round(
            agg_result["event_specific_ibs"][0]["mean_ibs"], 3
        )
        std_ibs = round(
            agg_result["event_specific_ibs"][0]["std_ibs"], 3
        )
        ibs = f"{mean_ibs} ± {std_ibs}"

        mean_censlog = round(agg_result["mean_censlog"], 3)
        std_censlog = round(agg_result["std_censlog"], 3)
        censlog = f"{mean_censlog} ± {std_censlog}"

        results.append(
            {
                "IBS": ibs,
                "Cen-log-simple": censlog,
                "model_name": model_name,
                "dataset_name": dataset_name,
            }
        )

df = pd.DataFrame(results)
df["model_name"] = df["model_name"].map(model_remaming)
df["dataset_name"] = df["dataset_name"].str.upper()

df = df.pivot(index="model_name", columns="dataset_name", values=["IBS", "Cen-log-simple"])
df.columns = df.columns.swaplevel(0, 1).sortlevel()[0]

order = {
    "Random Survival Forests": -1,
    "DeepHit": 0,
    "PCHazard": 1,
    "Han et al. (nll)": 2,
    "Han et al. (game)": 3,
    "DQS": 4,
    "SumoNet": 5,
    "SurvTRACE": 6,
    "Gradient Boosting Survival": 7,
    "MultiIncidence": 8,
}
df = df.sort_index(key=lambda x: x.map(order))

def bold_and_underline(x):
    style = [""] * len(x)
    order = np.argsort(x.values)
    style[order[0]] = "font-weight: bold"
    style[order[1]] = "text-decoration: underline"
    return style

df = df.style.apply(bold_and_underline, axis=0)
open(filename, "w").write(df.to_latex())
df
# %%
