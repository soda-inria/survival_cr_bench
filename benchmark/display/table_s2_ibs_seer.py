# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


model_remaming = {
    "gbmi": "MultiIncidence",
    "survtrace": "SurvTRACE",
    "deephit": "DeepHit",
    "DSM": "DSM",
    "DeSurv": "DeSurv",
    "random_survival_forest": "RandomSurvivalForest",
    "fine_and_gray": "Fine & Gray",
    "aalen_johansen": "Aalen Johansen",
}

include_datasets = ["seer"]
filename = "table_s1_ibs_seer.txt"

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

        for event_result in agg_result["event_specific_ibs"]:
            mean_ibs = round(event_result["mean_ibs"], 4)
            std_ibs = round(event_result["std_ibs"], 4)
            ibs = f"{mean_ibs} ± {std_ibs}"
            event_id = event_result["event"]

            results.append(
                {
                    "IBS": ibs,
                    "Event": event_id,
                    "model_name": model_name,
                }
            )

df = pd.DataFrame(results)
df["model_name"] = df["model_name"].map(model_remaming)

df = df.pivot(index="model_name", columns="Event", values=["IBS"])

order = {
    "Aalen Johansen": 0,
    "Fine & Gray": 1,
    "RandomSurvivalForest": 2,
    "DeepHit": 3,
    "DSM": 4,
    "DeSurv": 5,
    "SurvTRACE": 6,
    "MultiIncidence": 7,
}
df = df.join(pd.Series(order, name="order").to_frame())
df = df.sort_values("order").drop("order", axis=1)

def bold_and_underline(x):
    style = [""] * len(x)
    order = np.argsort(x)
    style[order[0]] = "font-weight: bold"
    style[order[1]] = "text-decoration: underline"
    return style

df = df.style.apply(bold_and_underline, axis=0)
open(filename, "w").write(df.to_latex())
df
# %%
