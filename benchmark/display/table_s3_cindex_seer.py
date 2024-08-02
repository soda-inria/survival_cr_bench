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
filename = "table_s3_cindex_seer.txt"

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

        for q_result in agg_result["c_index"]:
            q = str(round(q_result["time_quantile"], 2))
            mean_c_index = q_result["mean_c_index"]
            std_c_index = q_result["std_c_index"]
            event_ids = q_result["event"]
            
            for mean, std, event_id in zip(mean_c_index, std_c_index, event_ids):
                c_index = f"{mean} ± {std}"
                results.append(
                    {
                        "Time Horizon Quantile": q,
                        "Event": event_id,
                        "c_index": c_index,
                        "model_name": model_name,
                    }
                )

df = pd.DataFrame(results)
df["model_name"] = df["model_name"].map(model_remaming)

df = df.pivot(index="model_name", columns=["Time Horizon Quantile", "Event"], values=["c_index"])

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
df = df.sort_index(key=lambda x: x.map(order))

def bold_and_underline(x):
    style = [""] * len(x)
    order = np.argsort(x)[::-1]
    style[order[0]] = "font-weight: bold"
    style[order[1]] = "text-decoration: underline"
    return style

df_style = df.style.apply(bold_and_underline, axis=0)
open(filename, "w").write(df_style.to_latex())
df_style

# %%
