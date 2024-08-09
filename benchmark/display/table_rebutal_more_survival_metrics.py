# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd


model_remaming = {
    "gbmi": " MultiIncidence",
    "survtrace": "SurvTRACE",
    # "deephit": "DeepHit",
    # "sumonet": "SumoNet",
    "dqs": "DQS",
    # "han-nll": "Han et al. (nll)",
    # "han-bll_game": "Han et al. (bll_game)",
    "sksurv_boosting": "Gradient Boosting Survival",
    "random_survival_forest": "Random Survival Forests",
    # "fine_and_gray": "Fine & Gray",
    # "aalen_johansen": "Aalen Johansen",
    # "pchazard": "PCHazard",
}

include_datasets = ["support", "metabric"]
metabric_filename = "table_s4_metabric_cindex_ibs.txt"
support_filename = "table_s5_suppport_cindex_ibs.txt"

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
        
        mean_mse = round(agg_result["mean_mse"], 1)
        std_mse = round(agg_result["std_mse"], 1)
        mse = f"{mean_mse} ± {std_mse}"

        mean_mae = round(agg_result["mean_mae"], 1)
        std_mae = round(agg_result["std_mae"], 1)
        mae = f"{mean_mae} ± {std_mae}"

        mean_auc = round(agg_result["mean_auc"], 4)
        std_auc = round(agg_result["std_auc"], 4)
        auc = f"{mean_auc} ± {std_auc}"

        result = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "MSE": mse,
            "MAE": mae,
            "AUC": auc,
        }

        results.append(result)


df = pd.DataFrame(results)
df["model_name"] = df["model_name"].map(model_remaming)

order = {
    # "DeepHit": 0,
    # "PCHazard": 1,
    # "Han et al. (nll)": 2,
    # "Han et al. (bll_game)": 3,
    "DQS": 4,
    # "SumoNet": 5,
    "SurvTRACE": 6,
    "Random Survival Forests": 7,
    "Gradient Boosting Survival": 8,
    "MultiIncidence": 9,
}
df["order"] = df["model_name"].map(order)
df = df.sort_values("order").drop("order", axis=1)

df_support = (
    df.query("dataset_name == 'support'")
    .drop("dataset_name", axis=1)
    .reset_index(drop=True)
)
df_metabric = (
    df.query("dataset_name == 'metabric'")
    .drop("dataset_name", axis=1)
    .reset_index(drop=True)
)

def bold_and_underline(x):
    style = [""] * len(x)
    if x.name == "model_name":
        return style
    
    means = [float(cell.split("±")[0]) for cell in x.values]
    order = np.asarray(np.argsort(means))
    if "C-index" in x.name or "AUC" in x.name:
        order = order[::-1]
    style[order[0]] = "font-weight: bold"
    style[order[1]] = "text-decoration: underline"

    return style

df_metabric_style = df_metabric.style.apply(bold_and_underline, axis=0)
open(metabric_filename, "w").write(df_metabric_style.to_latex())
df_metabric_style

# %%
df_support_style = df_support.style.apply(bold_and_underline, axis=0)
open(support_filename, "w").write(df_support_style.to_latex())
df_support_style


# %%
