# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd


model_remaming = {
    "kaplan_meier": "Kaplan-Meier",
    "gbmi": " MultiIncidence",
    "survtrace": "SurvTRACE",
    "deephit": "DeepHit",
    "sumonet": "SumoNet",
    "dqs": "DQS",
    "han-bs_game": "Han et al. (bs_game)",
    "sksurv_boosting": "Gradient Boosting Survival",
    "random_survival_forest": "Random Survival Forests",
    # "fine_and_gray": "Fine & Gray",
    # "aalen_johansen": "Aalen Johansen",
    "pchazard": "PCHazard",
}

include_datasets = ["support", "metabric"]
metabric_filename = "table_rebutal_more_survival_metrics_metabric.txt"
support_filename = "table_rebutal_more_survival_metrics_support.txt"

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

        mean_ibs = agg_result["event_specific_ibs"][0]["mean_ibs"]
        std_ibs = agg_result["event_specific_ibs"][0]["std_ibs"]
        ibs = f"{mean_ibs} ± {std_ibs}"

        result = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "ibs (↓)": ibs,
        }

        for metric in [
            "mse", "mae", "auc", "km_calibration", "x_calibration",
            "d_calibration", "one_calibration",
        ]:
            mean = agg_result[f"mean_{metric}"]
            std = agg_result[f"std_{metric}"]

            if metric in ["mse", "mae"]:
                value =  f"{mean:.1f} ± {std:.1f}"
            elif metric == "d_calibration":
                value = f"{mean:.3E} ± {std:.3E}"
            elif "calibration" in metric:
                value = f"{mean:.6f} ± {std:.6f}"
            else:
                value =  f"{mean:.4f} ± {std:.4f}" 

            arrow = {
                "mse": "↓",
                "mae": "↓",
                "auc": "↓",
                "km_calibration": "↓",
                "x_calibration": "↓",
                "d_calibration": "↑",
                "one_calibration": "↑",
            }[metric]
            metric_with_arrow = f"{metric} ({arrow})"            
            result[metric_with_arrow] = value

        results.append(result)


df = pd.DataFrame(results)
df["model_name"] = df["model_name"].map(model_remaming)

order = {
    "Kaplan-Meier": -1,
    "DeepHit": 0,
    "PCHazard": 1,
    "Han et al. (bs_game)": 3,
    "DQS": 4,
    "SumoNet": 5,
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
    
    means = [float(cell.split("±")[0]) for cell in x.values[1:]] # Exclude KM
    order = np.asarray(np.argsort(means)) + 1
    if x.name.split(" (")[0] not in [
        "ibs", "mse", "mae", "x_calibration", "km_calibration"
    ]:
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

print(df_support.to_markdown())
# %%
print(df_metabric.to_markdown())

# %%
