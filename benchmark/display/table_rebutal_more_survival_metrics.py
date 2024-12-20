# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd


model_remaming = {
    "kaplan_meier": "Kaplan-Meier",
    "gbmi": "SurvivalBoost",
    "survtrace": "SurvTRACE",
    "deephit": "DeepHit",
    "sumonet": "SumoNet",
    "dqs": "DQS",
    "pchazard": "PCHazard",
    #"han-nll": "Han et al. (NLL)",
    "sksurv_boosting": "Gradient Boosting Survival",
    "random_survival_forest": "Random Survival Forests",
    # "fine_and_gray": "Fine & Gray",
    # "aalen_johansen": "Aalen Johansen",
    "pchazard": "PCHazard",
    "xgbse": "XGBSE Debiased BCE",
}

include_datasets = ["support", "metabric"] #, "kkbox", "metabric"]
metabric_filename = "table_rebutal_more_survival_metrics_metabric.txt"
support_filename = "table_rebutal_more_survival_metrics_support.txt"
kkbox_filename = "table_rebutal_more_survival_metrics_kkbox.txt"

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
        
        mean_cenlog = agg_result["mean_censlog"]
        std_cenlog = agg_result.get("std_cenlog") or agg_result.get("std_censlog")
        censlog = f"{mean_cenlog} ± {std_cenlog}"

        result = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "ibs (↓)": ibs,
            "S-cen-log-simple (↓)": censlog,
        }

        for c_index_q in agg_result["c_index"]:
            q = c_index_q["time_quantile"]
            mean = c_index_q["mean_c_index"][0]
            std = c_index_q["std_c_index"][0]
            result[f"C-index {q}"] = f"{mean} ± {std}"

        for metric in [
            "mse", "mae", "auc", "km_calibration", "x_calibration",
            "d_calibration", "one_calibration",
        ]:
            if f"mean_{metric}" not in agg_result:
                continue
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
                "auc": "↑",
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
    "Han et al. (NLL)": 3,
    "DQS": 4,
    "SumoNet": 5,
    "SurvTRACE": 6,
    "Random Survival Forests": 7,
    "Gradient Boosting Survival": 8,
    "XGBSE Debiased BCE": 9,
    "SurvivalBoost": 10,
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
df_kkbox = (
    df.query("dataset_name == 'kkbox'")
    .drop("dataset_name", axis=1)
    .reset_index(drop=True)
)


def bold_and_underline(x):
    style = [""] * len(x)
    if x.name == "model_name":
        return style
    
    means = [
        float(cell.split("±")[0])
        if cell is not np.nan else np.nan
        for cell in x.values[0:]
    ] # Exclude KM
    order = np.asarray(np.argsort(means))
    if x.name.split(" (")[0] not in [
        "ibs", "S-cen-log-simple", "mse", "mae", "x_calibration", "km_calibration"
    ]:
        order = order[::-1]
    style[order[0]] = "font-weight: bold"
    style[order[1]] = "text-decoration: underline"

    return style


df_kkbox_style = df_kkbox.style.apply(bold_and_underline, axis=0)
open(kkbox_filename, "w").write(df_kkbox_style.to_latex())
df_kkbox_style

# %%

df_metabric_style = df_metabric.style.apply(bold_and_underline, axis=0)
open(metabric_filename, "w").write(df_metabric_style.to_latex())
df_metabric_style

# %%
df_support_style = df_support.style.apply(bold_and_underline, axis=0)
open(support_filename, "w").write(df_support_style.to_latex())
df_support_style


# %%
print(df_kkbox.to_markdown())

# %%

print(df_support.to_markdown())
# %%
print(df_metabric.to_markdown())

# %%
