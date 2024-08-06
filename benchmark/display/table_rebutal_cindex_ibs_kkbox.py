# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd


model_remaming = {
    "gbmi": " MultiIncidence",
    "survtrace": "SurvTRACE",
    "deephit": "DeepHit",
    "sumonet": "SumoNet",
    "dqs": "DQS",
    "han-nll": "Han et al. (nll)",
    "han-bll_game": "Han et al. (bll_game)",
    "sksurv_boosting": "Gradient Boosting Survival",
    "random_survival_forest": "Random Survival Forests",
    "fine_and_gray": "Fine & Gray",
    "aalen_johansen": "Aalen Johansen",
    "pchazard": "PCHazard",
}

include_datasets = ["kkbox_100k", "kkbox_1M"]
filename = "table_rebutal_cindex_ibs_kkbox"

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
        
        mean_cenlog = round(agg_result["mean_censlog"], 3)
        std_cenlog = round(agg_result["std_censlog"], 3)
        cenlog = f"{mean_cenlog} ± {std_cenlog}"

        ibs_result = agg_result["event_specific_ibs"][0]
        mean_ibs = round(ibs_result["mean_ibs"], 3)
        std_ibs = round(ibs_result["std_ibs"], 3)
        ibs = f"{mean_ibs} ± {std_ibs}"

        result = {
            "dataset_name": dataset_name,
            "model_name": model_name,
        }

        for q_result in agg_result["c_index"]:
            q = str(round(q_result["time_quantile"], 2))
            mean_c_index = round(q_result["mean_c_index"][0], 3)
            std_c_index = round(q_result["std_c_index"][0], 3)
            event_ids = q_result["event"][0]
            
            c_index = f"{mean_c_index} ± {std_c_index}"
            result[f"C-index {q}"] = c_index

        result.update({
            "S Cen-log-simple": cenlog,
            "IBS": ibs,
        })

        results.append(result)


df = pd.DataFrame(results)
df["model_name"] = df["model_name"].map(model_remaming)

order = {
    # "DeepHit": 0,
    "PCHazard": 1,
    "Han et al. (nll)": 2,
    "Han et al. (bll_game)": 3,
    "DQS": 4,
    "SumoNet": 5,
    "SurvTRACE": 6,
    "Random Survival Forests": 7,
    "Gradient Boosting Survival": 8,
    "MultiIncidence": 9,
}
df["order"] = df["model_name"].map(order)
df = df.sort_values("order").drop("order", axis=1)

df_1m = df.query("dataset_name == 'kkbox_1M'").drop("dataset_name", axis=1)
df_100k = df.query("dataset_name == 'kkbox_100k'").drop("dataset_name", axis=1)

def bold_and_underline(x):
    style = [""] * len(x)
    if x.name == "model_name":
        return style
    
    order = np.asarray(np.argsort(x))
    if "C-index" in x.name:
        order = order[::-1]
    style[order[0]] = "font-weight: bold"
    style[order[1]] = "text-decoration: underline"

    return style

df_1m_style = df_1m.style.apply(bold_and_underline, axis=0)
open(filename, "w").write(df_1m_style.to_latex())
df_1m_style

# %%
df_100k_style = df_100k.style.apply(bold_and_underline, axis=0)
df_100k_style

# %%
