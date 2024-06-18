# %%
from tqdm import tqdm 
import pandas as pd
import numpy as np
from time import time
from collections import defaultdict
from sksurv.ensemble import RandomSurvivalForest 

from hazardous._deep_hit import _DeepHit
from hazardous.survtrace._model import SurvTRACE
from hazardous._gb_multi_incidence import GBMultiIncidence

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.survtrace._encoder import SurvFeatureEncoder
from hazardous.utils import (
    SurvStratifiedSingleSplit,
    SurvStratifiedShuffleSplit,
    CumulativeIncidencePipeline,
)


def make_recarray(y):
    event = y["event"].values
    duration = y["duration"].values
    return np.array(
        [(event[i], duration[i]) for i in range(y.shape[0])],
        dtype=[("e", bool), ("t", float)],
    )


def main():

    n_samples = [1e3, 1e4, 1e5]
    seeds = range(5)
    models = [
        RandomSurvivalForest(),
        _DeepHit(),
        SurvTRACE(),
        GBMultiIncidence(),
    ]    

    result = []
    for n_sample in n_samples:
        X, y = make_synthetic_competing_weibull(
            n_events=1,
            n_samples=int(n_sample),
            n_features=20,
            complex_features=True,
            return_X_y=True,
        )
        print(X.shape)
        print(y["event"].value_counts())

        for model in models:
            print(model)
            model_time = []
            for seed in tqdm(seeds):
                tic = time()

                if isinstance(model, RandomSurvivalForest):
                    if n_sample == 1e5:
                        break
                    y_ = make_recarray(y) 
                else:
                    y_ = y


                model.fit(X, y_)
                toc = time()
                model_time.append(toc - tic)
            
            mean, std = np.mean(model_time), np.std(model_time)
            row = {
                "model": model.__class__.__name__,
                "time": f"{mean:.2f}±{std:.2f}",
                "n_sample": n_sample,
            }
            result.append(row)
            print(row)

    result = pd.DataFrame(result)
    print(result)
    result.to_csv("timing.csv", index=False)


def run_deephit():    
    from sklearn.model_selection import train_test_split
    from pycox.datasets import metabric, support
    from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

    df_metabric = metabric.read_df()
    df_support = support.read_df()
    
    for df in [df_metabric]:#, df_support]:
        print("-"* 10)
        target_columns = ["event", "duration"]
        y = df[target_columns]
        X = df[list(set(df.columns) - set(target_columns))]

        all_ibs, all_cindex, fit_time = [], [], []

        for random_state in range(1):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

            est = _DeepHit()
            est.fit(X_train, y_train)
            y_surv_pred = deephit.predict_survival_function(X_test).T.values[:, 1:-1]
            print(y_surv_pred.shape)
            print(y_surv_pred)

            y_train, y_test = make_recarray(y_train), make_recarray(y_test)
            time_grid = deephit.labtrans.cuts[1:-1]
            print(time_grid)

            ibs = integrated_brier_score(
                survival_train=y_train,
                survival_test=y_test,
                estimate=y_surv_pred,
                times=time_grid,
            )
            print("ibs", round(ibs, 3))

            c_index = concordance_index_ipcw(
                y_train,
                y_test,
                estimate=(1 - y_surv_pred)[:, 3],
                tau=time_grid[3]
            )[0]
            print("cindex 0.5", round(c_index, 3))

            all_ibs.append(ibs)
            all_cindex.append(c_index)

        print(f"ibs: {np.mean(all_ibs):.3f}+-{np.std(all_ibs):.3f}")
        print(f"cindex: {np.mean(all_cindex):.3f}+-{np.std(all_cindex):.3f}")



def run_gbmi():    
    from sklearn.model_selection import train_test_split
    from pycox.datasets import metabric, support
    from sksurv.metrics import integrated_brier_score, concordance_index_ipcw
    from hazardous._fine_and_gray import FineGrayEstimator
    from hazardous._aalen_johansen import AalenJohansenEstimator

    from hazardous.data._seer import load_seer, CATEGORICAL_COLUMN_NAMES, NUMERIC_COLUMN_NAMES

    # df_metabric = metabric.read_df()
    # df_support = support.read_df()
    X, y = load_seer(
        input_path="../hazardous/data/seer_cancer_cardio_raw_data.txt",
        return_X_y=True,
        survtrace_preprocessing=True,
    ) 
    X = X[CATEGORICAL_COLUMN_NAMES + NUMERIC_COLUMN_NAMES]
    X = X.dropna()
    y = y.iloc[X.index]
    print(X.shape, y.shape)
    print(X.isna().sum())

    print("-"* 10)
    target_columns = ["event", "duration"]

    all_ibs, all_cindex, fit_time = [], [], []

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    tic = time()
    aj = AalenJohansenEstimator(calculate_variance=False)
    aj.fit(X_train, y_train)
    toc = time()
    fit_time.append(toc - tic)
    print(toc - tic)
    
    print("res", np.mean(fit_time), np.std(fit_time))

    #     y_surv_pred = deephit.predict_survival_function(X_test).T.values[:, 1:-1]
    #     print(y_surv_pred.shape)
    #     print(y_surv_pred)

    #     y_train, y_test = make_recarray(y_train), make_recarray(y_test)
    #     time_grid = deephit.labtrans.cuts[1:-1]
    #     print(time_grid)

    #     ibs = integrated_brier_score(
    #         survival_train=y_train,
    #         survival_test=y_test,
    #         estimate=y_surv_pred,
    #         times=time_grid,
    #     )
    #     print("ibs", round(ibs, 3))

    #     c_index = concordance_index_ipcw(
    #         y_train,
    #         y_test,
    #         estimate=(1 - y_surv_pred)[:, 3],
    #         tau=time_grid[3]
    #     )[0]
    #     print("cindex 0.5", round(c_index, 3))

    #     all_ibs.append(ibs)
    #     all_cindex.append(c_index)

    # print(f"ibs: {np.mean(all_ibs):.3f}+-{np.std(all_ibs):.3f}")
    # print(f"cindex: {np.mean(all_cindex):.3f}+-{np.std(all_cindex):.3f}")



def run_rsf():    
    from sklearn.model_selection import train_test_split
    from pycox.datasets import metabric, support
    from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

    from hazardous.utils import make_time_grid

    df_metabric = metabric.read_df()
    df_support = support.read_df()
    
    for df in [df_metabric, df_support]:
        print("-"* 10)
        target_columns = ["event", "duration"]
        y = df[target_columns]
        X = df[list(set(df.columns) - set(target_columns))]

        all_ibs, all_cindex, fit_time = [], [], []

        for random_state in range(5):
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
            y_train, y_test = make_recarray(y_train), make_recarray(y_test)

            tic = time()
            rsf = RandomSurvivalForest().fit(X_train, y_train)
            toc = time()
            fit_time.append(toc - tic) 
            print(toc - tic)

            # y_surv_pred = rsf.predict_survival_function(X_test, return_array=False)

            # time_grid = make_time_grid(y_test["t"], n_steps=10)[1:-1]
            # y_surv_pred = np.vstack([y_(time_grid) for y_ in y_surv_pred])

            # ibs = integrated_brier_score(
            #     survival_train=y_train,
            #     survival_test=y_test,
            #     estimate=y_surv_pred,
            #     times=time_grid,
            # )
            # print("ibs", round(ibs, 3))

            # c_index = concordance_index_ipcw(
            #     y_train,
            #     y_test,
            #     estimate=(1 - y_surv_pred)[:, 3],
            #     tau=time_grid[3]
            # )[0]
            # print("cindex 0.5", round(c_index, 3))

            # all_ibs.append(ibs)
            # all_cindex.append(c_index)

        # print(f"ibs: {np.mean(all_ibs)}+-{np.std(all_ibs)}")
        # print(f"cindex: {np.mean(all_cindex)}+-{np.std(all_cindex)}")

        print("res", np.mean(fit_time), np.std(fit_time))

# %%
# main()
#run_deephit()
run_gbmi()
# %%
