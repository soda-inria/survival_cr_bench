# survival_cr_bench

This repository contains the code to reproduce the benchmark of the paper "Teaching Models To Survive: Proper Scoring Rule and Stochastic Optimization with Competing Risks" by Alberge et al.

This project is an archive and won't be maintained.

Installing this project will allow you to run the following models (table A):


| Name                    | Competing risks | Proper loss | Implementation                                                                                                                                                      | Reference                                                                                   |
|-------------------------|-----------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| MultiIncidence          | ✔️               | ✔️           | ours                                                                                                                                                                 |Alberge et al.                                                                                 | 
| SurvTRACE               | ✔️               |             | ours, based on [github.com/RyanWangZf/SurvTRACE](https://github.com/RyanWangZf/SurvTRACE)                                                                           | [Wang and Sun (2022)](https://arxiv.org/abs/2110.00855)                                     |
| DeepHit                 | ✔️               |             | [github.com/havakv/pycox](https://github.com/havakv/pycox)                                                                                                          | [Lee et al. (2018)](https://ojs.aaai.org/index.php/AAAI/article/view/11842)                 |
| Random Survival Forests | ✔️               |             | [scikit-survival.readthedocs.io](https://scikit-survival.readthedocs.io/) for survival <br> and [randomforestsrc.org](www.randomforestsrc.org/) for competing risks |   [Ishwaran et al. (2014)](https://academic.oup.com/biostatistics/article/15/4/757/266340)                                                                                           |
| Fine and Gray           | ✔️               |             | [cran.r-project.org/package=cmprsk](cran.r-project.org/package=cmprsk)                                                                                              | [Fine and Gray (1999)](https://www.tandfonline.com/doi/epdf/10.1080/01621459.1999.10474144) |
| Aalen-Johansen          | ✔️               |             | ours, based on [lifelines.readthedocs.io](https://lifelines.readthedocs.io/)                                                                                        | [Aalen et al. (2008)](http://link.springer.com/10.1007/978-0-387-68560-1)                   |
| PCHazard                |                 |             | [github.com/havakv/pycox](https://github.com/havakv/pycox)                                                                                                          | [Kvamme and Borgan (2019b)](https://arxiv.org/abs/1910.06724)                               |


We also benchmark the following models, by adding some snippets to authors' code on our forked version (table B):


| Name                    | Competing risks | Proper loss | Implementation                                                                                                                                                      | Reference                                                                                   |
|-------------------------|-----------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| DSM                     | ✔️               |             | [autonlab.github.io/DeepSurvivalMachines](autonlab.github.io/DeepSurvivalMachines)                                                                                  | [Nagpal et al. (2021)](https://arxiv.org/abs/2003.01176)                                    |
| DeSurv                  | ✔️               |             | [github.com/djdanks/DeSurv](https://github.com/djdanks/DeSurv)                                                                                                      | [Danks and Yau (2022)](https://proceedings.mlr.press/v151/danks22a.html)                    |
| Han et al.              |                 |             | [github.com/rajesh-lab/Inverse-Weighted-Survival-Games](https://github.com/rajesh-lab/Inverse-Weighted-Survival-Games)                                              | [Han et al. (2021)](https://arxiv.org/abs/2111.08175) 
| Sumo-Net                |                 | ✔️           | [github.com/MrHuff/Sumo-Net](https://github.com/MrHuff/Sumo-Net)                                                                                                    | [Rindt et al. (2022)](https://arxiv.org/abs/2103.14755)                                     |
| DQS                     |                 | ✔️           | [ibm.github.io/dqs](ibm.github.io/dqs)                                                                                                                              | [Yanagisawa (2023)](https://arxiv.org/abs/2305.00621)                                       |

We used the following datasets (table C):

| Name              | Competing risks | Source                                                              | Need a license |
|-------------------|-----------------|---------------------------------------------------------------------|----------------|
| synthetic dataset | ✔️               | ours                                                                |                |
| Metabric          |                 | [pycox](https://github.com/havakv/pycox#datasets)                   |                |
| Support           |                 | [pycox](https://github.com/havakv/pycox#datasets)                   |                |
| SEER              | ✔️               | [NIH](https://soda-inria.github.io/hazardous/downloading_seer.html) | ✔️              |

See the [setup section](#13-download-the-seer-dataset) to learn how to download SEER.


## 1. Setup

#### 1.1 Clone the project

```shell
git clone git@github.com:soda-inria/survival_cr_bench.git
cd survival_cr_bench/ 
```

#### 1.2 Create a Python environment and install locally

Create and activate an environment, e.g.:

```shell
python -m venv <your_env_name>
source <your_env_name>/bin/activate
```

Then perform the local installation:

```shell
pip install -e .
```

#### 1.3 Download the SEER dataset

To use the SEER dataset in the benchmarks, you first have to make a request and be approved by the NIH before downloading it. [Here is a tutorial](https://soda-inria.github.io/hazardous/downloading_seer.html).

**Note that the waiting period can be up to several days.**


## 2. Run the benchmarks

#### 2.1 Benchmarks on this repository (table A)

##### 2.1.1 Running the hyperparameter search

We provide the best parameters that we found during our hyper parameters search.
To re-run this search operation, use the following:

```shell
cd benchmark
```
Then in a python shell:

```python
from hyper_parameter_search import search_all_dataset_params

search_all_dataset_params(dataset_name="seer", model_name="gbmi")
```

See `benchmark/_dataset.py` and `benchmark/_model.py` to see the options.

Running this function will create two files in the folder `benchmark/best_hyper_parameters/<model_name>/<dataset_name>/<dataset_params>`.

- the best hyper-parameters of the cross-validated model (`best_params.json`)
- the parameters used to generate the dataset (`dataset_params.json`).

##### 2.1.2 Fitting and creating the results


```python
from evaluate import evaluate_all_models

evaluate_all_models()
```

This will fit all models present in the `best_hyper_parameters` folder with their best hyper parameters, for each random seed.

Then, each fitted model is evaluated against the test set to compute metrics. The metrics are written at `benchmark/scores/raw/<model_name>/<dataset_name>.json` for each seed.

Finally, for each model and each dataset, the results are aggregated seed-wise and written at `benchmark/scores/agg/<model_name>/<dataset_name>.json`.

These aggregated metrics will be used to plot figures.


#### 2.2 Benchmarks on authors' repository (table B)

To run models from the table B, you have to go to the specific submodule, e.g. for `dqs`

```shell
cd dqs
```

Then, read the corresponding README of these projects to run their benchmark.


## 3. Display plots

As we already provide the results from our benchmarks, you don't have to run them all in order to reproduce the figures.

Each file corresponds to a figure introduced in the paper, with its number in the file name. Running a display file will create a .png file at the root of this repository.

```shell
python benchmark/display/display_06_brier_score.py
```