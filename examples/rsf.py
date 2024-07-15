# %%
import seaborn as sns
from hazardous._rsf import RSFEstimator
from time import time

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from sklearn.model_selection import train_test_split

seed = 1
independent_censoring = False
complex_features = False

bunch = make_synthetic_competing_weibull(
    n_samples=5000,
    n_events=3,
    return_X_y=True,
    censoring_relative_scale=1.5,
    random_state=seed,
)
X, y= bunch

censoring_rate = (y["event"] == 0).mean()
censoring_kind = "independent" if independent_censoring else "dependent"
ax = sns.histplot(
    y,
    x="duration",
    hue="event",
    multiple="stack",
    palette="magma",
)
ax.set_title(f"independent_censoring, censoring rate {censoring_rate:.2%}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


time_deb = time()
rsf = RSFEstimator()


rsf.fit(X_train, y_train)
time_fin = time()
res = rsf.predict_cumulative_incidence(X_test)
print(f'Time to train on {X_train.shape[0]} samples: {time_fin - time_deb}')

# %%
