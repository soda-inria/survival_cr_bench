import numpy as np
import torch
import torchtuples as tt
from pycox.models import PCHazard
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
    integrated_brier_score_incidence_oracle,
)

SEED = 0

np.random.seed(1234)
_ = torch.manual_seed(1234)


class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """

    def __init__(
        self,
        in_features,
        num_nodes_shared,
        num_nodes_indiv,
        num_risks,
        out_features,
        batch_norm=True,
        dropout=None,
    ):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features,
            num_nodes_shared[:-1],
            num_nodes_shared[-1],
            batch_norm,
            dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1],
                num_nodes_indiv,
                out_features,
                batch_norm,
                dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out


def get_x(df):
    return df.values.astype("float32")


def get_target(df):
    return (
        df["duration"].astype("float32").values,
        df["event"].astype("int32").values,
    )


class PCHazardEstimator(tt.Model):
    def __init__(
        self,
        num_nodes=[32, 32],
        batch_size=256,
        epochs=512,
        callbacks=[tt.callbacks.EarlyStoppingCycle()],
        verbose=False,
        num_durations=10,
        batch_norm=True,
        dropout=None,
        lr=0.005,
    ):
        self.num_durations = num_durations
        self.num_nodes = num_nodes
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.verbose = verbose
        self.lr = lr

    def fit(self, X, y):
        X_train_, X_val_, y_train_, y_val_ = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

        X_train = get_x(X_train_)
        X_val = get_x(X_val_)
        y_train = get_target(y_train_)
        y_val = get_target(y_val_)

        self.labtrans = PCHazard.label_transform(self.num_durations)

        y_train = self.labtrans.fit_transform(*y_train)
        y_val = self.labtrans.transform(*y_val)
        # TODO : REMOVE THIS
        self.y_train = y_train
        self.in_features = X_train.shape[1]
        self.num_risks = y_train[1].max()
        self.times_ = self.labtrans.cuts
        self.net = tt.practical.MLPVanilla(
            in_features=self.in_features, 
            num_nodes=self.num_nodes,
            out_features=self.labtrans.out_features, 
            batch_norm=self.batch_norm, 
            dropout=self.dropout)

        self.optimizer = tt.optim.AdamWR(
            lr=self.lr, decoupled_weight_decay=0.01, cycle_eta_multiplier=0.8
        )

        self.model = PCHazard(
            net=self.net,
            optimizer=self.optimizer,
            duration_index=self.labtrans.cuts,
        )
        self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=self.verbose,
            val_data=(X_val, y_val),
        )
        return self

    def predict_survival_function(self, X):
        X_ = get_x(X)
        return self.model.predict_surv_df(X_)

    def predict_cumulative_incidence(self, X, times=None):
        X_ = get_x(X)
        df_pred = self.model.predict_surv_df(X_)
        self.times_ = df_pred.index.values
        y_pred = np.swapaxes(df_pred.values, 0, 1)
        y_pred = y_pred[:, None, :]
        cif_pred = 1 - y_pred.sum(axis=1)
        cif_pred = cif_pred[:, None, :]
        y_pred = np.concatenate((y_pred, cif_pred), axis=1)

        all_event_y_pred = []

        for event in range(y_pred.shape[1]):
            event_pred = y_pred[:, event, :]
            if event == 0:
                y_pred_at_0 = np.ones((event_pred.shape[0], 1))
            else:
                y_pred_at_0 = np.zeros((event_pred.shape[0], 1))
        
            y_pred_t_max = event_pred[:, [-1]]
            event_pred = np.hstack([y_pred_at_0, event_pred, y_pred_t_max])

            if times is None:
                times = self.times_
            
            times_event = np.hstack([[0], self.times_, [np.inf]])

            all_y_pred = []
            for idx in range(event_pred.shape[0]):
                y_pred_ = interp1d(
                    x=times_event,
                    y=event_pred[idx, :],
                    kind="linear",
                )(times)
                all_y_pred.append(y_pred_)

            event_pred = np.vstack(all_y_pred)
            all_event_y_pred.append(event_pred)
        
        all_event_y_pred = np.asarray(all_event_y_pred)
        return all_event_y_pred
    

    def predict_proba(self, X):
        X_ = get_x(X)
        return self.model.predict_pmf(X_)

    def score(self, X, y, scale_censoring=None, shape_censoring=None):
        X_ = get_x(X)
        predicted_curves = self.model.predict_cif(X_)
        ibs_events = []
        for event in range(len(predicted_curves)):
            predicted_curves_for_event = predicted_curves[event]
            if scale_censoring is not None and shape_censoring is not None:
                ibs_event = integrated_brier_score_incidence_oracle(
                    y_train=self.y_train,
                    y_test=y,
                    y_pred=predicted_curves_for_event,
                    times=self.labtrans.cuts,
                    shape_censoring=shape_censoring,
                    scale_censoring=scale_censoring,
                    event_of_interest=event + 1,
                )

            else:
                ibs_event = integrated_brier_score_incidence(
                    y_train=self.y_train,
                    y_test=y,
                    y_pred=predicted_curves_for_event,
                    times=self.labtrans.cuts,
                    event_of_interest=event + 1,
                )

            ibs_events.append(ibs_event)
        return -np.mean(ibs_events)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        del deep
        out = dict()
        out["num_durations"] = self.num_durations
        #out["num_nodes_shared"] = self.num_nodes_shared
        #out["num_nodes_indiv"] = self.num_nodes_indiv
        out["batch_norm"] = self.batch_norm
        out["dropout"] = self.dropout
        #out["alpha"] = self.alpha
        #out["sigma"] = self.sigma
        out["optimizer"] = self.optimizer
        out["batch_size"] = self.batch_size
        out["epochs"] = self.epochs
        out["callbacks"] = self.callbacks
        out["verbose"] = self.verbose
        return out

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Returns
        -------
        self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
