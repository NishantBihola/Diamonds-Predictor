"""
Regressor.py
Wraps scikit-learn regressors and a Keras ANN regressor.
Functions:
- fit(X_train, y_train, model_name)
- predict(X)
- score(X,y, metric)  # metrics: r2, MSE, RMSE, MAE
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import models, layers

class Regressor:
    def __init__(self):
        self.sk_model = None
        self.model = None
        self.name = None

    def fit(self, X_train, y_train, model_name="linear", ann_params=None):
        model_name = model_name.lower()
        self.name = model_name
        if model_name == "linear":
            m = LinearRegression()
            m.fit(X_train, y_train)
            self.sk_model = m
        elif model_name == "knn":
            m = KNeighborsRegressor()
            m.fit(X_train, y_train)
            self.sk_model = m
        elif model_name == "decision_tree":
            m = DecisionTreeRegressor()
            m.fit(X_train, y_train)
            self.sk_model = m
        elif model_name == "random_forest":
            m = RandomForestRegressor(n_estimators=100)
            m.fit(X_train, y_train)
            self.sk_model = m
        elif model_name == "svr":
            m = SVR()
            m.fit(X_train, y_train)
            self.sk_model = m
        elif model_name == "ann":
            params = ann_params or {"hidden_units":[128,64], "lr":1e-3, "epochs":50, "batch_size":32}
            model = models.Sequential()
            model.add(layers.Input(shape=(X_train.shape[1],)))
            for u in params["hidden_units"]:
                model.add(layers.Dense(u, activation="relu"))
                model.add(layers.Dropout(0.2))
            model.add(layers.Dense(1, activation="linear"))
            model.compile(optimizer=tf.keras.optimizers.Adam(params["lr"]), loss="mse", metrics=["mae"])
            model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"], verbose=1)
            self.model = model
        else:
            raise ValueError(f"Unknown model {model_name}")
        return self

    def predict(self, X):
        if self.sk_model is not None:
            return self.sk_model.predict(X)
        elif self.model is not None:
            return self.model.predict(X).ravel()
        else:
            raise RuntimeError("No model trained")

    def score(self, X, y, metric="r2"):
        y_pred = self.predict(X)
        if metric == "r2":
            return r2_score(y, y_pred)
        elif metric == "MSE":
            return mean_squared_error(y, y_pred)
        elif metric == "RMSE":
            return np.sqrt(mean_squared_error(y, y_pred))
        elif metric == "MAE":
            return mean_absolute_error(y, y_pred)
        else:
            raise ValueError("Unknown metric")
