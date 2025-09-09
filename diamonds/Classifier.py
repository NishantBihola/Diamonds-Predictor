"""
Classifier.py
A convenience class wrapping scikit-learn classifiers and a Keras ANN classifier.
Provides:
- fit(name)
- predict(X)
- score(X,y, metric='accuracy')
- plot_confusionMatrix(y_true, y_pred)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

class Classifier:
    def __init__(self):
        self.model = None
        self.name = None
        self.sk_model = None

    def fit(self, X_train, y_train, model_name="logistic", ann_params=None):
        model_name = model_name.lower()
        self.name = model_name

        if model_name == "logistic":
            m = LogisticRegression(max_iter=200)
            m.fit(X_train, y_train)
            self.sk_model = m
        elif model_name == "knn":
            m = KNeighborsClassifier()
            m.fit(X_train, y_train)
            self.sk_model = m
        elif model_name == "decision_tree":
            m = DecisionTreeClassifier()
            m.fit(X_train, y_train)
            self.sk_model = m
        elif model_name == "random_forest":
            m = RandomForestClassifier(n_estimators=100)
            m.fit(X_train, y_train)
            self.sk_model = m
        elif model_name == "svc":
            m = SVC(probability=True)
            m.fit(X_train, y_train)
            self.sk_model = m
        elif model_name == "ann":
            # simple dense ANN for classification
            ipt_dim = X_train.shape[1]
            params = ann_params or {"hidden_units":[128,64], "lr":1e-3, "epochs":20, "batch_size":32}
            model = models.Sequential()
            model.add(layers.Input(shape=(ipt_dim,)))
            for u in params["hidden_units"]:
                model.add(layers.Dense(u, activation="relu"))
                model.add(layers.Dropout(0.2))
            model.add(layers.Dense(len(np.unique(y_train)), activation="softmax"))
            model.compile(optimizer=tf.keras.optimizers.Adam(params["lr"]),
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"], verbose=1)
            self.model = model
        else:
            raise ValueError(f"Unknown model {model_name}")
        return self

    def predict(self, X):
        if self.sk_model is not None:
            return self.sk_model.predict(X)
        elif self.model is not None:
            probs = self.model.predict(X)
            return probs.argmax(axis=1)
        else:
            raise RuntimeError("No model trained")

    def predict_proba(self, X):
        if self.sk_model is not None:
            if hasattr(self.sk_model, "predict_proba"):
                return self.sk_model.predict_proba(X)
            else:
                # fallback: decision function to probs
                return None
        elif self.model is not None:
            return self.model.predict(X)
        else:
            raise RuntimeError("No model trained")

    def score(self, X, y, metric="accuracy"):
        y_pred = self.predict(X)
        if metric == "accuracy":
            return accuracy_score(y, y_pred)
        elif metric == "f1":
            return f1_score(y, y_pred, average="weighted")
        else:
            raise ValueError("metric must be 'accuracy' or 'f1'")

    def plot_confusionMatrix(self, y_true, y_pred, labels=None, out_path=None):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix: {self.name}")
        if out_path:
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            return out_path
        else:
            plt.show()
            return fig
