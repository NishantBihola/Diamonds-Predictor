import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Classifier:
    def __init__(self):
        self.model = None
        self.model_name = None

    def fit(self, X, y, model_name="random_forest"):
        self.model_name = model_name
        if model_name == "random_forest":
            self.model = RandomForestClassifier(random_state=42)
        elif model_name == "svc":
            self.model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        self.model.fit(X, y)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)

    def score(self, X, y, metric="accuracy"):
        y_pred = self.predict(X)
        if metric == "accuracy":
            return accuracy_score(y, y_pred)
        elif metric == "f1":
            return f1_score(y, y_pred, average="weighted")
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def plot_confusionMatrix(self, y_true, y_pred, labels=None, out_path=None):
        if labels is None:
            labels = np.unique(np.concatenate((y_true, y_pred)))
        labels = list(labels)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            print(f"Confusion matrix saved at {out_path}")
        plt.show()
        return cm
