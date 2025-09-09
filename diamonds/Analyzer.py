"""
Analyzer.py
- load dataset
- drop columns
- encode categorical features
- shuffle, sample
- plotting functions: correlation matrix, histograms categorical, boxplot, pairplot
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import save_plot
from sklearn.preprocessing import LabelEncoder

class Analyzer:
    def __init__(self, csv_path=None, df: pd.DataFrame=None):
        if df is not None:
            self.df = df.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Provide csv_path or df")
        self.encoders = {}

    def info(self):
        return self.df.info()

    def head(self, n=5):
        return self.df.head(n)

    def drop_columns(self, cols):
        self.df.drop(columns=cols, inplace=True, errors='ignore')
        return self.df

    def encode_features(self, cols):
        """Fit LabelEncoder per column and replace in df. Save encoders."""
        for c in cols:
            le = LabelEncoder()
            self.df[c] = le.fit_transform(self.df[c].astype(str))
            self.encoders[c] = le
        return self.df

    def encode_target(self, col):
        le = LabelEncoder()
        y = le.fit_transform(self.df[col].astype(str))
        self.encoders[col] = le
        return y

    def shuffle(self, seed=42):
        self.df = self.df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return self.df

    def sample(self, reduction_factor=1.0, seed=42):
        if not (0.0 < reduction_factor <= 1.0):
            raise ValueError("reduction_factor in (0.0, 1.0]")
        n = int(len(self.df) * reduction_factor)
        self.df = self.df.sample(n=n, random_state=seed).reset_index(drop=True)
        return self.df

    # ---- plotting ----
    def plot_correlationMatrix(self, out_dir="outputs/plots", figsize=(10,8), annot=False):
        numeric = self.df.select_dtypes(include=[np.number])
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=annot, cmap="coolwarm", ax=ax)
        path = save_plot(fig, out_dir, "correlation_matrix.png")
        return path

    def plot_histograms_categorical(self, cols=None, out_dir="outputs/plots", bins=20):
        if cols is None:
            cols = self.df.select_dtypes(include=['object','category']).columns.tolist()
        saved=[]
        for c in cols:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(y=c, data=self.df, ax=ax, order=self.df[c].value_counts().index)
            ax.set_title(f"Histogram: {c}")
            path = save_plot(fig, out_dir, f"hist_{c}.png")
            saved.append(path)
        return saved

    def plot_boxPlot(self, cols=None, out_dir="outputs/plots"):
        if cols is None:
            cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        saved=[]
        for c in cols:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(x=self.df[c], ax=ax)
            ax.set_title(f"Boxplot: {c}")
            path = save_plot(fig, out_dir, f"box_{c}.png")
            saved.append(path)
        return saved

    def plot_pairPlot(self, cols=None, out_dir="outputs/plots", sample=2000):
        if cols is None:
            cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        df = self.df[cols].sample(n=min(sample, len(self.df)), random_state=42)
        # pairplot can be heavy, we save it
        pp = sns.pairplot(df)
        path = os.path.join(out_dir, "pairplot.png")
        os.makedirs(out_dir, exist_ok=True)
        pp.fig.savefig(path)
        plt.close(pp.fig)
        return path

    # helper to split into features and target
    def get_X_y(self, target_col, drop_cols=None):
        df = self.df.copy()
        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y
