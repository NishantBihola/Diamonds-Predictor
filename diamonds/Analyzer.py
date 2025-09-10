# diamonds/Analyzer.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Analyzer:
    def __init__(self, csv_path=None, df=None):
        if df is not None:
            self.df = df.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Provide csv_path or df")

    # -----------------------
    # Correlation matrix
    def plot_correlationMatrix(self, out_path=None):
        corr = self.df.corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            print(f"Correlation matrix saved at {out_path}")
        plt.show()

    # -----------------------
    # Histograms of categorical variables
    def plot_histograms_categorical(self, out_path=None):
        cat_cols = self.df.select_dtypes(include="object").columns
        n_cols = 3
        n_rows = (len(cat_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols,4*n_rows))
        axes = axes.flatten()
        for i, col in enumerate(cat_cols):
            sns.countplot(data=self.df, x=col, ax=axes[i])
            axes[i].set_title(col)
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            print(f"Categorical histograms saved at {out_path}")
        plt.show()

    # -----------------------
    # Boxplots for all numeric features
    def plot_boxPlot(self, out_path=None):
        num_cols = self.df.select_dtypes(include="number").columns
        n_cols = 3
        n_rows = (len(num_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols,4*n_rows))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.boxplot(y=self.df[col], ax=axes[i])
            axes[i].set_title(col)
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            print(f"Boxplots saved at {out_path}")
        plt.show()

    # -----------------------
    # Pairplot for numeric features
    def plot_pairPlot(self, out_path=None):
        num_cols = self.df.select_dtypes(include="number").columns
        pair_plot = sns.pairplot(self.df[num_cols])
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pair_plot.savefig(out_path)
            print(f"Pairplot saved at {out_path}")
        plt.show()
