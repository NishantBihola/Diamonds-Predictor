"""
main.py
An example runner for the diamonds capstone:
- loads dataset
- runs Analyzer EDA
- trains classifier on 'clarity'
- trains regressor on 'price'
- runs clustering
"""

import os
import pandas as pd
from diamonds.Analyzer import Analyzer
from diamonds.Classifier import Classifier
from diamonds.Regressor import Regressor
from diamonds.Clustering import Clustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/diamonds.csv"  # set your path

def main():
    # 1. Load and basic preprocessing
    analyzer = Analyzer(csv_path=DATA_PATH)
    print("Dataset shape:", analyzer.df.shape)
    # Drop Unnamed: 0 if present
    analyzer.drop_columns(["Unnamed: 0"])
    analyzer.shuffle(seed=42)

    # quick EDA: save correlation matrix
    os.makedirs("outputs/plots", exist_ok=True)
    analyzer.plot_correlationMatrix(out_dir="outputs/plots")

    # Encode categorical features: cut, color, clarity
    cat_cols = []
    for c in ["cut", "color", "clarity"]:
        if c in analyzer.df.columns:
            cat_cols.append(c)
    analyzer.encode_features(cat_cols)

    # prepare data for classification (clarity)
    if "clarity" not in analyzer.df.columns:
        raise RuntimeError("clarity column not found")

    # Features/target for classification
    X_clf = analyzer.df.drop(columns=["clarity"])
    y_clf = analyzer.df["clarity"]

    # train/test split
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    # scale numeric features
    scaler = StandardScaler()
    Xc_train = scaler.fit_transform(Xc_train)
    Xc_test = scaler.transform(Xc_test)

    # Classifier: try a few models
    clf = Classifier()
    clf.fit(Xc_train, yc_train, model_name="random_forest")
    acc = clf.score(Xc_test, yc_test, metric="accuracy")
    print("Classifier (random_forest) accuracy:", acc)

    # plot confusion matrix
    ypred = clf.predict(Xc_test)
    clf.plot_confusionMatrix(yc_test, ypred, labels=None, out_path="outputs/plots/confusion_clarity.png")

    # ---- Regression: predict 'price' ----
    if "price" not in analyzer.df.columns:
        raise RuntimeError("price column not found")
    # drop price and clarity if present, keep numeric/categorical encoded
    X_reg = analyzer.df.drop(columns=["price"])
    y_reg = analyzer.df["price"].astype(float)

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    scaler_r = StandardScaler()
    Xr_train = scaler_r.fit_transform(Xr_train)
    Xr_test = scaler_r.transform(Xr_test)

    reg = Regressor()
    reg.fit(Xr_train, yr_train, model_name="random_forest")
    r2 = reg.score(Xr_test, yr_test, metric="r2")
    rmse = reg.score(Xr_test, yr_test, metric="RMSE")
    print(f"Regressor (random_forest) R2: {r2:.4f} | RMSE: {rmse:.2f}")

    # ---- Clustering ----
    # cluster using a subset of numeric features
    numeric = analyzer.df.select_dtypes(include=['number'])
    # scale numeric
    from sklearn.preprocessing import StandardScaler
    ns = StandardScaler()
    X_num = ns.fit_transform(numeric)
    cl = Clustering()
    cl.fit(X_num, model_name="kmeans", n_clusters=4)
    labels = cl.model.labels_
    print("Clustering sample labels counts:", pd.Series(labels).value_counts().to_dict())

    print("Done. Outputs saved to outputs/plots/")

if __name__ == "__main__":
    main()
