
# Diamonds Predictor Application (Capstone)

## Overview
This repository implements a mini machine learning platform that:
- Performs Exploratory Data Analysis (EDA)
- Builds classification models to predict diamond clarity
- Builds regression models to estimate diamond price
- Performs clustering for segmentation

## Files
- `diamonds/` : Python package with modules:
  - `Analyzer.py` : EDA, preprocessing, plotting
  - `Classifier.py` : classification wrapper (sklearn + Keras ANN)
  - `Regressor.py` : regression wrapper (sklearn + Keras ANN)
  - `Clustering.py` : clustering wrappers
  - `utils.py` : small helpers
- `main.py` : example runner script
- `notebooks/Diamonds_Predictor.ipynb` : demonstration notebook (execute and save outputs)
- `data/diamonds.csv` : place the dataset here (not included)
- `outputs/` : plots and saved artifacts

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
