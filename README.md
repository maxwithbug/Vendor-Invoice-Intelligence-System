# Freight Cost Prediction

This repository contains a freight cost prediction project built with Python and scikit-learn. The project uses vendor invoice data from an SQLite database to train a model that predicts freight cost based on invoice values.

## Project Structure

- `Copy of sonar data.csv` - sample dataset file stored at the root
- `data/` - local data storage
  - `inventory.db` - SQLite database used by the project
- `notebooks/` - Jupyter notebooks and project code
  - `Predicting Freight Cost.ipynb` - notebook for analysis and modeling
  - `Untitled.ipynb` - extra notebook
  - `freight_cost_prediction/` - Python modules for the prediction workflow
    - `data_preprocessing.py` - data loading, feature selection, and train/test split
    - `train.py` - training script for the prediction model
    - `modeling_evaluation.py` - evaluation helper functions
    - `models/` - saved model artifacts (ignored by git)

## Setup

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install pandas scikit-learn jupyter
```

## Usage

Run the training script from the project folder:

```bash
cd notebooks/freight_cost_prediction
python train.py
```

Or open the notebook in `notebooks/Predicting Freight Cost.ipynb` to explore the data and model results.

## Notes

- `data/*.db` and model artifacts are excluded from version control via `.gitignore`.
- Keep source data and notebooks tracked, but avoid committing generated files and environment artifacts.
