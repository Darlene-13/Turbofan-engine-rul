# Turbofan Engine Remaining Useful Life (RUL) Prediction

An end-to-end Machine Learning project that predicts how many operational cycles remain before a turbofan engine fails — from raw sensor data all the way to a deployed API.

---

## Project Overview

This project uses the **NASA C-MAPS Turbofan Engine Degradation Simulation** dataset. Engines are fitted with multiple sensors that record readings across every operational cycle. The goal is to predict the **Remaining Useful Life (RUL)** — the number of cycles left before an engine breaks down.

We start with **FD001** (the simplest subset: 1 operating condition, 1 fault mode) and build the full pipeline before scaling to the other subsets.

---

## Dataset

- **Source:** [NASA C-MAPS via Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
- **Subsets used:** FD001 (primary), FD002, FD003, FD004
- **Features:** 21 sensor readings + 3 operational settings per cycle
- **Target:** Remaining Useful Life (RUL) — engineered from the training data

---

## Project Structure

```
├── data/               # Raw and processed datasets
├── notebooks/          # EDA and experimentation notebooks
├── src/                # Source code (preprocessing, training, evaluation)
├── models/             # Saved trained models
├── api/                # FastAPI prediction endpoint
├── app/                # Django web application
└── README.md
```

---

## Pipeline

1. **Exploratory Data Analysis (EDA)** — understand sensor behaviour and degradation patterns
2. **Feature Engineering** — create RUL labels, drop useless sensors, scale features
3. **Modelling** — train and evaluate ML models (baseline → advanced)
4. **Model Packaging** — save the best model for serving
5. **FastAPI** — expose predictions via a REST endpoint
6. **Django** — wrap the service in a web application

---

## Tech Stack

- **Python** — core language
- **Pandas, NumPy, Matplotlib, Seaborn** — data handling and visualisation
- **Scikit-learn, XGBoost** — modelling
- **FastAPI** — prediction API
- **Django** — web application layer




## Written by:
Darlene Wendy