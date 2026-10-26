# Titanic Survival Prediction Using Random Forest Classification

## Project Overview
This project builds an end-to-end machine learning pipeline to predict passenger survival on the RMS Titanic using Python, Pandas, Matplotlib, and scikit-learn. Moving beyond basic decision trees, this implementation incorporates advanced feature engineering (extracting social titles, family sizes, and solo traveler indicators) and an ensemble **Random Forest Classifier** to maximize predictive accuracy.

## Problem Statement
Using historical passenger data from the Titanic disaster, the objective is to build a binary classification model that predicts whether a given passenger survived (`1`) or did not survive (`0`) based on demographic and ticketing attributes.

## Dataset Source
* **Official Kaggle Competition:** [Titanic – Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)
* **Files Used:** `train.csv` (891 passenger records with survival labels).

## Features & Target Variable
* **Target:** `Survived` (0 = No, 1 = Yes)
* **Input Features:**
  * `Pclass` — Ticket class (1st, 2nd, 3rd)
  * `Sex` — Passenger gender
  * `Age` — Passenger age (imputed with training median)
  * `Fare` — Ticket fare (imputed with training median)
  * `Title` — Extracted social title from passenger name (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`)
  * `FamilySize` — Total family members aboard (`SibSp` + `Parch` + 1)
  * `IsAlone` — Binary indicator for solo travelers

## Technologies Used
* **Python 3**
* **Pandas & NumPy** — Data manipulation and numerical operations
* **Matplotlib** — Data visualization and feature importance plotting
* **scikit-learn** — Preprocessing pipelines, imputation, encoding, and Random Forest modeling
* **Jupyter Notebook** — Interactive educational walkthrough
* **Git & GitHub** — Version control and repository publishing

## Repository Structure
titanic-decision-tree/
│
├── data/
│   └── train.csv
├── notebooks/
│   └── titanic_analysis.ipynb
├── src/
│   └── train_model.py
├── outputs/
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   └── metrics.json
├── screenshots/
│   └── .gitkeep
├── requirements.txt
├── .gitignore
└── README.md
