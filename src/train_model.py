import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def extract_title(name):
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return "Unknown"

def main():
    print("1. Loading dataset...")
    data_path = os.path.join("data", "train.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}.")
    
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    # 2. Advanced Feature Engineering
    print("2. Engineering features (Title, FamilySize, IsAlone)...")
    df['Title'] = df['Name'].apply(extract_title)
    
    # Consolidate rare or redundant titles
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Expanded feature set
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'FamilySize', 'IsAlone']
    target = 'Survived'
    
    X = df[features]
    y = df[target]

    # 3. Train / Validation Split (80/20, stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    # 4. Preprocessing Pipeline
    numeric_features = ['Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone']
    categorical_features = ['Sex', 'Title']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 5. Ensemble Pipeline (Random Forest Classifier)
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    print("Training Optimized Random Forest model...")
    clf.fit(X_train, y_train)

    # 6. Evaluation on Validation Data
    y_pred = clf.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    print("\n--- Advanced Validation Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # 7. Outputs & Persistence
    os.makedirs("outputs", exist_ok=True)
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
    with open(os.path.join("outputs", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 8. Feature Importance Analysis
    cat_encoder = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + encoded_cat_features

    importances = clf.named_steps['classifier'].feature_importances_
    fi_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    fi_df.to_csv(os.path.join("outputs", "feature_importance.csv"), index=False)

    print("\n--- Top Feature Importances ---")
    print(fi_df.head(10))

    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    plt.barh(fi_df['Feature'].head(10), fi_df['Importance'].head(10), color='teal')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importance (Top 10)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "feature_importance.png"))
    plt.close()

    print("\nExecution complete! Advanced outputs saved.")

if __name__ == "__main__":
    main()