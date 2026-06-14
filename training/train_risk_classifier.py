# -*- coding: utf-8 -*-
"""
Training script for XGBoost Risk Classifier with SHAP
"""
import os
import json
import xgboost as xgb
import pandas as pd
import joblib
import shap
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
import sys

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.output.risk_classifier import RiskClassifier

def main():
    print("🚀 Starting Risk Classifier Training Pipeline...")
    classifier = RiskClassifier()
    
    print("\n1. Generating synthetic training data (1000 samples)...")
    df = classifier.generate_synthetic_training_data(n_samples=1000)
    
    X = df.drop(columns=["label"])
    y = df["label"]
    
    print("\n2. Training XGBoost model with cross-validation...")
    # Using specific hyperparameters from prompt
    model = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=4,
        random_state=42
    )
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    print(f"✅ 5-Fold CV Mean F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["LOW", "MEDIUM", "HIGH", "CRITICAL"]))
    
    print("\n3. Saving model and feature names...")
    os.makedirs(os.path.dirname(classifier.model_path), exist_ok=True)
    joblib.dump(model, classifier.model_path)
    
    feature_names = list(X.columns)
    with open(classifier.feature_names_path, "w") as f:
        json.dump(feature_names, f)
    print(f"✅ Saved model to {classifier.model_path}")
    
    print("\n4. Running SHAP analysis on 50 test samples...")
    explainer = shap.TreeExplainer(model)
    shap_values_obj = explainer.shap_values(X_test.head(50))
    
    # Calculate mean absolute SHAP value per feature
    if isinstance(shap_values_obj, list):
        # Multiclass: take mean over all classes
        mean_shap = pd.DataFrame(
            [pd.DataFrame(sv, columns=X.columns).abs().mean() for sv in shap_values_obj]
        ).mean()
    else:
        # Binary (fallback)
        mean_shap = pd.DataFrame(shap_values_obj, columns=X.columns).abs().mean()
        
    print("\nTop 5 Most Important Features (Mean |SHAP|):")
    print(mean_shap.sort_values(ascending=False).head(5))

if __name__ == "__main__":
    main()
