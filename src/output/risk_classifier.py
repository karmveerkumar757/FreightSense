# -*- coding: utf-8 -*-
"""
Risk Classifier with SHAP Explainability for FreightSense
Model: XGBoost classifier trained on constraint features
SHAP: TreeExplainer for fast, exact Shapley values
"""
import shap
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
import os
import random
import json

class RiskClassifier:
    def __init__(self):
        self.model_path = os.path.join("models", "risk_classifier", "xgb_model.joblib")
        self.feature_names_path = os.path.join("models", "risk_classifier", "feature_names.json")
        self.model = None
        self.feature_names = []
        
        self.risk_mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.feature_names_path):
            self.model = joblib.load(self.model_path)
            with open(self.feature_names_path, "r") as f:
                self.feature_names = json.load(f)
        else:
            print(f"⚠️ Risk Classifier model not found at {self.model_path}. Run training script.")

    def extract_features(self, constraints: dict) -> pd.DataFrame:
        """
        Convert the constraint dict from NER output into a numerical feature vector.
        """
        features = {
            "hours_to_deadline": 999.0,
            "has_time_window": 0,
            "time_window_width_hours": 24.0,
            "is_pharmaceutical": 0,
            "is_hazardous": 0,
            "is_perishable": 0,
            "is_fragile": 0,
            "num_route_restrictions": 0,
            "has_city_restriction": 0,
            "num_stops": 1,
            "eway_bill_hours_remaining": 999.0,
            "has_hazmat_requirement": 0,
            "has_permit_requirement": 0,
            "special_vehicle_required": 0,
            "weather_risk_score": 0.0,
            "is_urgent": 0,
            "is_high_priority": 0
        }
        
        # Populate from constraints
        cargo_types = [c.lower() for c in constraints.get("cargo_type", [])]
        if any("pharma" in c or "medicine" in c for c in cargo_types):
            features["is_pharmaceutical"] = 1
            
        handling = [h.lower() for h in constraints.get("special_handling", [])]
        if any("frozen" in h or "refrigerated" in h or "cold" in h for h in handling):
            features["is_perishable"] = 1
        if any("fragile" in h or "glass" in h for h in handling):
            features["is_fragile"] = 1
        if any("hazmat" in h or "flammable" in h or "chemical" in h for h in handling):
            features["is_hazardous"] = 1
            
        routes = constraints.get("route_constraints", [])
        features["num_route_restrictions"] = len(routes)
        if any("ring road" in r.lower() or "city" in r.lower() for r in routes):
            features["has_city_restriction"] = 1
            
        if constraints.get("vehicle_type"):
            features["special_vehicle_required"] = 1
            
        return pd.DataFrame([features])

    def generate_synthetic_training_data(self, n_samples=1000) -> pd.DataFrame:
        """
        Generate realistic synthetic training data using logical rules.
        """
        data = []
        for _ in range(n_samples):
            f = {
                "hours_to_deadline": random.choice([2, 12, 24, 48, 999]),
                "has_time_window": random.choice([0, 1]),
                "time_window_width_hours": random.choice([2, 4, 8, 24]),
                "is_pharmaceutical": random.choices([0, 1], weights=[0.8, 0.2])[0],
                "is_hazardous": random.choices([0, 1], weights=[0.9, 0.1])[0],
                "is_perishable": random.choices([0, 1], weights=[0.8, 0.2])[0],
                "is_fragile": random.choices([0, 1], weights=[0.85, 0.15])[0],
                "num_route_restrictions": random.choice([0, 1, 2]),
                "has_city_restriction": random.choices([0, 1], weights=[0.7, 0.3])[0],
                "num_stops": random.choice([1, 2, 3]),
                "eway_bill_hours_remaining": random.choice([1, 10, 24, 999]),
                "has_hazmat_requirement": 0,
                "has_permit_requirement": random.choices([0, 1], weights=[0.8, 0.2])[0],
                "special_vehicle_required": random.choices([0, 1], weights=[0.7, 0.3])[0],
                "weather_risk_score": round(random.uniform(0, 1), 2),
                "is_urgent": random.choices([0, 1], weights=[0.8, 0.2])[0],
                "is_high_priority": random.choices([0, 1], weights=[0.7, 0.3])[0]
            }
            if f["is_hazardous"]:
                f["has_hazmat_requirement"] = 1

            # Rule-based labeling
            label = 0 # LOW
            
            if f["eway_bill_hours_remaining"] < 2 and f["hours_to_deadline"] < 3:
                label = 3 # CRITICAL
            elif f["is_pharmaceutical"] and not f["is_perishable"]:
                label = 2 # HIGH
            elif f["has_city_restriction"] and f["is_urgent"]:
                label = 2 # HIGH
            elif f["is_hazardous"] and not f["has_hazmat_requirement"]:
                label = 3 # CRITICAL
            elif f["weather_risk_score"] > 0.8:
                label = 2 # HIGH
            elif f["weather_risk_score"] > 0.5 or f["num_route_restrictions"] > 0 or f["is_high_priority"]:
                label = 1 # MEDIUM
                
            # Add noise (10% chance to flip to adjacent class)
            if random.random() < 0.1:
                label = max(0, min(3, label + random.choice([-1, 1])))
                
            f["label"] = label
            data.append(f)
            
        return pd.DataFrame(data)

    def predict_with_explanation(self, constraints: dict) -> dict:
        """
        Main method - returns risk prediction and SHAP explanation.
        """
        if self.model is None:
            return {
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "shap_values": {},
                "top_risk_factors": [],
                "explanation_text": "Risk classifier model not trained."
            }

        df = self.extract_features(constraints)
        
        # Ensure column order matches training
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]

        # Predict probability
        probs = self.model.predict_proba(df)[0]
        pred_class = int(np.argmax(probs))
        risk_level = self.risk_mapping.get(pred_class, "UNKNOWN")
        risk_score = float(probs[pred_class])

        # SHAP Explainability
        explainer = shap.TreeExplainer(self.model)
        # For multi-class, shap_values is a list of arrays (one per class). We take the values for the predicted class.
        shap_values_obj = explainer.shap_values(df)
        
        if isinstance(shap_values_obj, list):
            class_shap_values = shap_values_obj[pred_class][0]
        else:
            class_shap_values = shap_values_obj[0]
            
        feature_contributions = dict(zip(df.columns, class_shap_values))
        
        # Sort factors by absolute magnitude to find top contributors
        sorted_factors = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        top_risk_factors = []
        explanation_parts = []
        
        for feature, contribution in sorted_factors[:5]:
            # Positive SHAP value means it pushed the prediction toward this class
            direction = "increases_risk" if contribution > 0 else "decreases_risk"
            if abs(contribution) > 0.05:
                top_risk_factors.append({
                    "factor": feature.replace("_", " ").title(),
                    "contribution": float(round(contribution, 3)),
                    "direction": direction
                })
                
                impact = "increased" if direction == "increases_risk" else "decreased"
                explanation_parts.append(f"{feature.replace('_', ' ')} {impact} the likelihood")

        explanation_text = f"The risk is classified as {risk_level} (confidence: {risk_score:.2f}). "
        if explanation_parts:
            explanation_text += "Primary factors: " + ", ".join(explanation_parts) + "."

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "shap_values": {k: float(v) for k, v in feature_contributions.items()},
            "top_risk_factors": top_risk_factors,
            "explanation_text": explanation_text
        }
