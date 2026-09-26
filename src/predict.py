"""
predict.py

Loads the saved model, scalers, and threshold, and exposes a single
predict_transaction() function that takes raw transaction values and
returns a fraud probability and decision.

This is the module app.py (Gradio) or a FastAPI wrapper would import from,
instead of duplicating model-loading logic in the UI layer.
"""

import os
import json
import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(filename: str) -> str:
    return os.path.join(PROJECT_ROOT, filename)


class FraudPredictor:
    """Wraps the trained model, scalers, and threshold behind one clean interface."""

    def __init__(self,
                 model_path: str = "fraud_model.pkl",
                 scaler_amount_path: str = "scaler_amount.pkl",
                 scaler_time_path: str = "scaler_time.pkl",
                 threshold_path: str = "threshold.json"):
        self.model = joblib.load(_path(model_path))
        self.scaler_amount = joblib.load(_path(scaler_amount_path))
        self.scaler_time = joblib.load(_path(scaler_time_path))

        with open(_path(threshold_path), "r") as f:
            self.threshold = json.load(f)["threshold"]

        print(f"Loaded model, scalers, and threshold ({self.threshold:.4f}) successfully.")

    def predict(self, amount: float, time: float, v_features: list) -> dict:
        """
        amount: raw dollar amount
        time: raw seconds-since-first-transaction value
        v_features: list of 28 floats, V1 through V28 in order
        """
        if len(v_features) != 28:
            raise ValueError(f"Expected 28 V-features, got {len(v_features)}")

        scaled_amount = self.scaler_amount.transform([[amount]])[0][0]
        scaled_time = self.scaler_time.transform([[time]])[0][0]

        features = np.array(v_features + [scaled_amount, scaled_time]).reshape(1, -1)
        prob = float(self.model.predict_proba(features)[0][1])
        is_fraud = prob >= self.threshold

        return {
            "fraud_probability": prob,
            "is_fraud": bool(is_fraud),
            "threshold_used": self.threshold,
        }


if __name__ == "__main__":
    # Quick manual smoke test using a real fraud example from the ULB test set
    predictor = FraudPredictor()

    fraud_example_v = [
        -1.54878809850026, 1.80869795041448, -0.953509033832342, 2.21308539346999,
        -2.01572779170327, -0.913456844516923, -2.35601298316433, 1.19716896702387,
        -1.67837405659509, -3.53865023182429, 3.1020899271543, -3.99337305447702,
        -1.93741062327519, -3.82289410599595, 0.830970110708369, -2.47535885382925,
        -5.21187516766885, -0.413871678166879, 0.933262164554872, 0.390785963777347,
        0.855138263312025, 0.77474482148342, 0.0590371520063436, 0.343199807900813,
        -0.468937928609185, -0.278337986906642, 0.625922215184372, 0.395573378256676
    ]

    result = predictor.predict(amount=76.94, time=74159.0, v_features=fraud_example_v)
    print("Prediction for known fraud example:", result)