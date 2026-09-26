"""
test_predict.py

Basic sanity tests for the FraudPredictor class in src/predict.py.
These only READ the existing saved model/scalers - they don't modify,
retrain, or overwrite anything, so the deployed app is unaffected.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import FraudPredictor


FRAUD_EXAMPLE_V = [
    -1.54878809850026, 1.80869795041448, -0.953509033832342, 2.21308539346999,
    -2.01572779170327, -0.913456844516923, -2.35601298316433, 1.19716896702387,
    -1.67837405659509, -3.53865023182429, 3.1020899271543, -3.99337305447702,
    -1.93741062327519, -3.82289410599595, 0.830970110708369, -2.47535885382925,
    -5.21187516766885, -0.413871678166879, 0.933262164554872, 0.390785963777347,
    0.855138263312025, 0.77474482148342, 0.0590371520063436, 0.343199807900813,
    -0.468937928609185, -0.278337986906642, 0.625922215184372, 0.395573378256676
]

LEGIT_EXAMPLE_V = [
    1.2288211502379, -0.0634077165201056, 0.274145142235826, 0.647465021810117,
    -0.0481345611508765, 0.372073028593297, -0.22423058741343, 0.0799390492455152,
    0.640758817066441, -0.273053702248503, -1.25272793883718, 0.465078770741453,
    0.400502115321077, -0.292841860600363, -0.10177401599731, -0.399835897844616,
    0.0343356567914817, -0.783550254934187, 0.141344900433949, -0.0965659023514416,
    -0.129554448055005, -0.0837793282428063, -0.151661473916324, -0.700371597289218,
    0.598550164523483, 0.491409070563651, 0.0029892597250263, 0.0017822861144491
]


def test_model_loads_successfully():
    predictor = FraudPredictor()
    assert predictor.model is not None
    assert predictor.scaler_amount is not None
    assert predictor.scaler_time is not None
    assert 0.0 <= predictor.threshold <= 1.0


def test_prediction_returns_valid_probability():
    predictor = FraudPredictor()
    result = predictor.predict(amount=100.0, time=50000.0, v_features=[0.0] * 28)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert isinstance(result["is_fraud"], bool)


def test_known_fraud_example_is_flagged():
    predictor = FraudPredictor()
    result = predictor.predict(amount=76.94, time=74159.0, v_features=FRAUD_EXAMPLE_V)
    assert result["is_fraud"] is True
    assert result["fraud_probability"] > predictor.threshold


def test_known_legit_example_is_not_flagged():
    predictor = FraudPredictor()
    result = predictor.predict(amount=11.5, time=61290.0, v_features=LEGIT_EXAMPLE_V)
    assert result["is_fraud"] is False


def test_wrong_number_of_v_features_raises_error():
    predictor = FraudPredictor()
    try:
        predictor.predict(amount=100.0, time=50000.0, v_features=[0.0] * 10)
        assert False, "Expected a ValueError for wrong feature count"
    except ValueError:
        pass