"""
preprocessing.py

Loads the raw credit card transaction data, removes duplicates,
scales Amount and Time, and returns train/test splits ready for modeling.
"""

import os
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Project root = one level up from this file (src/preprocessing.py -> project folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(filename: str) -> str:
    """Find a file whether it lives in the project root or was passed as a full path."""
    if os.path.exists(filename):
        return filename
    candidate = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Could not find '{filename}' in current folder or project root.")


def load_and_clean_data(csv_path: str = "creditcard.csv") -> pd.DataFrame:
    """Load the raw CSV and remove duplicate rows."""
    df = pd.read_csv(_resolve_path(csv_path))
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Loaded {before} rows, removed {before - after} duplicates, {after} remain.")
    return df


def scale_features(df: pd.DataFrame, save_scalers: bool = True):
    """
    Scale Amount and Time using separate RobustScaler objects
    (kept separate on purpose - reusing one scaler for both columns
    silently overwrites its fitted parameters).
    """
    scaler_amount = RobustScaler()
    scaler_time = RobustScaler()

    df = df.copy()
    df["scaled_amount"] = scaler_amount.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"] = scaler_time.fit_transform(df["Time"].values.reshape(-1, 1))
    df = df.drop(["Amount", "Time"], axis=1)

    if save_scalers:
        joblib.dump(scaler_amount, os.path.join(PROJECT_ROOT, "scaler_amount.pkl"))
        joblib.dump(scaler_time, os.path.join(PROJECT_ROOT, "scaler_time.pkl"))
        print("Saved scaler_amount.pkl and scaler_time.pkl to project root")

    return df, scaler_amount, scaler_time


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Separate features/target and perform a stratified train/test split."""
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Fraud ratio - train: {y_train.mean():.4f}, test: {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


def run_preprocessing(csv_path: str = "creditcard.csv"):
    """Convenience function that runs the full preprocessing pipeline end to end."""
    df = load_and_clean_data(csv_path)
    df, scaler_amount, scaler_time = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test, scaler_amount, scaler_time


if __name__ == "__main__":
    run_preprocessing()