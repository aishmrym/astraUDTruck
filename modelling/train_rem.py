# modelling/train_rem.py
"""Prediksi tebal kampas rem (mm) — Polynomial Regression"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = Path("data/processed/fleet_clean.csv")
OUT  = Path("modelling/models/model_rem.pkl")

FEATURES = ["hard_brake", "jarak_km", "overspeed",
            "hard_brake_roll7", "cumulative_km", "hard_brake_std7"]
TARGET   = "tebal_rem_mm"

def train():
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_p = poly.fit_transform(X_train)
    X_test_p  = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    print("=" * 50)
    print("MODEL KAMPAS REM")
    print("=" * 50)
    print(f"  MAE  : {mae:.4f} mm")
    print(f"  RMSE : {rmse:.4f} mm")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"model": model, "poly": poly, "features": FEATURES,
                 "mae": mae, "rmse": rmse}, open(OUT, "wb"))
    print(f"  💾 Model disimpan ke {OUT}")

if __name__ == "__main__":
    train()