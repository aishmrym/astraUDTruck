import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = Path("data/processed/fleet_clean.csv")
OUT  = Path("modelling/models/model_ban.pkl")
FEATURES = ["jarak_km","muatan_ton","overspeed","jarak_km_roll7","cumulative_km"]
TARGET = "kondisi_ban_pct"

def train():
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    model = LinearRegression()
    model.fit(poly.fit_transform(X_train), y_train)
    y_pred = model.predict(poly.transform(X_test))
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print("=" * 50)
    print("MODEL BAN")
    print(f"  MAE  : {mae:.4f} %")
    print(f"  RMSE : {rmse:.4f} %")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"model": model, "poly": poly, "features": FEATURES, "mae": mae, "rmse": rmse}, open(OUT, "wb"))
    print(f"  Disimpan ke {OUT}")

train()