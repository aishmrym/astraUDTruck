import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

RAW = Path("data/raw/data_dummy_fleetsight_FINAL.csv")
OUT = Path("data/processed/fleet_clean.csv")

def preprocess():
    df = pd.read_csv(RAW)
    print(f"Load: {len(df)} baris")
    df = df.drop_duplicates()
    df = df.sort_values(["plat", "hari"]).reset_index(drop=True)
    df = df.groupby("plat", group_keys=False).apply(lambda x: x.ffill()).reset_index(drop=True)
    num_cols = ["hard_brake","idle_minutes","overspeed","jarak_km","suhu_mesin","muatan_ton","tebal_rem_mm","kondisi_ban_pct","kondisi_aki_pct"]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())
    df["cumulative_km"] = df.groupby("plat")["jarak_km"].cumsum()
    for col in ["hard_brake", "suhu_mesin", "jarak_km"]:
        df[f"{col}_roll7"] = df.groupby("plat")[col].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["hard_brake_std7"] = df.groupby("plat")["hard_brake"].transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
    for col in ["hard_brake", "overspeed", "jarak_km", "suhu_mesin"]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        df[col] = df[col].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))
    scale_cols = ["hard_brake","idle_minutes","overspeed","jarak_km","suhu_mesin","muatan_ton","cumulative_km","hard_brake_roll7","suhu_mesin_roll7","jarak_km_roll7","hard_brake_std7"]
    df[scale_cols] = MinMaxScaler().fit_transform(df[scale_cols])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Selesai. {len(df)} baris disimpan ke {OUT}")

preprocess()
