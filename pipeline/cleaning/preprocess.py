# pipeline/cleaning/preprocess.py
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw/data_dummy_fleetsight_FINAL.csv")
OUT = Path("data/processed/fleet_clean.csv")

def preprocess():
    df = pd.read_csv(RAW)
    print(f"🔄 Load: {len(df)} baris")

    # 1. Hapus duplikat eksak
    before = len(df)
    df = df.drop_duplicates()
    print(f"   Duplikat dihapus: {before - len(df)} baris")

    # 2. Forward fill missing values (logika time-series per truk)
    df = df.sort_values(["plat", "hari"])
    df = df.groupby("plat", group_keys=False).apply(lambda x: x.ffill())

    # 3. Mean imputation sebagai fallback
    num_cols = ["hard_brake","idle_minutes","overspeed","jarak_km",
                "suhu_mesin","muatan_ton","tebal_rem_mm","kondisi_ban_pct","kondisi_aki_pct"]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    # 4. Cumulative distance per truk
    df["cumulative_km"] = df.groupby("plat")["jarak_km"].cumsum()

    # 5. Rolling average 7 hari (per truk)
    for col in ["hard_brake", "suhu_mesin", "jarak_km"]:
        df[f"{col}_roll7"] = (
            df.groupby("plat")[col]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )

    # 6. Rolling std 7 hari (volatilitas berkendara)
    df["hard_brake_std7"] = (
        df.groupby("plat")["hard_brake"]
        .transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
    )

    # 7. Outlier capping (IQR) — hanya kolom input
    for col in ["hard_brake", "overspeed", "jarak_km", "suhu_mesin"]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    # 8. Normalisasi fitur input (Min-Max)
    from sklearn.preprocessing import MinMaxScaler
    scale_cols = ["hard_brake","idle_minutes","overspeed","jarak_km",
                  "suhu_mesin","muatan_ton","cumulative_km",
                  "hard_brake_roll7","suhu_mesin_roll7","jarak_km_roll7","hard_brake_std7"]
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"   ✅ {len(df)} baris bersih")
    print(f"   Kolom baru: cumulative_km, *_roll7, hard_brake_std7")
    print(f"   💾 Disimpan ke {OUT}")

if __name__ == "__main__":
    preprocess()