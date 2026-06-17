# pipeline/ingestion/load_data.py
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/data_dummy_fleetsight_FINAL.csv")

def load_data():
    df = pd.read_csv(RAW)
    print("=" * 50)
    print("DATASET INFO")
    print("=" * 50)
    print(f"  Total baris     : {len(df)}")
    print(f"  Total kolom     : {len(df.columns)}")
    print(f"  Jumlah truk     : {df['plat'].nunique()}")
    print(f"  Model truk      : {df['model'].unique().tolist()}")
    print(f"  Rentang hari    : {df['hari'].min()} – {df['hari'].max()}")
    print(f"  Missing values  :\n{df.isnull().sum()}")
    print(f"\nSample data (5 baris pertama):")
    print(df.head())
    return df

if __name__ == "__main__":
    load_data()