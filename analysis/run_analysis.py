# analysis/run_analysis.py
import pandas as pd
import duckdb
from pathlib import Path

DATA   = Path("data/processed/fleet_clean.csv")
DB     = Path("data/processed/warehouse.duckdb")

def run():
    df = pd.read_csv(DATA)
    con = duckdb.connect(str(DB))
    con.execute("CREATE OR REPLACE TABLE fleet AS SELECT * FROM df")

    views = {
        "v_kondisi_per_truk": (
            """SELECT plat, model,
               ROUND(MIN(tebal_rem_mm),3) as rem_min,
               ROUND(MIN(kondisi_ban_pct),2) as ban_min,
               ROUND(MIN(kondisi_aki_pct),2) as aki_min,
               ROUND(AVG(hard_brake),3) as avg_hard_brake,
               ROUND(SUM(jarak_km),1) as total_km
               FROM fleet GROUP BY plat, model ORDER BY rem_min""",
            "Kondisi komponen terendah tiap truk — semakin kecil nilainya semakin kritis."
        ),
        "v_tren_harian": (
            """SELECT hari,
               ROUND(AVG(tebal_rem_mm),4) as avg_rem,
               ROUND(AVG(kondisi_ban_pct),3) as avg_ban,
               ROUND(AVG(kondisi_aki_pct),3) as avg_aki,
               ROUND(AVG(hard_brake),3) as avg_hard_brake
               FROM fleet GROUP BY hari ORDER BY hari""",
            "Tren penurunan kondisi komponen dari hari ke hari untuk seluruh armada."
        ),
        "v_risiko_tinggi": (
            """SELECT plat, model, sopir, hari,
               tebal_rem_mm, kondisi_ban_pct, kondisi_aki_pct
               FROM fleet
               WHERE tebal_rem_mm < 7 OR kondisi_ban_pct < 85 OR kondisi_aki_pct < 85
               ORDER BY tebal_rem_mm""",
            "Truk dengan kondisi komponen kritis yang membutuhkan perhatian segera."
        ),
        "v_perilaku_sopir": (
            """SELECT sopir, plat,
               ROUND(AVG(hard_brake),3) as avg_hard_brake,
               ROUND(AVG(overspeed),3) as avg_overspeed,
               ROUND(AVG(idle_minutes),2) as avg_idle,
               ROUND(SUM(jarak_km),1) as total_km
               FROM fleet GROUP BY sopir, plat ORDER BY avg_hard_brake DESC""",
            "Perbandingan perilaku berkendara antar sopir — identifikasi sopir berisiko tinggi."
        ),
        "v_model_comparison": (
            """SELECT model,
               ROUND(AVG(tebal_rem_mm),4) as avg_rem,
               ROUND(AVG(kondisi_ban_pct),3) as avg_ban,
               ROUND(AVG(kondisi_aki_pct),3) as avg_aki,
               COUNT(DISTINCT plat) as jumlah_unit
               FROM fleet GROUP BY model""",
            "Perbandingan rata-rata kondisi komponen antar model truk."
        ),
    }

    pd.set_option("display.max_columns", 8)
    pd.set_option("display.width", 120)

    for name, (query, desc) in views.items():
        con.execute(f"CREATE OR REPLACE VIEW {name} AS {query}")
        result = con.execute(f"SELECT * FROM {name}").df()
        print("=" * 65)
        print(f"View: {name}")
        print("=" * 65)
        print(result.to_string(index=False))
        print(f"\n{len(result)} baris | {result.columns.tolist()}")
        print(f"\n  {desc}\n")

    con.close()

if __name__ == "__main__":
    run()