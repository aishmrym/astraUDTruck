import streamlit as st
import duckdb
import pandas as pd

DB = "data/processed/warehouse.duckdb"
st.title("📊 Overview Armada")
con = duckdb.connect(DB, read_only=True)

df = con.execute("SELECT * FROM v_kondisi_per_truk").df()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Truk", len(df))
col2.metric("Tebal Rem Minimum", f"{df['rem_min'].min():.2f} mm")
col3.metric("Kondisi Ban Minimum", f"{df['ban_min'].min():.1f}%")
col4.metric("Kondisi Aki Minimum", f"{df['aki_min'].min():.1f}%")

st.subheader("Kondisi Komponen per Truk")
st.dataframe(df, use_container_width=True)

st.subheader("Truk Berisiko Tinggi 🔴")
df_risk = con.execute("SELECT * FROM v_risiko_tinggi LIMIT 20").df()
st.dataframe(df_risk, use_container_width=True)
con.close()