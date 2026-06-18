# dashboard/pages/prediksi_aki.py
import streamlit as st
import numpy as np
import pickle

st.title("🔋 Prediksi Kondisi Aki")

model_data = pickle.load(open("modelling/models/model_aki.pkl", "rb"))
model   = model_data["model"]
poly    = model_data["poly"]

col1, col2 = st.columns(2)
col1.metric("MAE Model", f"{model_data['mae']:.4f} %")
col2.metric("RMSE Model", f"{model_data['rmse']:.4f} %")

st.subheader("Simulasi Prediksi")
jarak_km       = st.slider("Jarak KM (normalized)", 0.0, 1.0, 0.5)
suhu_mesin     = st.slider("Suhu Mesin (normalized)", 0.0, 1.0, 0.5)
idle_minutes   = st.slider("Idle Minutes (normalized)", 0.0, 1.0, 0.3)
suhu_mesin_r7  = st.slider("Suhu Mesin Roll7 (normalized)", 0.0, 1.0, 0.5)
cumulative_km  = st.slider("Cumulative KM (normalized)", 0.0, 1.0, 0.4)

if st.button("Prediksi"):
    X = np.array([[jarak_km, suhu_mesin, idle_minutes, suhu_mesin_r7, cumulative_km]])
    pred = model.predict(poly.transform(X))[0]
    st.success(f"Estimasi kondisi aki: **{pred:.2f}%**")
    if pred < 70:
        st.error("⚠️ Aki harus segera diganti!")
    elif pred < 80:
        st.warning("🟡 Kondisi aki mendekati batas aman.")
    else:
        st.info("✅ Kondisi aki masih aman.")