# dashboard/pages/prediksi_ban.py
import streamlit as st
import numpy as np
import pickle

st.title("🛞 Prediksi Kondisi Ban")

model_data = pickle.load(open("modelling/models/model_ban.pkl", "rb"))
model   = model_data["model"]
poly    = model_data["poly"]

col1, col2 = st.columns(2)
col1.metric("MAE Model", f"{model_data['mae']:.4f} %")
col2.metric("RMSE Model", f"{model_data['rmse']:.4f} %")

st.subheader("Simulasi Prediksi")
jarak_km      = st.slider("Jarak KM (normalized)", 0.0, 1.0, 0.5)
muatan_ton    = st.slider("Muatan Ton (normalized)", 0.0, 1.0, 0.5)
overspeed     = st.slider("Overspeed (normalized)", 0.0, 1.0, 0.3)
jarak_km_r7   = st.slider("Jarak KM Roll7 (normalized)", 0.0, 1.0, 0.5)
cumulative_km = st.slider("Cumulative KM (normalized)", 0.0, 1.0, 0.4)

if st.button("Prediksi"):
    X = np.array([[jarak_km, muatan_ton, overspeed, jarak_km_r7, cumulative_km]])
    pred = model.predict(poly.transform(X))[0]
    st.success(f"Estimasi kondisi ban: **{pred:.2f}%**")
    if pred < 70:
        st.error("⚠️ Ban harus segera diganti!")
    elif pred < 80:
        st.warning("🟡 Kondisi ban mendekati batas aman.")
    else:
        st.info("✅ Kondisi ban masih aman.")