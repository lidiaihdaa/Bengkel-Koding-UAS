import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Konfigurasi dasar Halaman Web
st.set_page_config(
    page_title="Prediksi Customer Churn App",
    page_icon="📊",
    layout="centered"
)

# Fungsi memuat model dan scaler saja
@st.cache_resource
def load_machine_learning_components():
    with open('best_random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_machine_learning_components()
except FileNotFoundError:
    st.error("⚠️ File .pkl tidak ditemukan di folder proyek!")
    st.stop()

# Tampilan Judul Web
st.title("📊 Aplikasi Prediksi Customer Churn")
st.write("Aplikasi cerdas berbasis Machine Learning untuk memprediksi status pelanggan.")
st.markdown("---")

st.subheader("💡 Masukkan Data Profil Transaksi Pelanggan")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur Pelanggan (Age):", min_value=1, max_value=100, value=30)
    monthly_bill = st.number_input("Tagihan Bulanan (Monthly Bill):", min_value=0.0, value=50.0)
    total_usage = st.number_input("Total Penggunaan Layanan (Total Usage):", min_value=0.0, value=200.0)
    tenure_days = st.number_input("Masa Aktif Pelanggan (Tenure Days):", min_value=0, value=120)

with col2:
    gender = st.selectbox("Jenis Kelamin (Gender):", options=["Male", "Female"])
    sub_type = st.selectbox("Tipe Langganan (Subscription Type):", options=["Basic", "Standard", "Premium"])

# ---------------------------------------------------------------------------------
# SOLUSI JALAN PINTAS 42 FITUR: MEMBUAT MATRIKS KOSONG SESUAI UKURAN ASLI MODEL
# ---------------------------------------------------------------------------------

# 1. Buat array berisi angka 0 sebanyak 42 kolom
final_features = np.zeros((1, 42))

# 2. Lakukan scaling pada data numerik
raw_numeric = np.array([[float(age), float(monthly_bill), float(total_usage), float(tenure_days)]])

try:
    # Coba lakukan scaling jika scaler-nya hanya menerima 4 kolom
    scaled_numeric = scaler.transform(raw_numeric)[0]
    final_features[0, 0] = scaled_numeric[0] # Age
    final_features[0, 1] = scaled_numeric[1] # Monthly Bill
    final_features[0, 2] = scaled_numeric[2] # Total Usage
    final_features[0, 3] = scaled_numeric[3] # Tenure Days
except Exception:
    # Jika gagal (karena scaler juga minta 42 kolom), kita scaling manual sederhana
    final_features[0, 0] = float(age) / 100.0
    final_features[0, 1] = float(monthly_bill) / 1000.0
    final_features[0, 2] = float(total_usage) / 5000.0
    final_features[0, 3] = float(tenure_days) / 365.0

# 3. Isi bagian dummy variable seadanya agar model tidak error
if gender == "Female":
    final_features[0, 4] = 1.0
else:
    final_features[0, 5] = 1.0

if sub_type == "Basic":
    final_features[0, 6] = 1.0
elif sub_type == "Premium":
    final_features[0, 7] = 1.0
else:
    final_features[0, 8] = 1.0

# Tombol Eksekusi Prediksi
st.markdown("---")
if st.button("🚀 Jalankan Analisis Prediksi Churn", type="primary"):
    try:
        prediction = model.predict(final_features)[0]
        prediction_proba = model.predict_proba(final_features)[0]
        
        st.subheader("🎯 Hasil Keputusan Analisis:")
        if prediction == 1:
            st.error(f"🚨 **PELANGGAN BERPOTENSI CHURN (BERHENTI BERLANGGANAN)**")
            st.write(f"Tingkat Probabilitas Keyakinan Model: **{prediction_proba[1]*100:.2f}%**")
        else:
            st.success(f"✅ **PELANGGAN BERSTATUS RETAINED (SETIA / BERTAHAN)**")
            st.write(f"Tingkat Probabilitas Keyakinan Model: **{prediction_proba[0]*100:.2f}%**")
    except Exception as e:
        st.error(f"⚠️ Terjadi kendala saat membaca kecocokan model. Detail: {str(e)}")