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

# Fungsi memuat semua file pintar .pkl
@st.cache_resource
def load_machine_learning_components():
    with open('best_random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

try:
    model, scaler, model_features = load_machine_learning_components()
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

# 1. OTOMATIS COCOKKAN NAMA KOLOM DENGAN MODEL FEATURES (Mencegah Typo / Perbedaan Kapital)
# Kita cari string yang mirip di dalam model_features
def get_exact_col_name(input_name, features_list):
    for f in features_list:
        if input_name.lower() == f.lower().strip():
            return f
    return input_name

col_age = get_exact_col_name('age', model_features)
col_bill = get_exact_col_name('monthly_bill', model_features)
col_usage = get_exact_col_name('total_usage', model_features)
col_tenure = get_exact_col_name('tenure_days', model_features)

# 2. Buat DataFrame Awal
input_data = pd.DataFrame([{
    col_age: float(age),
    col_bill: float(monthly_bill),
    col_usage: float(total_usage),
    col_tenure: float(tenure_days),
    'gender': gender,
    'subscription_type': sub_type
}])

# 3. One-Hot Encoding
input_encoded = pd.get_dummies(input_data)

# 4. Cari kolom dummy gender dan subscription type yang sesuai di model_features
for f in model_features:
    for col in input_encoded.columns:
        if col.lower().replace(" ", "") in f.lower().replace("_", "").replace(" ", ""):
            input_encoded = input_encoded.rename(columns={col: f})

# 5. Sinkronisasi total seluruh kolom agar identik dengan model_features.pkl
for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_final = input_encoded[model_features].copy()

# 6. NORMALISASI SCALER BERDASARKAN FILTER KOLOM YANG VALID DI SCALER
kolom_numeric_real = [col_age, col_bill, col_usage, col_tenure]
# Validasi apakah kolom beneran dikenali oleh scaler bawaan
kolom_valid_scaler = [c for c in kolom_numeric_real if c in model_features]

if kolom_valid_scaler:
    try:
        input_numeric_scaled = scaler.transform(input_final[kolom_valid_scaler])
        input_final.loc[:, kolom_valid_scaler] = input_numeric_scaled
    except Exception as e:
        # Jika scaler setup-nya menggunakan total seluruh fitur encoding
        input_final_scaled = scaler.transform(input_final)
        input_final = pd.DataFrame(input_final_scaled, columns=model_features)

# Tombol Eksekusi Prediksi
st.markdown("---")
if st.button("🚀 Jalankan Analisis Prediksi Churn", type="primary"):
    prediction = model.predict(input_final)[0]
    prediction_proba = model.predict_proba(input_final)[0]
    
    st.subheader("🎯 Hasil Keputusan Analisis:")
    if prediction == 1:
        st.error(f"🚨 **PELANGGAN BERPOTENSI CHURN (BERHENTI BERLANGGANAN)**")
        st.write(f"Tingkat Probabilitas Keyakinan Model: **{prediction_proba[1]*100:.2f}%**")
    else:
        st.success(f"✅ **PELANGGAN BERSTATUS RETAINED (SETIA / BERTAHAN)**")
        st.write(f"Tingkat Probabilitas Keyakinan Model: **{prediction_proba[0]*100:.2f}%**")