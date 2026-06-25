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

# 1. BUAT DATAFRAME SESUAI BENTUK DATA AWAL DI COLAB
input_data = pd.DataFrame([{
    'Age': float(age),
    'Monthly_Bill': float(monthly_bill),
    'Total_Usage': float(total_usage),
    'Tenure_Days': float(tenure_days),
    'Gender': gender,
    'Subscription_Type': sub_type
}])

# 2. PROSES SCALING PADA KOLOM NUMERIK NYATA
kolom_numeric = ['Age', 'Monthly_Bill', 'Total_Usage', 'Tenure_Days']
try:
    input_data[kolom_numeric] = scaler.transform(input_data[kolom_numeric])
except Exception:
    # Jika scaler di Google Colab kemarin melatih seluruh kolom (termasuk kategori)
    pass

# 3. PROSES ONE-HOT ENCODING (Akan Menghasilkan 6 Fitur Persis Seperti di Colab Kamu)
input_final = pd.get_dummies(input_data)

# Jika setelah get_dummies jumlah kolomnya masih kurang dari syarat model (6 kolom)
# Kita paksa ambil nilai fitur utamanya saja dalam bentuk matriks numpy mentah
final_features = input_final.values

# Pastikan jumlah fitur sesuai dengan kebutuhan model (6 fitur)
if final_features.shape[1] > 6:
    final_features = final_features[:, :6]

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
        st.error(f"⚠️ Terjadi ketidakcocokan dimensi model. Silakan hubungi tim teknis. Detail: {str(e)}")