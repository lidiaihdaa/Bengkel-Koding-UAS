import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── KONFIGURASI HALAMAN UTAMA WEB ──────────────────────────────────
st.set_page_config(
    page_title="Prediksi Churn Pelanggan", 
    page_icon="📊", 
    layout="centered"
)

# ── LOAD 5 ARTEFAK PINTAR DARI FOLDER DEPLOYMENT ───────────────────
@st.cache_resource
def load_machine_learning_artifacts():
    model        = joblib.load("deployment/best_model.pkl")
    scaler       = joblib.load("deployment/scaler.pkl")
    features     = joblib.load("deployment/feature_names.pkl")
    median_train = joblib.load("deployment/median_train.pkl")
    iqr_bounds   = joblib.load("deployment/iqr_bounds.pkl")
    return model, scaler, features, median_train, iqr_bounds

try:
    model, scaler, FEATURES, MEDIAN_TRAIN, IQR_BOUNDS = load_machine_learning_artifacts()
except Exception as e:
    st.error(f"⚠️ Gagal memuat file .pkl! Pastikan folder 'deployment' beserta 5 filenya sudah ada. Detail error: {e}")
    st.stop()

# ── ANTARMUKA PENGGUNA (UI DESIGN) ─────────────────────────────────
st.title("📊 Aplikasi Prediksi Customer Churn")
st.markdown("**UAS Bengkel Koding Data Science** — Prediksi Akurat Menggunakan Model Hasil Preprocessing & Tuning.")
st.markdown("---")

st.subheader("💡 Masukkan Profil & Data Transaksi Utama Pelanggan")
st.write("Silakan isi data di bawah ini untuk menganalisis tingkat risiko churn:")

col1, col2 = st.columns(2)

with col1:
    tenure_days = st.number_input("Masa Aktif Pelanggan (Tenure Days):", min_value=0, value=365, step=10)
    total_spent = st.number_input("Total Pengeluaran Pelanggan (Rp):", min_value=0.0, value=500000.0, step=50000.0)
    support_tickets = st.number_input("Jumlah Komplain (Support Tickets):", min_value=0, value=1, step=1)

with col2:
    satisfaction_score = st.slider("Skor Kepuasan Pelanggan (1-10):", min_value=1.0, max_value=10.0, value=7.0, step=0.5)
    last_3_month_freq = st.number_input("Frekuensi Belanja (3 Bulan Terakhir):", min_value=0, value=5, step=1)
    age = st.number_input("Usia Pelanggan (Age):", min_value=15, max_value=100, value=30, step=1)

st.markdown("---")

# ── PROSES PREDIKSI BACKEND (SINKRONISASI DATA) ─────────────────────
if st.button("🚀 Jalankan Analisis Prediksi Churn", type="primary", use_container_width=True):
    
    # 1. RESET REFRESH: Bikin form isian dasar pakai nilai median data latih
    data_mapping = {}
    for col in FEATURES:
        if col in MEDIAN_TRAIN.index:
            data_mapping[col] = float(MEDIAN_TRAIN[col])
        else:
            data_mapping[col] = 0.0 
            
    # 2. Fungsi pelacak kolom
    def temukan_kolom(keyword):
        for f in FEATURES:
            if keyword.lower() in f.lower():
                return f
        return None

    # 3. TIMPA data median dengan inputan segar dari pengguna web
    kol_tenure = temukan_kolom('tenure')
    kol_spent = temukan_kolom('spent') or temukan_kolom('bill')
    kol_ticket = temukan_kolom('ticket')
    kol_sat = temukan_kolom('satisfaction')
    kol_freq = temukan_kolom('freq') or temukan_kolom('purchase')
    kol_age = temukan_kolom('age')

    if kol_tenure: data_mapping[kol_tenure] = float(tenure_days)
    if kol_spent: data_mapping[kol_spent] = float(total_spent)
    if kol_ticket: data_mapping[kol_ticket] = float(support_tickets)
    if kol_sat: data_mapping[kol_sat] = float(satisfaction_score)
    if kol_freq: data_mapping[kol_freq] = float(last_3_month_freq)
    if kol_age: data_mapping[kol_age] = float(age)

    # Convert ke matriks Pandas 1 baris
    input_df = pd.DataFrame([data_mapping])[FEATURES]

    # 4. ANTI-OUTLIER: Potong nilai aneh pakai IQR
    for col, (lower, upper) in IQR_BOUNDS.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].clip(lower=lower, upper=upper)

    # 5. ANTI-LEAKAGE: Scaling disamakan dengan cetakan training asli
    try:
        if hasattr(scaler, 'feature_names_in_'):
            df_scale_template = pd.DataFrame(0.0, index=[0], columns=scaler.feature_names_in_)
            for col in input_df.columns:
                if col in df_scale_template.columns:
                    df_scale_template[col] = input_df[col]
            df_scaled = pd.DataFrame(scaler.transform(df_scale_template), columns=scaler.feature_names_in_)
            input_final = df_scaled[FEATURES]
        else:
            input_final = input_df
    except Exception:
        input_final = input_df

    # 6. EKSEKUSI AI & CUSTOM THRESHOLD (Batas 35%)
    try:
        # Ambil persentase (probabilitas) pelanggan masuk ke kelas Churn (Indeks ke-1)
        probas = model.predict_proba(input_final)[0] if hasattr(model, 'predict_proba') else [0.0, 0.0]
        prob_churn = float(probas[1])
        
        # ── LOGIKA BISNIS: JIKA RISIKO >= 35%, LANGSUNG ANGGAP CHURN ──
        prediksi = 1 if prob_churn >= 0.35 else 0
        
        st.subheader("🎯 Hasil Keputusan Analisis Model:")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if prediksi == 1:
                st.error("🚨 **PELANGGAN BERPOTENSI CHURN**\n\n(Pelanggan memiliki kecenderungan tinggi untuk berhenti berlangganan/pergi).")
            else:
                st.success("✅ **PELANGGAN BERSTATUS RETAINED**\n\n(Pelanggan terdeteksi aman, loyal, dan akan tetap bertahan).")
                
        with col_res2:
            st.metric(label="Tingkat Risiko Churn:", value=f"{prob_churn * 100:.2f}%")
            # Limit progress bar max ke 1.0 agar tidak error UI
            progress_value = prob_churn if prob_churn <= 1.0 else 1.0
            st.progress(progress_value)
            
        st.divider()
        st.subheader("📌 Rekomendasi Tindakan Bisnis:")
        if prediksi == 1:
            st.warning("""
            **Rekomendasi Penyelamatan Pelanggan:**
            1. Segera tugaskan Tim Customer Success untuk menghubungi pelanggan via telepon/email.
            2. Kirimkan email penawaran eksklusif berupa diskon khusus (*retention coupon*).
            3. Prioritaskan penyelesaian sisa tiket komplain yang mereka miliki agar skor kepuasan kembali naik.
            """)
        else:
            st.info("""
            **Rekomendasi Mempertahankan Loyalitas:**
            1. Masukkan pelanggan ke dalam program loyalitas berbasis poin atau reward khusus.
            2. Kirimkan pesan personalisasi apresiasi atas kesetiaan mereka menggunakan layanan Anda.
            3. Tawarkan promosi ringan untuk *upgrade* paket ke tingkat langganan Premium.
            """)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis saat model memproses data: {str(e)}")