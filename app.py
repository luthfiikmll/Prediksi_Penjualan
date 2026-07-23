"""
app.py — Aplikasi Prediksi Penjualan Kopi Per Produk (Harian).

Alur (sesuai Bab 5.3.3 Proses Prediksi):
  Input tanggal -> Pembentukan fitur otomatis -> Model produk dipanggil
  -> Prediksi jumlah cup -> Hasil ditampilkan

Artifact yang dibutuhkan di folder yang sama:
  models_produk.pkl, product_daily.csv, eval_summary.csv, utils.py
"""

import pickle
from datetime import timedelta

import pandas as pd
import streamlit as st

import utils

st.set_page_config(page_title="Prediksi Penjualan Kopi", page_icon="☕", layout="centered")


@st.cache_resource
def load_artifacts():
    with open("models_produk.pkl", "rb") as f:
        bundle = pickle.load(f)
    models = {}
    for produk, meta in bundle.items():
        models[produk] = {
            "model": utils.load_xgb_model_from_bytes(meta["model_bytes"]),
            "features": meta["features"],
            "optimizer": meta["optimizer"],
        }
    product_daily = utils.load_product_daily("product_daily.csv")
    eval_summary = pd.read_csv("eval_summary.csv")
    return models, product_daily, eval_summary


# ── Halaman Utama ──
st.title("☕ Prediksi Penjualan Kopi")
st.caption(
    "Coffee Shop Sans Your Day — prediksi jumlah cup terjual per produk "
    "untuk tanggal yang dipilih, menggunakan model XGBoost per produk."
)

try:
    models_produk, product_daily, eval_summary = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Artifact model tidak ditemukan di folder ini "
        f"(models_produk.pkl / product_daily.csv / eval_summary.csv). Detail: {e}"
    )
    st.stop()

last_date = product_daily["tanggal"].max().date()
st.markdown(f"📅 Data histori tersedia sampai **{last_date.strftime('%d %B %Y')}**.")

# ── Input Prediksi ──
st.subheader("Input Prediksi")
target_date = st.date_input(
    "Pilih tanggal yang ingin diprediksi",
    value=last_date + timedelta(days=1),
    min_value=last_date + timedelta(days=1),
    max_value=last_date + timedelta(days=30),
)

semua_produk = sorted(models_produk.keys())
produk_dipilih = st.multiselect(
    "Filter produk (opsional — kosongkan untuk prediksi semua produk)",
    options=semua_produk,
    default=[],
)
produk_target = produk_dipilih if produk_dipilih else semua_produk

if st.button("🔮 Prediksi", type="primary"):
    with st.spinner("Membentuk fitur & memanggil model tiap produk..."):
        hasil = []
        gagal = []
        for produk in produk_target:
            model_info = models_produk[produk]
            try:
                pred = utils.predict_for_date(produk, product_daily, model_info, target_date)
                hasil.append({"Produk": produk, "Prediksi (cup)": pred})
            except ValueError as e:
                gagal.append((produk, str(e)))

    # ── Hasil Prediksi ──
    st.subheader(f"Hasil Prediksi — {target_date.strftime('%d %B %Y')}")
    df_hasil = pd.DataFrame(hasil).sort_values("Prediksi (cup)", ascending=False).reset_index(drop=True)
    st.dataframe(df_hasil, use_container_width=True, hide_index=True)

    total_cup = df_hasil["Prediksi (cup)"].sum()
    label_total = "Total Prediksi Produk Terpilih" if produk_dipilih else "Total Prediksi Seluruh Produk"
    st.metric(label_total, f"{total_cup:.0f} cup")

    if gagal:
        with st.expander(f"⚠️ {len(gagal)} produk gagal diprediksi"):
            for produk, msg in gagal:
                st.write(f"- **{produk}**: {msg}")

st.markdown("---")
with st.expander("ℹ️ Tentang Model"):
    st.dataframe(eval_summary, use_container_width=True, hide_index=True)
    st.caption(
        "Tabel di atas menunjukkan performa masing-masing model produk pada "
        "data pengujian (train vs test) saat pelatihan."
    )

st.caption("Skripsi — Prediksi Penjualan Kopi Per Produk (Harian) dengan XGBoost (CRISP-DM).")
