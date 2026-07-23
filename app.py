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


# Cache hasil prediksi per (produk, tanggal) supaya reload/interaksi ulang
# dengan input yang sama tidak menghitung ulang fitur & memanggil model lagi
# (mengurangi beban CPU yang bisa memicu throttle di Streamlit Cloud).
# Parameter berawalan "_" (mis. _model_info, _product_daily) sengaja tidak
# di-hash oleh st.cache_data karena isinya objek model/dataframe.
@st.cache_data(show_spinner=False)
def cached_predict_for_date(produk, target_date, _model_info, _product_daily):
    return utils.predict_for_date(produk, _product_daily, _model_info, target_date)


@st.cache_data(show_spinner=False)
def cached_predict_week(produk, start_date, n_days, _model_info, _product_daily):
    return utils.predict_week(produk, _product_daily, _model_info, start_date, n_days=n_days)


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

mode = st.radio(
    "Mode Prediksi",
    options=["🗓️ Harian (1 tanggal)", "📈 Mingguan (7 hari ke depan)"],
    horizontal=True,
)

semua_produk = sorted(models_produk.keys())
produk_dipilih = st.multiselect(
    "Filter produk (opsional — kosongkan untuk prediksi semua produk)",
    options=semua_produk,
    default=[],
)
produk_target = produk_dipilih if produk_dipilih else semua_produk

if mode.startswith("🗓️"):
    # ── Mode Harian ──
    target_date = st.date_input(
        "Pilih tanggal yang ingin diprediksi",
        value=last_date + timedelta(days=1),
        min_value=last_date + timedelta(days=1),
        max_value=last_date + timedelta(days=30),
    )

    if st.button("🔮 Prediksi", type="primary"):
        with st.spinner("Membentuk fitur & memanggil model tiap produk..."):
            hasil = []
            gagal = []
            for produk in produk_target:
                model_info = models_produk[produk]
                try:
                    pred = cached_predict_for_date(produk, target_date, model_info, product_daily)
                    hasil.append({"Produk": produk, "Prediksi (cup)": pred})
                except ValueError as e:
                    gagal.append((produk, str(e)))

        # ── Hasil Prediksi ──
        st.subheader(f"Hasil Prediksi — {target_date.strftime('%d %B %Y')}")
        df_hasil = pd.DataFrame(hasil).sort_values("Prediksi (cup)", ascending=False).reset_index(drop=True)
        df_hasil["Prediksi (cup)"] = df_hasil["Prediksi (cup)"].round().astype(int)
        st.dataframe(df_hasil, use_container_width=True, hide_index=True)

        total_cup = df_hasil["Prediksi (cup)"].sum()
        label_total = "Total Prediksi Produk Terpilih" if produk_dipilih else "Total Prediksi Seluruh Produk"
        st.metric(label_total, f"{total_cup:.0f} cup")

        if gagal:
            with st.expander(f"⚠️ {len(gagal)} produk gagal diprediksi"):
                for produk, msg in gagal:
                    st.write(f"- **{produk}**: {msg}")

else:
    # ── Mode Mingguan (7 hari ke depan) ──
    start_date = st.date_input(
        "Mulai prediksi dari tanggal",
        value=last_date + timedelta(days=1),
        min_value=last_date + timedelta(days=1),
        max_value=last_date + timedelta(days=24),  # sisakan ruang 7 hari sebelum batas 30 hari
    )
    end_date_preview = start_date + timedelta(days=6)
    st.caption(f"Rentang prediksi: **{start_date.strftime('%d %b %Y')} – {end_date_preview.strftime('%d %b %Y')}** (7 hari).")

    if st.button("🔮 Prediksi 7 Hari", type="primary"):
        with st.spinner("Membentuk fitur & memanggil model tiap produk (rekursif 7 hari)..."):
            per_produk_rows = {}  # produk -> list of {'tanggal', 'prediksi'}
            gagal = []
            for produk in produk_target:
                model_info = models_produk[produk]
                try:
                    rows = cached_predict_week(produk, start_date, 7, model_info, product_daily)
                    per_produk_rows[produk] = rows
                except ValueError as e:
                    gagal.append((produk, str(e)))

        if per_produk_rows:
            # ── Tabel Harian (baris = tanggal, kolom = produk) ──
            st.subheader(f"Prediksi Harian — {start_date.strftime('%d %b')} s/d {end_date_preview.strftime('%d %b %Y')}")
            tanggal_list = [r["tanggal"] for r in next(iter(per_produk_rows.values()))]
            df_harian = pd.DataFrame(
                {"Tanggal": [t.strftime("%a, %d %b %Y") for t in tanggal_list]}
            )
            for produk, rows in per_produk_rows.items():
                df_harian[produk] = [r["prediksi"] for r in rows]
            df_harian["Total Harian"] = df_harian[list(per_produk_rows.keys())].sum(axis=1)
            st.dataframe(df_harian, use_container_width=True, hide_index=True)

            # ── Total Mingguan per Produk ──
            st.subheader("Total Mingguan per Produk")
            df_total_minggu = pd.DataFrame(
                [
                    {"Produk": produk, "Total 7 Hari (cup)": sum(r["prediksi"] for r in rows)}
                    for produk, rows in per_produk_rows.items()
                ]
            ).sort_values("Total 7 Hari (cup)", ascending=False).reset_index(drop=True)
            st.dataframe(df_total_minggu, use_container_width=True, hide_index=True)

            grand_total = df_total_minggu["Total 7 Hari (cup)"].sum()
            label_total = (
                "Total Prediksi 7 Hari — Produk Terpilih" if produk_dipilih
                else "Total Prediksi 7 Hari — Seluruh Produk"
            )
            st.metric(label_total, f"{grand_total:.0f} cup")

        if gagal:
            with st.expander(f"⚠️ {len(gagal)} produk gagal diprediksi"):
                for produk, msg in gagal:
                    st.write(f"- **{produk}**: {msg}")

st.markdown("---")
st.caption("Skripsi — Prediksi Penjualan Kopi Per Produk (Harian) dengan XGBoost (CRISP-DM).")
