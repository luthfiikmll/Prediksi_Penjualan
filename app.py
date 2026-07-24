"""
app.py — Aplikasi Prediksi Penjualan Kopi Per Produk (Harian/Mingguan/Bulanan).

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
def cached_predict_horizon(produk, start_date, n_days, _model_info, _product_daily):
    return utils.predict_week(produk, _product_daily, _model_info, start_date, n_days=n_days)


@st.cache_data(show_spinner=False)
def cached_dashboard_summary(_product_daily):
    return utils.get_dashboard_summary(_product_daily)


# ── Halaman Utama ──
st.title("☕ Prediksi Penjualan Kopi")
st.caption(
    "Coffee Shop Sans Your Day — prediksi penjualan harian minuman kopi "
    ", menggunakan model XGBoost dan optimasi Random Search."
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

# ══════════════════════════════════════════════════════════════
# ── Dashboard Ringkasan Penjualan (dari data histori) ──
# ══════════════════════════════════════════════════════════════
st.header("📊 Dashboard Penjualan")

total_per_produk, tren_harian, stats = cached_dashboard_summary(product_daily)

col1, col2, col3, col4 = st.columns(4)
# col1.metric("Total Terjual (histori)", f"{stats['total_cup']:,.0f} cup")
# col2.metric("Rata-rata per Hari", f"{stats['rata_rata_harian']:,.0f} cup")
col3.metric("Jumlah Produk", f"{stats['jumlah_produk']}")
# col4.metric("Rentang Data", f"{stats['jumlah_hari']} hari")

produk_terlaris = total_per_produk.iloc[0]
st.markdown(
    f"🏆 **Produk terlaris:** {produk_terlaris['Produk']} "
    f"({produk_terlaris['Total Terjual (cup)']:,.0f} cup terjual sepanjang histori)"
)

tab_top, tab_tren = st.tabs(["🏆 Produk Terlaris", "📈 Tren Penjualan Harian"])

with tab_top:
    top_n = st.slider("Tampilkan top N produk", min_value=3, max_value=len(total_per_produk),
                       value=min(10, len(total_per_produk)))
    df_top = total_per_produk.head(top_n).set_index("Produk")
    st.bar_chart(df_top["Total Terjual (cup)"])
    st.dataframe(total_per_produk, use_container_width=True, hide_index=True)

with tab_tren:
    st.line_chart(tren_harian.set_index("tanggal")["Total Cup"])
    st.caption("Total cup terjual (semua produk digabung) per hari, sepanjang data histori.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# ── Input Prediksi ──
# ══════════════════════════════════════════════════════════════
st.header("🔮 Prediksi Penjualan")

mode = st.radio(
    "Mode Prediksi",
    options=["🗓️ Harian (1 tanggal)", "📈 Mingguan (7 hari ke depan)", "🗓️📦 Bulanan (30 hari ke depan)"],
    horizontal=True,
)

semua_produk = sorted(models_produk.keys())
produk_dipilih = st.multiselect(
    "Filter produk (opsional — kosongkan untuk prediksi semua produk)",
    options=semua_produk,
    default=[],
)
produk_target = produk_dipilih if produk_dipilih else semua_produk


def render_horizon_prediction(n_days, tombol_label, judul_prefix, max_start_offset):
    """Render UI + hasil untuk mode Mingguan/Bulanan (n_days berturut-turut).
    Sama-sama pakai predict_week (generik lewat cached_predict_horizon), cuma beda n_days."""
    start_date = st.date_input(
        "Mulai prediksi dari tanggal",
        value=last_date + timedelta(days=1),
        min_value=last_date + timedelta(days=1),
        max_value=last_date + timedelta(days=max_start_offset),
    )
    end_date_preview = start_date + timedelta(days=n_days - 1)
    st.caption(
        f"Rentang prediksi: **{start_date.strftime('%d %b %Y')} – "
        f"{end_date_preview.strftime('%d %b %Y')}** ({n_days} hari)."
    )

    if st.button(tombol_label, type="primary"):
        with st.spinner(f"Membentuk fitur & memanggil model tiap produk (rekursif {n_days} hari)..."):
            per_produk_rows = {}  # produk -> list of {'tanggal', 'prediksi'}
            gagal = []
            for produk in produk_target:
                model_info = models_produk[produk]
                try:
                    rows = cached_predict_horizon(produk, start_date, n_days, model_info, product_daily)
                    per_produk_rows[produk] = rows
                except ValueError as e:
                    gagal.append((produk, str(e)))

        if per_produk_rows:
            # ── Tabel Harian ──
            st.subheader(f"{judul_prefix} — {start_date.strftime('%d %b')} s/d {end_date_preview.strftime('%d %b %Y')}")
            tanggal_list = [r["tanggal"] for r in next(iter(per_produk_rows.values()))]
            df_harian = pd.DataFrame({"tanggal": tanggal_list})
            for produk, rows in per_produk_rows.items():
                df_harian[produk] = [r["prediksi"] for r in rows]
            df_harian["Total Harian"] = df_harian[list(per_produk_rows.keys())].sum(axis=1)

            df_tampil = df_harian.copy()
            df_tampil.insert(0, "Tanggal", df_tampil["tanggal"].dt.strftime("%a, %d %b %Y"))
            df_tampil = df_tampil.drop(columns=["tanggal"])
            st.dataframe(df_tampil, use_container_width=True, hide_index=True)

            # ── Grafik Tren ──
            st.markdown(f"**📈 Grafik Tren {n_days} Hari**")
            chart_cols = list(per_produk_rows.keys()) if len(per_produk_rows) <= 8 else []
            if chart_cols:
                st.line_chart(df_harian.set_index("tanggal")[chart_cols])
                st.caption("Tren prediksi harian per produk. Kolom Total Harian ditampilkan terpisah di bawah.")
            st.line_chart(df_harian.set_index("tanggal")["Total Harian"])

            # ── Total per Produk ──
            label_periode = "7 Hari" if n_days == 7 else ("30 Hari" if n_days == 30 else f"{n_days} Hari")
            st.subheader(f"Total {label_periode} per Produk")
            df_total = pd.DataFrame(
                [
                    {"Produk": produk, f"Total {label_periode} (cup)": sum(r["prediksi"] for r in rows)}
                    for produk, rows in per_produk_rows.items()
                ]
            ).sort_values(f"Total {label_periode} (cup)", ascending=False).reset_index(drop=True)
            st.bar_chart(df_total.set_index("Produk")[f"Total {label_periode} (cup)"])
            st.dataframe(df_total, use_container_width=True, hide_index=True)

            grand_total = df_total[f"Total {label_periode} (cup)"].sum()
            label_metric = (
                f"Total Prediksi {label_periode} — Produk Terpilih" if produk_dipilih
                else f"Total Prediksi {label_periode} — Seluruh Produk"
            )
            st.metric(label_metric, f"{grand_total:,.0f} cup")

        if gagal:
            with st.expander(f"⚠️ {len(gagal)} produk gagal diprediksi"):
                for produk, msg in gagal:
                    st.write(f"- **{produk}**: {msg}")


if mode.startswith("🗓️ Harian"):
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

        st.markdown("**📊 Grafik Prediksi per Produk**")
        st.bar_chart(df_hasil.set_index("Produk")["Prediksi (cup)"])

        total_cup = df_hasil["Prediksi (cup)"].sum()
        label_total = "Total Prediksi Produk Terpilih" if produk_dipilih else "Total Prediksi Seluruh Produk"
        st.metric(label_total, f"{total_cup:.0f} cup")

        if gagal:
            with st.expander(f"⚠️ {len(gagal)} produk gagal diprediksi"):
                for produk, msg in gagal:
                    st.write(f"- **{produk}**: {msg}")

elif mode.startswith("📈 Mingguan"):
    # ── Mode Mingguan (7 hari ke depan) ──
    render_horizon_prediction(
        n_days=7,
        tombol_label="🔮 Prediksi 7 Hari",
        judul_prefix="Prediksi Harian",
        max_start_offset=24,  # sisakan ruang 7 hari sebelum batas horizon 30 hari
    )

else:
    # ── Mode Bulanan (30 hari ke depan) ──
    render_horizon_prediction(
        n_days=30,
        tombol_label="🔮 Prediksi 30 Hari",
        judul_prefix="Prediksi Harian",
        max_start_offset=1,  # horizon model dibatasi maksimal 30 hari dari data histori terakhir
    )

st.markdown("---")
with st.expander("ℹ️ Tentang Model"):
    st.dataframe(eval_summary, use_container_width=True, hide_index=True)
    st.caption(
        "Tabel di atas menunjukkan performa masing-masing model produk pada "
        "data pengujian (train vs test) saat pelatihan."
    )

st.caption("Skripsi — Prediksi Penjualan harian Minuman Kopi dengan XGBoost dan Random Search.")
