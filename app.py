"""
app.py — Streamlit demo untuk model forecasting penjualan kopi (XGBoost).

Fungsi utama: EVALUASI / DEMO MODEL.
  - User upload data export Luna POS (CSV, sep=';', kolom: Tanggal, Produk, Qty)
  - App menjalankan feature engineering yang identik dengan notebook training
  - Menampilkan prediksi vs aktual, metrik (MAE/RMSE/MAPE/SMAPE), dan feature importance
  - Tambahan: forecast singkat ke depan (7 hari untuk Skenario 1, 4 minggu untuk Skenario 2)

Artifact yang dibutuhkan di folder yang sama:
  model_xgb.pkl, dow_map.pkl, models_s2.pkl, eval_s1.csv, eval_s2.csv, utils.py
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import utils

st.set_page_config(page_title="Forecast Penjualan Kopi — XGBoost", page_icon="☕", layout="wide")


@st.cache_resource
def load_artifacts():
    with open("model_xgb.pkl", "rb") as f:
        s1 = pickle.load(f)
    with open("dow_map.pkl", "rb") as f:
        dow_map = pickle.load(f)
    with open("models_s2.pkl", "rb") as f:
        s2 = pickle.load(f)
    eval_s1 = pd.read_csv("eval_s1.csv")
    eval_s2 = pd.read_csv("eval_s2.csv")
    return s1, dow_map, s2, eval_s1, eval_s2


def plot_pred_vs_actual(dates, actual, pred, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode="lines+markers", name="Aktual",
                              line=dict(color="#4B4B4B")))
    fig.add_trace(go.Scatter(x=dates, y=pred, mode="lines+markers", name="Prediksi",
                              line=dict(color="darkorange")))
    fig.update_layout(title=title, xaxis_title="Tanggal", yaxis_title="Cup",
                       legend=dict(orientation="h", y=1.1), height=420,
                       margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_feature_importance(model, features, title):
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    fig = go.Figure(go.Bar(
        x=[imp[i] for i in order],
        y=[features[i] for i in order],
        orientation="h",
        marker_color="darkorange",
    ))
    fig.update_layout(title=title, yaxis=dict(autorange="reversed"), height=420,
                       margin=dict(l=10, r=10, t=60, b=10))
    return fig


def metric_cards(m):
    cols = st.columns(4)
    cols[0].metric("MAE", f"{m['MAE']:.2f} cup")
    cols[1].metric("RMSE", f"{m['RMSE']:.2f} cup")
    cols[2].metric("MAPE", f"{m['MAPE']:.2f}%" if pd.notna(m['MAPE']) else "N/A")
    cols[3].metric("SMAPE", f"{m['SMAPE']:.2f}%" if pd.notna(m['SMAPE']) else "N/A")


st.title("☕ Forecast Penjualan Kopi")
st.caption(
    "prediksi vs aktual, metrik evaluasi, dan feature importance."
)

try:
    model_s1_obj, dow_map, models_s2, eval_s1_bench, eval_s2_bench = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Artifact model tidak ditemukan di folder ini "
        "(model_xgb.pkl / dow_map.pkl / models_s2.pkl / eval_s1.csv / eval_s2.csv). "
        f"Detail: {e}"
    )
    st.stop()

with st.sidebar:
    st.header("📂 Data")
    uploaded = st.file_uploader("Upload export Luna POS (.csv)", type=["csv"])
    st.markdown("---")
    st.markdown(
        "**Kolom wajib:** `Tanggal` (dd/mm/yyyy [hh:mm]), `Produk`, `Qty`\n\n"
        "Delimiter file harus `;` (sesuai export Luna POS)."
    )

tab1, tab2 = st.tabs(["📊 Skenario 1 — Total Cup Harian", "📦 Skenario 2 — Per Produk Mingguan"])

# ───────────────────────── SKENARIO 1 ─────────────────────────
with tab1:
    st.subheader("Benchmark model saat training")
    st.dataframe(eval_s1_bench, use_container_width=True, hide_index=True)

    if uploaded is None:
        st.info("⬅️ Upload file di sidebar untuk menjalankan evaluasi pada data barumu.")
    else:
        try:
            raw = utils.load_luna_pos(uploaded)
            daily = utils.build_daily(raw)
            ds = utils.build_features(daily, required_cols=utils.FEATURES_DAILY_BASE)

            if len(ds) < 10:
                st.warning(
                    f"Data valid setelah feature engineering hanya {len(ds)} baris. "
                    "Butuh histori lebih panjang (idealnya > 30 hari) agar lag/rolling feature terisi."
                )
            else:
                global_mean = float(np.mean(list(dow_map.values())))
                ds_enc = utils.apply_dow_encoding(ds, dow_map, global_mean)

                model_s1 = model_s1_obj["model"]
                features_s1 = model_s1_obj["features"]
                X = ds_enc[features_s1]
                pred = np.expm1(model_s1.predict(X))
                actual = ds_enc["penjualan"].values

                m = utils.get_metrics(actual, pred)
                st.subheader("Hasil pada data yang diupload")
                metric_cards(m)

                st.plotly_chart(
                    plot_pred_vs_actual(ds_enc["tanggal"], actual, pred, "Total Cup Harian — Aktual vs Prediksi"),
                    use_container_width=True,
                )

                st.plotly_chart(
                    plot_feature_importance(model_s1, features_s1, "Feature Importance — Skenario 1"),
                    use_container_width=True,
                )

                st.markdown("---")
                st.subheader("🔮 Forecast 7 hari ke depan")
                if st.button("Jalankan forecast 7 hari", key="fc_s1"):
                    fc = utils.recursive_forecast_daily(daily, model_s1, features_s1, dow_map, n_ahead=7)
                    st.dataframe(fc, use_container_width=True, hide_index=True)
                    fig = go.Figure(go.Scatter(x=fc["tanggal"], y=fc["prediksi_cup"],
                                                mode="lines+markers", line=dict(color="darkorange")))
                    fig.update_layout(title="Forecast 7 Hari — Total Cup", height=350,
                                       margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Forecast dihitung rekursif: prediksi hari t dipakai sebagai histori "
                        "untuk menghitung fitur lag/rolling hari t+1, dst. Akurasi menurun "
                        "semakin jauh horizon-nya."
                    )

        except ValueError as e:
            st.error(str(e))

# ───────────────────────── SKENARIO 2 ─────────────────────────
with tab2:
    st.subheader("Benchmark model saat training (per produk)")
    st.dataframe(eval_s2_bench, use_container_width=True, hide_index=True)

    produk_list = sorted(models_s2.keys())
    produk_pilihan = st.selectbox("Pilih produk", produk_list)

    if uploaded is None:
        st.info("⬅️ Upload file di sidebar untuk menjalankan evaluasi pada data barumu.")
    else:
        try:
            raw = utils.load_luna_pos(uploaded)
            weekly_all = utils.build_weekly_per_product(raw)

            if produk_pilihan not in weekly_all["Produk"].unique():
                st.warning(f"Produk '{produk_pilihan}' tidak ditemukan di data yang diupload.")
            else:
                weekly_prod = weekly_all[weekly_all["Produk"] == produk_pilihan][["tanggal", "penjualan"]]
                ds = utils.build_features(weekly_prod, required_cols=utils.FEATURES_WEEKLY)

                if len(ds) < 6:
                    st.warning(
                        f"Data mingguan valid untuk '{produk_pilihan}' hanya {len(ds)} baris. "
                        "Butuh histori lebih panjang (idealnya > 10 minggu)."
                    )
                else:
                    model_info = models_s2[produk_pilihan]
                    model_s2 = model_info["model"]
                    features_s2 = model_info["features"]

                    X = ds[features_s2]
                    pred = np.expm1(model_s2.predict(X))
                    actual = ds["penjualan"].values

                    m = utils.get_metrics(actual, pred)
                    st.subheader(f"Hasil pada data yang diupload — {produk_pilihan}")
                    st.caption(f"Model terpilih saat training: **{model_info['optimizer']}**")
                    metric_cards(m)

                    st.plotly_chart(
                        plot_pred_vs_actual(ds["tanggal"], actual, pred,
                                            f"{produk_pilihan} — Aktual vs Prediksi (Mingguan)"),
                        use_container_width=True,
                    )

                    st.plotly_chart(
                        plot_feature_importance(model_s2, features_s2, f"Feature Importance — {produk_pilihan}"),
                        use_container_width=True,
                    )

                    st.markdown("---")
                    st.subheader("🔮 Forecast 4 minggu ke depan")
                    if st.button("Jalankan forecast 4 minggu", key="fc_s2"):
                        fc = utils.recursive_forecast_weekly(weekly_prod, model_s2, features_s2, n_ahead=4)
                        st.dataframe(fc, use_container_width=True, hide_index=True)
                        fig = go.Figure(go.Scatter(x=fc["tanggal"], y=fc["prediksi_cup"],
                                                    mode="lines+markers", line=dict(color="darkorange")))
                        fig.update_layout(title=f"Forecast 4 Minggu — {produk_pilihan}", height=350,
                                           margin=dict(l=10, r=10, t=50, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            "Forecast dihitung rekursif per minggu. Untuk produk dengan penjualan "
                            "sangat jarang, MAPE bisa N/A (semua actual test = 0) — SMAPE jadi acuan utama."
                        )

        except ValueError as e:
            st.error(str(e))

st.markdown("---")
st.caption("Skripsi — Forecasting Penjualan Kopi dengan XGBoost (CRISP-DM). Demo model, bukan sistem produksi.")
