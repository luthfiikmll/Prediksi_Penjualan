"""
utils.py — Fungsi shared untuk app.py (Streamlit).
Logika feature engineering & metrik di sini HARUS identik dengan yang dipakai
saat training di notebook (xgb_kopi_all_products_v2.ipynb), supaya prediksi
konsisten antara training dan deployment.
"""

import numpy as np
import pandas as pd
import holidays

# ── Daftar fitur — HARUS sama urutan/isinya dengan notebook ──
FEATURES_DAILY_BASE = [
    'dayofweek', 'month', 'day', 'week_of_year',
    'is_holiday', 'is_weekend',
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
    'ewm_7',
]
FEATURES_DAILY = FEATURES_DAILY_BASE + ['dow_target_enc']

FEATURES_WEEKLY = [
    'month', 'week_of_year', 't_index',
    'is_holiday', 'is_payday',
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_7',
    'rolling_mean_7', 'rolling_mean_14',
    'rolling_std_7',
    'ewm_7', 'ewm_14',
]

DOW_LABEL = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


def load_luna_pos(uploaded_file):
    """Baca file export Luna POS (CSV, delimiter ';'). Kolom wajib: Tanggal, Produk, Qty."""
    df = pd.read_csv(uploaded_file, sep=';')
    df = df.drop_duplicates().reset_index(drop=True)

    required = {'Tanggal', 'Produk', 'Qty'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}. "
            f"Pastikan file adalah export Luna POS dengan kolom Tanggal, Produk, Qty."
        )

    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y %H:%M', errors='coerce')
    if df['Tanggal'].isna().any():
        # fallback kalau format jam tidak ada
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    df = df.dropna(subset=['Tanggal']).reset_index(drop=True)
    df['date'] = df['Tanggal'].dt.date
    return df


def build_daily(df):
    """Agregasi harian total cup — untuk Skenario 1."""
    daily = df.groupby('date')['Qty'].sum().reset_index()
    daily.columns = ['tanggal', 'penjualan']
    daily['tanggal'] = pd.to_datetime(daily['tanggal'])
    return daily


def build_weekly_per_product(df):
    """Agregasi mingguan per produk — untuk Skenario 2."""
    df = df.copy()
    df['week'] = df['Tanggal'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly = df.groupby(['week', 'Produk'])['Qty'].sum().reset_index()
    weekly.columns = ['tanggal', 'Produk', 'penjualan']
    return weekly


def build_features(series_df, required_cols=None):
    """Identik dengan build_features() di notebook."""
    ds = series_df.copy()

    q99 = ds['penjualan'].quantile(0.99)
    rolling_med = ds['penjualan'].rolling(7, min_periods=1, center=False).median()
    ds['penjualan'] = np.where(ds['penjualan'] > q99, rolling_med, ds['penjualan'])

    ds['penjualan_log'] = np.log1p(ds['penjualan'])

    ds['dayofweek'] = ds['tanggal'].dt.dayofweek
    ds['month'] = ds['tanggal'].dt.month
    ds['day'] = ds['tanggal'].dt.day
    ds['week_of_year'] = ds['tanggal'].dt.isocalendar().week.astype(int)
    ds['t_index'] = np.arange(len(ds))

    years = ds['tanggal'].dt.year.unique().tolist()
    id_holidays = holidays.Indonesia(years=years)
    ds['is_holiday'] = ds['tanggal'].isin(id_holidays).astype(int)
    ds['is_weekend'] = (ds['dayofweek'] >= 5).astype(int)

    tgl_sorted = ds['tanggal'].sort_values()
    freq_days = tgl_sorted.diff().median()
    span_days = max(int(freq_days.days), 1) if pd.notna(freq_days) else 1

    def _contains_payday(start_date, span):
        rng = pd.date_range(start_date, periods=span)
        return int((rng.day >= 25).any())

    ds['is_payday'] = ds['tanggal'].apply(lambda d: _contains_payday(d, span_days)).astype(int)

    for lag in [1, 2, 3, 4, 7, 14, 21, 30]:
        ds[f'lag_{lag}'] = ds['penjualan_log'].shift(lag)

    for w in [7, 14, 30]:
        ds[f'rolling_mean_{w}'] = ds['penjualan_log'].rolling(w).mean()
        ds[f'rolling_std_{w}'] = ds['penjualan_log'].rolling(w).std()

    ds['ewm_7'] = ds['penjualan_log'].ewm(span=7, adjust=False).mean()
    ds['ewm_14'] = ds['penjualan_log'].ewm(span=14, adjust=False).mean()

    if required_cols is None:
        ds = ds.dropna().reset_index(drop=True)
    else:
        keep_cols = list(dict.fromkeys(list(required_cols) + ['tanggal', 'penjualan', 'penjualan_log']))
        keep_cols = [c for c in keep_cols if c in ds.columns]
        ds = ds.dropna(subset=keep_cols).reset_index(drop=True)
    return ds


def get_metrics(y_true, y_pred):
    """Identik dengan get_metrics() di notebook: MAE, RMSE, MAPE, SMAPE.
    MAPE dihitung hanya pada baris actual != 0 (dan dilaporkan sebagai NaN
    kalau semua actual = 0, konsisten dengan perilaku training)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    denom = np.abs(y_true) + np.abs(y_pred)
    smape_mask = denom != 0
    if smape_mask.any():
        smape = np.mean(2 * np.abs(y_true[smape_mask] - y_pred[smape_mask]) / denom[smape_mask]) * 100
    else:
        smape = np.nan

    return {
        'MAE': round(float(mae), 4),
        'RMSE': round(float(rmse), 4),
        'MAPE': round(float(mape), 2) if not np.isnan(mape) else np.nan,
        'SMAPE': round(float(smape), 2) if not np.isnan(smape) else np.nan,
    }


def apply_dow_encoding(ds, dow_map, global_mean):
    ds = ds.copy()
    ds['dow_target_enc'] = ds['dayofweek'].map(dow_map).fillna(global_mean)
    return ds


def recursive_forecast_daily(history_daily, model, features, dow_map, n_ahead=7):
    """Forecast N hari ke depan (Skenario 1) secara rekursif: prediksi hari t+1
    dimasukkan kembali sebagai 'histori' untuk menghitung lag/rolling hari t+2, dst.
    """
    global_mean = float(np.mean(list(dow_map.values())))
    work = history_daily[['tanggal', 'penjualan']].copy()
    last_date = work['tanggal'].max()
    preds = []

    for step in range(n_ahead):
        next_date = last_date + pd.Timedelta(days=step + 1)
        # baris dummy untuk hari yang diprediksi (nilai penjualan diisi 0, tidak dipakai)
        dummy = pd.DataFrame({'tanggal': [next_date], 'penjualan': [0.0]})
        ds = pd.concat([work, dummy], ignore_index=True)
        ds_feat = build_features(ds, required_cols=FEATURES_DAILY_BASE)
        row = ds_feat[ds_feat['tanggal'] == next_date]
        if row.empty:
            break
        row = apply_dow_encoding(row, dow_map, global_mean)
        X_next = row[features]
        pred_log = model.predict(X_next)[0]
        pred = float(np.expm1(pred_log))
        preds.append({'tanggal': next_date, 'prediksi_cup': round(pred, 1)})
        work = pd.concat(
            [work, pd.DataFrame({'tanggal': [next_date], 'penjualan': [pred]})],
            ignore_index=True,
        )

    return pd.DataFrame(preds)


def recursive_forecast_weekly(history_weekly, model, features, n_ahead=4):
    """Forecast N minggu ke depan (Skenario 2, per produk) secara rekursif."""
    work = history_weekly[['tanggal', 'penjualan']].copy().sort_values('tanggal').reset_index(drop=True)
    last_date = work['tanggal'].max()
    preds = []

    for step in range(n_ahead):
        next_date = last_date + pd.Timedelta(weeks=step + 1)
        dummy = pd.DataFrame({'tanggal': [next_date], 'penjualan': [0.0]})
        ds = pd.concat([work, dummy], ignore_index=True)
        ds_feat = build_features(ds, required_cols=FEATURES_WEEKLY)
        row = ds_feat[ds_feat['tanggal'] == next_date]
        if row.empty:
            break
        X_next = row[features]
        pred_log = model.predict(X_next)[0]
        pred = float(np.expm1(pred_log))
        preds.append({'tanggal': next_date, 'prediksi_cup': round(max(pred, 0), 1)})
        work = pd.concat(
            [work, pd.DataFrame({'tanggal': [next_date], 'penjualan': [pred]})],
            ignore_index=True,
        )

    return pd.DataFrame(preds)
