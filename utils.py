import pandas as pd
import numpy as np
import holidays

# ── Fitur -- identik dengan notebook training lu ─────────────────────────────
FEATURES_DAILY_BASE = [
    'month', 'day', 'week_of_year', 'quarter',
    'is_holiday', 'is_weekend', 'is_payday',
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21', 'lag_30',
    'rolling_mean_7',  'rolling_mean_14', 'rolling_mean_30',
    'rolling_std_7',   'rolling_std_14',  'rolling_std_30',
    'rolling_min_7',   'rolling_max_7',
    'ewm_7', 'ewm_14', 'ewm_30',
    'diff_1', 'diff_7',
]
FEATURES_DAILY = FEATURES_DAILY_BASE + ['dow_target_enc']

# Top 3 produk -- sesuai models_s2.pkl
TOP3_PRODUCTS = ['Kopi Aren / ICE', 'Americano / ICE', 'Capuccino / ICE']


def build_features(series_df):
    """Identik dengan build_features() di notebook training."""
    ds = series_df.copy()

    q99         = ds['penjualan'].quantile(0.99)
    rolling_med = ds['penjualan'].rolling(7, min_periods=1, center=False).median()
    ds['penjualan'] = np.where(ds['penjualan'] > q99, rolling_med, ds['penjualan'])

    ds['penjualan_log'] = np.log1p(ds['penjualan'])

    ds['dayofweek']    = ds['tanggal'].dt.dayofweek
    ds['month']        = ds['tanggal'].dt.month
    ds['day']          = ds['tanggal'].dt.day
    ds['week_of_year'] = ds['tanggal'].dt.isocalendar().week.astype(int)
    ds['quarter']      = ds['tanggal'].dt.quarter

    years = ds['tanggal'].dt.year.unique().tolist()
    id_holidays = holidays.Indonesia(years=years)
    ds['is_holiday'] = ds['tanggal'].isin(id_holidays).astype(int)
    ds['is_weekend']  = (ds['dayofweek'] >= 5).astype(int)
    ds['is_payday']   = (ds['tanggal'].dt.day >= 25).astype(int)

    for lag in [1, 2, 3, 7, 14, 21, 30]:
        ds[f'lag_{lag}'] = ds['penjualan_log'].shift(lag)

    for w in [7, 14, 30]:
        ds[f'rolling_mean_{w}'] = ds['penjualan_log'].rolling(w).mean()
        ds[f'rolling_std_{w}']  = ds['penjualan_log'].rolling(w).std()
        ds[f'rolling_min_{w}']  = ds['penjualan_log'].rolling(w).min()
        ds[f'rolling_max_{w}']  = ds['penjualan_log'].rolling(w).max()

    ds['ewm_7']  = ds['penjualan_log'].ewm(span=7,  adjust=False).mean()
    ds['ewm_14'] = ds['penjualan_log'].ewm(span=14, adjust=False).mean()
    ds['ewm_30'] = ds['penjualan_log'].ewm(span=30, adjust=False).mean()

    ds['diff_1'] = ds['penjualan_log'].diff(1)
    ds['diff_7'] = ds['penjualan_log'].diff(7)

    ds = ds.dropna().reset_index(drop=True)
    return ds


def build_input_row_daily(target_date, daily_sales, dow_map):
    """Build satu baris fitur untuk prediksi Skenario 1."""
    hist = daily_sales[['tanggal', 'penjualan']].copy()
    new_row = pd.DataFrame({
        'tanggal':   [pd.Timestamp(target_date)],
        'penjualan': [0.0]
    })
    hist = pd.concat([hist, new_row], ignore_index=True)
    hist = hist.drop_duplicates('tanggal').sort_values('tanggal').reset_index(drop=True)

    ds  = build_features(hist)
    row = ds[ds['tanggal'] == pd.Timestamp(target_date)].copy()
    if row.empty:
        return None

    # DOW target encoding -- sesuai dow_map dari pkl
    dow = row['dayofweek'].values[0]
    global_mean = dow_map.mean()
    row['dow_target_enc'] = dow_map.get(dow, global_mean)

    missing = [f for f in FEATURES_DAILY if f not in row.columns]
    if missing:
        return None

    return row[FEATURES_DAILY]


def build_input_row_s2(target_date, product_daily, produk, daily_sales, dow_map):
    """Build satu baris fitur untuk prediksi Skenario 2 (per produk)."""
    prod_data = product_daily[product_daily['Produk'] == produk][['tanggal','penjualan']].copy()

    # Merge dengan semua tanggal -- fillna=0 untuk hari tanpa transaksi
    all_dates = daily_sales[['tanggal']].copy()
    hist = all_dates.merge(prod_data, on='tanggal', how='left').fillna(0)
    hist['penjualan'] = hist['penjualan'].astype(float)

    new_row = pd.DataFrame({
        'tanggal':   [pd.Timestamp(target_date)],
        'penjualan': [0.0]
    })
    hist = pd.concat([hist, new_row], ignore_index=True)
    hist = hist.drop_duplicates('tanggal').sort_values('tanggal').reset_index(drop=True)

    ds  = build_features(hist)
    row = ds[ds['tanggal'] == pd.Timestamp(target_date)].copy()
    if row.empty:
        return None

    dow = row['dayofweek'].values[0]
    global_mean = dow_map.mean()
    row['dow_target_enc'] = dow_map.get(dow, global_mean)

    missing = [f for f in FEATURES_DAILY if f not in row.columns]
    if missing:
        return None

    return row[FEATURES_DAILY]


def mape_kategori(mape):
    if mape < 10:   return 'Highly Accurate'
    elif mape < 20: return 'Good'
    elif mape < 50: return 'Reasonable'
    else:           return 'Inaccurate'


def rekomendasi_barista(pred_cup):
    if pred_cup < 40:    return 1, 'Sepi -- cukup 1 barista'
    elif pred_cup < 80:  return 2, 'Normal -- siapkan 2 barista'
    elif pred_cup < 120: return 3, 'Ramai -- siapkan 3 barista'
    else:                return 4, 'Sangat ramai -- siapkan 4+ barista'
