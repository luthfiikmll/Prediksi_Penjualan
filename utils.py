import pandas as pd
import numpy as np
import holidays

# Fitur -- IDENTIK dengan notebook skripsi_xgb
FEATURES_DAILY_BASE = [
    'dayofweek', 'month', 'day', 'week_of_year',
    'is_holiday', 'is_weekend',
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21', 'lag_30',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
    'rolling_std_7',  'rolling_std_14',  'rolling_std_30',
    'ewm_7', 'ewm_14',
]
FEATURES_DAILY = FEATURES_DAILY_BASE + ['dow_target_enc']


def build_features(series_df):
    """Identik dengan build_features() di notebook. JANGAN diubah."""
    ds = series_df.copy()

    q99         = ds['penjualan'].quantile(0.99)
    rolling_med = ds['penjualan'].rolling(7, min_periods=1, center=False).median()
    ds['penjualan'] = np.where(ds['penjualan'] > q99, rolling_med, ds['penjualan'])

    ds['penjualan_log'] = np.log1p(ds['penjualan'])
    ds['dayofweek']    = ds['tanggal'].dt.dayofweek
    ds['month']        = ds['tanggal'].dt.month
    ds['day']          = ds['tanggal'].dt.day
    ds['week_of_year'] = ds['tanggal'].dt.isocalendar().week.astype(int)

    years = ds['tanggal'].dt.year.unique().tolist()
    id_holidays = holidays.Indonesia(years=years)
    ds['is_holiday'] = ds['tanggal'].isin(id_holidays).astype(int)
    ds['is_weekend']  = (ds['dayofweek'] >= 5).astype(int)

    for lag in [1, 2, 3, 7, 14, 21, 30]:
        ds[f'lag_{lag}'] = ds['penjualan_log'].shift(lag)

    for w in [7, 14, 30]:
        ds[f'rolling_mean_{w}'] = ds['penjualan_log'].rolling(w).mean()
        ds[f'rolling_std_{w}']  = ds['penjualan_log'].rolling(w).std()

    ds['ewm_7']  = ds['penjualan_log'].ewm(span=7,  adjust=False).mean()
    ds['ewm_14'] = ds['penjualan_log'].ewm(span=14, adjust=False).mean()

    ds = ds.dropna().reset_index(drop=True)
    return ds


def build_input_row(target_date, daily_sales, dow_map):
    """Build fitur untuk satu tanggal prediksi."""
    hist = daily_sales[['tanggal', 'penjualan']].copy()
    new_row = pd.DataFrame({'tanggal': [pd.Timestamp(target_date)], 'penjualan': [0.0]})
    hist = pd.concat([hist, new_row], ignore_index=True)
    hist = hist.drop_duplicates('tanggal').sort_values('tanggal').reset_index(drop=True)

    ds  = build_features(hist)
    row = ds[ds['tanggal'] == pd.Timestamp(target_date)].copy()
    if row.empty:
        return None

    dow = row['dayofweek'].values[0]
    row['dow_target_enc'] = dow_map.get(dow, dow_map.mean())

    if any(f not in row.columns for f in FEATURES_DAILY):
        return None

    return row[FEATURES_DAILY]


def build_7day_forecast(model, daily_sales, dow_map):
    """
    Forecasting 7 hari ke depan secara ITERATIF.
    Hasil hari sebelumnya dimasukkan ke historis agar
    lag features hari berikutnya akurat.
    """
    from datetime import timedelta

    hist      = daily_sales[['tanggal', 'penjualan']].copy()
    last_date = hist['tanggal'].max().date()
    hari_id   = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
    hasil     = []

    for i in range(7):
        tgl = last_date + timedelta(days=i + 1)
        row = build_input_row(tgl, hist, dow_map)
        if row is None:
            continue

        pred_cup = max(0, round(np.expm1(model.predict(row)[0])))
        hasil.append({
            'tanggal': tgl,
            'Tanggal': tgl.strftime('%d %b %Y'),
            'Hari':    hari_id[tgl.weekday()],
            'Prediksi (cup)': pred_cup,
            'Barista': rekomendasi_barista(pred_cup)[0]
        })

        # Masukkan hasil ke historis untuk hari berikutnya
        hist = pd.concat([hist, pd.DataFrame({
            'tanggal': [pd.Timestamp(tgl)], 'penjualan': [float(pred_cup)]
        })], ignore_index=True)
        hist = hist.drop_duplicates('tanggal').sort_values('tanggal').reset_index(drop=True)

    return hasil


def process_luna_upload(uploaded_file):
    """Proses file export Luna POS (CSV atau Excel)."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=';')
        else:
            df = pd.read_excel(uploaded_file)

        required = ['Tanggal', 'Produk', 'Qty']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return None, f'Kolom tidak ditemukan: {missing}. Pastikan ini file export Luna POS.'

        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y %H:%M', errors='coerce')
        df = df.dropna(subset=['Tanggal'])
        df['date'] = df['Tanggal'].dt.date

        if len(df) == 0:
            return None, 'Tidak ada data valid setelah parsing tanggal.'

        daily = df.groupby('date')['Qty'].sum().reset_index()
        daily.columns = ['tanggal', 'penjualan']
        daily['tanggal'] = pd.to_datetime(daily['tanggal'])
        daily = daily.sort_values('tanggal').reset_index(drop=True)

        if len(daily) < 35:
            return None, f'Data terlalu sedikit ({len(daily)} hari). Minimal 35 hari.'

        return daily, None

    except Exception as e:
        return None, f'Error: {str(e)}'


def mape_kategori(mape):
    if mape < 10:   return 'Highly Accurate'
    elif mape < 20: return 'Good'
    elif mape < 50: return 'Reasonable'
    else:           return 'Inaccurate'


def rekomendasi_barista(pred_cup):
    if pred_cup < 8:     return 1, 'Sepi -- cukup 1 barista'
    elif pred_cup < 18:  return 2, 'Normal -- siapkan 2 barista'
    elif pred_cup < 28:  return 3, 'Ramai -- siapkan 3 barista'
    else:                return 4, 'Sangat ramai -- siapkan 4+ barista'
