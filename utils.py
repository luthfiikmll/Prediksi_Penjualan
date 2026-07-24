"""
utils.py — Fungsi shared untuk app.py (Streamlit).
Logika feature engineering di sini HARUS identik dengan
model_prediksi_per_produk_harian.ipynb, supaya prediksi konsisten
antara training dan deployment.
"""

import numpy as np
import pandas as pd
import holidays

FEATURES_DAILY = [
    'dayofweek', 'is_weekend', 'month', 'day', 't_index',
    'is_holiday', 'is_payday',
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_7', 'lag_14',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7',
    'ewm_7', 'ewm_14',
]


def load_product_daily(path="product_daily.csv"):
    """Baca histori penjualan harian per produk yang dibundel bareng model."""
    df = pd.read_csv(path)
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    return df


def build_features(series_df, required_cols=None):
    """Identik dengan build_features() di notebook."""
    ds = series_df.copy()

    q99 = ds['penjualan'].quantile(0.99)
    rolling_med = ds['penjualan'].rolling(7, min_periods=1, center=False).median()
    ds['penjualan'] = np.where(ds['penjualan'] > q99, rolling_med, ds['penjualan'])
    ds['penjualan_log'] = np.log1p(ds['penjualan'])

    ds['dayofweek'] = ds['tanggal'].dt.dayofweek
    ds['is_weekend'] = (ds['dayofweek'] >= 5).astype(int)
    ds['month'] = ds['tanggal'].dt.month
    ds['day'] = ds['tanggal'].dt.day
    ds['t_index'] = np.arange(len(ds))

    years = ds['tanggal'].dt.year.unique().tolist()
    id_holidays = holidays.Indonesia(years=years)
    ds['is_holiday'] = ds['tanggal'].isin(id_holidays).astype(int)

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


def load_xgb_model_from_bytes(model_bytes):
    """Load model XGBoost dari raw bytes format JSON, BUKAN dari pickle objek
    XGBRegressor langsung — supaya tidak corrupt kalau versi xgboost server
    beda dengan versi training."""
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(bytearray(model_bytes))
    return model


def _forecast_range(produk, product_daily, model_info, end_date):
    """Helper internal: prediksi rekursif dari hari terakhir histori sampai end_date.

    Dihitung sekali jalan dari last_date+1 s/d end_date (inklusif): prediksi
    hari t dipakai sebagai histori untuk menghitung fitur lag/rolling hari
    t+1, dst. Mengembalikan (last_date, list of (tanggal, pred_float)) untuk
    SEMUA hari di rentang tsb, supaya predict_for_date maupun predict_week
    bisa reuse loop yang sama tanpa hitung ulang dari nol.
    """
    prod_hist = product_daily[product_daily['Produk'] == produk][['tanggal', 'penjualan']].copy()
    prod_hist = prod_hist.sort_values('tanggal').reset_index(drop=True)
    last_date = prod_hist['tanggal'].max()

    end_date = pd.Timestamp(end_date)
    if end_date <= last_date:
        raise ValueError(f"Tanggal harus setelah {last_date.date()} (data histori terakhir).")

    n_ahead = (end_date - last_date).days
    features = model_info['features']
    model = model_info['model']

    work = prod_hist.copy()
    daily_preds = []
    for step in range(n_ahead):
        next_date = last_date + pd.Timedelta(days=step + 1)
        dummy = pd.DataFrame({'tanggal': [next_date], 'penjualan': [0.0]})
        ds = pd.concat([work, dummy], ignore_index=True)
        ds_feat = build_features(ds, required_cols=features)
        row = ds_feat[ds_feat['tanggal'] == next_date]
        if row.empty:
            raise ValueError(
                f"Histori untuk produk '{produk}' tidak cukup panjang untuk membentuk "
                f"fitur pada {next_date.date()}."
            )
        X_next = row[features]
        pred_log = model.predict(X_next)[0]
        pred = float(max(np.expm1(pred_log), 0))
        work = pd.concat(
            [work, pd.DataFrame({'tanggal': [next_date], 'penjualan': [pred]})],
            ignore_index=True,
        )
        daily_preds.append((next_date, pred))

    return last_date, daily_preds


def predict_for_date(produk, product_daily, model_info, target_date):
    """Pembentukan Fitur Otomatis + Prediksi untuk SATU produk pada SATU tanggal target.

    Alur: Input tanggal -> Pembentukan fitur otomatis -> Model produk dipanggil
    -> Prediksi jumlah cup -> (dikembalikan ke pemanggil untuk ditampilkan).
    """
    _, daily_preds = _forecast_range(produk, product_daily, model_info, target_date)
    pred_final = daily_preds[-1][1]
    return int(round(pred_final))


def predict_week(produk, product_daily, model_info, start_date, n_days=7):
    """Prediksi rekursif untuk n_days (default 7) hari berturut-turut mulai
    start_date (inklusif). Mengembalikan list of dict:
    [{'tanggal': Timestamp, 'prediksi': int}, ...] sebanyak n_days,
    diurutkan dari start_date sampai start_date + (n_days - 1) hari.
    """
    start_date = pd.Timestamp(start_date)
    end_date = start_date + pd.Timedelta(days=n_days - 1)
    _, daily_preds = _forecast_range(produk, product_daily, model_info, end_date)
    return [
        {'tanggal': tgl, 'prediksi': int(round(pred))}
        for tgl, pred in daily_preds
        if tgl >= start_date
    ]


def get_dashboard_summary(product_daily):
    """Ringkasan dashboard dari data HISTORI penjualan (product_daily), bukan
    hasil prediksi. Dipakai untuk menampilkan produk terlaris & tren harian
    di bagian atas app.

    Mengembalikan:
      total_per_produk : DataFrame ['Produk', 'Total Terjual (cup)'], urut
                          menurun berdasarkan total penjualan.
      tren_harian       : DataFrame ['tanggal', 'Total Cup'] — total semua
                          produk digabung per hari, untuk grafik tren harian.
      tren_mingguan     : DataFrame ['minggu_mulai', 'Total Cup'] — agregasi
                          mingguan (Senin s/d Minggu) dari tren_harian.
      tren_bulanan      : DataFrame ['bulan', 'Total Cup'] — agregasi bulanan
                          (per awal bulan) dari tren_harian.
      stats             : dict ringkasan (total_cup, jumlah_produk,
                          jumlah_hari, rata_rata_harian, tanggal_awal,
                          tanggal_akhir).
    """
    total_per_produk = (
        product_daily.groupby('Produk')['penjualan'].sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'penjualan': 'Total Terjual (cup)'})
    )

    tren_harian = (
        product_daily.groupby('tanggal')['penjualan'].sum()
        .reset_index()
        .rename(columns={'penjualan': 'Total Cup'})
        .sort_values('tanggal')
        .reset_index(drop=True)
    )

    tren_mingguan = (
        tren_harian.set_index('tanggal')['Total Cup']
        .resample('W-MON', label='left', closed='left')
        .sum()
        .reset_index()
        .rename(columns={'tanggal': 'minggu_mulai'})
    )

    tren_bulanan = (
        tren_harian.set_index('tanggal')['Total Cup']
        .resample('MS')
        .sum()
        .reset_index()
        .rename(columns={'tanggal': 'bulan'})
    )

    tanggal_awal = product_daily['tanggal'].min()
    tanggal_akhir = product_daily['tanggal'].max()
    stats = {
        'total_cup': float(product_daily['penjualan'].sum()),
        'jumlah_produk': int(product_daily['Produk'].nunique()),
        'jumlah_hari': int(tren_harian.shape[0]),
        'rata_rata_harian': float(tren_harian['Total Cup'].mean()),
        'tanggal_awal': tanggal_awal,
        'tanggal_akhir': tanggal_akhir,
    }

    return total_per_produk, tren_harian, tren_mingguan, tren_bulanan, stats
