"""
utils.py — Fungsi shared untuk app.py (Streamlit).

"""

import numpy as np
import pandas as pd

# ── Data Hari Libur Nasional Indonesia (SUMBER RESMI PEMERINTAH) ──
HARI_LIBUR_NASIONAL_DICT = {
    # 2024
    '2024-01-01': 'Libur Nasional: Tahun Baru 2024 Masehi',
    '2024-02-08': 'Libur Nasional: Isra Mikraj Nabi Muhammad SAW',
    '2024-02-09': 'Cuti Bersama: Tahun Baru Imlek 2575 Kongzili',
    '2024-02-10': 'Libur Nasional: Tahun Baru Imlek 2575 Kongzili',
    '2024-03-11': 'Libur Nasional: Hari Suci Nyepi Tahun Baru Saka 1946',
    '2024-03-12': 'Cuti Bersama: Hari Suci Nyepi Tahun Baru Saka 1946',
    '2024-03-29': 'Libur Nasional: Wafat Isa Al Masih',
    '2024-03-31': 'Libur Nasional: Hari Paskah',
    '2024-04-08': 'Cuti Bersama: Hari Raya Idul Fitri 1445 Hijriah',
    '2024-04-09': 'Cuti Bersama: Hari Raya Idul Fitri 1445 Hijriah',
    '2024-04-10': 'Libur Nasional: Hari Raya Idul Fitri 1445 Hijriah',
    '2024-04-11': 'Libur Nasional: Hari Raya Idul Fitri 1445 Hijriah',
    '2024-04-12': 'Cuti Bersama: Hari Raya Idul Fitri 1445 Hijriah',
    '2024-04-15': 'Cuti Bersama: Hari Raya Idul Fitri 1445 Hijriah',
    '2024-05-01': 'Libur Nasional: Hari Buruh Internasional',
    '2024-05-09': 'Libur Nasional: Kenaikan Isa Al Masih',
    '2024-05-10': 'Cuti Bersama: Kenaikan Isa Al Masih',
    '2024-05-23': 'Libur Nasional: Hari Raya Waisak 2568 BE',
    '2024-05-24': 'Cuti Bersama: Hari Raya Waisak',
    '2024-06-01': 'Libur Nasional: Hari Lahir Pancasila',
    '2024-06-17': 'Libur Nasional: Hari Raya Idul Adha 1445 Hijriah',
    '2024-06-18': 'Cuti Bersama: Hari Raya Idul Adha 1445 Hijriah',
    '2024-07-07': 'Libur Nasional: Tahun Baru Islam 1446 Hijriah',
    '2024-08-17': 'Libur Nasional: Hari Kemerdekaan Republik Indonesia',
    '2024-09-16': 'Libur Nasional: Maulid Nabi Muhammad SAW',
    '2024-12-25': 'Libur Nasional: Hari Raya Natal',
    '2024-12-26': 'Cuti Bersama: Hari Raya Natal',
    # 2025
    '2025-01-01': 'Libur Nasional: Tahun Baru 2025 Masehi',
    '2025-01-27': 'Libur Nasional: Isra Mikraj Nabi Muhammad SAW',
    '2025-01-28': 'Cuti Bersama: Tahun Baru Imlek 2576 Kongzili',
    '2025-01-29': 'Libur Nasional: Tahun Baru Imlek 2576 Kongzili',
    '2025-03-28': 'Cuti Bersama: Hari Suci Nyepi Tahun Baru Saka 1947',
    '2025-03-29': 'Libur Nasional: Hari Suci Nyepi Tahun Baru Saka 1947',
    '2025-03-31': 'Libur Nasional: Hari Raya Idul Fitri 1446 Hijriah',
    '2025-04-01': 'Libur Nasional: Hari Raya Idul Fitri 1446 Hijriah',
    '2025-04-02': 'Cuti Bersama: Idul Fitri 1446 Hijriah',
    '2025-04-03': 'Cuti Bersama: Idul Fitri 1446 Hijriah',
    '2025-04-04': 'Cuti Bersama: Idul Fitri 1446 Hijriah',
    '2025-04-07': 'Cuti Bersama: Idul Fitri 1446 Hijriah',
    '2025-04-18': 'Libur Nasional: Wafat Yesus Kristus',
    '2025-04-20': 'Libur Nasional: Kebangkitan Yesus Kristus (Paskah)',
    '2025-05-01': 'Libur Nasional: Hari Buruh Internasional',
    '2025-05-12': 'Libur Nasional: Hari Raya Waisak 2569 BE',
    '2025-05-13': 'Cuti Bersama: Hari Raya Waisak 2569 BE',
    '2025-05-29': 'Libur Nasional: Kenaikan Yesus Kristus',
    '2025-05-30': 'Cuti Bersama: Kenaikan Yesus Kristus',
    '2025-06-01': 'Libur Nasional: Hari Lahir Pancasila',
    '2025-06-06': 'Libur Nasional: Hari Raya Idul Adha 1446 Hijriah',
    '2025-06-09': 'Cuti Bersama: Idul Adha 1446 Hijriah',
    '2025-06-27': 'Libur Nasional: 1 Muharram Tahun Baru Islam 1447 Hijriah',
    '2025-08-17': 'Libur Nasional: Hari Kemerdekaan Republik Indonesia',
    '2025-09-05': 'Libur Nasional: Maulid Nabi Muhammad SAW',
    '2025-12-25': 'Libur Nasional: Hari Raya Natal',
    '2025-12-26': 'Cuti Bersama: Kelahiran Yesus Kristus',
    # 2026
    '2026-01-01': 'Libur Nasional: Tahun Baru 2026 Masehi',
    '2026-01-16': 'Libur Nasional: Isra Mikraj Nabi Muhammad SAW',
    '2026-02-16': 'Cuti Bersama: Tahun Baru Imlek 2577 Kongzili',
    '2026-02-17': 'Libur Nasional: Tahun Baru Imlek 2577 Kongzili',
    '2026-03-18': 'Cuti Bersama: Hari Suci Nyepi Tahun Baru Saka 1948',
    '2026-03-19': 'Libur Nasional: Hari Suci Nyepi Tahun Baru Saka 1948',
    '2026-03-20': 'Cuti Bersama: Idulfitri 1447 Hijriah',
    '2026-03-21': 'Libur Nasional: Hari Raya Idulfitri 1447 Hijriah',
    '2026-03-22': 'Libur Nasional: Hari Raya Idulfitri 1447 Hijriah',
    '2026-03-23': 'Cuti Bersama: Idulfitri 1447 Hijriah',
    '2026-03-24': 'Cuti Bersama: Idulfitri 1447 Hijriah',
    '2026-04-03': 'Libur Nasional: Wafat Yesus Kristus',
    '2026-04-05': 'Libur Nasional: Kebangkitan Yesus Kristus (Paskah)',
    '2026-05-01': 'Libur Nasional: Hari Buruh Internasional',
    '2026-05-14': 'Libur Nasional: Kenaikan Yesus Kristus',
    '2026-05-15': 'Cuti Bersama: Kenaikan Yesus Kristus',
    '2026-05-27': 'Libur Nasional: Hari Raya Iduladha 1447 Hijriah',
    '2026-05-28': 'Cuti Bersama: Iduladha 1447 Hijriah',
    '2026-05-31': 'Libur Nasional: Hari Raya Waisak 2570 BE',
    '2026-06-01': 'Libur Nasional: Hari Lahir Pancasila',
    '2026-06-16': 'Libur Nasional: 1 Muharram Tahun Baru Islam 1448 Hijriah',
    '2026-08-17': 'Libur Nasional: Proklamasi Kemerdekaan Republik Indonesia',
    '2026-08-25': 'Libur Nasional: Maulid Nabi Muhammad SAW',
    '2026-12-24': 'Cuti Bersama: Kelahiran Yesus Kristus',
    '2026-12-25': 'Libur Nasional: Hari Raya Natal',
}
HARI_LIBUR_NASIONAL = pd.to_datetime(list(HARI_LIBUR_NASIONAL_DICT.keys()))

FEATURES_DAILY = [
    'dayofweek', 'is_weekend', 'month', 'day',
    'is_holiday',
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_7', 'lag_14',
    'rolling_mean_7', 'rolling_mean_14',
    'ewm_7',
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


    ds['dayofweek'] = ds['tanggal'].dt.dayofweek
    ds['is_weekend'] = (ds['dayofweek'] >= 5).astype(int)
    ds['month'] = ds['tanggal'].dt.month
    ds['day'] = ds['tanggal'].dt.day

    ds['is_holiday'] = ds['tanggal'].isin(HARI_LIBUR_NASIONAL).astype(int)

    for lag in [1, 2, 3, 4, 7, 14]:
        ds[f'lag_{lag}'] = ds['penjualan'].shift(lag)
    ds['rolling_mean_7'] = ds['penjualan'].rolling(7).mean()
    ds['rolling_mean_14'] = ds['penjualan'].rolling(14).mean()
    ds['ewm_7'] = ds['penjualan'].ewm(span=7, adjust=False).mean()

    if required_cols is None:
        ds = ds.dropna().reset_index(drop=True)
    else:
        keep_cols = list(dict.fromkeys(list(required_cols) + ['tanggal', 'penjualan']))
        keep_cols = [c for c in keep_cols if c in ds.columns]
        ds = ds.dropna(subset=keep_cols).reset_index(drop=True)
    return ds


def load_xgb_model_from_bytes(model_bytes):
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(bytearray(model_bytes))
    return model


def _forecast_range(produk, product_daily, model_info, end_date):
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
        pred_raw = model.predict(X_next)[0]
        pred = float(max(pred_raw, 0))
        work = pd.concat(
            [work, pd.DataFrame({'tanggal': [next_date], 'penjualan': [pred]})],
            ignore_index=True,
        )
        daily_preds.append((next_date, pred))

    return last_date, daily_preds


def predict_for_date(produk, product_daily, model_info, target_date):

    _, daily_preds = _forecast_range(produk, product_daily, model_info, target_date)
    pred_final = daily_preds[-1][1]
    return int(round(pred_final))


def predict_week(produk, product_daily, model_info, start_date, n_days=7):
    start_date = pd.Timestamp(start_date)
    end_date = start_date + pd.Timedelta(days=n_days - 1)
    _, daily_preds = _forecast_range(produk, product_daily, model_info, end_date)
    return [
        {'tanggal': tgl, 'prediksi': float(pred)}
        for tgl, pred in daily_preds
        if tgl >= start_date
    ]


def evaluate_on_test(produk, product_daily, model_info, test_ratio=0.2):
    """Uji akurasi model pada DATA HISTORIS.
    """
    features = model_info['features']
    model = model_info['model']

    prod_hist = product_daily[product_daily['Produk'] == produk][['tanggal', 'penjualan']].copy()
    prod_hist = prod_hist.sort_values('tanggal').reset_index(drop=True)

    feat = build_features(prod_hist, required_cols=features)
    if len(feat) == 0:
        return None

    split_idx = int(len(feat) * (1 - test_ratio))
    test = feat.iloc[split_idx:].reset_index(drop=True)
    if len(test) == 0:
        return None

    y_true = test['penjualan'].to_numpy(dtype=float)
    X_test = test[features]
    y_pred = np.asarray(model.predict(X_test), dtype=float)
    y_pred = np.clip(y_pred, 0, None)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() > 0 else float('nan')

    return {
        'tanggal': test['tanggal'],
        'aktual': y_true,
        'prediksi': y_pred,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'n_test': len(test),
    }


def get_dashboard_summary(product_daily):

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

    return total_per_produk, tren_harian, stats
