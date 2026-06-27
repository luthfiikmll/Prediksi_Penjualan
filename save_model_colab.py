# ── Jalankan cell ini di AKHIR notebook Colab setelah semua training selesai ─
# Menyimpan semua file yang dibutuhkan Streamlit (2 skenario)

import joblib
import pandas as pd
import numpy as np

print('Menyimpan file untuk Streamlit...')
print()

# ════════════════════════════════════════════
# 1. MODEL SKENARIO 1 — model terbaik (GS)
# ════════════════════════════════════════════
# Cek mana yang terbaik berdasarkan MAPE
# Evaluasi hanya Baseline dan RS
df_s1_rows = []
for opt in ['Baseline','RS']:
    r = res_total_xgb[opt]
    df_s1_rows.append({'Optimizer':opt,'MAPE (%)':r['MAPE']})
df_s1 = pd.DataFrame(df_s1_rows)
best_opt_s1 = df_s1.loc[df_s1['MAPE (%)'].idxmin(), 'Optimizer']
model_s1 = res_total_xgb[best_opt_s1]['model']
joblib.dump(model_s1, 'model_s1.pkl')
print(f'model_s1.pkl — XGBoost {best_opt_s1} (MAPE: {df_s1["MAPE (%)"].min():.2f}%)')

# ════════════════════════════════════════════
# 2. DOW TARGET ENCODING (Skenario 1)
# ════════════════════════════════════════════
split    = int(len(ds_total) * 0.8)
train_s1 = ds_total.iloc[:split]
dow_map  = train_s1.groupby('dayofweek')['penjualan'].mean()
joblib.dump(dow_map, 'dow_map.pkl')
print(f'dow_map.pkl — {len(dow_map)} hari dalam seminggu')

# ════════════════════════════════════════════
# 3. MODEL SKENARIO 2 — model terbaik per produk
# ════════════════════════════════════════════
models_s2 = {}
for produk in TOP3_PRODUCTS:
    df_p_rows = [{'Optimizer':o,'MAPE (%)':results_s2_xgb[produk][o]['MAPE']} for o in ['Baseline','RS']]
    df_p   = pd.DataFrame(df_p_rows)
    best_p = df_p.loc[df_p['MAPE (%)'].idxmin(), 'Optimizer']
    models_s2[produk] = results_s2_xgb[produk][best_p]['model']
    print(f'  {produk}: XGBoost {best_p} (MAPE: {df_p["MAPE (%)"].min():.2f}%)')

joblib.dump(models_s2, 'models_s2.pkl')
print(f'models_s2.pkl — {len(models_s2)} produk')

# ════════════════════════════════════════════
# 4. DATA HISTORIS
# ════════════════════════════════════════════
daily_sales[['tanggal','penjualan']].to_csv('data_harian.csv', index=False)
weekly_sales[['tanggal','penjualan']].to_csv('data_mingguan.csv', index=False)
product_weekly.to_csv('data_produk_mingguan.csv', index=False)
print('data_harian.csv, data_mingguan.csv, data_produk_mingguan.csv')

# ════════════════════════════════════════════
# 5. HASIL EVALUASI (untuk halaman Evaluasi di Streamlit)
# ════════════════════════════════════════════
rows_s1 = []
for opt in ['Baseline','RS']:
    r = res_total_xgb[opt]
    rows_s1.append({'Optimizer': opt, 'MAE': r['MAE'], 'RMSE': r['RMSE'],
                    'MAPE (%)': r['MAPE'], 'Waktu (dtk)': round(r['waktu'],1)})
pd.DataFrame(rows_s1).to_csv('eval_s1.csv', index=False)
print('eval_s1.csv')

rows_s2 = []
for produk in TOP3_PRODUCTS:
    for opt in ['Baseline','RS']:
        r = results_s2_xgb[produk][opt]
        rows_s2.append({'Produk': produk, 'Optimizer': opt,
                        'MAE': r['MAE'], 'RMSE': r['RMSE'], 'MAPE (%)': r['MAPE'],
                        'Waktu (dtk)': round(r['waktu'],1)})
pd.DataFrame(rows_s2).to_csv('eval_s2.csv', index=False)
print('eval_s2.csv')

# ════════════════════════════════════════════
# 6. DOWNLOAD SEMUA FILE
# ════════════════════════════════════════════
from google.colab import files

print()
print('Download semua file...')
for fname in ['model_s1.pkl','dow_map.pkl','models_s2.pkl',
              'data_harian.csv','data_mingguan.csv','data_produk_mingguan.csv',
              'eval_s1.csv','eval_s2.csv']:
    files.download(fname)
    print(f'  ✅ {fname}')

print()
print('Selesai! Upload semua file ke folder Streamlit:')
print('''
  streamlit_app/
  ├── app.py
  ├── utils.py
  ├── requirements.txt
  ├── model_s1.pkl
  ├── dow_map.pkl
  ├── models_s2.pkl
  ├── data_harian.csv
  ├── data_mingguan.csv
  ├── data_produk_mingguan.csv
  ├── eval_s1.csv
  └── eval_s2.csv
''')
