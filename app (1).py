import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from utils import (
    build_features, compute_dow_enc,
    build_input_row_daily, build_input_row_weekly,
    FEATURES_DAILY, FEATURES_WEEKLY,
    mape_kategori, rekomendasi_barista, minggu_depan_start
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Forecast Penjualan Kopi',
    page_icon='☕',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card{background:#f8f9fa;border-radius:10px;padding:14px 18px;
             border-left:4px solid #1D9E75;margin-bottom:8px}
.metric-label{font-size:12px;color:#666;margin-bottom:3px}
.metric-value{font-size:24px;font-weight:600;color:#1a1a1a}
.metric-sub{font-size:11px;color:#888;margin-top:3px}
.badge-green{background:#d4edda;color:#155724;padding:2px 10px;
             border-radius:20px;font-size:11px;font-weight:500}
.badge-yellow{background:#fff3cd;color:#856404;padding:2px 10px;
              border-radius:20px;font-size:11px;font-weight:500}
.badge-red{background:#f8d7da;color:#721c24;padding:2px 10px;
           border-radius:20px;font-size:11px;font-weight:500}
.pred-box{background:#E1F5EE;border-left:4px solid #1D9E75;
          padding:14px 16px;border-radius:0 8px 8px 0;margin-bottom:10px}
.pred-val{font-size:34px;font-weight:600;color:#085041}
.pred-lbl{font-size:11px;color:#0F6E56;margin-bottom:4px}
.pred-sub{font-size:11px;color:#0F6E56;margin-top:4px}
.warn-box{background:#fff8e1;border-left:4px solid #f9a825;
          padding:10px 14px;border-radius:0 8px 8px 0;font-size:12px;color:#5d4037}
.info-box{background:#e3f2fd;border-left:4px solid #1976d2;
          padding:10px 14px;border-radius:0 8px 8px 0;font-size:12px;color:#0d47a1}
h1{font-size:22px!important}h2{font-size:17px!important}h3{font-size:14px!important}
</style>
""", unsafe_allow_html=True)


# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m_s1     = joblib.load('model_s1.pkl')
    dow_map  = joblib.load('dow_map.pkl')
    models_s2= joblib.load('models_s2.pkl')   # dict: {produk: model}
    return m_s1, dow_map, models_s2

@st.cache_data
def load_data():
    daily   = pd.read_csv('data_harian.csv',   parse_dates=['tanggal'])
    weekly  = pd.read_csv('data_mingguan.csv', parse_dates=['tanggal'])
    prod_wk = pd.read_csv('data_produk_mingguan.csv', parse_dates=['tanggal'])
    daily   = daily.sort_values('tanggal').reset_index(drop=True)
    weekly  = weekly.sort_values('tanggal').reset_index(drop=True)
    return daily, weekly, prod_wk

try:
    model_s1, dow_map, models_s2 = load_models()
    daily_sales, weekly_sales, product_weekly = load_data()
    global_mean = daily_sales['penjualan'].mean()
    TOP3        = list(models_s2.keys())
    all_weeks   = weekly_sales[['tanggal']].copy()
    loaded      = True
except Exception as e:
    st.error(f'Gagal load model/data: {e}')
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## ☕ Forecast Kopi')
    st.markdown('---')
    page = st.radio(
        'Navigasi',
        ['🏠 Dashboard', '📈 Skenario 1 — Harian',
         '📦 Skenario 2 — Per Produk', '📊 Evaluasi Model', 'ℹ️ Tentang'],
        label_visibility='collapsed'
    )
    st.markdown('---')
    st.markdown('**Model aktif:** XGBoost')
    st.markdown('**Optimizer:** Baseline + RS')
    st.caption(f'Data: {daily_sales["tanggal"].min().date()} — {daily_sales["tanggal"].max().date()}')
    st.caption(f'Total: {len(daily_sales)} hari | {len(weekly_sales)} minggu')


# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == '🏠 Dashboard':
    st.title('Dashboard Forecast Penjualan Kopi')
    st.markdown('Sistem prediksi berbasis **XGBoost** — 2 skenario untuk operasional dan stok bahan baku.')
    st.markdown('---')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Rata-rata harian</div>
            <div class="metric-value">{daily_sales['penjualan'].mean():.0f} cup</div>
            <div class="metric-sub">Sepanjang periode data</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Penjualan tertinggi</div>
            <div class="metric-value">{daily_sales['penjualan'].max():.0f} cup</div>
            <div class="metric-sub">Dalam satu hari</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">MAPE Skenario 1</div>
            <div class="metric-value">lihat evaluasi</div>
            <div class="metric-sub">Total cup harian</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Top produk</div>
            <div class="metric-value" style="font-size:15px;padding-top:4px">{TOP3[0] if TOP3 else '—'}</div>
            <div class="metric-sub">Penjualan tertinggi</div></div>""", unsafe_allow_html=True)

    st.markdown('---')

    col_left, col_right = st.columns([3,1])
    with col_right:
        periode = st.selectbox('Tampilkan', ['30 hari terakhir','90 hari terakhir','Semua data'])
    with col_left:
        st.subheader('Tren penjualan harian')

    plot_df = {'30 hari terakhir': daily_sales.tail(30),
               '90 hari terakhir': daily_sales.tail(90),
               'Semua data': daily_sales}[periode]
    roll7   = plot_df['penjualan'].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['tanggal'], y=plot_df['penjualan'],
        name='Harian', line=dict(color='#B4B2A9', width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=plot_df['tanggal'], y=roll7,
        name='Rolling 7 hari', line=dict(color='#1D9E75', width=2.5)))
    fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor='white', legend=dict(orientation='h',y=1.05,x=1,xanchor='right'),
        yaxis=dict(title='Cup',gridcolor='#f0f0f0'),
        xaxis=dict(gridcolor='#f0f0f0'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    cl, cr = st.columns(2)
    with cl:
        st.subheader('Pola per hari dalam seminggu')
        daily_sales['dow'] = daily_sales['tanggal'].dt.day_name()
        dow_order  = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow_label  = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
        dow_avg    = daily_sales.groupby('dow')['penjualan'].mean().reindex(dow_order)
        fig2 = go.Figure(go.Bar(
            x=dow_label, y=dow_avg.values,
            marker_color=['#9FE1CB']*5+['#1D9E75']*2,
            text=[f'{v:.0f}' for v in dow_avg.values], textposition='outside'))
        fig2.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor='white', showlegend=False,
            yaxis=dict(title='Rata-rata cup',gridcolor='#f0f0f0'),
            xaxis=dict(gridcolor='#f0f0f0'))
        st.plotly_chart(fig2, use_container_width=True)
        daily_sales.drop(columns=['dow'], inplace=True, errors='ignore')

    with cr:
        st.subheader('Kontribusi Top 3 produk (mingguan)')
        tot_per_produk = product_weekly.groupby('Produk')['penjualan'].sum()
        top3_vals = tot_per_produk[TOP3] if TOP3 else tot_per_produk.head(3)
        fig3 = go.Figure(go.Pie(
            labels=top3_vals.index.tolist(),
            values=top3_vals.values.tolist(),
            hole=0.45,
            marker_colors=['#1D9E75','#9FE1CB','#5DCAA5']))
        fig3.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation='h',y=-0.1))
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 2 — SKENARIO 1 (HARIAN)
# ══════════════════════════════════════════════════════════════════════════════
elif page == '📈 Skenario 1 — Harian':
    st.title('Skenario 1 — Prediksi Total Cup Harian')
    st.markdown('Tujuan: **perencanaan SDM / shift barista** berdasarkan prediksi total cup yang akan terjual.')
    st.markdown('---')

    col_in, col_out = st.columns([1,1])

    with col_in:
        st.subheader('Input')
        tgl_min = daily_sales['tanggal'].max().date() + timedelta(days=1)
        tgl_max = daily_sales['tanggal'].max().date() + timedelta(days=30)
        target_date = st.date_input('Pilih tanggal prediksi',
            value=tgl_min, min_value=tgl_min, max_value=tgl_max,
            help=f'Tersedia {tgl_min} s/d {tgl_max}')
        btn = st.button('Prediksi', type='primary', use_container_width=True)

    with col_out:
        st.subheader('Hasil')
        if btn:
            with st.spinner('Menghitung...'):
                row = build_input_row_daily(target_date, daily_sales, dow_map, global_mean)
            if row is None:
                st.error('Gagal membangun fitur. Pastikan data historis cukup.')
            else:
                pred_cup  = max(0, round(np.expm1(model_s1.predict(row)[0])))
                n_bar, pesan = rekomendasi_barista(pred_cup)
                hari_id   = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
                nama_hari = hari_id[target_date.weekday()]
                diff_pct  = (pred_cup - global_mean) / global_mean * 100
                tanda     = '↑' if diff_pct > 0 else '↓'

                st.markdown(f"""<div class="pred-box">
                    <div class="pred-lbl">{nama_hari}, {target_date.strftime('%d %B %Y')}</div>
                    <div class="pred-val">{pred_cup} cup</div>
                    <div class="pred-sub">Prediksi total penjualan hari ini</div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f'**Rekomendasi SDM:** {pesan}')
                st.markdown(f'{tanda} `{abs(diff_pct):.1f}%` dari rata-rata harian ({global_mean:.0f} cup)')
        else:
            st.info('Pilih tanggal lalu klik **Prediksi**.')

    st.markdown('---')
    st.subheader('Prediksi 7 hari ke depan')

    if st.button('Hitung 7 hari ke depan', use_container_width=False):
        hasil = []
        tgl_mulai = daily_sales['tanggal'].max().date() + timedelta(days=1)
        hari_id   = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
        bar = st.progress(0)

        for i in range(7):
            tgl = tgl_mulai + timedelta(days=i)
            row = build_input_row_daily(tgl, daily_sales, dow_map, global_mean)
            if row is not None:
                pred = max(0, round(np.expm1(model_s1.predict(row)[0])))
                n_b, _= rekomendasi_barista(pred)
                hasil.append({'Tanggal': tgl.strftime('%d %b %Y'),
                              'Hari': hari_id[tgl.weekday()],
                              'Prediksi (cup)': pred,
                              'Barista': n_b})
            bar.progress((i+1)/7)
        bar.empty()

        if hasil:
            df7 = pd.DataFrame(hasil)
            fig4 = go.Figure(go.Bar(
                x=df7['Hari'], y=df7['Prediksi (cup)'],
                marker_color='#1D9E75',
                text=df7['Prediksi (cup)'], textposition='outside'))
            fig4.update_layout(height=240, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor='white', showlegend=False,
                yaxis=dict(gridcolor='#f0f0f0'), xaxis=dict(gridcolor='#f0f0f0'))
            st.plotly_chart(fig4, use_container_width=True)
            st.dataframe(df7, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 3 — SKENARIO 2 (PER PRODUK MINGGUAN)
# ══════════════════════════════════════════════════════════════════════════════
elif page == '📦 Skenario 2 — Per Produk':
    st.title('Skenario 2 — Prediksi Per Produk (Mingguan)')
    st.markdown('Tujuan: **perencanaan stok bahan baku** per produk untuk order ke supplier minggu depan.')

    st.markdown("""<div class="warn-box">
        ⚠️ <strong>Catatan akurasi:</strong> Prediksi per produk memiliki tingkat error lebih tinggi
        dibanding total cup harian, karena pola penjualan mingguan per produk lebih bervariasi.
        Gunakan sebagai <em>estimasi</em>, bukan angka pasti. Tambah buffer ±15% saat order bahan baku.
    </div>""", unsafe_allow_html=True)
    st.markdown('---')

    # Minggu depan
    minggu_start = minggu_depan_start(daily_sales)
    minggu_end   = minggu_start + timedelta(days=6)
    st.subheader(f'Prediksi minggu {minggu_start.strftime("%d %b")} — {minggu_end.strftime("%d %b %Y")}')

    if st.button('Hitung prediksi minggu depan', type='primary'):
        hasil_s2 = {}
        bar = st.progress(0)

        for i, produk in enumerate(TOP3):
            row = build_input_row_weekly(minggu_start, product_weekly, produk, all_weeks)
            if row is not None and produk in models_s2:
                pred = max(0, round(np.expm1(models_s2[produk].predict(row)[0])))
            else:
                pred = None
            hasil_s2[produk] = pred
            bar.progress((i+1)/len(TOP3))
        bar.empty()

        col1, col2, col3 = st.columns(3)
        for col, produk in zip([col1,col2,col3], TOP3):
            with col:
                pred = hasil_s2[produk]
                if pred is not None:
                    pred_low  = round(pred * 0.85)
                    pred_high = round(pred * 1.15)
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">{produk}</div>
                        <div class="metric-value">{pred} cup</div>
                        <div class="metric-sub">Buffer ±15%: {pred_low}–{pred_high} cup</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.warning(f'{produk}: data tidak cukup')

        if any(v is not None for v in hasil_s2.values()):
            st.markdown('---')
            st.subheader('Visualisasi prediksi per produk')
            valid = {k:v for k,v in hasil_s2.items() if v is not None}
            fig5 = go.Figure(go.Bar(
                x=list(valid.keys()),
                y=list(valid.values()),
                marker_color=['#1D9E75','#9FE1CB','#5DCAA5'],
                text=list(valid.values()), textposition='outside'))
            fig5.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor='white', showlegend=False,
                yaxis=dict(title='Prediksi cup/minggu',gridcolor='#f0f0f0'),
                xaxis=dict(gridcolor='#f0f0f0'))
            st.plotly_chart(fig5, use_container_width=True)

    st.markdown('---')
    st.subheader('Histori penjualan mingguan per produk')

    produk_pilih = st.selectbox('Pilih produk', TOP3)
    hist_produk  = product_weekly[product_weekly['Produk']==produk_pilih].tail(12)

    fig6 = go.Figure(go.Bar(
        x=hist_produk['tanggal'].dt.strftime('%d %b'),
        y=hist_produk['penjualan'],
        marker_color='#9FE1CB'))
    fig6.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor='white', showlegend=False,
        yaxis=dict(title='Cup/minggu',gridcolor='#f0f0f0'),
        xaxis=dict(gridcolor='#f0f0f0',title='Minggu'))
    st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 4 — EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == '📊 Evaluasi Model':
    st.title('Evaluasi Model XGBoost')
    st.markdown('Performa dihitung dari **test set (20%)** — data yang tidak dipakai saat training.')
    st.markdown('---')

    tab1, tab2 = st.tabs(['Skenario 1 — Harian', 'Skenario 2 — Per Produk'])

    with tab1:
        st.subheader('Skenario 1 — Total Cup Harian')
        eval_s1 = {
            'Optimizer':   ['Baseline','RS'],
            'MAE':         ['—','—','—'],
            'RMSE':        ['—','—','—'],
            'MAPE (%)':    ['—','—','—'],
            'Kategori':    ['—','—','—'],
            'vs Baseline': ['—','—','—'],
        }
        try:
            eval_s1_df = pd.read_csv('eval_s1.csv')
            st.dataframe(eval_s1_df, use_container_width=True, hide_index=True)
        except:
            st.markdown("""<div class="info-box">
                File <code>eval_s1.csv</code> belum tersedia. Jalankan cell simpan hasil di Colab terlebih dahulu.
            </div>""", unsafe_allow_html=True)
            st.markdown("""
            **Panduan:** Setelah training selesai di Colab, jalankan:
            ```python
            import pandas as pd
            rows = []
            for opt in ['Baseline','RS']:
                r = res_total_xgb[opt]
                rows.append({'Optimizer':opt,'MAE':r['MAE'],'RMSE':r['RMSE'],
                             'MAPE (%)':r['MAPE'],'Waktu (dtk)':round(r['waktu'],1)})
            pd.DataFrame(rows).to_csv('eval_s1.csv', index=False)
            ```
            """)

    with tab2:
        st.subheader('Skenario 2 — Per Produk Mingguan')
        try:
            eval_s2_df = pd.read_csv('eval_s2.csv')
            st.dataframe(eval_s2_df, use_container_width=True, hide_index=True)
        except:
            st.markdown("""<div class="info-box">
                File <code>eval_s2.csv</code> belum tersedia.
            </div>""", unsafe_allow_html=True)
            st.markdown("""
            **Panduan:**
            ```python
            rows = []
            for produk in TOP3_PRODUCTS:
                for opt in ['Baseline','RS']:
                    r = results_s2_xgb[produk][opt]
                    rows.append({'Produk':produk,'Optimizer':opt,
                                 'MAE':r['MAE'],'RMSE':r['RMSE'],'MAPE (%)':r['MAPE']})
            pd.DataFrame(rows).to_csv('eval_s2.csv', index=False)
            ```
            """)

    st.markdown('---')
    st.subheader('Referensi kategori MAPE (Lewis, 1982)')
    lewis = pd.DataFrame({
        'MAPE':      ['< 10%','10% – 20%','20% – 50%','> 50%'],
        'Kategori':  ['Highly Accurate','Good','Reasonable','Inaccurate'],
    })
    st.dataframe(lewis, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 5 — TENTANG
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'ℹ️ Tentang':
    st.title('Tentang Aplikasi')
    st.markdown('---')
    st.markdown("""
    ### Latar belakang
    Aplikasi ini adalah prototype implementasi dari penelitian skripsi S1 tentang
    **prediksi penjualan kopi menggunakan algoritma XGBoost**.
    Data bersumber dari Luna POS (export Excel) milik kedai kopi yang menjadi objek penelitian.

    ### Dua skenario prediksi
    | Skenario | Granularitas | Tujuan bisnis |
    |---|---|---|
    | Skenario 1 | Harian | Prediksi total cup → keputusan jumlah barista/shift |
    | Skenario 2 | Mingguan per produk | Estimasi penjualan Top 3 menu → order bahan baku ke supplier |

    ### Metodologi
    - **Algoritma:** XGBoost (Extreme Gradient Boosting)
    - **Optimizer:** Baseline, RandomizedSearch
    - **Validasi CV:** TimeSeriesSplit (n_splits=5, gap=30 hari)
    - **Split data:** 80% train / 20% test
    - **Fitur S1:** 30 fitur (kalender, lag, rolling, EWM, tren, DOW encoding)
    - **Fitur S2:** 28 fitur (kalender, lag, rolling, EWM, tren — tanpa DOW)
    - **Sumber data:** Luna POS — export laporan penjualan (Excel)

    ### Keterbatasan sistem
    Aplikasi ini bersifat **prototype** — data tidak terupdate otomatis dari POS.
    Untuk pengembangan lebih lanjut, disarankan integrasi langsung dengan API Luna POS
    agar prediksi dapat diperbarui secara berkala tanpa upload manual.

    ### Referensi
    - Lewis, C.D. (1982). *Industrial and Business Forecasting Methods.* Butterworth.
    - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016.*
    - Micci-Barreca, D. (2001). Preprocessing for high-cardinality categorical attributes. *ACM SIGKDD.*
    - Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures. *Statistics Surveys, 4*, 40–79.
    - Pargent et al. (2022). Regularized target encoding. *Computational Statistics, 37*, 2671–2692.
    """)
    st.markdown('---')
    st.caption('Prototype sistem prediksi untuk keperluan penelitian akademik.')
