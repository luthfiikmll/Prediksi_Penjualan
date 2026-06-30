import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from utils import (
    build_features, build_input_row_daily, build_input_row_s2,
    FEATURES_DAILY, TOP3_PRODUCTS,
    mape_kategori, rekomendasi_barista
)

# ── Konfigurasi ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Forecast Penjualan Kopi',
    page_icon='☕',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
.metric-card{background:#f8f9fa;border-radius:10px;padding:14px 18px;
             border-left:4px solid #1D9E75;margin-bottom:10px}
.metric-label{font-size:12px;color:#666;margin-bottom:3px}
.metric-value{font-size:24px;font-weight:600;color:#1a1a1a}
.metric-sub{font-size:11px;color:#888;margin-top:3px}
.pred-box{background:#E1F5EE;border-left:4px solid #1D9E75;
          padding:16px 18px;border-radius:0 10px 10px 0;margin-bottom:12px}
.pred-val{font-size:36px;font-weight:600;color:#085041}
.pred-lbl{font-size:12px;color:#0F6E56;margin-bottom:4px}
.pred-sub{font-size:12px;color:#0F6E56;margin-top:6px}
.warn-box{background:#fff8e1;border-left:4px solid #f9a825;
          padding:12px 16px;border-radius:0 10px 10px 0;
          font-size:12px;color:#5d4037;margin-bottom:12px}
.info-box{background:#e3f2fd;border-left:4px solid #1976d2;
          padding:12px 16px;border-radius:0 10px 10px 0;
          font-size:12px;color:#0d47a1;margin-bottom:12px}
.success-box{background:#e8f5e9;border-left:4px solid #1D9E75;
             padding:12px 16px;border-radius:0 10px 10px 0;
             font-size:12px;color:#1b5e20;margin-bottom:12px}
.badge-green{background:#d4edda;color:#155724;padding:2px 10px;
             border-radius:20px;font-size:11px;font-weight:500}
.badge-yellow{background:#fff3cd;color:#856404;padding:2px 10px;
              border-radius:20px;font-size:11px;font-weight:500}
h1{font-size:22px!important}h2{font-size:17px!important}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Memuat model...')
def load_models():
    model_s1  = joblib.load('model_xgb.pkl')
    dow_map   = joblib.load('dow_map.pkl')
    models_s2 = joblib.load('models_s2.pkl')
    return model_s1, dow_map, models_s2

@st.cache_data(show_spinner='Memuat data...')
def load_default_data():
    daily  = pd.read_csv('data_harian.csv',        parse_dates=['tanggal'])
    produk = pd.read_csv('data_produk_harian.csv', parse_dates=['tanggal'])
    e1     = pd.read_csv('eval_s1.csv')
    e2     = pd.read_csv('eval_s2.csv')
    daily  = daily.sort_values('tanggal').reset_index(drop=True)
    produk = produk.sort_values('tanggal').reset_index(drop=True)
    return daily, produk, e1, e2

try:
    model_s1, dow_map, models_s2 = load_models()
    default_daily, default_produk, eval_s1, eval_s2 = load_default_data()
except Exception as e:
    st.error(f'Gagal memuat model/data: {e}')
    st.stop()


# ── Fungsi proses upload Luna POS ─────────────────────────────────────────────
def process_luna_upload(uploaded_file):
    """
    Proses file export Luna POS (CSV atau Excel).
    Format Luna POS: No Transaksi, Tanggal, Produk, Qty, Harga
    Tanggal format: dd/mm/yyyy HH:MM
    """
    try:
        # Baca file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=';')
        else:
            df = pd.read_excel(uploaded_file)

        # Validasi kolom wajib
        required = ['Tanggal', 'Produk', 'Qty']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return None, None, f'Kolom tidak ditemukan: {missing}. Pastikan ini file export Luna POS.'

        # Parse tanggal
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y %H:%M', errors='coerce')
        if df['Tanggal'].isna().all():
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], infer_datetime_format=True, errors='coerce')

        df = df.dropna(subset=['Tanggal'])
        df['date'] = df['Tanggal'].dt.date

        if len(df) == 0:
            return None, None, 'Tidak ada data valid setelah parsing tanggal.'

        # Buat daily_sales
        daily = df.groupby('date')['Qty'].sum().reset_index()
        daily.columns = ['tanggal', 'penjualan']
        daily['tanggal'] = pd.to_datetime(daily['tanggal'])
        daily = daily.sort_values('tanggal').reset_index(drop=True)

        # Buat product_daily
        produk_daily = df.groupby(['date','Produk'])['Qty'].sum().reset_index()
        produk_daily.columns = ['tanggal', 'Produk', 'penjualan']
        produk_daily['tanggal'] = pd.to_datetime(produk_daily['tanggal'])
        produk_daily = produk_daily.sort_values('tanggal').reset_index(drop=True)

        return daily, produk_daily, None

    except Exception as e:
        return None, None, f'Error memproses file: {str(e)}'


# ── Session state untuk data ──────────────────────────────────────────────────
if 'daily_sales' not in st.session_state:
    st.session_state.daily_sales    = default_daily
    st.session_state.product_daily  = default_produk
    st.session_state.data_source    = 'default'
    st.session_state.upload_info    = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## ☕ Forecast Kopi')
    st.markdown('---')
    page = st.radio('Navigasi', [
        '🏠 Dashboard',
        '📈 Skenario 1 -- Harian',
        '📦 Skenario 2 -- Per Produk',
        '📊 Evaluasi Model',
        'ℹ️ Tentang'
    ], label_visibility='collapsed')

    st.markdown('---')

    # ── Upload data baru ──────────────────────────────────────────────────────
    st.markdown('### 📂 Update Data Penjualan')
    st.caption('Upload export terbaru dari Luna POS untuk prediksi yang lebih akurat.')

    uploaded = st.file_uploader(
        'Export Luna POS (.csv atau .xlsx)',
        type=['csv','xlsx'],
        help='Export laporan dari Luna POS, format: No Transaksi, Tanggal, Produk, Qty, Harga'
    )

    if uploaded is not None:
        daily_new, produk_new, err = process_luna_upload(uploaded)
        if err:
            st.error(err)
        else:
            st.session_state.daily_sales   = daily_new
            st.session_state.product_daily = produk_new
            st.session_state.data_source   = 'upload'
            st.session_state.upload_info   = {
                'nama':   uploaded.name,
                'baris':  len(daily_new),
                'mulai':  daily_new['tanggal'].min().date(),
                'selesai':daily_new['tanggal'].max().date(),
            }
            st.success(f'Data berhasil diupload!')

    # Status data aktif
    if st.session_state.data_source == 'upload' and st.session_state.upload_info:
        info = st.session_state.upload_info
        st.markdown(f"""<div class="success-box">
            <strong>Data aktif: {info['nama']}</strong><br>
            {info['mulai']} s/d {info['selesai']}<br>
            Total: {info['baris']} hari
        </div>""", unsafe_allow_html=True)
        if st.button('Kembali ke data default', use_container_width=True):
            st.session_state.daily_sales   = default_daily
            st.session_state.product_daily = default_produk
            st.session_state.data_source   = 'default'
            st.session_state.upload_info   = None
            st.rerun()
    else:
        d = st.session_state.daily_sales
        st.caption(f'Data default: {d["tanggal"].min().date()} s/d {d["tanggal"].max().date()}')
        st.caption(f'Total: {len(d)} hari')

    st.markdown('---')
    st.markdown('**Model:** XGBoost + RandomizedSearch')


# ── Shortcut ke data aktif ────────────────────────────────────────────────────
daily_sales   = st.session_state.daily_sales
product_daily = st.session_state.product_daily
global_mean   = daily_sales['penjualan'].mean()
last_date     = daily_sales['tanggal'].max()


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == '🏠 Dashboard':
    st.title('Dashboard Forecast Penjualan Kopi')

    if st.session_state.data_source == 'upload':
        info = st.session_state.upload_info
        st.markdown(f"""<div class="success-box">
            Menggunakan data terbaru: <strong>{info['nama']}</strong>
            ({info['mulai']} s/d {info['selesai']})
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="info-box">
            Menggunakan data default. Upload export Luna POS terbaru di sidebar
            untuk prediksi yang lebih akurat.
        </div>""", unsafe_allow_html=True)

    st.markdown('---')

    # Metric cards
    mape_s1 = eval_s1.loc[eval_s1['MAPE (%)'].idxmin(), 'MAPE (%)']
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Rata-rata harian</div>
            <div class="metric-value">{global_mean:.0f} cup</div>
            <div class="metric-sub">Periode aktif</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Penjualan tertinggi</div>
            <div class="metric-value">{daily_sales['penjualan'].max():.0f} cup</div>
            <div class="metric-sub">Dalam satu hari</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">MAPE Skenario 1</div>
            <div class="metric-value">{mape_s1:.2f}%</div>
            <div class="metric-sub"><span class="badge-green">{mape_kategori(mape_s1)}</span></div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total data aktif</div>
            <div class="metric-value">{len(daily_sales)}</div>
            <div class="metric-sub">hari</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('---')

    col_l, col_r = st.columns([3,1])
    with col_r:
        periode = st.selectbox('Periode', ['30 hari terakhir','90 hari terakhir','Semua data'])
    with col_l:
        st.subheader('Tren penjualan harian')

    n = {'30 hari terakhir':30,'90 hari terakhir':90,'Semua data':len(daily_sales)}[periode]
    plot_df = daily_sales.tail(n)
    roll7   = plot_df['penjualan'].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['tanggal'], y=plot_df['penjualan'],
        name='Harian', line=dict(color='#B4B2A9', width=1), opacity=0.6))
    fig.add_trace(go.Scatter(x=plot_df['tanggal'], y=roll7,
        name='Rolling 7 hari', line=dict(color='#1D9E75', width=2.5)))
    fig.update_layout(height=280, margin=dict(l=0,r=0,t=5,b=0),
        plot_bgcolor='white',
        legend=dict(orientation='h', y=1.05, x=1, xanchor='right'),
        yaxis=dict(title='Cup', gridcolor='#f0f0f0'),
        xaxis=dict(gridcolor='#f0f0f0'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader('Pola per hari dalam seminggu')
        daily_sales['dow'] = daily_sales['tanggal'].dt.day_name()
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow_label = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
        dow_avg   = daily_sales.groupby('dow')['penjualan'].mean().reindex(dow_order)
        fig2 = go.Figure(go.Bar(
            x=dow_label, y=dow_avg.values,
            marker_color=['#9FE1CB']*5+['#1D9E75']*2,
            text=[f'{v:.0f}' for v in dow_avg.values],
            textposition='outside'))
        fig2.update_layout(height=230, margin=dict(l=0,r=0,t=5,b=0),
            plot_bgcolor='white', showlegend=False,
            yaxis=dict(title='Rata-rata cup', gridcolor='#f0f0f0'))
        st.plotly_chart(fig2, use_container_width=True)
        daily_sales.drop(columns=['dow'], inplace=True, errors='ignore')

    with col_b:
        st.subheader('Komposisi Top 3 produk')
        top3_data = product_daily[product_daily['Produk'].isin(TOP3_PRODUCTS)]
        if len(top3_data) > 0:
            tot = top3_data.groupby('Produk')['penjualan'].sum()
            fig3 = go.Figure(go.Pie(
                labels=tot.index.tolist(),
                values=tot.values.tolist(),
                hole=0.45,
                marker_colors=['#1D9E75','#9FE1CB','#5DCAA5']))
            fig3.update_layout(height=230, margin=dict(l=0,r=0,t=5,b=0),
                legend=dict(orientation='h', y=-0.15))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info('Data produk Top 3 tidak tersedia di file yang diupload.')


# ══════════════════════════════════════════════════════════════════════════════
# SKENARIO 1 -- PREDIKSI HARIAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == '📈 Skenario 1 -- Harian':
    st.title('Skenario 1 -- Prediksi Total Cup Harian')
    st.markdown('Tujuan: **perencanaan SDM / shift barista**.')

    if st.session_state.data_source == 'upload':
        info = st.session_state.upload_info
        st.markdown(f"""<div class="success-box">
            Data aktif: <strong>{info['nama']}</strong> ({info['mulai']} s/d {info['selesai']})
            -- prediksi menggunakan data terbaru.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="warn-box">
            Menggunakan data default. Upload data terbaru di sidebar untuk prediksi yang lebih akurat.
        </div>""", unsafe_allow_html=True)

    st.markdown('---')
    col_in, col_out = st.columns([1,1])

    with col_in:
        st.subheader('Input')
        tgl_min = last_date.date() + timedelta(days=1)
        tgl_max = last_date.date() + timedelta(days=30)
        target_date = st.date_input(
            'Pilih tanggal prediksi',
            value=tgl_min, min_value=tgl_min, max_value=tgl_max,
            help=f'Tersedia {tgl_min} s/d {tgl_max}'
        )
        btn = st.button('Prediksi', type='primary', use_container_width=True)

    with col_out:
        st.subheader('Hasil prediksi')
        if btn:
            with st.spinner('Menghitung prediksi...'):
                row = build_input_row_daily(target_date, daily_sales, dow_map)
            if row is None:
                st.error('Gagal membangun fitur. Data historis tidak cukup (minimal 30 hari).')
            else:
                pred_cup  = max(0, round(np.expm1(model_s1.predict(row)[0])))
                n_bar, pesan = rekomendasi_barista(pred_cup)
                hari_id   = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
                nama_hari = hari_id[target_date.weekday()]
                diff_pct  = (pred_cup - global_mean) / global_mean * 100

                st.markdown(f"""<div class="pred-box">
                    <div class="pred-lbl">{nama_hari}, {target_date.strftime('%d %B %Y')}</div>
                    <div class="pred-val">{pred_cup} cup</div>
                    <div class="pred-sub">Prediksi total penjualan</div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f'**Rekomendasi SDM:** {pesan}')
                st.caption(
                    f'{"↑" if diff_pct > 0 else "↓"} {abs(diff_pct):.1f}% '
                    f'{"di atas" if diff_pct > 0 else "di bawah"} rata-rata ({global_mean:.0f} cup)'
                )
        else:
            st.info('Pilih tanggal lalu klik **Prediksi**.')

    st.markdown('---')
    st.subheader('Prediksi 7 hari ke depan')
    if st.button('Hitung 7 hari ke depan', use_container_width=False):
        hari_id   = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
        tgl_mulai = last_date.date() + timedelta(days=1)
        hasil     = []
        bar       = st.progress(0)
        for i in range(7):
            tgl = tgl_mulai + timedelta(days=i)
            row = build_input_row_daily(tgl, daily_sales, dow_map)
            if row is not None:
                pred   = max(0, round(np.expm1(model_s1.predict(row)[0])))
                n_b, _ = rekomendasi_barista(pred)
                hasil.append({'Tanggal': tgl.strftime('%d %b %Y'),
                              'Hari': hari_id[tgl.weekday()],
                              'Prediksi (cup)': pred, 'Barista': n_b})
            bar.progress((i+1)/7)
        bar.empty()
        if hasil:
            df7 = pd.DataFrame(hasil)
            fig4 = go.Figure(go.Bar(
                x=df7['Hari'], y=df7['Prediksi (cup)'],
                marker_color='#1D9E75',
                text=df7['Prediksi (cup)'], textposition='outside'))
            fig4.update_layout(height=250, margin=dict(l=0,r=0,t=5,b=0),
                plot_bgcolor='white', showlegend=False,
                yaxis=dict(gridcolor='#f0f0f0'), xaxis=dict(gridcolor='#f0f0f0'))
            st.plotly_chart(fig4, use_container_width=True)
            st.dataframe(df7, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# SKENARIO 2 -- PER PRODUK
# ══════════════════════════════════════════════════════════════════════════════
elif page == '📦 Skenario 2 -- Per Produk':
    st.title('Skenario 2 -- Prediksi Per Produk Harian')
    st.markdown('Tujuan: **estimasi kebutuhan stok bahan baku** per produk.')

    if st.session_state.data_source == 'upload':
        info = st.session_state.upload_info
        st.markdown(f"""<div class="success-box">
            Data aktif: <strong>{info['nama']}</strong> ({info['mulai']} s/d {info['selesai']})
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="warn-box">
            Menggunakan data default. Upload data terbaru di sidebar untuk prediksi yang lebih akurat.
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="warn-box">
        <strong>Catatan akurasi:</strong> Prediksi per produk memiliki error lebih tinggi
        dari total cup. Gunakan sebagai estimasi dengan buffer +/- 15%.
    </div>""", unsafe_allow_html=True)

    col_in2, col_out2 = st.columns([1,1])

    with col_in2:
        st.subheader('Input')
        tgl_min2 = last_date.date() + timedelta(days=1)
        tgl_max2 = last_date.date() + timedelta(days=30)
        target_date2 = st.date_input(
            'Pilih tanggal prediksi',
            value=tgl_min2, min_value=tgl_min2, max_value=tgl_max2, key='s2_date')
        btn2 = st.button('Prediksi per produk', type='primary', use_container_width=True)

    with col_out2:
        st.subheader('Hasil prediksi')
        if btn2:
            hari_id   = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
            nama_hari = hari_id[target_date2.weekday()]
            hasil_s2  = {}

            with st.spinner('Menghitung prediksi per produk...'):
                for produk in TOP3_PRODUCTS:
                    row = build_input_row_s2(target_date2, product_daily, produk, daily_sales, dow_map)
                    pred = None
                    if row is not None and produk in models_s2:
                        pred = max(0, round(np.expm1(models_s2[produk].predict(row)[0])))
                    hasil_s2[produk] = pred

            st.markdown(f'**{nama_hari}, {target_date2.strftime("%d %B %Y")}**')
            for produk in TOP3_PRODUCTS:
                pred = hasil_s2[produk]
                if pred is not None:
                    low  = round(pred * 0.85)
                    high = round(pred * 1.15)
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">{produk}</div>
                        <div class="metric-value">{pred} cup</div>
                        <div class="metric-sub">Buffer ±15%: {low} -- {high} cup</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.warning(f'{produk}: data tidak cukup (minimal 30 hari)')
        else:
            st.info('Pilih tanggal lalu klik **Prediksi per produk**.')

    if btn2 and any(v is not None for v in hasil_s2.values()):
        st.markdown('---')
        st.subheader('Visualisasi prediksi per produk')
        valid = {k:v for k,v in hasil_s2.items() if v is not None}
        fig5 = go.Figure(go.Bar(
            x=list(valid.keys()), y=list(valid.values()),
            marker_color=['#1D9E75','#9FE1CB','#5DCAA5'],
            text=list(valid.values()), textposition='outside'))
        fig5.update_layout(height=250, margin=dict(l=0,r=0,t=5,b=0),
            plot_bgcolor='white', showlegend=False,
            yaxis=dict(title='Prediksi cup', gridcolor='#f0f0f0'),
            xaxis=dict(gridcolor='#f0f0f0'))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown('---')
    st.subheader('Histori penjualan per produk (30 hari terakhir)')
    produk_pilih = st.selectbox('Pilih produk', TOP3_PRODUCTS)
    hist_p = product_daily[product_daily['Produk']==produk_pilih].tail(30)
    if len(hist_p) > 0:
        fig6 = go.Figure(go.Bar(
            x=hist_p['tanggal'].dt.strftime('%d %b'),
            y=hist_p['penjualan'], marker_color='#9FE1CB'))
        fig6.update_layout(height=200, margin=dict(l=0,r=0,t=5,b=0),
            plot_bgcolor='white', showlegend=False,
            yaxis=dict(title='Cup/hari', gridcolor='#f0f0f0'),
            xaxis=dict(gridcolor='#f0f0f0'))
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info('Tidak ada data histori untuk produk ini.')


# ══════════════════════════════════════════════════════════════════════════════
# EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == '📊 Evaluasi Model':
    st.title('Evaluasi Model XGBoost')
    st.markdown('Performa dihitung dari **test set (20%)** yang tidak dipakai saat training.')
    st.markdown('---')

    tab1, tab2 = st.tabs(['Skenario 1 -- Total Harian','Skenario 2 -- Per Produk'])

    with tab1:
        st.subheader('Skenario 1 -- Total Cup Harian')
        best_s1 = eval_s1.loc[eval_s1['MAPE (%)'].idxmin()]
        c1, c2, c3 = st.columns(3)
        with c1: st.metric('MAE terbaik',  f'{best_s1["MAE"]:.4f} cup')
        with c2: st.metric('RMSE terbaik', f'{best_s1["RMSE"]:.4f} cup')
        with c3: st.metric('MAPE terbaik', f'{best_s1["MAPE (%)"]:.2f}%')
        st.markdown('---')

        e1_display = eval_s1.copy()
        e1_display['Kategori'] = e1_display['MAPE (%)'].apply(mape_kategori)
        base_mape = e1_display[e1_display['Optimizer']=='Baseline']['MAPE (%)'].values[0]
        e1_display['vs Baseline'] = e1_display['MAPE (%)'].apply(
            lambda x: '-' if x == base_mape
            else f'-{base_mape-x:.2f}pp' if x < base_mape else f'+{x-base_mape:.2f}pp')

        def hl1(row):
            return ['background-color:#e8f5e9']*len(row) if row['MAPE (%)']==eval_s1['MAPE (%)'].min() else ['']*len(row)

        st.dataframe(e1_display.style.apply(hl1, axis=1), use_container_width=True, hide_index=True)
        st.markdown(f"""<div class="info-box">
            XGBoost RS mencapai MAPE <strong>{best_s1['MAPE (%)']:.2f}%</strong> --
            kategori <strong>{mape_kategori(best_s1['MAPE (%)'])}</strong> (Lewis, 1982).
            Kriteria MAPE &lt; 20% tercapai.
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.subheader('Skenario 2 -- Per Produk Harian')
        st.markdown("""<div class="warn-box">
            Akurasi per produk lebih rendah karena data harian per produk lebih sparse.
        </div>""", unsafe_allow_html=True)

        for produk in TOP3_PRODUCTS:
            st.markdown(f'**{produk}**')
            df_p = eval_s2[eval_s2['Produk']==produk].copy()
            df_p['Kategori'] = df_p['MAPE (%)'].apply(mape_kategori)
            base_p = df_p[df_p['Optimizer']=='Baseline']['MAPE (%)'].values[0]
            df_p['vs Baseline'] = df_p['MAPE (%)'].apply(
                lambda x: '-' if x==base_p
                else f'-{base_p-x:.2f}pp' if x<base_p else f'+{x-base_p:.2f}pp')

            def hl2(row, df=df_p):
                return ['background-color:#e8f5e9']*len(row) if row['MAPE (%)']==df['MAPE (%)'].min() else ['']*len(row)

            st.dataframe(df_p.drop(columns=['Produk']).style.apply(hl2, axis=1),
                         use_container_width=True, hide_index=True)
            st.markdown('---')

    st.subheader('Referensi MAPE (Lewis, 1982)')
    st.dataframe(pd.DataFrame({
        'MAPE':     ['< 10%','10% - 20%','20% - 50%','> 50%'],
        'Kategori': ['Highly Accurate','Good','Reasonable','Inaccurate'],
    }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TENTANG
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'ℹ️ Tentang':
    st.title('Tentang Aplikasi')
    st.markdown('---')
    st.markdown("""
    ### Latar Belakang
    Prototype sistem prediksi penjualan kopi berbasis **XGBoost** untuk penelitian skripsi S1.
    Data bersumber dari ekspor laporan **Luna POS** milik kedai kopi.

    ### Cara Menggunakan
    1. Export laporan penjualan dari **Luna POS** (menu Laporan -> Export CSV/Excel)
    2. Upload file di **sidebar kiri** (bagian Update Data Penjualan)
    3. Gunakan halaman Skenario 1 atau Skenario 2 untuk prediksi

    ### Dua Skenario Prediksi
    | Skenario | Tujuan bisnis |
    |---|---|
    | Skenario 1 -- Total cup harian | Keputusan jumlah barista/shift |
    | Skenario 2 -- Per produk harian | Estimasi kebutuhan stok bahan baku |

    ### Metodologi
    - **Algoritma:** XGBoost (Extreme Gradient Boosting)
    - **Optimizer:** Baseline vs RandomizedSearch (N_ITER=150)
    - **Validasi:** TimeSeriesSplit (n_splits=5, gap=30 hari)
    - **Split data:** 80% train / 20% test
    - **Fitur:** 28 fitur (kalender, lag, rolling, EWM, tren, DOW encoding)

    ### Keterbatasan
    - Model dilatih sekali (statis) -- tidak retrain otomatis dari data baru
    - Data baru yang diupload hanya dipakai untuk menghitung lag features prediksi,
      bukan untuk melatih ulang model
    - Luna POS tidak menyediakan API publik sehingga upload manual diperlukan

    ### Referensi
    - Lewis, C.D. (1982). *Industrial and Business Forecasting Methods.* Butterworth.
    - Chen & Guestrin (2016). XGBoost. *KDD 2016.*
    - Micci-Barreca (2001). Target encoding. *ACM SIGKDD.*
    - Arlot & Celisse (2010). Cross-validation. *Statistics Surveys, 4*, 40-79.
    """)
    st.caption('Prototype untuk keperluan penelitian akademik.')
