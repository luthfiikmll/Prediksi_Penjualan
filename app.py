import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from utils import (
    build_features, build_input_row, build_7day_forecast,
    process_luna_upload, FEATURES_DAILY,
    mape_kategori, 
)

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
h1{font-size:22px!important}h2{font-size:17px!important}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner='Memuat model...')
def load_model():
    model   = joblib.load('model_xgb.pkl')
    dow_map = joblib.load('dow_map.pkl')
    return model, dow_map

@st.cache_data(show_spinner='Memuat data...')
def load_default_data():
    daily = pd.read_csv('data_harian.csv', parse_dates=['tanggal'])
    e1    = pd.read_csv('eval_s1.csv')
    return daily.sort_values('tanggal').reset_index(drop=True), e1

try:
    model_s1, dow_map = load_model()
    default_daily, eval_s1 = load_default_data()
except Exception as e:
    st.error(f'Gagal memuat model/data: {e}')
    st.stop()

if 'daily_sales' not in st.session_state:
    st.session_state.daily_sales = default_daily
    st.session_state.data_source = 'default'
    st.session_state.upload_info = None

with st.sidebar:
    st.markdown('## ☕ Forecast Kopi')
    st.markdown('---')
    page = st.radio('Navigasi', [
        '🏠 Dashboard',
        '📈 Forecasting Harian',
        '📊 Evaluasi Model',
        'ℹ️ Tentang'
    ], label_visibility='collapsed')

    st.markdown('---')
    st.markdown('### 📂 Update Data Penjualan')
    st.caption('Upload export terbaru dari Luna POS agar forecasting lebih akurat.')
    uploaded = st.file_uploader('Export Luna POS (.csv/.xlsx)', type=['csv','xlsx'])

    if uploaded is not None:
        daily_new, err = process_luna_upload(uploaded)
        if err:
            st.error(err)
        else:
            st.session_state.daily_sales = daily_new
            st.session_state.data_source = 'upload'
            st.session_state.upload_info = {
                'nama':    uploaded.name,
                'baris':   len(daily_new),
                'mulai':   daily_new['tanggal'].min().date(),
                'selesai': daily_new['tanggal'].max().date(),
            }
            st.success('Data berhasil diupload!')

    if st.session_state.data_source == 'upload' and st.session_state.upload_info:
        info = st.session_state.upload_info
        st.markdown(f"""<div class="success-box">
            <strong>{info['nama']}</strong><br>
            {info['mulai']} s/d {info['selesai']}<br>
            Total: {info['baris']} hari
        </div>""", unsafe_allow_html=True)
        if st.button('Reset ke data default', use_container_width=True):
            st.session_state.daily_sales = default_daily
            st.session_state.data_source = 'default'
            st.session_state.upload_info = None
            st.rerun()
    else:
        d = st.session_state.daily_sales
        st.caption(f'{d["tanggal"].min().date()} s/d {d["tanggal"].max().date()}')
        st.caption(f'Total: {len(d)} hari')

    st.markdown('---')
    st.markdown('**Model:** XGBoost + RandomizedSearch')

daily_sales = st.session_state.daily_sales
global_mean = daily_sales['penjualan'].mean()
last_date   = daily_sales['tanggal'].max()


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == '🏠 Dashboard':
    st.title('Dashboard Forecast Penjualan Kopi')
   

    if st.session_state.data_source == 'upload':
        info = st.session_state.upload_info
        st.markdown(f"""<div class="success-box">Data aktif: <strong>{info['nama']}</strong>
            ({info['mulai']} s/d {info['selesai']})</div>""", unsafe_allow_html=True)

   

    st.markdown('---')
    mape_s1 = eval_s1.loc[eval_s1['MAPE (%)'].idxmin(), 'MAPE (%)']
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Rata-rata harian</div>
            <div class="metric-value">{global_mean:.0f} cup</div>
            <div class="metric-sub">Periode aktif</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Penjualan tertinggi</div>
            <div class="metric-value">{daily_sales['penjualan'].max():.0f} cup</div>
            <div class="metric-sub">Dalam satu hari</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">MAPE model</div>
            <div class="metric-value">{mape_s1:.2f}%</div>
            <div class="metric-sub"><span class="badge-green">{mape_kategori(mape_s1)}</span></div>
            </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total data aktif</div>
            <div class="metric-value">{len(daily_sales)}</div>
            <div class="metric-sub">hari</div></div>""", unsafe_allow_html=True)

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
        fig2 = go.Figure(go.Bar(x=dow_label, y=dow_avg.values,
            marker_color=['#9FE1CB']*5+['#1D9E75']*2,
            text=[f'{v:.0f}' for v in dow_avg.values], textposition='outside'))
        fig2.update_layout(height=230, margin=dict(l=0,r=0,t=5,b=0),
            plot_bgcolor='white', showlegend=False,
            yaxis=dict(title='Rata-rata cup', gridcolor='#f0f0f0'))
        st.plotly_chart(fig2, use_container_width=True)
        daily_sales.drop(columns=['dow'], inplace=True, errors='ignore')

    with col_b:
        st.subheader('Distribusi penjualan harian')
        fig3 = go.Figure(go.Histogram(x=daily_sales['penjualan'],
            nbinsx=20, marker_color='#1D9E75', opacity=0.8))
        fig3.add_vline(x=global_mean, line_dash='dash', line_color='red',
            annotation_text=f'Rata-rata: {global_mean:.0f}', annotation_position='top right')
        fig3.update_layout(height=230, margin=dict(l=0,r=0,t=5,b=0),
            plot_bgcolor='white', showlegend=False,
            xaxis=dict(title='Total cup', gridcolor='#f0f0f0'),
            yaxis=dict(title='Frekuensi', gridcolor='#f0f0f0'))
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING HARIAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == '📈 Forecasting Harian':
    st.title('Forecasting Total Cup Harian')
    

    if st.session_state.data_source == 'upload':
        info = st.session_state.upload_info
        st.markdown(f"""<div class="success-box">Data aktif: <strong>{info['nama']}</strong>
            ({info['mulai']} s/d {info['selesai']})</div>""", unsafe_allow_html=True)


    st.markdown('---')
    col_in, col_out = st.columns([1,1])
    with col_in:
        st.subheader('Input')
        tgl_min = last_date.date() + timedelta(days=1)
        tgl_max = last_date.date() + timedelta(days=30)
        target_date = st.date_input('Pilih tanggal forecast',
            value=tgl_min, min_value=tgl_min, max_value=tgl_max)
        btn = st.button('Forecast', type='primary', use_container_width=True)

    with col_out:
        st.subheader('Hasil forecast')
        if btn:
            with st.spinner('Menghitung...'):
                row = build_input_row(target_date, daily_sales, dow_map)
            if row is None:
                st.error('Gagal membangun fitur. Data historis tidak cukup .')
            else:
                pred_cup     = max(0, round(np.expm1(model_s1.predict(row)[0])))
                
                hari_id      = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
                nama_hari    = hari_id[target_date.weekday()]
                diff_pct     = (pred_cup - global_mean) / global_mean * 100

                st.markdown(f"""<div class="pred-box">
                    <div class="pred-lbl">{nama_hari}, {target_date.strftime('%d %B %Y')}</div>
                    <div class="pred-val">{pred_cup} cup</div>
                    <div class="pred-sub">Hasil forecasting total penjualan</div>
                </div>""", unsafe_allow_html=True)
            
                st.caption(f'{"↑" if diff_pct>0 else "↓"} {abs(diff_pct):.1f}% '
                           f'{"di atas" if diff_pct>0 else "di bawah"} rata-rata ({global_mean:.0f} cup)')
        else:
            st.info('Pilih tanggal lalu klik **Forecast**.')

    st.markdown('---')
    st.subheader('Forecast 7 Hari ke Depan')
    st.caption('Dihitung secara iteratif -- hasil hari sebelumnya dipakai untuk lag features hari berikutnya.')

    if st.button('Hitung forecast 7 hari'):
        with st.spinner('Menghitung forecast 7 hari...'):
            hasil = build_7day_forecast(model_s1, daily_sales, dow_map)

        if hasil:
            df7 = pd.DataFrame(hasil).drop(columns=['tanggal'])
            fig4 = go.Figure(go.Bar(x=df7['Hari'], y=df7['Prediksi (cup)'],
                marker_color='#1D9E75', text=df7['Prediksi (cup)'], textposition='outside'))
            fig4.add_hline(y=global_mean, line_dash='dash', line_color='red',
                annotation_text=f'Rata-rata ({global_mean:.0f})', annotation_position='top right')
            fig4.update_layout(height=280, margin=dict(l=0,r=0,t=5,b=0),
                plot_bgcolor='white', showlegend=False,
                yaxis=dict(title='Cup', gridcolor='#f0f0f0'),
                xaxis=dict(gridcolor='#f0f0f0'))
            st.plotly_chart(fig4, use_container_width=True)
            st.dataframe(df7, use_container_width=True, hide_index=True)
        else:
            st.error('Gagal menghitung forecast 7 hari.')


# ══════════════════════════════════════════════════════════════════════════════
# EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == '📊 Evaluasi Model':
    st.title('Evaluasi Model XGBoost')
    
    st.markdown('---')

    best_s1 = eval_s1.loc[eval_s1['MAPE (%)'].idxmin()]
    c1, c2, c3 = st.columns(3)
    with c1: st.metric('MAE',  f'{best_s1["MAE"]:.4f} cup')
    with c2: st.metric('RMSE', f'{best_s1["RMSE"]:.4f} cup')
    with c3: st.metric('MAPE', f'{best_s1["MAPE (%)"]:.2f}%')

    st.markdown('---')
    st.subheader('Perbandingan Baseline vs RandomizedSearch')

    e1d = eval_s1.copy()
    e1d['Kategori'] = e1d['MAPE (%)'].apply(mape_kategori)
    base = e1d[e1d['Optimizer']=='Baseline']['MAPE (%)'].values[0]
    e1d['vs Baseline'] = e1d['MAPE (%)'].apply(
        lambda x: '-' if x==base else f'-{base-x:.2f}pp' if x<base else f'+{x-base:.2f}pp')

    def hl(row):
        return ['background-color:#e8f5e9']*len(row) if row['MAPE (%)']==eval_s1['MAPE (%)'].min() else ['']*len(row)

    st.dataframe(e1d.style.apply(hl, axis=1), use_container_width=True, hide_index=True)

    st.markdown(f"""<div class="info-box">
        XGBoost RandomizedSearch mencapai MAPE <strong>{best_s1['MAPE (%)']:.2f}%</strong> --
    </div>""", unsafe_allow_html=True)

    st.markdown('---')



# ══════════════════════════════════════════════════════════════════════════════
# TENTANG
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'ℹ️ Tentang':
    st.title('Tentang Aplikasi')
    st.markdown('---')
    st.markdown("""
    ### Latar Belakang


    ### Cara Menggunakan
    1. Export laporan dari Luna POS (Laporan -> Export CSV/Excel)
    2. Upload di sidebar kiri (Update Data Penjualan)
    3. Pilih tanggal di halaman **Forecasting Harian** dan klik Forecast


    """)
    st.caption('Prototype untuk keperluan penelitian akademik.')
