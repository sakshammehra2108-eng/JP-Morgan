import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy import stats
from datetime import datetime, timedelta

# --- 1. DATA ENGINE (High-Fidelity JPMC Ledger) ---
@st.cache_data
def load_jpmc_ledger(n=10000):
    np.random.seed(42)
    start_date = datetime(2025, 1, 1)
    
    # Precise timestamps for "Ghost Hour" detection
    dates = [start_date + timedelta(seconds=np.random.randint(0, 31536000)) for _ in range(n)]
    
    segments = ['Retail', 'Corporate', 'Private Banking', 'Institutional']
    channels = ['SWIFT', 'Mobile App', 'ATM', 'API Gateway', 'In-Branch']
    categories = ['Utilities', 'Luxury Retail', 'Gambling', 'Crypto', 'Healthcare', 'Tax Haven']
    
    df = pd.DataFrame({
        'Transaction_ID': [f"TXN-{i:06d}" for i in range(n)],
        'Timestamp': dates,
        'Amount': np.random.lognormal(mean=7.5, sigma=1.8, size=n) + 1.0,
        'Segment': np.random.choice(segments, n, p=[0.5, 0.3, 0.1, 0.1]),
        'Channel': np.random.choice(channels, n),
        'Category': np.random.choice(categories, n),
        'Merchant': np.random.choice(['Amazon', 'Shell', 'Unknown Casino', 'Crypto_Exch', 'Tax Haven Ltd', 'Apple'], n),
        'Region': np.random.choice(['North America', 'EMEA', 'APAC', 'LATAM'], n)
    })
    
    # Statistical Calculations
    df['Z_Score'] = stats.zscore(df['Amount'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['First_Digit'] = df['Amount'].apply(lambda x: int(str(x).replace('.', '').lstrip('0')[0]))
    
    # Regex Risk Detection
    risk_pattern = r"(Casino|Crypto|Haven|Unknown|Ltd)"
    df['Risk_Flag'] = df['Merchant'].apply(lambda x: 1 if re.search(risk_pattern, x, re.IGNORECASE) else 0)
    
    return df

df = load_jpmc_ledger()

# --- 2. ADVANCED INTERFACE ---
st.set_page_config(page_title="JPMC Forensic Risk Suite", layout="wide")
st.title("🏦 J.P. Morgan Chase: Global Risk & Transaction Monitoring")
st.markdown("---")

# --- 3. DYNAMIC SIDEBAR FILTERS ---
st.sidebar.header("🎛️ Risk Control Center")
with st.sidebar:
    date_range = st.date_input("Time Period", [df['Timestamp'].min(), df['Timestamp'].max()])
    selected_regions = st.multiselect("Region", df['Region'].unique(), default=df['Region'].unique())
    selected_segments = st.multiselect("Client Segment", df['Segment'].unique(), default=df['Segment'].unique())
    selected_channels = st.multiselect("Transaction Channel", df['Channel'].unique(), default=df['Channel'].unique())
    
    st.divider()
    z_threshold = st.slider("Anomaly Sensitivity (Z-Score)", 1.5, 5.0, 3.0)
    min_amt, max_amt = st.slider("Transaction Amount Range ($)", 0, 100000, (0, 100000))

# Filter Application
mask = (df['Timestamp'].dt.date >= date_range[0]) & (df['Timestamp'].dt.date <= date_range[1]) & \
       (df['Region'].isin(selected_regions)) & (df['Segment'].isin(selected_segments)) & \
       (df['Channel'].isin(selected_channels)) & (df['Amount'].between(min_amt, max_amt))
f_df = df[mask]

# --- 4. EXECUTIVE SUMMARY METRICS ---
anomalies = f_df[f_df['Z_Score'] > z_threshold]
risk_hits = f_df[f_df['Risk_Flag'] == 1]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Exposure", f"${f_df['Amount'].sum()/1e6:.2f}M")
m2.metric("Filtered Vol.", len(f_df))
m3.metric("Critical Anomalies", len(anomalies), delta=f"{len(anomalies)/len(f_df)*100:.1f}% of total")
m4.metric("High-Risk Entities", len(risk_hits))

st.markdown("---")

# --- 5. DATA VISUALIZATION GRID ---

# ROW 1: Forensic Pattern Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Benford's Law (Fraud Detection)")
    st.info("Reason: Detects if transaction amounts are natural or manually 'fudged'.")
    # Calculate actual vs expected
    actual_digits = f_df['First_Digit'].value_counts(normalize=True).sort_index()
    st.bar_chart(actual_digits)

with col2:
    st.subheader("🕵️ 'Ghost Hour' Analysis (Temporal Risk)")
    st.info("Reason: Money laundering typically spikes between 12 AM and 5 AM.")
    hourly_vol = f_df.groupby('Hour').size()
    st.line_chart(hourly_vol)

# ROW 2: Segment & Regional Exposure
col3, col4 = st.columns(2)

with col3:
    st.subheader("🌎 Regional Risk Concentration")
    region_risk = f_df.groupby('Region')['Amount'].sum()
    st.bar_chart(region_risk)

with col4:
    st.subheader("🏦 Segmented Exposure Distribution")
    segment_dist = f_df.groupby('Segment')['Amount'].mean()
    st.bar_chart(segment_dist)

st.markdown("---")

# ROW 3: High-Density Scatter
st.subheader("📍 Multi-Dimensional Outlier Mapping")
st.info("Color represents the customer segment. Outliers appearing higher on the Y-axis require immediate AML (Anti-Money Laundering) review.")
st.scatter_chart(data=f_df, x='Timestamp', y='Amount', color='Segment', size='Z_Score')

# --- 6. FORENSIC AUDIT TRAIL ---
st.subheader("📝 Critical Audit Trail")
st.write("Displaying transactions filtered by the current Z-Score threshold and Risk Flags.")
# Combine anomalies and regex risk hits
final_audit = pd.concat([anomalies, risk_hits]).drop_duplicates().sort_values(by='Amount', ascending=False)
st.dataframe(final_audit, use_container_width=True)
