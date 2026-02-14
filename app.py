import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy import stats
from datetime import datetime, timedelta

# --- 1. DATA ENGINE ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 2000 
    start_date = datetime(2025, 1, 1)
    
    # Generate unique timestamps with HH:MM:SS
    dates = [start_date + timedelta(seconds=np.random.randint(0, 31536000)) for _ in range(n)]
    
    df = pd.DataFrame({
        'Timestamp': dates,
        'Amount': np.random.lognormal(mean=7, sigma=1.5, size=n),
        'Segment': np.random.choice(['Retail', 'Corporate', 'Private Banking', 'Institutional'], n),
        'Merchant': np.random.choice(['Amazon', 'Shell', 'Unknown Casino', 'Crypto_Exch', 'Tax Haven Ltd', 'Apple', 'Walmart'], n)
    })
    
    # Business Analyst Logic: Z-Score for Anomaly Detection
    df['Z_Score'] = stats.zscore(df['Amount'])
    df['Hour'] = df['Timestamp'].dt.hour
    
    # Regex Risk Detection (Forensic Text Mining)
    risk_pattern = r"(Casino|Crypto|Haven|Unknown|Ltd)"
    df['Risk_Flag'] = df['Merchant'].apply(lambda x: 1 if re.search(risk_pattern, x, re.IGNORECASE) else 0)
    
    return df

df = load_data()

# --- 2. DASHBOARD UI ---
st.set_page_config(page_title="JPMC Risk Monitor", layout="wide")
st.title("🏦 JPMC: Transaction Risk & Monitoring")
st.markdown("Developed by Saksham Mehra - Business Analyst Suite")

# Sidebar Filters
st.sidebar.header("Risk Controls")
seg_filter = st.sidebar.multiselect("Select Segment", df['Segment'].unique(), default=df['Segment'].unique())
z_threshold = st.sidebar.slider("Anomaly Sensitivity (Z-Score)", 1.0, 5.0, 3.0)

# Apply Filters
filtered_df = df[(df['Segment'].isin(seg_filter))]
anomalies = filtered_df[filtered_df['Z_Score'] > z_threshold]

# KPI Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Volume", f"${filtered_df['Amount'].sum():,.0f}")
c2.metric("Critical Anomalies", len(anomalies))
c3.metric("High-Risk Entities", filtered_df['Risk_Flag'].sum())

# --- 3. NATIVE CHARTS (No Matplotlib Needed) ---
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Temporal Risk (Hourly Volume)")
    # Identifies "Ghost Hour" activity automatically
    hourly_counts = filtered_df.groupby('Hour').size()
    st.bar_chart(hourly_counts)

with col_right:
    st.subheader("Transaction Distribution")
    # Native Scatter: Interactive and stable
    st.scatter_chart(data=filtered_df, x='Timestamp', y='Amount', color='Segment')

# --- 4. AUDIT TABLE ---
st.subheader("Forensic Audit Trail (Filtered by Risk)")
st.dataframe(anomalies.sort_values(by='Z_Score', ascending=False), use_container_width=True)