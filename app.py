import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy import stats
from datetime import datetime, timedelta

# --- 1. DATA ENGINE (High-Fidelity Simulation) ---
@st.cache_data
def load_jpmc_ledger(n=5000):
    np.random.seed(42)
    start_date = datetime(2025, 1, 1)
    
    # Generate unique timestamps (HH:MM:SS)
    dates = [start_date + timedelta(seconds=np.random.randint(0, 31536000)) for _ in range(n)]
    
    df = pd.DataFrame({
        'Transaction_ID': [f"TXN-{i:06d}" for i in range(n)],
        'Timestamp': dates,
        'Amount': np.random.lognormal(mean=7.5, sigma=1.5, size=n),
        'Segment': np.random.choice(['Retail', 'Corporate', 'Private Banking', 'Institutional'], n),
        'Merchant': np.random.choice(['Amazon', 'Shell', 'Unknown Casino', 'Crypto_Exch', 'Tax Haven Ltd', 'Apple'], n)
    })
    
    # Forensic Analytics: Z-Score and Hour Extraction
    df['Z_Score'] = stats.zscore(df['Amount'])
    df['Hour'] = df['Timestamp'].dt.hour
    
    # Regex Risk Detection
    risk_pattern = r"(Casino|Crypto|Haven|Unknown)"
    df['Risk_Flag'] = df['Merchant'].apply(lambda x: 1 if re.search(risk_pattern, x, re.IGNORECASE) else 0)
    
    return df

df = load_jpmc_ledger()

# --- 2. DASHBOARD INTERFACE ---
st.set_page_config(page_title="JPMC Risk Monitor", layout="wide")
st.title("🏦 J.P. Morgan Chase: Risk & Monitoring Dashboard")
st.write("Forensic Analysis Suite for Banking Transactions")

# Summary Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Flagged (Z > 3)", len(df[df['Z_Score'] > 3]))
col2.metric("High-Risk Entities", df['Risk_Flag'].sum())
col3.metric("Avg Transaction Value", f"${df['Amount'].mean():,.2f}")

st.divider()

# --- 3. NATIVE VISUALIZATIONS (No Matplotlib Dependencies) ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("Temporal Risk: Hourly Volume")
    # Identifies 'Ghost Hour' activity (red flag for automated washing)
    hourly_vol = df.groupby('Hour').size()
    st.bar_chart(hourly_vol)

with c2:
    st.subheader("Transaction Outlier Mapping")
    # Interactive scatter for fat-tail risk detection
    st.scatter_chart(data=df, x='Timestamp', y='Amount', color='Segment')

st.divider()

# --- 4. AUDIT TRAIL ---
st.subheader("Critical Audit Table (Z-Score > 2)")
# Displaying high-deviation transactions for manual BA review
audit_data = df[df['Z_Score'] > 2].sort_values(by='Amount', ascending=False)
st.dataframe(audit_data, use_container_width=True)