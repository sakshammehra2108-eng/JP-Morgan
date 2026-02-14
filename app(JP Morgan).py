import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import stats
from datetime import datetime, timedelta

# --- 1. DATA ENGINE (Audit-Ready) ---
@st.cache_data
def load_bank_ledger(n=5000):
    np.random.seed(42)
    start_date = datetime(2025, 1, 1)
    
    # Generate unique timestamps with full HH:MM:SS
    dates = [start_date + timedelta(seconds=np.random.randint(0, 31536000)) for _ in range(n)]
    
    segments = ['Retail', 'Corporate', 'Private Banking', 'Institutional']
    merchants = ['Amazon', 'Shell', 'Unknown Casino', 'Crypto_Exch_01', 'TaxHaven_Ltd', 'Walmart', 'Apple']
    
    df = pd.DataFrame({
        'Transaction_ID': [f"TXN-{i:06d}" for i in range(n)],
        'Timestamp': dates,
        'Amount': np.random.lognormal(mean=7, sigma=1.8, size=n) + 1,
        'Segment': np.random.choice(segments, n, p=[0.6, 0.2, 0.1, 0.1]),
        'Merchant': np.random.choice(merchants, n),
        'Hour': [d.hour for d in dates]
    })
    
    # Forensic Columns
    df['Z_Score'] = stats.zscore(df['Amount'])
    df['First_Digit'] = df['Amount'].apply(lambda x: int(str(x).replace('.', '').lstrip('0')[0]))
    
    # Regex Fraud Search
    risk_regex = r"(Casino|Crypto|Haven|Unknown|Ltd)"
    df['High_Risk_Entity'] = df['Merchant'].apply(lambda x: 1 if re.search(risk_regex, x, re.IGNORECASE) else 0)
    
    return df

df = load_bank_ledger()

# --- 2. INTERFACE ---
st.set_page_config(page_title="JPMC Risk Dashboard", layout="wide")
st.title("🏦 J.P. Morgan Chase & Co.")
st.subheader("Transaction Monitoring & Operational Risk Dashboard")

# Sidebar
st.sidebar.header("Filter Analytics")
target_segment = st.sidebar.multiselect("Client Segment", df['Segment'].unique(), default=df['Segment'].unique())
min_amount = st.sidebar.number_input("Min Transaction Amount ($)", 0, 1000000, 0)

filtered_df = df[(df['Segment'].isin(target_segment)) & (df['Amount'] >= min_amount)]

# --- 3. BUSINESS ANALYTICS MODULES ---

col1, col2, col3 = st.columns(3)
col1.metric("Total Exposure", f"${filtered_df['Amount'].sum():,.2f}")
col2.metric("Flagged Anomalies", len(filtered_df[filtered_df['Z_Score'] > 3]))
col3.metric("High-Risk Merchant Hits", filtered_df['High_Risk_Entity'].sum())

st.divider()

# --- 4. DATA VISUALIZATION (Error-Checked Matplotlib) ---

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Benford's Law: Fraud Detection")
    
    # Calculate distributions
    actual_freq = filtered_df['First_Digit'].value_counts(normalize=True).sort_index()
    # Ensure all digits 1-9 are present even if not in filtered data to avoid label mismatch
    full_index = range(1, 10)
    actual_freq = actual_freq.reindex(full_index, fill_value=0)
    expected_freq = np.log10(1 + 1/np.array(full_index))
    
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(full_index, actual_freq.values, color='#004a99', alpha=0.6, label='Actual Data')
    ax1.plot(full_index, expected_freq, color='red', marker='o', label='Benford Expected')
    
    ax1.set_title("First Digit Frequency Distribution")
    ax1.set_xlabel("Leading Digit")
    ax1.set_ylabel("Probability")
    ax1.set_xticks(full_index) # Force 1-9 labels to prevent auto-scaling errors
    ax1.legend()
    st.pyplot(fig1)



with chart_col2:
    st.markdown("#### Hourly Transaction Density")
    
    # Ensure all 24 hours are represented to keep the X-axis stable
    hourly_counts = filtered_df.groupby('Hour')['Amount'].count().reindex(range(0, 24), fill_value=0)
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    # Red for 'Ghost Hours' (0-5 AM), Blue for Business Hours
    bar_colors = ['#d9534f' if (0 <= h <= 5) else '#004a99' for h in range(24)]
    
    ax2.bar(hourly_counts.index, hourly_counts.values, color=bar_colors)
    ax2.set_title("Transaction Count by Hour (24h Clock)")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Volume")
    ax2.set_xticks(range(0, 24)) # Explicitly label 0-23
    st.pyplot(fig2)

st.divider()

# --- 5. RISK SCATTER ---
st.markdown("#### Statistical Exposure: Amount vs. Z-Score")

fig3, ax3 = plt.subplots(figsize=(12, 5))
# Using scatter with timestamp and amount
scatter = ax3.scatter(filtered_df['Timestamp'], filtered_df['Amount'], 
                      c=filtered_df['Z_Score'], cmap='Reds', alpha=0.6, edgecolors='w')
plt.colorbar(scatter, ax=ax3, label='Z-Score (Deviation)')
ax3.set_title("Anomaly Identification (Fat-Tail Events)")
ax3.set_ylabel("Transaction Amount ($)")
st.pyplot(fig3)

# --- 6. AUDIT TABLE ---
st.markdown("#### Forensic Audit Trail (High-Risk Selection)")
# Using regex search again for the table to demonstrate Business Analyst capability
st.dataframe(filtered_df.sort_values(by='Z_Score', ascending=False).head(50), use_container_width=True)