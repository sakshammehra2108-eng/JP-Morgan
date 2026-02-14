import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy import stats
from datetime import datetime, timedelta

# --- 1. SYNTHETIC LEDGER GENERATION ---
def generate_jpmc_data(n=10000):
    np.random.seed(42)
    start_date = datetime(2025, 1, 1)
    
    # Precise timestamps (HH:MM:SS) for temporal risk mapping
    dates = [start_date + timedelta(seconds=np.random.randint(0, 31536000)) for _ in range(n)]
    
    segments = ['Retail', 'Corporate', 'Private Banking', 'Institutional']
    merchants = [
        'Amazon', 'Apple', 'Shell Oil', 'Unknown Casino', 'Global Shell Corp', 
        'Crypto Exchange X', 'Walmart', 'Starbucks', 'Tax Haven Holdings', 'Azure Cloud'
    ]
    
    df = pd.DataFrame({
        'Transaction_ID': [f"JPMC-{i:06d}" for i in range(n)],
        'Timestamp': dates,
        'Amount': np.random.lognormal(mean=7.5, sigma=1.8, size=n) + 0.50,
        'Segment': np.random.choice(segments, n, p=[0.5, 0.3, 0.1, 0.1]),
        'Merchant': np.random.choice(merchants, n),
        'Hour': [d.hour for d in dates]
    })
    
    # Extraction of Lead Digit for Benford's Law Analysis
    df['Lead_Digit'] = df['Amount'].apply(lambda x: int(str(x).replace('.', '').lstrip('0')[0]))
    
    return df

df = generate_jpmc_data()

# --- 2. RISK MONITORING LOGIC (The Business Analyst Engine) ---

# A. Statistical Anomaly Detection (Z-Score)
# Reason: Identifies transactions that are standard deviations away from the mean within their segment.
df['Z_Score'] = stats.zscore(df['Amount'])
anomalies = df[df['Z_Score'] > 3.0]

# B. Regex-Based Forensic Mining
# Reason: Isolates transactions involving high-risk entities like tax havens or casinos.
risk_pattern = r"(Casino|Crypto|Shell|Haven|Unknown)"
df['High_Risk_Flag'] = df['Merchant'].apply(lambda x: 1 if re.search(risk_pattern, x, re.IGNORECASE) else 0)

# C. Benford’s Law Statistical Validation
# Reason: Checks if the data follows a natural logarithmic distribution; deviations suggest fraud.
actual_counts = df['Lead_Digit'].value_counts(normalize=True).sort_index()
expected_benford = np.log10(1 + 1/np.arange(1, 10))

# --- 3. DATA VISUALIZATION REPORT ---
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
plt.subplots_adjust(hspace=0.4)

# Plot 1: Benford's Law Compliance
axs[0, 0].bar(actual_counts.index, actual_counts.values, color='navy', alpha=0.7, label='Actual')
axs[0, 0].plot(range(1, 10), expected_benford, color='red', marker='o', label='Benford Expected')
axs[0, 0].set_title("Forensic Audit: Benford's Law Distribution")
axs[0, 0].set_xlabel("Leading Digit")
axs[0, 0].legend()



# Plot 2: Ghost Hour Analysis (Temporal Risk)
# Reason: High-volume activity during off-peak hours (2AM-4AM) is a red flag for automated laundry.
hourly_vol = df.groupby('Hour')['Amount'].count()
colors = ['firebrick' if (2 <= h <= 4) else 'navy' for h in hourly_vol.index]
axs[0, 1].bar(hourly_vol.index, hourly_vol.values, color=colors)
axs[0, 1].set_title("Temporal Risk: Transaction Volume by Hour")
axs[0, 1].set_xticks(range(0, 24))

# Plot 3: Z-Score Outlier Distribution
# Reason: Visualizes 'Fat-Tail' risks where amounts exceed the normal operational threshold.
axs[1, 0].scatter(df['Timestamp'], df['Amount'], c=df['Z_Score'], cmap='Reds', alpha=0.5)
axs[1, 0].set_title("Statistical Outliers (Amount vs. Deviation)")
axs[1, 0].set_ylabel("Transaction Amount ($)")



# Plot 4: Risk Flag Concentration
# Reason: Shows which segments are most exposed to high-risk merchant categories.
risk_by_segment = df.groupby('Segment')['High_Risk_Flag'].sum()
axs[1, 1].pie(risk_by_segment, labels=risk_by_segment.index, autopct='%1.1f%%', colors=['#004a99', '#ffcc00', '#7a7a7a', '#000000'])
axs[1, 1].set_title("Risk Exposure by Customer Segment")

plt.show()

# --- 4. EXPORT AUDIT TRAIL ---
# Extracting the top 50 high-priority transactions for manual review.
audit_trail = df[df['Z_Score'] > 3.0].sort_values(by='Z_Score', ascending=False)
print("--- HIGH-RISK AUDIT TRAIL ---")
print(audit_trail[['Transaction_ID', 'Merchant', 'Amount', 'Z_Score']].head(10))