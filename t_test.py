import pandas as pd
from scipy.stats import ttest_rel

# =========================
# User-configurable values
# =========================
file1 = "results/red/cardiff_base_reward_stealth_flood_3.csv"
file2 = "results/red/cardiff_rbt_stealth_flood_0.csv"

# Significance threshold
p_val_threshold = 0.001

# Column to compare
column_name = "red_impact"

# =========================
# Load data
# =========================
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Ensure same length
if len(df1) != len(df2):
    raise ValueError("Files must have the same number of rows for a paired t-test.")

# Extract paired samples
sample1 = df1[column_name]
sample2 = df2[column_name]

# =========================
# Paired t-test
# =========================
t_stat, p_value = ttest_rel(sample1, sample2)

# =========================
# Results
# =========================
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

if p_value < p_val_threshold:
    print(
        f"Result is statistically significant "
        f"(p < {p_val_threshold})"
    )
else:
    print(
        f"Result is NOT statistically significant "
        f"(p >= {p_val_threshold})"
    )