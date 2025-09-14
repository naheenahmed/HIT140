# HIT140 — Assessment 2 (Investigation A)
# Clean, minimal script using statsmodels only
# - Cleans and merges dataset1.csv and dataset2.csv
# - Saves cleaned CSVs + merged CSV
# - Generates plots + stats summary
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import statsmodels.api as sm

DATASET1 = "dataset1.csv"  # bats
DATASET2 = "dataset2.csv"  # rats

def parse_dt(series, primary_fmt="%d/%m/%Y %H:%M"):
    """Parse datetimes robustly (try known format, then infer)."""
    s = pd.to_datetime(series, format=primary_fmt, errors="coerce")
    if s.isna().mean() > 0.2:
        s = pd.to_datetime(series, errors="coerce", dayfirst=True, infer_datetime_format=True)
    return s

def main():
    if not os.path.exists(DATASET1) or not os.path.exists(DATASET2):
        raise FileNotFoundError("Place dataset1.csv and dataset2.csv in the same folder as this script.")

# Load datasets
    d1 = pd.read_csv(DATASET1)  # bats
    d2 = pd.read_csv(DATASET2)  # rats

# Clean datetime columns
    d1["start_time"]       = parse_dt(d1.get("start_time"))
    d1["rat_period_start"] = parse_dt(d1.get("rat_period_start"))
    d1["rat_period_end"]   = parse_dt(d1.get("rat_period_end"))
    d1["sunset_time"]      = parse_dt(d1.get("sunset_time"))
    d2["time"]             = parse_dt(d2.get("time"))
    
    if "habit" in d1.columns:
        d1["habit"] = d1["habit"].fillna("unknown")
        
# Merge on 30-minute blocks
    d1["time_block"] = d1["start_time"].dt.floor("30min")
    d2["time_block"] = d2["time"].dt.floor("30min")

    merged = pd.merge(
        d1,
        d2[["time_block", "rat_minutes", "rat_arrival_number", "bat_landing_number", "food_availability"]],
        on="time_block",
        how="left"
    )

# Save cleaned datasets
    d1.to_csv("clean_dataset1_bats.csv", index=False)
    d2.to_csv("clean_dataset2_rats.csv", index=False)
    merged.to_csv("merged_dataset.csv", index=False)
    print("Cleaned data saved as clean_dataset1_bats.csv, clean_dataset2_rats.csv, merged_dataset.csv")

# Histogram: seconds_after_rat_arrival by risk
    plt.figure(figsize=(7, 4))
    for r in sorted(merged["risk"].dropna().unique()):
        vals = merged.loc[merged["risk"] == r, "seconds_after_rat_arrival"].dropna()
        if len(vals):
            color = "blue" if r == 0 else "orange"
            plt.hist(vals, bins=30, alpha=0.7, color=color, edgecolor="black", label=f"Risk {r}")
    plt.title("Seconds After Rat Arrival by Risk")
    plt.xlabel("Seconds After Rat Arrival")
    plt.ylabel("Count")
    plt.legend(title="Risk Behaviour")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("histogram_risk_vs_time.png", dpi=300)
    plt.close()

# Bar chart: mean risk vs rat_minutes bins
    merged_nonnull = merged.dropna(subset=["rat_minutes"]).copy()
    if not merged_nonnull.empty:
        merged_nonnull["rat_minutes_bin"] = pd.cut(
            merged_nonnull["rat_minutes"],
            bins=[-0.1, 0, 1, 5, 10, 20, 60, 120],
            include_lowest=True
        )
        risk_by_ratmins = merged_nonnull.groupby("rat_minutes_bin", observed=False)["risk"].mean()

        plt.figure(figsize=(7, 4))
        risk_by_ratmins.plot(kind="bar", color="red", edgecolor="black")
        plt.title("Mean Risk-Taking vs Rat Minutes")
        plt.xlabel("Rat Minutes (per 30-min window)")
        plt.ylabel("Mean Risk (0=avoid, 1=risk-taking)")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("bar_risk_vs_rat_minutes.png", dpi=300)
        plt.close()

# Stats for Investigation A
    stats_lines = []

# (A) Chi-square: rat presence vs risk
    merged["rat_presence"] = (merged["rat_minutes"] > 0).astype(int)
    contingency = pd.crosstab(merged["rat_presence"], merged["risk"])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    stats_lines.append("Chi-square (rat presence x risk)")
    stats_lines.append(str(contingency))
    stats_lines.append(f"Chi2={chi2:.4f}, p={p_value:.6f}, dof={dof}\n")

# (B) Chi-square: binned rat_minutes vs risk
    merged_nonnull = merged.dropna(subset=["rat_minutes", "risk"]).copy()
    if not merged_nonnull.empty:
        merged_nonnull["rat_minutes_bin"] = pd.cut(
            merged_nonnull["rat_minutes"],
            bins=[-0.1, 0, 1, 5, 10, 20, 60, 120],
            include_lowest=True
        )
        contingency_bins = pd.crosstab(merged_nonnull["rat_minutes_bin"], merged_nonnull["risk"])
        chi2b, p_value_b, dof_b, expected_b = chi2_contingency(contingency_bins)
        stats_lines.append("Chi-square (rat_minutes_bin x risk)")
        stats_lines.append(str(contingency_bins))
        stats_lines.append(f"Chi2={chi2b:.4f}, p={p_value_b:.6f}, dof={dof_b}\n")
    else:
        stats_lines.append("Chi-square with rat_minutes bins not possible (empty data).\n")

# (C) Logistic regression
    X = merged[["rat_minutes", "seconds_after_rat_arrival", "hours_after_sunset"]].replace([np.inf, -np.inf], np.nan).dropna()
    y = merged.loc[X.index, "risk"]

    if len(X) > 0 and y.nunique() > 1:
        try:
            X_sm = sm.add_constant(X, has_constant="add")
            model = sm.Logit(y, X_sm).fit(disp=False)
            stats_lines.append(str(model.summary()))
            odds_ratios = np.exp(model.params)
            stats_lines.append("\nOdds ratios:\n" + str(odds_ratios))
        except Exception as e:
            stats_lines.append(f"Logit failed: {e}")
    else:
        stats_lines.append("Insufficient variation in y or predictors for logistic regression.")

# Save stats summary
    with open("stats_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))
    print("Saved stats_summary.txt, plus plots")

if __name__ == "__main__":
    main()

