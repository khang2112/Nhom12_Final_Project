# SOURCES: https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN
# https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
REPORTS = ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

# 1. Read the integrated dataset
source = pd.read_csv(
    DATA / "multisource_lifeexp_filled.csv"
)

assert len(source) == 3104

sim_cols = [
    "sim_health_index",
    "sim_education_index",
    "sim_trend_index",
    "sim_policy_group"
]

# 2. Audit missing values before cleaning
missing_before = source[sim_cols].isna().sum()

# 3. Clean the unique country-year observations first
base = source[
    source["data_source"] == "Gapminder"
].copy()

base = base.sort_values(
    ["country", "year"]
).reset_index(drop=True)

assert len(base) == 1704
assert base["record_id"].is_unique

# Time-series: forward-fill/backward-fill within country
base["sim_trend_index"] = (
    base.groupby("country")["sim_trend_index"]
    .transform(lambda s: s.ffill().bfill())
)

# Numerical: median by continent
for col in [
    "sim_health_index",
    "sim_education_index"
]:
    base[col] = base[col].fillna(
        base.groupby("continent")[col].transform("median")
    )
    base[col] = base[col].fillna(base[col].median())

# Fallback if a whole country's time series was missing
base["sim_trend_index"] = base[
    "sim_trend_index"
].fillna(base["sim_trend_index"].median())

# Categorical: explicit missing category
base["sim_policy_group"] = (
    base["sim_policy_group"].fillna("Unknown")
)

assert base[sim_cols].isna().sum().sum() == 0

# 4. Apply consistent values to both data sources
clean_values = base[["record_id"] + sim_cols].copy()

cleaned = source.drop(columns=sim_cols).merge(
    clean_values,
    on="record_id",
    how="left",
    validate="many_to_one",
    indicator=True
)

assert cleaned["_merge"].eq("both").all()
cleaned = cleaned.drop(columns="_merge")

# Original value is retained in the source and audit files.
# Remove this intentionally sparse audit column from clean data.
cleaned = cleaned.drop(
    columns=["life_expectancy_original"],
    errors="ignore"
)

# 5. Validate
assert len(cleaned) == 3104
assert cleaned["record_id"].nunique() == 1704
assert not cleaned.duplicated(
    ["data_source", "iso_alpha", "year"]
).any()
assert cleaned.isna().sum().sum() == 0

# 6. Save the new clean dataset
output = DATA / "multisource_clean.csv"
cleaned.to_csv(output, index=False)

# 7. Before-and-after missingness report
comparison = pd.DataFrame({
    "column": sim_cols,
    "missing_before": missing_before.values,
    "missing_after": cleaned[sim_cols].isna().sum().values
})

comparison["filled_count"] = (
    comparison["missing_before"]
    - comparison["missing_after"]
)

comparison.to_csv(
    REPORTS / "multisource_imputation_report.csv",
    index=False
)

# 8. Distribution integrity report
# Compare unique country-year observations to avoid double-counting
impact = []

for col in [
    "sim_health_index",
    "sim_education_index",
    "sim_trend_index"
]:
    original = source.loc[
        source["data_source"] == "Gapminder", col
    ].dropna()

    final = base[col].dropna()

    impact.append({
        "column": col,
        "mean_before": original.mean(),
        "mean_after": final.mean(),
        "median_before": original.median(),
        "median_after": final.median(),
        "std_before": original.std(),
        "std_after": final.std()
    })

pd.DataFrame(impact).to_csv(
    REPORTS / "multisource_distribution_impact.csv",
    index=False
)

# 9. Final results
print("MULTISOURCE CLEANING SUCCESS")
print("Total rows:", len(cleaned))
print("Unique country-year:", cleaned["record_id"].nunique())
print("Missing simulated values before:", int(missing_before.sum()))
print("Missing simulated values after:",
      int(cleaned[sim_cols].isna().sum().sum()))
print("Total missing in clean dataset:",
      int(cleaned.isna().sum().sum()))
print("Saved:", output)

print("\n=== IMPUTATION REPORT ===")
print(comparison.to_string(index=False))
