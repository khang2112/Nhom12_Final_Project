# SOURCES: https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN
# https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
SYN = ROOT / "data/synthetic"
REPORTS = ROOT / "reports"

# 1. Load the unified data from two real sources
multi = pd.read_csv(
    DATA / "unified_static_master_multisource.csv"
)

# 2. Load original Gapminder reference
reference = pd.read_csv(
    DATA / "unified_static_master.csv"
)

# Correct North Korea ISO code in working data
reference.loc[
    reference["country"] == "Korea, Dem. Rep.",
    "iso_alpha"
] = "PRK"

# Unique country-year reference
keys = ["iso_alpha", "year"]

assert not reference.duplicated(keys).any()
assert not multi.duplicated(
    ["data_source", "iso_alpha", "year"]
).any()

# 3. Add reference record IDs and economic context
reference_cols = [
    "iso_alpha",
    "year",
    "record_id",
    "population",
    "gdp_per_capita"
]

merged = multi.merge(
    reference[reference_cols],
    on=keys,
    how="left",
    validate="many_to_one",
    indicator="reference_match"
)

assert merged["reference_match"].eq("both").all()

merged = merged.drop(columns="reference_match")

# World Bank provides life expectancy only.
# Population and GDP context come from Gapminder.
merged["economic_data_source"] = "Gapminder"

# 4. Load simulated dynamic indicators
dynamic = pd.read_csv(
    SYN / "simulated_country_year_indicators.csv"
)

assert dynamic["record_id"].is_unique

dynamic = dynamic.drop(
    columns=["country", "year"]
)

# 5. Horizontal merge with dynamic data
merged = merged.merge(
    dynamic,
    on="record_id",
    how="left",
    validate="many_to_one",
    indicator="dynamic_match"
)

assert merged["dynamic_match"].eq("both").all()

merged = merged.drop(columns="dynamic_match")

# 6. Keep source observations distinct
assert len(merged) == len(multi)

assert not merged.duplicated(
    ["data_source", "iso_alpha", "year"]
).any()

# 7. Save without overwriting the AI master
output = DATA / "multisource_with_dynamic.csv"
merged.to_csv(output, index=False)

# 8. Generate an integration audit
audit = merged.groupby(
    "data_source"
).agg(
    rows=("record_id", "size"),
    unique_records=("record_id", "nunique"),
    missing_life_expectancy=(
        "life_expectancy",
        lambda x: int(x.isna().sum())
    )
).reset_index()

REPORTS.mkdir(parents=True, exist_ok=True)

audit.to_csv(
    REPORTS / "multisource_merge_audit.csv",
    index=False
)

print("MULTISOURCE MERGE SUCCESS")
print("Total rows:", len(merged))
print("Unique country-year records:",
      merged["record_id"].nunique())
print("Missing life expectancy:",
      merged["life_expectancy"].isna().sum())
print("Dynamic join failures:", 0)
print("Duplicate source-country-year:",
      merged.duplicated(
          ["data_source", "iso_alpha", "year"]
      ).sum())
print("Saved:", output)
print(audit.to_string(index=False))
