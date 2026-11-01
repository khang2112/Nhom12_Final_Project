# SOURCES: https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN
# https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# SOURCE 1: GAPMINDER
gap = pd.read_csv(
    RAW / "source_gapminder_original.csv"
)

# Correct the original ISO code issue in working data
gap.loc[
    gap["country"] == "Korea, Dem. Rep.",
    "iso_alpha"
] = "PRK"

gap["data_source"] = "Gapminder"

# SOURCE 2: WORLD BANK
wb = pd.read_csv(
    PROCESSED / "worldbank_life_expectancy_standardized.csv"
)

# Use Gapminder's country mapping to align naming
country_map = gap[
    ["iso_alpha", "country", "continent"]
].drop_duplicates(subset=["iso_alpha"])

wb = wb.merge(
    country_map,
    on="iso_alpha",
    how="left",
    validate="many_to_one"
)

assert wb["country"].notna().all()
assert wb["continent"].notna().all()

wb["data_source"] = "World Bank"

# Standardize columns before concatenation
cols = [
    "country",
    "continent",
    "iso_alpha",
    "year",
    "life_expectancy",
    "data_source"
]

gap = gap[cols].copy()
wb = wb[cols].copy()

for df in (gap, wb):
    df["year"] = pd.to_numeric(
        df["year"], errors="raise"
    ).astype("int64")

    df["life_expectancy"] = pd.to_numeric(
        df["life_expectancy"], errors="coerce"
    )

    for col in (
        "country", "continent",
        "iso_alpha", "data_source"
    ):
        df[col] = df[col].astype("string")

# Vertical concatenation
unified = pd.concat(
    [gap, wb],
    ignore_index=True
)

# Each source-country-year should be unique
assert not unified.duplicated(
    ["data_source", "iso_alpha", "year"]
).any()

# Preserve missingness for analysis
unified = unified.sort_values(
    ["iso_alpha", "year", "data_source"]
).reset_index(drop=True)

# Save unified multi-source observations
out = PROCESSED / "unified_static_master_multisource.csv"
unified.to_csv(out, index=False)

# Save source audit
audit = (
    unified.groupby("data_source", dropna=False)
    .agg(
        records=("year", "size"),
        countries=("iso_alpha", "nunique"),
        missing_life_exp=(
            "life_expectancy",
            lambda s: int(s.isna().sum())
        )
    )
    .reset_index()
)

audit.to_csv(
    REPORTS / "multisource_audit.csv",
    index=False
)

print("CONCAT SUCCESS")
print("Gapminder records:", len(gap))
print("World Bank records:", len(wb))
print("Unified records:", len(unified))
print("Missing life expectancy:",
      unified["life_expectancy"].isna().sum())
print("Duplicate source-country-year:",
      unified.duplicated(
          ["data_source", "iso_alpha", "year"]
      ).sum())
print("Saved:", out)
print(audit.to_string(index=False))
