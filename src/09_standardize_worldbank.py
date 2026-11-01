# SOURCES: https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN
# https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv

from pathlib import Path
import json
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/worldbank_life_expectancy.json"
OUTPUT = ROOT / "data/processed/worldbank_life_expectancy_standardized.csv"

# 1. Read World Bank data
with open(SOURCE, "r", encoding="utf-8") as f:
    raw = json.load(f)

wb = pd.DataFrame(raw[1])

# 2. Standardize columns
wb["iso_alpha"] = wb["countryiso3code"]
wb["year"] = pd.to_numeric(wb["date"], errors="coerce")
wb["life_expectancy"] = pd.to_numeric(wb["value"], errors="coerce")
wb["country_worldbank"] = wb["country"].map(
    lambda x: x.get("value", "") if isinstance(x, dict) else ""
)
wb["data_source"] = "World Bank"

# 3. Build the corrected Gapminder country reference
gap = px.data.gapminder().copy()
gap.loc[gap["country"] == "Korea, Dem. Rep.", "iso_alpha"] = "PRK"

assert not gap.duplicated(["iso_alpha", "year"]).any()

valid_countries = set(gap["iso_alpha"])
valid_years = set(gap["year"])

# 4. Keep corresponding countries and historical years
wb = wb[
    wb["iso_alpha"].isin(valid_countries)
    & wb["year"].isin(valid_years)
].copy()

wb["year"] = wb["year"].astype(int)

# 5. Keep standardized data
columns = [
    "iso_alpha",
    "year",
    "country_worldbank",
    "life_expectancy",
    "data_source"
]
wb = wb[columns].sort_values(["iso_alpha", "year"])

assert not wb.duplicated(["iso_alpha", "year"]).any()

# Keep missing values for the subsequent cleaning audit
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
wb.to_csv(OUTPUT, index=False)

print("STANDARDIZATION SUCCESS")
print("Total rows:", len(wb))
print("Valid life expectancy:", wb["life_expectancy"].notna().sum())
print("Missing life expectancy:", wb["life_expectancy"].isna().sum())
print("Unique countries:", wb["iso_alpha"].nunique())
print("Duplicate country/year:", wb.duplicated(["iso_alpha", "year"]).sum())
print("Saved file:", OUTPUT)
