# SOURCES: https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN
# https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
REPORTS = ROOT / "reports"

# 1. Read the integrated dataset
df = pd.read_csv(DATA / "multisource_with_dynamic.csv")

assert len(df) == 3104
assert not df.duplicated(
    ["data_source", "iso_alpha", "year"]
).any()

# 2. Preserve original life expectancy
df["life_expectancy_original"] = df["life_expectancy"]
df["life_expectancy_imputed"] = False
df["life_expectancy_value_source"] = df["data_source"]

missing_before = int(df["life_expectancy"].isna().sum())

# 3. Obtain reference values from Gapminder
gap = df.loc[
    df["data_source"].eq("Gapminder"),
    ["record_id", "life_expectancy"]
].copy()

assert gap["record_id"].is_unique

gap = gap.rename(
    columns={
        "life_expectancy": "gapminder_reference_life_expectancy"
    }
)

df = df.merge(
    gap,
    on="record_id",
    how="left",
    validate="many_to_one"
)

# 4. Identify missing World Bank records
mask = (
    df["data_source"].eq("World Bank")
    & df["life_expectancy"].isna()
)

assert missing_before == int(mask.sum())

assert df.loc[
    mask, "gapminder_reference_life_expectancy"
].notna().all()

# 5. Fill using matching Gapminder values
df.loc[mask, "life_expectancy"] = df.loc[
    mask, "gapminder_reference_life_expectancy"
]

df.loc[mask, "life_expectancy_imputed"] = True
df.loc[mask, "life_expectancy_value_source"] = (
    "Gapminder fallback"
)

# 6. Save an audit of the six replaced values
audit = df.loc[
    mask,
    [
        "country",
        "year",
        "record_id",
        "data_source",
        "life_expectancy_original",
        "life_expectancy",
        "life_expectancy_value_source"
    ]
].copy()

REPORTS.mkdir(parents=True, exist_ok=True)
audit.to_csv(
    REPORTS / "worldbank_missing_value_audit.csv",
    index=False
)

# 7. Validate and export
df = df.drop(
    columns=["gapminder_reference_life_expectancy"]
)

assert len(df) == 3104
assert df["life_expectancy"].notna().all()
assert int(df["life_expectancy_imputed"].sum()) == missing_before

output = DATA / "multisource_lifeexp_filled.csv"
df.to_csv(output, index=False)

print("LIFE EXPECTANCY CLEANING SUCCESS")
print("Total rows:", len(df))
print("Missing before:", missing_before)
print("Values filled:", int(mask.sum()))
print("Missing after:", df["life_expectancy"].isna().sum())
print("Original missing values preserved:",
      df["life_expectancy_original"].isna().sum())
print("Saved:", output)
print("Audit:", REPORTS / "worldbank_missing_value_audit.csv")
