# SOURCES: https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN
# https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv

from pathlib import Path
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]

# Source 1: Gapminder
gap = px.data.gapminder().copy()

gap.loc[
    gap["country"] == "Korea, Dem. Rep.",
    "iso_alpha"
] = "PRK"

gap = gap.rename(
    columns={"lifeExp": "life_expectancy_gapminder"}
)

# Source 2: World Bank
wb = pd.read_csv(
    ROOT / "data/processed/worldbank_life_expectancy_standardized.csv"
)

wb = wb.rename(
    columns={"life_expectancy": "life_expectancy_worldbank"}
)

# Match by country ISO code and year
matched = pd.merge(
    gap[["country", "iso_alpha", "year",
         "life_expectancy_gapminder"]],
    wb[["iso_alpha", "year",
        "life_expectancy_worldbank"]],
    on=["iso_alpha", "year"],
    how="inner",
    validate="one_to_one"
)

# Compare records having values in both sources
valid = matched.dropna(
    subset=["life_expectancy_gapminder",
            "life_expectancy_worldbank"]
).copy()

valid["absolute_difference"] = (
    valid["life_expectancy_gapminder"]
    - valid["life_expectancy_worldbank"]
).abs()

# Export the comparison for the report
output = ROOT / "reports/worldbank_gapminder_comparison.csv"
output.parent.mkdir(parents=True, exist_ok=True)
valid.to_csv(output, index=False)

print("COMPARISON SUCCESS")
print("Matched records:", len(matched))
print("Valid comparable records:", len(valid))
print("Mean absolute difference:",
      round(valid["absolute_difference"].mean(), 3))
print("Median absolute difference:",
      round(valid["absolute_difference"].median(), 3))
print("Records with difference > 5 years:",
      int((valid["absolute_difference"] > 5).sum()))
print("Saved:", output)
