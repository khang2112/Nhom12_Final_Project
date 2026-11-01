"""Prepare real Gapminder data and simulated country-year indicators.

Original public dataset: https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv
Dataset information: https://www.gapminder.org/data/
Offline bundled copy: plotly.express.data.gapminder()

Both geographic feeds are extracts of the SAME public dataset, not independent
external publications. Simulated indicators are not real-world observations.
Run from the project root: python src/01_prepare_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
SYN = ROOT / "data/synthetic"
OUT = ROOT / "data/processed"
SEED = 20261010

REAL_COLUMNS = [
    "country", "continent", "year", "population",
    "gdp_per_capita", "life_expectancy", "iso_alpha", "iso_num",
]
NUMERIC_COLUMNS = ["population", "gdp_per_capita", "life_expectancy"]
TEXT_COLUMNS = ["country", "continent", "iso_alpha"]


def get_public_source():
    """Load the packaged Gapminder dataset and preserve its original CSV."""
    df = px.data.gapminder().rename(
        columns={
            "lifeExp": "life_expectancy",
            "gdpPercap": "gdp_per_capita",
            "pop": "population",
        }
    )
    assert len(df) == 1704 and df.country.nunique() == 142

    df = df[REAL_COLUMNS].copy()
    df.to_csv(RAW / "source_gapminder_original.csv", index=False)

    # Correct North Korea ISO codes in the working copy only.
    north_korea = df["country"] == "Korea, Dem. Rep."
    df.loc[north_korea, "iso_alpha"] = "PRK"
    df.loc[north_korea, "iso_num"] = 408
    return df


def generate_dynamic(master):
    """Generate synthetic indicators without using the regression/class target."""
    rg = np.random.default_rng(SEED)
    d = master[
        ["record_id", "country", "year", "gdp_per_capita", "population"]
    ].copy()

    gdp = np.log1p(d.gdp_per_capita.to_numpy())
    relative_time = (d.year.to_numpy() - 1952) / 55

    health = np.clip(
        10 + 7 * gdp + 9 * relative_time + rg.normal(0, 11, len(d)), 0, 100
    )
    education = np.clip(
        9 + 7.3 * gdp + 8 * relative_time + rg.normal(0, 10, len(d)), 0, 100
    )
    trend = np.clip(
        15 + 6.5 * gdp + 12 * relative_time + rg.normal(0, 9, len(d)), 0, 100
    )
    tier = np.where(
        health >= 72,
        "Higher simulated investment",
        np.where(
            health >= 53,
            "Moderate simulated investment",
            "Lower simulated investment",
        ),
    ).astype(object)

    simulated = pd.DataFrame(
        {
            "record_id": d.record_id,
            "country": d.country,
            "year": d.year,
            "sim_health_index": np.round(health, 2),
            "sim_education_index": np.round(education, 2),
            "sim_trend_index": np.round(trend, 2),
            "sim_policy_group": tier,
            "synthetic_record": True,
        }
    )
    missing_ratios = [
        ("sim_health_index", 0.055),
        ("sim_education_index", 0.07),
        ("sim_trend_index", 0.045),
        ("sim_policy_group", 0.05),
    ]
    for col, ratio in missing_ratios:
        k = int(round(ratio * len(simulated)))
        ix = rg.choice(len(simulated), k, replace=False)
        simulated.loc[ix, col] = np.nan

    simulated.to_csv(SYN / "simulated_country_year_indicators.csv", index=False)
    return simulated


def main():
    for folder in (RAW, SYN, OUT):
        folder.mkdir(parents=True, exist_ok=True)

    raw = get_public_source()
    geographic_sources = {
        "Feed_A_Africa_Asia": raw.loc[
            raw.continent.isin(["Africa", "Asia"])
        ].copy(),
        "Feed_B_Other_Regions": raw.loc[
            ~raw.continent.isin(["Africa", "Asia"])
        ].copy(),
    }

    chunks = []
    for name, part in geographic_sources.items():
        part = part.rename(columns=str.lower)
        assert set(part.columns) == set(map(str.lower, raw.columns))
        part["year"] = part.year.astype("int64")

        for numeric in NUMERIC_COLUMNS:
            part[numeric] = pd.to_numeric(
                part[numeric], errors="raise"
            ).astype("float64")
        for col in TEXT_COLUMNS:
            part[col] = part[col].astype("string")

        part["data_feed"] = name
        part.to_csv(RAW / f"{name}.csv", index=False)
        chunks.append(part)

    common_cols = sorted(set.intersection(*(set(d.columns) for d in chunks)))
    assert all(set(t.columns) == set(common_cols) for t in chunks)
    static = pd.concat([c[common_cols] for c in chunks], ignore_index=True)
    static["record_id"] = (
        static.country.astype(str)
        .str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
        .str.strip("_")
        + "_"
        + static.year.astype(str)
    )
    assert static.record_id.is_unique and len(static) == 1704
    static.to_csv(OUT / "unified_static_master.csv", index=False)

    dynamic = generate_dynamic(static)
    integrated = static.merge(
        dynamic.drop(columns=["country", "year"]),
        on="record_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    assert integrated["_merge"].eq("both").all()
    integrated = integrated.drop(columns="_merge")
    integrated["target_high_life_expectancy"] = (
        integrated["life_expectancy"] >= 70
    ).astype(int)
    integrated.to_csv(OUT / "master_sparse.csv", index=False)

    print(
        "Real source:", len(raw),
        "| feed A:", len(chunks[0]),
        "| feed B:", len(chunks[1]),
    )
    print(
        "Merged:", integrated.shape,
        "| synthetic missing:", int(integrated.isna().sum().sum()),
    )


if __name__ == "__main__":
    main()
