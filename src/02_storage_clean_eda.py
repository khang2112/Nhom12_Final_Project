
# ORIGINAL REAL-WORLD DATASET URL:
# https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv
"""Stage 1: SQLite storage, missing-value handling, validation, and EDA."""

import json
import sqlite3
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
CH = ROOT / "charts"
DB = ROOT / "databases"
REP = ROOT / "reports"
DB_PATH = DB / "gapminder_project.sqlite"

for folder in (CH, DB, REP):
    folder.mkdir(parents=True, exist_ok=True)

# Plot configuration: keep the original colors and rendering settings.
PALETTE = ["#0b7f85", "#10ac9c", "#ff9e50", "#6245a4", "#bd6488"]

sns.set_theme(style="whitegrid", context="notebook", palette=PALETTE)
plt.rcParams.update(
    {
        "figure.dpi": 145,
        "savefig.dpi": 190,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
    }
)

SIMULATED_NUMERIC_COLUMNS = [
    "sim_health_index",
    "sim_education_index",
    "sim_trend_index",
]

NUMERIC_COLUMNS = [
    "life_expectancy",
    "gdp_per_capita",
    "population",
    *SIMULATED_NUMERIC_COLUMNS,
]

CORRELATION_COLUMNS = [
    "year",
    "life_expectancy",
    "gdp_per_capita",
    "population",
    *SIMULATED_NUMERIC_COLUMNS,
]


def save(name):
    """Save the current chart using the project's original export settings."""
    plt.tight_layout()
    plt.savefig(CH / name, bbox_inches="tight", facecolor="white")
    plt.close()


def missingness_table(df):
    """Summarize missing values without changing the DataFrame."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": (100 * df.isna().mean()).round(3).values,
        }
    )


def prepare():
    """Round-trip data through SQLite, impute missing values, and export audits."""
    raw = pd.read_csv(DATA / "master_sparse.csv")

    # SQLite round-trip for the original, uncleaned dataset.
    with sqlite3.connect(DB_PATH) as conn:
        raw.to_sql("master_sparse", conn, if_exists="replace", index=False)
        from_db = pd.read_sql("SELECT * FROM master_sparse", conn)

    assert len(raw) == len(from_db) == 1704
    assert set(raw.columns) == set(from_db.columns)

    missingness_table(from_db).to_csv(
        REP / "missingness_before.csv", index=False
    )
    before = from_db.copy()

    # Sort chronologically within each country before time-series imputation.
    from_db = from_db.sort_values(["country", "year"]).reset_index(drop=True)

    # Forward/backward filling is restricted to each country.
    from_db["sim_trend_index"] = from_db.groupby("country")[
        "sim_trend_index"
    ].transform(lambda series: series.ffill().bfill())

    # Continental median first, then global median as a fallback.
    for col in ("sim_health_index", "sim_education_index"):
        from_db[col] = from_db[col].fillna(
            from_db.groupby("continent")[col].transform("median")
        )
        from_db[col] = from_db[col].fillna(from_db[col].median())

    from_db["sim_policy_group"] = from_db["sim_policy_group"].fillna(
        "Unknown"
    )
    from_db["sim_trend_index"] = from_db["sim_trend_index"].fillna(
        from_db["sim_trend_index"].median()
    )

    # Validate the cleaned data and the unchanged classification target.
    assert from_db.isna().sum().sum() == 0, "Missing data remained"
    assert from_db.record_id.is_unique
    assert from_db["life_expectancy"].between(0, 100).all()
    assert from_db["target_high_life_expectancy"].eq(
        (from_db.life_expectancy >= 70).astype(int)
    ).all()

    # Verify that SQLite preserves the cleaned values on reloading.
    with sqlite3.connect(DB_PATH) as conn:
        from_db.to_sql("master_clean", conn, if_exists="replace", index=False)
        reloaded = pd.read_sql("SELECT * FROM master_clean", conn)

    pd.testing.assert_frame_equal(
        from_db.reset_index(drop=True),
        reloaded.reset_index(drop=True),
        check_dtype=False,
    )

    from_db.to_csv(DATA / "master_clean.csv", index=False)
    missingness_table(from_db).to_csv(REP / "missingness_after.csv", index=False)

    # Track the effect of imputation on descriptive distributions.
    integrity = []
    for col in SIMULATED_NUMERIC_COLUMNS:
        left = before[col].dropna()
        right = from_db[col].dropna()
        integrity.append(
            {
                "column": col,
                "n_before": len(left),
                "mean_before": left.mean(),
                "mean_after": right.mean(),
                "median_before": left.median(),
                "median_after": right.median(),
                "std_before": left.std(),
                "std_after": right.std(),
                "mean_change_percent": 100
                * (right.mean() - left.mean())
                / max(abs(left.mean()), 1e-9),
            }
        )

    pd.DataFrame(integrity).round(4).to_csv(
        REP / "imputation_impact.csv", index=False
    )

    # Univariate statistics.
    stats = from_db[NUMERIC_COLUMNS].agg(
        ["mean", "median", "std", "skew", "kurt"]
    ).T
    stats.to_csv(REP / "univariate_statistics.csv")

    # IQR-based outlier audit (same thresholds as the original script).
    outliers = []
    for col in NUMERIC_COLUMNS:
        q1, q3 = from_db[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        count = int(
            (
                (from_db[col] < q1 - 1.5 * iqr)
                | (from_db[col] > q3 + 1.5 * iqr)
            ).sum()
        )
        outliers.append(
            {
                "feature": col,
                "iqr": iqr,
                "outliers": count,
                "outlier_rate_pct": 100 * count / len(from_db),
            }
        )

    pd.DataFrame(outliers).to_csv(REP / "iqr_outliers.csv", index=False)

    # Categorical frequencies and continent-by-policy cross-tabulation.
    for col in ("continent", "sim_policy_group", "target_high_life_expectancy"):
        counts = (
            from_db[col]
            .value_counts(dropna=False)
            .rename_axis(col)
            .reset_index(name="count")
        )
        counts["percent"] = 100 * counts["count"] / len(from_db)
        counts.to_csv(REP / f"categorical_{col}.csv", index=False)

    pd.crosstab(
        from_db["continent"],
        from_db["sim_policy_group"],
        normalize="index",
    ).to_csv(REP / "crosstab_continent_policy.csv")

    return before, from_db


def charts(before, df):
    """Generate the same 10 descriptive and exploratory charts."""

    # Chart 01: Missing values before imputation.
    counts = before.isna().sum()
    counts = counts[counts > 0].sort_values()
    plt.figure(figsize=(9, 4))
    ax = sns.barplot(x=counts.values, y=counts.index, color=PALETTE[0])
    ax.set(
        title="Missing entries before imputation",
        xlabel="Number of missing entries",
        ylabel="Field",
    )
    save("01_missingness.png")

    # Chart 02: Distribution before and after health-index imputation.
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(
        before.sim_health_index.dropna(),
        kde=True,
        ax=axs[0],
        color=PALETTE[3],
        bins=23,
    )
    axs[0].set(
        title="Before: synthetic health index",
        xlabel="Simulated index",
        ylabel="Observations",
    )
    sns.histplot(
        df.sim_health_index,
        kde=True,
        ax=axs[1],
        color=PALETTE[0],
        bins=23,
    )
    axs[1].set(
        title="After: continental median imputation",
        xlabel="Simulated index",
        ylabel="Observations",
    )
    save("02_imputation_distribution.png")

    # Chart 03: Real life-expectancy distribution.
    plt.figure(figsize=(9, 4.5))
    sns.histplot(
        data=df,
        x="life_expectancy",
        kde=True,
        bins=29,
        color=PALETTE[0],
    )
    plt.title("Distribution of life expectancy, 1952–2007")
    plt.xlabel("Life expectancy (years)")
    plt.ylabel("Country-year observations")
    save("03_life_expectancy_hist_kde.png")

    # Chart 04: Life-expectancy boxplot.
    plt.figure(figsize=(9, 4.5))
    sns.boxplot(data=df, x="life_expectancy", color=PALETTE[2])
    plt.title("Life expectancy and IQR outliers")
    plt.xlabel("Life expectancy (years)")
    save("04_life_expectancy_box_iqr.png")

    # Chart 05: Observation counts across continents.
    plt.figure(figsize=(9, 4.5))
    ax = sns.countplot(
        data=df,
        y="continent",
        order=df.continent.value_counts().index,
        color=PALETTE[0],
    )
    ax.set(
        title="Country-year coverage by continent",
        xlabel="Number of observations",
        ylabel="Continent",
    )
    save("05_continent_bar.png")

    # Chart 06: Spearman correlations (not Pearson).
    corr = df[CORRELATION_COLUMNS].corr(method="spearman")
    plt.figure(figsize=(9, 6.8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="BrBG",
        center=0,
        square=True,
        linewidths=0.45,
        cbar_kws={"label": "Spearman rho"},
    )
    plt.title("Multivariate Spearman correlation heatmap")
    save("06_correlation_heatmap.png")
    corr.to_csv(REP / "correlations_spearman.csv")

    # Chart 07: GDP per capita vs. life expectancy.
    plt.figure(figsize=(9, 5))
    ax = sns.scatterplot(
        data=df,
        x="gdp_per_capita",
        y="life_expectancy",
        hue="continent",
        alpha=0.5,
        s=25,
        palette="Set2",
    )
    ax.set_xscale("log")
    ax.set(
        title="Life expectancy vs. GDP per capita",
        xlabel="GDP per capita (log scale)",
        ylabel="Life expectancy (years)",
    )
    plt.legend(title="Continent", loc="lower right")
    save("07_gdp_lifeexp_scatter.png")

    # Chart 08: Life expectancy grouped by continent.
    plt.figure(figsize=(9, 5))
    ax = sns.boxplot(
        data=df,
        x="continent",
        y="life_expectancy",
        hue="continent",
        palette="Set2",
        legend=False,
    )
    ax.set(
        title="Life expectancy by continent",
        xlabel="Continent",
        ylabel="Life expectancy (years)",
    )
    save("08_continent_lifeexp_grouped_box.png")

    # Chart 09: Life-expectancy class percentages by continent.
    cross = pd.crosstab(
        df.continent,
        df.target_high_life_expectancy,
        normalize="index",
    ) * 100
    ax = cross.rename(
        columns={0: "Below 70 years", 1: "At least 70 years"}
    ).plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5),
        color=[PALETTE[2], PALETTE[0]],
    )
    ax.set(
        title="Life expectancy category mix by continent",
        xlabel="Continent",
        ylabel="Share of observations (%)",
    )
    ax.legend(
        title="Life expectancy class",
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )
    plt.xticks(rotation=0)
    save("09_continent_lifeexp_stacked_bar.png")

    # Chart 10: Average life expectancy over time by continent.
    trend = df.groupby(["year", "continent"], as_index=False).life_expectancy.mean()
    plt.figure(figsize=(9, 5))
    ax = sns.lineplot(
        data=trend,
        x="year",
        y="life_expectancy",
        hue="continent",
        marker="o",
        palette="Set2",
    )
    ax.set(
        title="Historical life expectancy trend by continent",
        xlabel="Year",
        ylabel="Average life expectancy (years)",
    )
    save("10_historical_trend.png")


def main():
    before, df = prepare()
    charts(before, df)

    summary = {
        "n_records": len(df),
        "countries": int(df.country.nunique()),
        "year_min": int(df.year.min()),
        "year_max": int(df.year.max()),
        "raw_missing_count": int(before.isna().sum().sum()),
        "clean_missing_count": int(df.isna().sum().sum()),
        "mean_life_expectancy": round(float(df.life_expectancy.mean()), 3),
        "high_life_expectancy_pct": round(
            100 * df.target_high_life_expectancy.mean(), 2
        ),
        "gdp_lifeexp_spearman": round(
            float(
                df[["gdp_per_capita", "life_expectancy"]]
                .corr("spearman")
                .iloc[0, 1]
            ),
            4,
        ),
    }

    summary_text = json.dumps(summary, indent=2)
    (REP / "eda_summary.json").write_text(summary_text, encoding="utf8")
    print(summary_text)


if __name__ == "__main__":
    main()
