# ORIGINAL REAL-WORLD DATASET URL:
# https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv
"""Stage 2: PCA/K-Means and country-held-out supervised model comparisons."""

import json
import sqlite3
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
CH = ROOT / "charts"
MOD = ROOT / "models"
REP = ROOT / "reports"
MOD.mkdir(exist_ok=True)

SEED = 42
NUM = [
    "year",
    "log_population",
    "log_gdp",
    "sim_health_index",
    "sim_education_index",
    "sim_trend_index",
]
CAT = ["continent", "sim_policy_group"]
FEATS = NUM + CAT


def features(df):
    """Create the original model features, excluding life expectancy targets."""
    data = df.copy()
    data["log_population"] = np.log1p(data["population"].clip(lower=0))
    data["log_gdp"] = np.log1p(data["gdp_per_capita"].clip(lower=0))
    return data[FEATS].copy()


def preprocess():
    """Build the same numerical scaling and one-hot encoding pipeline."""
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUM),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def fig(name):
    """Export one chart with the original rendering parameters."""
    plt.tight_layout()
    plt.savefig(CH / name, bbox_inches="tight", dpi=190, facecolor="white")
    plt.close()


def clustering(df):
    """Run PCA and select the highest-silhouette K-Means configuration."""
    X = features(df)
    prep = preprocess()
    Z = prep.fit_transform(X)

    # Retain the minimum number of components explaining at least 90% variance.
    pca_full = PCA().fit(Z)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    ncomp = int(np.searchsorted(cum, 0.90) + 1)
    assert 0.85 <= cum[ncomp - 1] <= 0.95, (
        f"Chosen explained variance outside 85-95%: {cum[ncomp - 1]:.3f}"
    )
    pca = PCA(n_components=ncomp, random_state=SEED)
    reduced = pca.fit_transform(Z)

    variance = pd.DataFrame(
        {
            "component": np.arange(1, len(cum) + 1),
            "cumulative_explained_variance": cum,
        }
    )
    variance.to_csv(REP / "pca_variance.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(
        variance.component,
        variance.cumulative_explained_variance,
        marker="o",
        color="#0b7f85",
    )
    plt.axhline(0.90, color="#e5864e", ls="--", label="Target: 90% variance")
    plt.axvline(
        ncomp,
        color="#6245a4",
        ls="--",
        label=f"Selected {ncomp} PCs = {cum[ncomp - 1] * 100:.1f}%",
    )
    plt.title("PCA – Cumulative explained variance")
    plt.xlabel("Principal components")
    plt.ylabel("Cumulative explained variance")
    plt.legend()
    fig("11_pca_cumulative_variance.png")

    # Preserve K order, representation order, and KMeans random state.
    compared = []
    models = {}
    for k in range(2, 9):
        for name, matrix in [("Original features", Z), ("PCA features", reduced)]:
            model = KMeans(n_clusters=k, n_init=15, random_state=SEED)
            labels = model.fit_predict(matrix)
            sil = silhouette_score(
                matrix,
                labels,
                random_state=SEED,
                sample_size=min(len(df), 1000),
            )
            compared.append(
                {
                    "representation": name,
                    "k": k,
                    "wcss": float(model.inertia_),
                    "silhouette": float(sil),
                }
            )
            models[(name, k)] = model

    cv = pd.DataFrame(compared)
    cv.to_csv(REP / "kmeans_k_benchmark.csv", index=False)
    selection = cv.sort_values(
        ["silhouette", "k"], ascending=[False, True]
    ).iloc[0]
    representation = str(selection.representation)
    chosen_k = int(selection.k)
    best = models[(representation, chosen_k)]
    cluster_data = reduced if representation == "PCA features" else Z
    labels = best.predict(cluster_data)
    df["Cluster_Label"] = labels.astype(int)

    # Save labels into SQLite and export the original country-year rows.
    with sqlite3.connect(ROOT / "databases/gapminder_project.sqlite") as conn:
        df.to_sql("master_with_clusters", conn, if_exists="replace", index=False)
        audit = pd.read_sql(
            "SELECT COUNT(*) n, COUNT(DISTINCT record_id) distinct_ids "
            "FROM master_with_clusters",
            conn,
        ).iloc[0]
        assert int(audit.n) == int(audit.distinct_ids) == len(df)

    df.to_csv(DATA / "master_with_clusters.csv", index=False)
    joblib.dump(
        {
            "preprocessor": prep,
            "pca": pca,
            "model": best,
            "representation": representation,
            "feature_names": FEATS,
        },
        MOD / "clustering_pipeline.joblib",
    )

    _, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, group in cv.groupby("representation"):
        axs[0].plot(group.k, group.wcss, marker="o", label=name)
        axs[1].plot(group.k, group.silhouette, marker="o", label=name)
    axs[0].set(
        title="Elbow method – WCSS",
        xlabel="K (clusters)",
        ylabel="Within-cluster sum of squares",
    )
    axs[1].set(
        title="Silhouette comparison",
        xlabel="K (clusters)",
        ylabel="Silhouette score",
    )
    for ax in axs:
        ax.legend()
    fig("12_kmeans_elbow_silhouette.png")

    visual = PCA(n_components=2, random_state=SEED).fit_transform(Z)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x=visual[:, 0],
        y=visual[:, 1],
        hue=labels,
        palette="viridis",
        alpha=0.7,
        s=24,
    )
    plt.title(f"K-Means clusters (K={chosen_k}; {representation}) projected to 2D")
    plt.xlabel("PC 1 (for visualization)")
    plt.ylabel("PC 2 (for visualization)")
    plt.legend(title="Cluster label")
    fig("13_kmeans_2d_clusters.png")

    cluster_summary = df.groupby("Cluster_Label").agg(
        n=("record_id", "size"),
        life_expectancy_mean=("life_expectancy", "mean"),
        median_gdp=("gdp_per_capita", "median"),
        mean_health_sim=("sim_health_index", "mean"),
        share_high_life=("target_high_life_expectancy", "mean"),
    ).round(3)
    cluster_summary.to_csv(REP / "cluster_summary.csv")

    return df, {
        "pca_components": ncomp,
        "explained_variance": float(cum[ncomp - 1]),
        "chosen_k": chosen_k,
        "cluster_representation": representation,
        "winning_silhouette": float(selection.silhouette),
        "original_same_k_silhouette": float(
            cv[(cv.representation == "Original features") & (cv.k == chosen_k)]
            .silhouette.iloc[0]
        ),
        "pca_same_k_silhouette": float(
            cv[(cv.representation == "PCA features") & (cv.k == chosen_k)]
            .silhouette.iloc[0]
        ),
    }


def supervised(df):
    """Train and evaluate regressors/classifiers on unseen countries."""
    # Split on unique countries, not repeated rows from the same country.
    countries = df[["country", "continent"]].drop_duplicates()
    train_c, test_c = train_test_split(
        countries.country,
        test_size=0.2,
        random_state=SEED,
        stratify=countries.continent,
    )
    istrain = df.country.isin(train_c)
    iste = df.country.isin(test_c)
    assert not set(train_c).intersection(test_c)

    xtr, xts = features(df.loc[istrain]), features(df.loc[iste])
    rytr = df.loc[istrain, "life_expectancy"]
    ryts = df.loc[iste, "life_expectancy"]
    cytr = df.loc[istrain, "target_high_life_expectancy"]
    cyts = df.loc[iste, "target_high_life_expectancy"]

    test_df = df.loc[iste, ["record_id", "country", "year"]].copy()
    test_df.to_csv(REP / "held_out_test_records.csv", index=False)

    ratio = cytr.value_counts(normalize=True)
    imbalance = ratio.max() / ratio.min()
    weights = "balanced" if imbalance > 1.5 else None

    regs = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(
            max_depth=6, min_samples_leaf=12, random_state=SEED
        ),
    }
    clfs = {
        "LogisticRegression": LogisticRegression(
            max_iter=2200, class_weight=weights, random_state=SEED
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=16,
            class_weight=weights,
            random_state=SEED,
        ),
    }

    regscore, clfscore = [], []
    rp, cp = {}, {}
    trained_reg, trained_clf = {}, {}

    for name, model in regs.items():
        pipe = Pipeline([("prepare", preprocess()), ("model", model)]).fit(
            xtr, rytr
        )
        pred = pipe.predict(xts)
        regscore.append(
            {
                "algorithm": name,
                "MSE": mean_squared_error(ryts, pred),
                "RMSE": np.sqrt(mean_squared_error(ryts, pred)),
                "R2": r2_score(ryts, pred),
            }
        )
        rp[name] = pred
        trained_reg[name] = pipe

    for name, model in clfs.items():
        pipe = Pipeline([("prepare", preprocess()), ("model", model)]).fit(
            xtr, cytr
        )
        pred = pipe.predict(xts)
        clfscore.append(
            {
                "algorithm": name,
                "Accuracy": accuracy_score(cyts, pred),
                "Precision_macro": precision_score(
                    cyts, pred, average="macro", zero_division=0
                ),
                "Recall_macro": recall_score(
                    cyts, pred, average="macro", zero_division=0
                ),
                "F1_macro": f1_score(
                    cyts, pred, average="macro", zero_division=0
                ),
                "F1_positive": f1_score(
                    cyts, pred, pos_label=1, zero_division=0
                ),
            }
        )
        cp[name] = pred
        trained_clf[name] = pipe
        (REP / f"classification_report_{name}.txt").write_text(
            classification_report(cyts, pred, zero_division=0),
            encoding="utf8",
        )

    reg = pd.DataFrame(regscore).sort_values("RMSE")
    clf = pd.DataFrame(clfscore).sort_values("F1_macro", ascending=False)
    reg.to_csv(REP / "regression_benchmark.csv", index=False)
    clf.to_csv(REP / "classification_benchmark.csv", index=False)

    regwin = str(reg.iloc[0].algorithm)
    clfwin = str(clf.iloc[0].algorithm)
    joblib.dump(trained_reg[regwin], MOD / "regression_winner.joblib")
    joblib.dump(trained_clf[clfwin], MOD / "classification_winner.joblib")

    test_pred = test_df.copy()
    test_pred["actual_life_expectancy"] = ryts.to_numpy()
    test_pred["pred_life_expectancy"] = rp[regwin]
    test_pred["actual_class"] = cyts.to_numpy()
    test_pred["pred_class"] = cp[clfwin]
    test_pred.to_csv(REP / "test_set_predictions.csv", index=False)

    # Actual versus predicted regression values.
    plt.figure(figsize=(8, 5.5))
    plt.scatter(ryts, rp[regwin], s=23, alpha=0.65, c="#0b7f85")
    lim = [min(ryts.min(), min(rp[regwin])) - 2, max(ryts.max(), max(rp[regwin])) + 2]
    plt.plot(lim, lim, linestyle="--", color="#d34e43", label="Perfect prediction")
    plt.title(f"Actual vs predicted – {regwin}")
    plt.xlabel("Actual life expectancy (years)")
    plt.ylabel("Predicted life expectancy (years)")
    plt.legend()
    fig("14_regression_actual_vs_predicted.png")

    plt.figure(figsize=(8, 4.7))
    sns.scatterplot(
        x=rp[regwin],
        y=ryts.to_numpy() - rp[regwin],
        alpha=0.55,
        color="#6245a4",
    )
    plt.axhline(0, color="#d34e43", ls="--")
    plt.xlabel("Predicted life expectancy (years)")
    plt.ylabel("Residual (actual - predicted)")
    plt.title(f"Residual diagnostics – {regwin}")
    fig("15_regression_residual_plot.png")

    # Use the same explicit 0/1 ordering in the confusion matrix.
    cm = confusion_matrix(cyts, cp[clfwin], labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["True<70", "True>=70"],
        columns=["Pred<70", "Pred>=70"],
    ).to_csv(REP / "confusion_matrix.csv")
    plt.figure(figsize=(6.6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=["Below 70", "70 or above"],
        yticklabels=["Below 70", "70 or above"],
    )
    plt.title(f"Confusion matrix – {clfwin}")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    fig("16_confusion_matrix.png")

    return {
        "train_rows": len(xtr),
        "test_rows": len(xts),
        "train_countries": len(train_c),
        "test_countries": len(test_c),
        "class_train_distribution": {
            str(k): round(float(v), 4) for k, v in ratio.items()
        },
        "class_weight": str(weights),
        "winning_regression": regwin,
        "winning_classification": clfwin,
        "best_regression": {
            k: round(float(v), 5)
            for k, v in reg.iloc[0].to_dict().items()
            if k != "algorithm"
        },
        "best_classification": {
            k: round(float(v), 5)
            for k, v in clf.iloc[0].to_dict().items()
            if k != "algorithm"
        },
    }


def main():
    df = pd.read_csv(DATA / "master_clean.csv")
    df, uns = clustering(df)
    sup = supervised(df)
    result = {**uns, **sup}
    result_text = json.dumps(result, indent=2, ensure_ascii=False)
    (REP / "model_results.json").write_text(result_text, encoding="utf8")
    print(result_text)


if __name__ == "__main__":
    main()
