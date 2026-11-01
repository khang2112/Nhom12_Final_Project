# ORIGINAL REAL-WORLD DATASET URL:
# https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv
"""Persist country-year PCA embeddings in ChromaDB and verify retrieval.

Install project requirements, then run after src/03_model_stage2.py.
If ChromaDB is missing, record the failed status rather than fabricate ingestion.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from vector_store import client as db_client, read_frame


ROOT = Path(__file__).resolve().parents[1]


def main():
    """Upsert PCA embeddings and confirm a self-neighbor lookup succeeds."""
    try:
        import chromadb
    except ImportError as error:
        message = (
            "ChromaDB CHƯA được nạp: môi trường hiện tại chưa cài chromadb. "
            "Chạy pip install -r requirements.txt sau đó python src/04_chromadb_vector.py."
        )
        (ROOT / "reports/chroma_status.json").write_text(
            json.dumps({"ingested": False, "reason": message}, ensure_ascii=False, indent=2),
            encoding="utf8",
        )
        raise RuntimeError(message) from error

    import sys
    from importlib import import_module

    sys.path.insert(0, str(ROOT / "src"))
    model_stage2 = import_module("03_model_stage2")

    df = pd.read_csv(ROOT / "data/processed/master_with_clusters.csv")
    artifacts = joblib.load(ROOT / "models/clustering_pipeline.joblib")

    # Use the saved preprocessors and PCA rather than retraining them.
    features = model_stage2.features(df)
    original = artifacts["preprocessor"].transform(features)
    matrix = artifacts["pca"].transform(original)
    vectors = np.asarray(matrix, dtype="float32")

    client = db_client()
    collection = client.get_or_create_collection(
        "country_year_embeddings", metadata={"hnsw:space": "l2"}, embedding_function=None
    )

    documents = []
    metadata = []
    for record in df.itertuples(index=False):
        documents.append(
            f"{record.country}; {record.year}; "
            f"GDP per capita {record.gdp_per_capita:.2f}; "
            f"life expectancy {record.life_expectancy:.2f}; "
            f"group {record.Cluster_Label}"
        )
        metadata.append(
            {
                "country": str(record.country),
                "year": int(record.year),
                "continent": str(record.continent),
                "cluster": int(record.Cluster_Label),
            }
        )

    ids = df.record_id.astype(str).tolist()
    for start in range(0, len(ids), 200):
        stop = start + 200
        collection.upsert(
            ids=ids[start:stop],
            embeddings=vectors[start:stop].tolist(),
            metadatas=metadata[start:stop],
            documents=documents[start:stop],
        )

    # Remove stale IDs if the source snapshot changes.
    old_ids = set(collection.get(include=[])['ids']) - set(ids)
    if old_ids:
        collection.delete(ids=list(old_ids))
    assert collection.count() == len(df)
    reloaded = read_frame(collection)
    pc_cols = [f'pc_{i+1}' for i in range(vectors.shape[1])]
    aligned = reloaded.set_index('record_id').loc[ids, pc_cols].to_numpy()
    np.testing.assert_allclose(aligned, vectors, atol=1e-6)
    reloaded.to_csv(ROOT/'reports/chroma_roundtrip.csv', index=False)
    nearest = collection.query(query_embeddings=[vectors[0].tolist()], n_results=3)
    assert nearest["ids"] and ids[0] in nearest["ids"][0]

    status = {
        "ingested": True,
        "backend": type(client).__name__,
        "roundtrip_dataframe": True,
        "collection": collection.name,
        "count": collection.count(),
        "dimensions": vectors.shape[1],
        "probe_record": ids[0],
        "nearest_ids": nearest["ids"][0],
    }
    (ROOT / "reports/chroma_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
