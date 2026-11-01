"""One DB interface for local execution and Docker HTTP deployment."""
import os
from pathlib import Path
import chromadb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = 'country_year_embeddings'


def client():
    if os.getenv('CHROMA_HOST'):
        return chromadb.HttpClient(host=os.environ['CHROMA_HOST'],
                                   port=int(os.getenv('CHROMA_PORT', '8000')))
    return chromadb.PersistentClient(path=os.getenv('CHROMA_PATH', str(ROOT/'databases/chroma_db')))


def collection():
    return client().get_collection(COLLECTION, embedding_function=None)


def read_frame(store=None):
    """Read all IDs, metadata and embeddings back into a DataFrame."""
    store = store if store is not None else collection()
    result = store.get(include=['metadatas', 'embeddings'])
    rows = []
    for rid, meta, vec in zip(result['ids'], result['metadatas'], result['embeddings']):
        rows.append({'record_id': rid, **meta,
                     **{f'pc_{i+1}': float(v) for i, v in enumerate(vec)}})
    return pd.DataFrame(rows)


def nearest_frame(vector, n=5):
    result = collection().query(query_embeddings=[vector.tolist()], n_results=n,
                                include=['metadatas', 'distances'])
    return pd.DataFrame([{'record_id': rid, **meta, 'distance_l2': float(distance)}
                         for rid, meta, distance in zip(result['ids'][0], result['metadatas'][0], result['distances'][0])])
