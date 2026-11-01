import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from transforms import FeatureCleaner
from vector_store import collection, read_frame


def test_train_only_imputation_and_clipping():
    X = pd.DataFrame({'continent': ['Asia']*5, 'sim_policy_group': ['Unknown']*5,
                      'sim_health_index': [10, 20, 30, 40, 50],
                      'log_gdp': [1., 2., 3., 4., 5.], 'log_population': [2., 3., 4., 5., 6.]})
    cleaner = FeatureCleaner().fit(X)
    query = X.iloc[[0]].copy()
    query['sim_health_index'] = np.nan
    query['log_gdp'] = 1e9
    out = cleaner.transform(query)
    assert out.sim_health_index.iloc[0] == 30
    assert out.log_gdp.iloc[0] == 7
    extreme = query.copy(); extreme.sim_health_index = 1000000
    batch = cleaner.transform(pd.concat([query, extreme]))
    assert batch.sim_health_index.iloc[0] == 30
    assert cleaner.medians_['sim_health_index'] == 30


def test_splits_and_winner_selection():
    split = pd.read_csv(ROOT/'reports/data_splits.csv')
    assert split.groupby('country')['split'].nunique().eq(1).all()
    assert set(split['split']) == {'fit', 'validation', 'test'}
    val = pd.read_csv(ROOT/'reports/validation_benchmark.csv')
    result = json.loads((ROOT/'reports/model_results.json').read_text())
    assert result['winning_regression'] == val[val.task.eq('regression')].sort_values('score').iloc[0].algorithm
    assert result['winning_classification'] == val[val.task.eq('classification')].sort_values('score', ascending=False).iloc[0].algorithm
    assert split[split['split'].eq('test')].country.nunique() == result['test_countries']


def test_live_vector_roundtrip():
    frame = read_frame()
    data = pd.read_csv(ROOT/'data/processed/master_with_clusters.csv')
    metrics = json.loads((ROOT/'reports/model_results.json').read_text())
    pcs = [c for c in frame if c.startswith('pc_')]
    assert len(pcs) == metrics['pca_components']
    assert len(frame) == len(data) and set(frame.record_id) == set(data.record_id)
    query = frame[pcs].iloc[0].tolist()
    result = collection().query(query_embeddings=[query], n_results=1)
    assert result['ids'][0][0] == frame.record_id.iloc[0]


def test_model_versions_and_all_predictions():
    from importlib import import_module
    model_mod = import_module('03_model_stage2')
    meta = json.loads((ROOT/'reports/model_results.json').read_text())
    assert sklearn.__version__ == meta['sklearn_version']
    X = model_mod.features(pd.read_csv(ROOT/'data/processed/master_sparse.csv').iloc[:3])
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        for name, digest in meta['artifact_sha256'].items():
            path = ROOT/'models'/name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            obj = joblib.load(path)
            if isinstance(obj, dict):
                Z = obj['preprocessor'].transform(X)
                Z = obj['pca'].transform(Z) if obj['representation'] == 'PCA features' else Z
                result = obj['model'].predict(Z)
            else:
                result = obj.predict(X)
            assert len(result) == 3 and np.isfinite(result).all()


def test_crawl_provenance():
    audit = json.loads((ROOT/'reports/crawl_audit.json').read_text())
    raw = ROOT/'data/raw/worldbank_life_expectancy.json'
    df = pd.read_csv(ROOT/'data/raw/worldbank_scrapy_records.csv')
    assert audit['mode'] == 'live_http' and audit['tool'] == 'Scrapy'
    assert len(df) == audit['records'] >= 300 and len(df.columns) >= 8
    assert audit['sha256'] == hashlib.sha256(raw.read_bytes()).hexdigest()


def test_ui_inputs_and_vector_search():
    spec = importlib.util.spec_from_file_location('atlas_app', ROOT/'ui/app.py')
    app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
    args = ['Asia', 2002, 25000000, 5000, 70, 65, 70, 'Moderate simulated investment']
    text, rows = app.predict(*args)
    assert 'Tuổi thọ dự đoán' in text and len(rows) == 5
    assert rows.distance_l2.is_monotonic_increasing
    for index, invalid in [(2, 0), (3, np.nan), (4, 200), (1, 2026), (0, 'Invalid')]:
        bad = args.copy(); bad[index] = invalid
        _, rows = app.predict(*bad)
        assert rows.empty
