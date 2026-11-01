"""Train-only imputation and IQR clipping; serializable sklearn transformers."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FeatureCleaner(TransformerMixin, BaseEstimator):
    """Learn statistics only in fit. Never fill from other test records."""
    def fit(self, X, y=None):
        data = X.copy()
        self.numeric_ = data.select_dtypes(include=np.number).columns.tolist()
        self.medians_ = data[self.numeric_].median()
        self.group_medians_ = data.groupby('continent')[self.numeric_].median()
        self.bounds_ = {}
        for col in ('log_population', 'log_gdp'):
            q1, q3 = data[col].quantile([.25, .75])
            self.bounds_[col] = (q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))
        self.feature_names_in_ = np.array(data.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, 'medians_')
        data = X.copy()
        for col in self.numeric_:
            grouped = data['continent'].map(self.group_medians_[col])
            data[col] = data[col].fillna(grouped).fillna(self.medians_[col])
        for col in ('continent', 'sim_policy_group'):
            data[col] = data[col].fillna('Unknown')
        for col, (lo, hi) in self.bounds_.items():
            data[col] = data[col].clip(lo, hi)
        return data

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
