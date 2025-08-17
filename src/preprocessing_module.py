# preprocessing_module.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Iterable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

class CryptoPreprocessor(BaseEstimator, TransformerMixin):
    """
    - Standardize numeric features
    - SelectKBest for regression (f_regression)
    - Force-keep key drivers so symbol/volume ka effect model tak pahunchta rahe
    - Stable name-based selection to avoid order mismatch issues
    """

    def __init__(self, k: int = 20, must_keep: Iterable[str] | None = None):
        self.k = int(k)
        self.must_keep = list(must_keep) if must_keep is not None else [
            "symbol_encoded", "volumefrom", "volumeto", "volume_price_ratio"
        ]

        self.scaler: StandardScaler | None = None
        self.selector: SelectKBest | None = None

        self.feature_names_: List[str] | None = None
        self.selected_feature_names_: List[str] | None = None
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # store original order
        self.feature_names_ = list(X.columns)

        # scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # select K
        # Handle case: fewer than k features
        k_eff = min(self.k, X.shape[1])
        self.selector = SelectKBest(score_func=f_regression, k=k_eff)
        self.selector.fit(X_scaled, y)

        support = self.selector.get_support()  # boolean mask
        support = np.array(support, dtype=bool)

        # force-keep must_keep if present
        idx_map = {c: i for i, c in enumerate(self.feature_names_)}
        for c in self.must_keep:
            if c in idx_map:
                support[idx_map[c]] = True

        # finalize selected names
        self.selected_feature_names_ = [c for i, c in enumerate(self.feature_names_) if support[i]]
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("CryptoPreprocessor.transform called before fit().")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)

        # align and scale in the same order
        X = X.reindex(columns=self.feature_names_, fill_value=0.0)
        X_scaled = self.scaler.transform(X)

        # select by stored names (stable)
        idx_map = {c: i for i, c in enumerate(self.feature_names_)}
        sel_idx = [idx_map[c] for c in self.selected_feature_names_]
        X_selected = X_scaled[:, sel_idx]
        return X_selected

    def get_selected_feature_names(self) -> List[str]:
        return list(self.selected_feature_names_ or [])
