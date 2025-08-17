# feature_engineering_module.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

_REQUIRED_INPUT_COLS = [
    "symbol", "time", "open", "high", "low", "close", "volumefrom", "volumeto"
]

class CryptoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Minimal, stable feature engineering for OHLCV + symbol + time.
    - Uses row's timestamp for time features (no datetime.now()).
    - NEVER fabricates return_1/log_return at inference.
      If they exist in input, we pass them through; else we skip them.
    - Safe handling for unseen symbols.
    """

    def __init__(self) -> None:
        self.symbol_encoder: LabelEncoder | None = None
        self.fitted_: bool = False
        self.feature_names_out_: list[str] | None = None
        self._allow_optional_returns = ["return_1", "log_return"]  # pass-through if present

    # ---------- helpers ----------
    def _safe_symbol_encode(self, series: pd.Series) -> np.ndarray:
        # unseen -> map to first known class (or 'UNK' bootstrap on fit)
        assert self.symbol_encoder is not None
        known = set(self.symbol_encoder.classes_)
        syms = series.astype(str)
        safe_syms = syms.where(syms.isin(known), next(iter(known)))
        return self.symbol_encoder.transform(safe_syms)

    def _build_base_features(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()

        # Derived price/shape features
        hl_range = (Xc["high"] - Xc["low"]).replace(0, np.nan)
        candle_body = (Xc["close"] - Xc["open"])
        upper_shadow = (Xc["high"] - Xc[["close", "open"]].max(axis=1))
        lower_shadow = (Xc[["close", "open"]].min(axis=1) - Xc["low"])
        body_to_range = candle_body / hl_range

        close_open_ratio = (Xc["close"] / Xc["open"]).replace([np.inf, -np.inf], np.nan)
        high_low_ratio  = (Xc["high"] / Xc["low"]).replace([np.inf, -np.inf], np.nan)
        typical_price   = (Xc["high"] + Xc["low"] + Xc["close"]) / 3.0
        hlc_avg         = (Xc["high"] + Xc["low"] + Xc["close"]) / 3.0

        volume_ratio        = (Xc["volumefrom"] / Xc["volumeto"]).replace([np.inf, -np.inf], np.nan)
        volume_price_ratio  = ((Xc["volumefrom"] + Xc["volumeto"]) / (Xc["close"].replace(0, np.nan))).replace([np.inf, -np.inf], np.nan)
        log_price           = np.log(Xc["close"].replace(0, np.nan))
        log_volume          = np.log((Xc["volumefrom"] + 1.0))
        price_volume        = Xc["close"] * Xc["volumefrom"]
        range_volume        = (Xc["high"] - Xc["low"]) * Xc["volumefrom"]

        # Time features from row timestamp (not now())
        if "time" in Xc.columns:
            t = pd.to_datetime(Xc["time"], errors="coerce")
            hour    = t.dt.hour
            day     = t.dt.day
            month   = t.dt.month
            quarter = t.dt.quarter
            weekday = t.dt.weekday
        else:
            # fallback zeros (shouldn't happen in our pipeline)
            hour = day = month = quarter = weekday = pd.Series(0, index=Xc.index)

        # Assemble
        feats = pd.DataFrame({
            "high": Xc["high"].astype(float),
            "low": Xc["low"].astype(float),
            "open": Xc["open"].astype(float),
            "volumefrom": Xc["volumefrom"].astype(float),
            "volumeto": Xc["volumeto"].astype(float),
            "close": Xc["close"].astype(float),

            "hl_range": hl_range,
            "candle_body": candle_body,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            "body_to_range": body_to_range,

            "close_open_ratio": close_open_ratio,
            "high_low_ratio": high_low_ratio,
            "typical_price": typical_price,
            "hlc_avg": hlc_avg,

            "hour": hour,
            "day": day,
            "month": month,
            "quarter": quarter,
            "weekday": weekday,

            "volume_ratio": volume_ratio,
            "volume_price_ratio": volume_price_ratio,
            "log_price": log_price,
            "log_volume": log_volume,
            "price_volume": price_volume,
            "range_volume": range_volume,
        }, index=Xc.index)

        # Optional pass-through returns if present
        for col in self._allow_optional_returns:
            if col in Xc.columns:
                feats[col] = pd.to_numeric(Xc[col], errors="coerce")

        # Symbol encoding
        feats["symbol_encoded"] = self._safe_symbol_encode(Xc["symbol"].astype(str))

        # Clean NaNs/Infs
        feats = feats.replace([np.inf, -np.inf], np.nan)
        feats = feats.fillna(0.0)

        return feats

    # ---------- sklearn API ----------
    def fit(self, X: pd.DataFrame, y=None):
        # validate required columns
        missing = [c for c in _REQUIRED_INPUT_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"CryptoFeatureEngineer.fit: missing required columns: {missing}")

        # Fit symbol encoder
        self.symbol_encoder = LabelEncoder()
        syms = X["symbol"].astype(str)
        # guarantee at least one class
        if syms.nunique() == 0:
            syms = syms.fillna("UNK")
        self.symbol_encoder.fit(syms)

        # Build once to lock column order
        feats = self._build_base_features(X)
        self.feature_names_out_ = list(feats.columns)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("CryptoFeatureEngineer.transform called before fit().")
        feats = self._build_base_features(X)

        # ensure stable column order (same as in fit)
        feats = feats.reindex(columns=self.feature_names_out_, fill_value=0.0)
        return feats

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not self.fitted_:
            return np.array([])
        return np.array(self.feature_names_out_)
