# feature_engineering_module.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

_REQUIRED_INPUT_COLS = [
    "symbol", "time", "open", "high", "low", "close", "volumefrom", "volumeto"
]

_EPS = 1e-8


# ============================================================================
# TECHNICAL INDICATORS
# ----------------------------------------------------------------------------
# Pure pandas/numpy implementations - no black-box TA library.
#
# LEAKAGE CONTRACT: every function here computes the indicator value *as of the
# close of each bar*, i.e. the value at index t uses data up to and including
# bar t. That means the raw output is NOT safe to use as a feature for
# predicting bar t's own return. Callers MUST shift the result by >= 1 bar
# before using it as a predictor. Both call sites in this project
# (CryptoFeatureEngineer below, and model_training.train_and_save) do exactly
# that and name the resulting columns with a `_lag{k}` suffix.
# ============================================================================


def _wilder_rma(values: pd.Series, period: int) -> pd.Series:
    """
    Wilder's running moving average (RMA), the smoothing used by RSI/ATR/ADX.

    Wilder seeds the average with a SIMPLE mean of the first `period`
    observations and only then switches to the recursive form
        avg_t = (avg_{t-1} * (period - 1) + x_t) / period
    which is algebraically an EMA with alpha = 1/period.

    Seeding matters: feeding the raw series straight into
    `ewm(alpha=1/period, adjust=False)` seeds with the *first observation*
    instead of the SMA, which produces visibly different RSI values for
    several hundred bars. So we blank the warm-up region, plant the SMA seed
    at position `period`, and let ewm handle the recursion from there.

    `values` is expected to come from .diff(), i.e. position 0 is NaN and the
    first real observation sits at position 1.
    """
    v = values.astype(float).copy()
    n = len(v)
    if n <= period:
        return pd.Series(np.nan, index=values.index, dtype=float)

    seed = v.iloc[1:period + 1].mean()
    v.iloc[:period + 1] = np.nan
    v.iloc[period] = seed
    return v.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing.

    Returns values in [0, 100]. The first `period` values are NaN (warm-up).
    Edge cases: unbroken gains over the window -> RSI 100; unbroken losses -> 0.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)

    rs = avg_gain / (avg_loss + _EPS)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Unbroken gains -> RSI 100; unbroken losses -> RSI 0.
    rsi = rsi.mask((avg_loss <= 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain <= 0) & (avg_loss > 0), 0.0)
    # Perfectly flat window: RSI is 0/0, i.e. undefined. Return the neutral 50
    # rather than letting the formula collapse to 0, which would otherwise be
    # read downstream as "maximally oversold". This is not hypothetical here -
    # ~1.4% of bars in this dataset have a zero return, and illiquid stretches
    # can produce a flat 14-bar window.
    rsi = rsi.mask((avg_gain <= 0) & (avg_loss <= 0), 50.0)
    # Keep the warm-up region NaN regardless of what mask() did.
    rsi[avg_gain.isna() | avg_loss.isna()] = np.nan
    return rsi


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.

    Returns a DataFrame with columns:
      macd        - fast EMA minus slow EMA (price units)
      macd_signal - EMA of the MACD line   (price units)
      macd_hist   - macd minus macd_signal (price units)
      macd_norm, macd_signal_norm, macd_hist_norm
                  - the same three divided by close, so they are comparable
                    across assets whose price levels differ by 4+ orders of
                    magnitude (BTC ~1e4 vs DOGE ~1e-1). The normalised
                    versions are the ones that are actually useful as features
                    in a pooled multi-asset model; the raw ones are kept
                    because they are what "MACD" conventionally means.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    denom = close.abs() + _EPS
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
            "macd_norm": macd_line / denom,
            "macd_signal_norm": signal_line / denom,
            "macd_hist_norm": histogram / denom,
        },
        index=close.index,
    )


def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands: `period`-bar SMA +/- `num_std` standard deviations.

    Returns a DataFrame with columns:
      bb_middle    - the SMA (price units)
      bb_upper     - middle + num_std * rolling std (price units)
      bb_lower     - middle - num_std * rolling std (price units)
      bb_pct_b     - %B = (close - lower) / (upper - lower). Scale-free.
                     0 = on the lower band, 1 = on the upper band, and it can
                     go outside [0, 1] when price breaks out of the channel.
      bb_bandwidth - (upper - lower) / middle. Scale-free volatility measure.
      bb_upper_dist / bb_lower_dist
                   - distance from close to each band, expressed in standard
                     deviations. Scale-free, and directly comparable to the
                     bb_upper_lag*/bb_lower_lag* features that model_training
                     already used before these indicators were added.

    Uses the sample standard deviation (ddof=1), matching pandas' default and
    the usual charting-package convention.
    """
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()

    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower).replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pct_b": (close - lower) / (width + _EPS),
            "bb_bandwidth": (upper - lower) / (middle.abs() + _EPS),
            "bb_upper_dist": (close - upper) / (std + _EPS),
            "bb_lower_dist": (close - lower) / (std + _EPS),
        },
        index=close.index,
    )


def build_indicator_frame(
    close: pd.Series,
    rsi_period: int = 14,
    macd_params: tuple[int, int, int] = (12, 26, 9),
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    """
    Convenience wrapper: all three indicator families for one close series,
    returned as a single unshifted DataFrame.

    This is the single definition of the indicator set that both
    CryptoFeatureEngineer and model_training.train_and_save consume, so the
    feature semantics cannot drift between the two code paths.
    """
    fast, slow, signal = macd_params
    frames = [
        compute_rsi(close, period=rsi_period).rename(f"rsi_{rsi_period}"),
        compute_macd(close, fast=fast, slow=slow, signal=signal),
        compute_bollinger_bands(close, period=bb_period, num_std=bb_std),
    ]
    return pd.concat(frames, axis=1)


# Columns produced by build_indicator_frame(), in order. Used to generate
# stable lagged column names without having to run the computation first.
INDICATOR_COLUMNS = [
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "bb_middle", "bb_upper", "bb_lower",
    "bb_pct_b", "bb_bandwidth", "bb_upper_dist", "bb_lower_dist",
]


class CryptoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates lagged features to eliminate data leakage.
    Uses PAST data to predict FUTURE returns.
    """

    def __init__(
        self,
        lag_periods=1,
        create_momentum_features=True,
        create_technical_indicators=True,
    ) -> None:
        self.lag_periods = lag_periods
        self.create_momentum_features = create_momentum_features
        self.create_technical_indicators = create_technical_indicators
        self.feature_names_ = None
        self.symbol_encoder = None

    # ---------- helpers ----------
    def _safe_symbol_encode(self, series: pd.Series) -> np.ndarray:
        # unseen -> map to first known class (or 'UNK' bootstrap on fit)
        assert self.symbol_encoder is not None
        known = set(self.symbol_encoder.classes_)
        syms = series.astype(str)
        safe_syms = syms.where(syms.isin(known), next(iter(known)))
        return self.symbol_encoder.transform(safe_syms)

    def _create_lagged_features(self, df):
        """
        Create features using ONLY past data to avoid leakage.
        Uses period t-1, t-2, etc. to predict period t.
        """
        df_sorted = df.sort_values(['symbol', 'time']).copy()
        result_df = pd.DataFrame()

        for symbol, group in df_sorted.groupby('symbol'):
            group = group.sort_values('time').reset_index(drop=True)
            group_features = pd.DataFrame()

            # Basic lagged price features (PAST data only)
            for lag in range(1, self.lag_periods + 3):  # Use 1, 2, 3 periods back
                group_features[f'open_lag{lag}'] = group['open'].shift(lag)
                group_features[f'high_lag{lag}'] = group['high'].shift(lag)
                group_features[f'low_lag{lag}'] = group['low'].shift(lag)
                group_features[f'close_lag{lag}'] = group['close'].shift(lag)
                group_features[f'volumefrom_lag{lag}'] = group['volumefrom'].shift(lag)
                group_features[f'volumeto_lag{lag}'] = group['volumeto'].shift(lag)

            # Technical indicators using PAST data only
            if self.create_technical_indicators:
                for lag in range(1, self.lag_periods + 2):
                    # Price action features
                    high_lag = group['high'].shift(lag)
                    low_lag = group['low'].shift(lag)
                    open_lag = group['open'].shift(lag)
                    close_lag = group['close'].shift(lag)

                    group_features[f'hl_range_lag{lag}'] = high_lag - low_lag
                    group_features[f'candle_body_lag{lag}'] = abs(close_lag - open_lag)
                    group_features[f'upper_shadow_lag{lag}'] = high_lag - np.maximum(open_lag, close_lag)
                    group_features[f'lower_shadow_lag{lag}'] = np.minimum(open_lag, close_lag) - low_lag

                    # Ratios using past data
                    group_features[f'close_open_ratio_lag{lag}'] = close_lag / (open_lag + _EPS)
                    group_features[f'high_low_ratio_lag{lag}'] = high_lag / (low_lag + _EPS)

                    # Volume features
                    volumefrom_lag = group['volumefrom'].shift(lag)
                    volumeto_lag = group['volumeto'].shift(lag)
                    group_features[f'volume_ratio_lag{lag}'] = volumefrom_lag / (volumeto_lag + _EPS)
                    group_features[f'volume_price_ratio_lag{lag}'] = volumefrom_lag / (close_lag + _EPS)

                # --- RSI / MACD / Bollinger Bands ---
                # Computed once on the *unshifted* close series (so each bar's
                # indicator uses that bar's own close, as the definitions
                # require), then shifted by >= 1 so that the feature available
                # at time t only ever reflects bars t-1 and earlier.
                indicators = build_indicator_frame(group['close'])
                for lag in range(1, self.lag_periods + 2):
                    shifted = indicators.shift(lag)
                    for col in INDICATOR_COLUMNS:
                        group_features[f'{col}_lag{lag}'] = shifted[col]

            # Momentum features using PAST returns only
            if self.create_momentum_features and 'return_1' in group.columns:
                for lag in range(1, self.lag_periods + 4):  # More momentum lags
                    group_features[f'return_lag{lag}'] = group['return_1'].shift(lag)

                # Rolling momentum (past 3, 5 periods)
                group_features['return_ma3'] = group['return_1'].shift(1).rolling(3).mean()
                group_features['return_ma5'] = group['return_1'].shift(1).rolling(5).mean()
                group_features['return_std3'] = group['return_1'].shift(1).rolling(3).std()

            # Time-based features (these don't cause leakage)
            if 'time' in group.columns:
                time_series = pd.to_datetime(group['time'])
                group_features['hour'] = time_series.dt.hour
                group_features['day'] = time_series.dt.day
                group_features['month'] = time_series.dt.month
                group_features['quarter'] = time_series.dt.quarter
                group_features['weekday'] = time_series.dt.weekday

                # Cyclical encoding
                group_features['hour_sin'] = np.sin(2 * np.pi * group_features['hour'] / 24)
                group_features['hour_cos'] = np.cos(2 * np.pi * group_features['hour'] / 24)
                group_features['month_sin'] = np.sin(2 * np.pi * group_features['month'] / 12)
                group_features['month_cos'] = np.cos(2 * np.pi * group_features['month'] / 12)

            # Symbol encoding (doesn't cause temporal leakage)
            if self.symbol_encoder is not None:
                try:
                    group_features['symbol_encoded'] = self.symbol_encoder.transform([symbol] * len(group))
                except Exception:
                    group_features['symbol_encoded'] = 0  # Unknown symbol

            # Add index to maintain order
            group_features.index = group.index
            result_df = pd.concat([result_df, group_features])

        # Remove rows with too many NaN values (caused by lagging)
        # Keep only rows where at least 50% of features are not NaN
        threshold = len(result_df.columns) * 0.5
        result_df = result_df.dropna(thresh=threshold)

        # Fill remaining NaN with 0 (conservative approach)
        result_df = result_df.fillna(0)

        return result_df

    # ---------- sklearn API ----------
    def fit(self, X: pd.DataFrame, y=None):
        # validate required columns
        missing = [c for c in _REQUIRED_INPUT_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"CryptoFeatureEngineer.fit: missing required columns: {missing}")

        print(f"Fitting CryptoFeatureEngineer with {self.lag_periods} lag periods...")

        # Fit symbol encoder BEFORE building the sample features, so that the
        # sample already contains the symbol_encoded column and the stored
        # feature_names_ matches what transform() will produce.
        self.symbol_encoder = LabelEncoder()
        if 'symbol' in X.columns:
            self.symbol_encoder.fit(X['symbol'].astype(str))

        # Create a sample to determine feature names. Needs to be long enough
        # to clear the 20-bar Bollinger / 26-bar MACD warm-up, otherwise the
        # dropna(thresh=...) inside _create_lagged_features can empty it out.
        sample = X.head(500) if len(X) > 500 else X.copy()
        features_df = self._create_lagged_features(sample)

        # Store feature names (excluding target if present)
        self.feature_names_ = [col for col in features_df.columns if col != 'return_1']

        print(f"Feature engineering fitted. Will create {len(self.feature_names_)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_:
            raise RuntimeError("CryptoFeatureEngineer.transform called before fit().")
        features_df = self._create_lagged_features(X)

        # ensure stable column order (same as in fit)
        features_df = features_df.reindex(columns=self.feature_names_, fill_value=0)

        print(f"Lagged features created. Shape: {features_df.shape}")
        return features_df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not self.feature_names_:
            return np.array([])
        return np.array(self.feature_names_)
