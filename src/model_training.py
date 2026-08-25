# model_training.py
from __future__ import annotations
import os
import io
import json
import contextlib
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from joblib import dump

from feature_engineering_module import (
    CryptoFeatureEngineer,
    build_indicator_frame,
    INDICATOR_COLUMNS,
)
from preprocessing_module import CryptoPreprocessor

EPS = 1e-8

# Lag suffixes that mark a column as safe (built only from past bars).
_SAFE_LAG_SUFFIXES = tuple(f"_lag{k}" for k in range(1, 8))

# Time-derived columns that are known in advance and therefore leakage-free.
_SAFE_TIME_COLS = [
    "symbol", "hour", "day", "month", "quarter", "weekday", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos",
]


# ---------------------------------------------------------------------------
# Time-aware splitting
# ---------------------------------------------------------------------------
def time_aware_split(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    time_col: str = "time",
    target_col: str = "target_returns",
    test_size: float = 0.2,
    gap: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-symbol chronological split with an embargo gap.

    Retained for backward compatibility and for the final holdout evaluation.
    Cross-validated model selection now goes through
    per_symbol_time_series_splits() instead.
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found for time-aware split.")

    parts_train, parts_test = [], []
    for sym, g in df.groupby(symbol_col):
        g = g.sort_values(time_col)
        n = len(g)
        if n < 50:
            continue
        split_idx = int(n * (1 - test_size))
        gap_adjusted = max(split_idx - gap, 0)
        parts_train.append(g.iloc[:gap_adjusted])
        parts_test.append(g.iloc[split_idx:])

    train = pd.concat(parts_train, axis=0) if parts_train else pd.DataFrame()
    test = pd.concat(parts_test, axis=0) if parts_test else pd.DataFrame()

    if not train.empty:
        train = train[train[target_col].notna()]
    if not test.empty:
        test = test[test[target_col].notna()]

    print("Time-aware split completed:")
    print(f"   Training set: {len(train)} rows")
    print(f"   Test set: {len(test)} rows")
    return train, test


def per_symbol_time_series_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    gap: int = 24,
    symbol_col: str = "symbol",
    time_col: str = "time",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window time-series CV that respects per-symbol chronology.

    sklearn's TimeSeriesSplit is applied to each symbol's own chronologically
    ordered rows, and fold k for every symbol is then concatenated into one
    global fold. That gives 5 folds where, *within each symbol*, every training
    row strictly precedes every test row - which a single global TimeSeriesSplit
    over the stacked frame would not guarantee, because the frame interleaves
    eight assets whose histories start at different dates.

    `gap` rows are embargoed between each train block and its test block so
    that the 20-bar rolling windows / 26-bar MACD warm-up of the first test row
    cannot overlap the last training row.

    Returns a list of (train_labels, test_labels) index-label arrays.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    ordered_labels: Dict[Any, np.ndarray] = {}
    per_symbol_folds: Dict[Any, list] = {}
    for sym, g in df.groupby(symbol_col, sort=True):
        labels = g.sort_values(time_col).index.to_numpy()
        if len(labels) < (n_splits + 1) * (gap + 2):
            print(f"   [skip] {sym}: only {len(labels)} rows, too few for {n_splits}-fold CV")
            continue
        ordered_labels[sym] = labels
        per_symbol_folds[sym] = list(tscv.split(labels))

    if not ordered_labels:
        raise ValueError("No symbol has enough rows for time-series cross-validation.")

    folds = []
    for k in range(n_splits):
        tr_parts, te_parts = [], []
        for sym, labels in ordered_labels.items():
            tr_pos, te_pos = per_symbol_folds[sym][k]
            tr_parts.append(labels[tr_pos])
            te_parts.append(labels[te_pos])
        folds.append((np.concatenate(tr_parts), np.concatenate(te_parts)))
    return folds


# ---------------------------------------------------------------------------
# Stationarity testing (statsmodels)
# ---------------------------------------------------------------------------
def run_adf_tests(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    price_col: str = "close",
    time_col: str = "time",
    max_obs: int = 20000,
    maxlag: int = 24,
) -> Dict[str, Any]:
    """
    Augmented Dickey-Fuller test on raw price vs. log returns, per symbol.

    This is the empirical justification for modelling returns rather than
    price. The ADF null hypothesis is "the series has a unit root", i.e. is
    non-stationary. We expect:
      - raw close  -> FAIL to reject H0 (p large)  -> non-stationary
      - log return -> REJECT H0        (p ~ 0)     -> stationary

    Fitting a model on a non-stationary target is what produced the original
    R2 = 0.9998 result: the model was simply carrying the price level forward.

    Each series is truncated to its most recent `max_obs` observations to keep
    the test tractable; ADF is O(n * maxlag) per candidate lag under AIC search.
    """
    print("\n" + "=" * 78)
    print("AUGMENTED DICKEY-FULLER STATIONARITY TEST (statsmodels)")
    print("=" * 78)
    print("H0: series has a unit root (non-stationary).  Reject when p < 0.05.\n")

    results: Dict[str, Any] = {"price": {}, "log_return": {}}

    header = f"{'Symbol':<8}{'Series':<13}{'ADF stat':>12}{'p-value':>12}{'1% crit':>11}  Verdict"
    print(header)
    print("-" * len(header))

    for sym, g in df.groupby(symbol_col, sort=True):
        g = g.sort_values(time_col)
        close = g[price_col].astype(float)
        close = close[close > 0]
        if len(close) < 100:
            continue

        log_ret = np.log(close / close.shift(1)).dropna()

        for label, series in (("price", close), ("log_return", log_ret)):
            s = series.iloc[-max_obs:].to_numpy()
            # ADF needs variation; a constant window would raise.
            if np.allclose(s, s[0]):
                continue
            try:
                stat, pvalue, usedlag, nobs, crit, _ = adfuller(
                    s, maxlag=maxlag, autolag="AIC"
                )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"{sym:<8}{label:<13}  ADF failed: {exc}")
                continue

            stationary = pvalue < 0.05
            results[label][sym] = {
                "adf_statistic": float(stat),
                "p_value": float(pvalue),
                "used_lag": int(usedlag),
                "n_obs": int(nobs),
                "critical_values": {k: float(v) for k, v in crit.items()},
                "stationary_at_5pct": bool(stationary),
            }
            verdict = "STATIONARY" if stationary else "non-stationary"
            print(
                f"{sym:<8}{label:<13}{stat:>12.4f}{pvalue:>12.4g}"
                f"{crit['1%']:>11.3f}  {verdict}"
            )

    n_price_stat = sum(v["stationary_at_5pct"] for v in results["price"].values())
    n_ret_stat = sum(v["stationary_at_5pct"] for v in results["log_return"].values())
    n_sym = len(results["log_return"])

    print("\nSummary:")
    print(f"   Raw close price stationary : {n_price_stat}/{len(results['price'])} symbols")
    print(f"   Log returns stationary     : {n_ret_stat}/{n_sym} symbols")

    if n_ret_stat > n_price_stat:
        conclusion = (
            "ADF confirms log returns are stationary while raw price is not. "
            "Modelling target is therefore the next-period LOG RETURN, not price."
        )
    else:
        conclusion = (
            "ADF did not show the expected stationarity gap; inspect the series "
            "before trusting the returns target."
        )
    print(f"   Conclusion: {conclusion}")
    results["conclusion"] = conclusion
    results["n_symbols"] = n_sym
    return results


# ---------------------------------------------------------------------------
# Interpretable statsmodels baselines
# ---------------------------------------------------------------------------
def fit_statsmodels_baselines(
    X_train: pd.DataFrame,
    y_train_reg: np.ndarray,
    y_train_clf: np.ndarray,
    X_test: pd.DataFrame,
    y_test_reg: np.ndarray,
    y_test_clf: np.ndarray,
    k_best: int,
    model_dir: str,
) -> Dict[str, Any]:
    """
    OLS (returns) and Logit (direction) baselines on the same preprocessed
    features the ML models see.

    Purpose: an interpretable benchmark with coefficients and p-values, so the
    tree ensembles can be judged against a transparent linear model rather than
    only against each other. If a 100-tree GBM cannot beat OLS on this data,
    that is worth knowing.

    Preprocessing is fit on TRAIN ONLY (same contract as the sklearn Pipeline),
    then applied to test.
    """
    print("\n" + "=" * 78)
    print("STATSMODELS INTERPRETABLE BASELINES (OLS / Logit)")
    print("=" * 78)

    out: Dict[str, Any] = {}

    pre = CryptoPreprocessor(k=k_best, task_type="regression")
    Xtr = pre.fit_transform(X_train, y_train_reg)
    Xte = pre.transform(X_test)
    names = list(pre.get_feature_names_out())

    Xtr_c = sm.add_constant(pd.DataFrame(Xtr, columns=names), has_constant="add")
    Xte_c = sm.add_constant(pd.DataFrame(Xte, columns=names), has_constant="add")

    # ---- OLS on log returns ----
    try:
        ols = sm.OLS(y_train_reg, Xtr_c).fit()
        pred = ols.predict(Xte_c).to_numpy()
        out["ols"] = {
            "r2": float(r2_score(y_test_reg, pred)),
            "mae": float(mean_absolute_error(y_test_reg, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test_reg, pred))),
            "aic": float(ols.aic),
            "n_significant_5pct": int((ols.pvalues < 0.05).sum()),
            "n_features": len(names),
        }
        summary_path = os.path.join(model_dir, "statsmodels_ols_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(str(ols.summary()))
        out["ols"]["summary_path"] = summary_path
        print(ols.summary())
        print(f"\n   OLS test R2={out['ols']['r2']:.6f}  MAE={out['ols']['mae']:.6f}")
        print(f"   Significant coefficients at 5%: "
              f"{out['ols']['n_significant_5pct']}/{len(ols.pvalues)}")
        print(f"   Full summary saved to {summary_path}")
    except Exception as exc:
        print(f"   OLS baseline failed: {exc}")
        out["ols"] = {"error": str(exc)}

    # ---- Logit on direction ----
    pre_clf = CryptoPreprocessor(k=k_best, task_type="classification")
    Xtr_c2 = sm.add_constant(
        pd.DataFrame(pre_clf.fit_transform(X_train, y_train_clf),
                     columns=list(pre_clf.get_feature_names_out())),
        has_constant="add",
    )
    Xte_c2 = sm.add_constant(
        pd.DataFrame(pre_clf.transform(X_test),
                     columns=list(pre_clf.get_feature_names_out())),
        has_constant="add",
    )
    try:
        logit = sm.Logit(y_train_clf, Xtr_c2).fit(disp=0, maxiter=200, method="lbfgs")
        proba = logit.predict(Xte_c2).to_numpy()
        pred_clf = (proba >= 0.5).astype(int)
        out["logit"] = {
            "accuracy": float(accuracy_score(y_test_clf, pred_clf)),
            "precision": float(precision_score(y_test_clf, pred_clf, zero_division=0)),
            "recall": float(recall_score(y_test_clf, pred_clf, zero_division=0)),
            "f1": float(f1_score(y_test_clf, pred_clf, zero_division=0)),
            "pseudo_r2": float(logit.prsquared),
            "n_significant_5pct": int((logit.pvalues < 0.05).sum()),
        }
        summary_path = os.path.join(model_dir, "statsmodels_logit_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(str(logit.summary()))
        out["logit"]["summary_path"] = summary_path
        print(logit.summary())
        print(f"\n   Logit test accuracy={out['logit']['accuracy']:.4f}  "
              f"pseudo-R2={out['logit']['pseudo_r2']:.5f}")
        print(f"   Significant coefficients at 5%: "
              f"{out['logit']['n_significant_5pct']}/{len(logit.pvalues)}")
        print(f"   Full summary saved to {summary_path}")
    except Exception as exc:
        print(f"   Logit baseline failed: {exc}")
        out["logit"] = {"error": str(exc)}

    return out


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def build_pipeline_no_leakage(
    model_type: str = "random_forest",
    task_type: str = "regression",
    k_best: int = 15,
    model_params: Dict[str, Any] | None = None,
) -> Pipeline:
    """
    Preprocessing + estimator. Feature engineering happens upstream (on the raw
    frame) so that lag construction can see each symbol's full history; only
    scaling and feature selection live inside the Pipeline, where they are fit
    on training folds alone.
    """
    if model_params is None:
        model_params = {}

    defaults = {
        "regression": {
            "random_forest": dict(n_estimators=100, max_depth=8, min_samples_split=10,
                                  min_samples_leaf=5, n_jobs=-1, random_state=42),
            "decision_tree": dict(max_depth=10, min_samples_split=10,
                                  min_samples_leaf=5, random_state=42),
            "xgboost": dict(n_estimators=100, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1),
            "lightgbm": dict(n_estimators=100, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=-1, verbose=-1),
        },
        "classification": {
            "random_forest": dict(n_estimators=100, max_depth=8, min_samples_split=10,
                                  min_samples_leaf=5, n_jobs=-1, random_state=42,
                                  class_weight="balanced"),
            "decision_tree": dict(max_depth=10, min_samples_split=10, min_samples_leaf=5,
                                  random_state=42, class_weight="balanced"),
            "xgboost": dict(n_estimators=100, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            n_jobs=-1, objective="binary:logistic"),
            "lightgbm": dict(n_estimators=100, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             n_jobs=-1, verbose=-1, objective="binary",
                             class_weight="balanced"),
        },
    }

    if task_type not in defaults:
        raise ValueError(f"Unsupported task type: {task_type}")
    if model_type not in defaults[task_type]:
        raise ValueError(f"Unsupported model type for {task_type}: {model_type}")

    params = {**defaults[task_type][model_type], **model_params}
    ctor = {
        ("regression", "random_forest"): RandomForestRegressor,
        ("regression", "decision_tree"): DecisionTreeRegressor,
        ("regression", "xgboost"): xgb.XGBRegressor,
        ("regression", "lightgbm"): lgb.LGBMRegressor,
        ("classification", "random_forest"): RandomForestClassifier,
        ("classification", "decision_tree"): DecisionTreeClassifier,
        ("classification", "xgboost"): xgb.XGBClassifier,
        ("classification", "lightgbm"): lgb.LGBMClassifier,
    }[(task_type, model_type)]

    return Pipeline(steps=[
        ("preprocessing", CryptoPreprocessor(k=k_best, task_type=task_type)),
        ("model", ctor(**params)),
    ])


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def _engineer_symbol_group(group: pd.DataFrame) -> pd.DataFrame:
    """
    Build every predictor column for ONE symbol's chronologically sorted bars.

    Shared by the training path (build_feature_frame) and the inference path
    (build_features_for_inference) so the two cannot drift apart - a mismatch
    between training-time and serving-time feature semantics is the single
    most common way a working model silently starts returning garbage.

    `group` must already be sorted by time with a clean RangeIndex, and must
    still carry the raw open/high/low/close/volumefrom/volumeto columns.
    """
    # 1. Lagged OHLCV
    for lag in range(1, 6):
        group[f"open_lag{lag}"] = group["open"].shift(lag)
        group[f"high_lag{lag}"] = group["high"].shift(lag)
        group[f"low_lag{lag}"] = group["low"].shift(lag)
        group[f"close_lag{lag}"] = group["close"].shift(lag)
        group[f"volumefrom_lag{lag}"] = group["volumefrom"].shift(lag)
        group[f"volumeto_lag{lag}"] = group["volumeto"].shift(lag)

    # 2. Rolling statistics. shift() FIRST, then rolling(), so the window
    #    never contains the current bar.
    for lag in (1, 2):
        close_lag = group["close"].shift(lag)
        high_lag = group["high"].shift(lag)
        low_lag = group["low"].shift(lag)
        vol_lag = group["volumefrom"].shift(lag)

        group[f"sma_5_lag{lag}"] = close_lag.rolling(5).mean()
        group[f"sma_10_lag{lag}"] = close_lag.rolling(10).mean()
        group[f"sma_20_lag{lag}"] = close_lag.rolling(20).mean()
        group[f"price_std_5_lag{lag}"] = close_lag.rolling(5).std()
        group[f"price_std_10_lag{lag}"] = close_lag.rolling(10).std()
        group[f"vol_sma_5_lag{lag}"] = vol_lag.rolling(5).mean()
        group[f"vol_sma_10_lag{lag}"] = vol_lag.rolling(10).mean()

        hl_range = high_lag - low_lag
        group[f"hl_range_sma_5_lag{lag}"] = hl_range.rolling(5).mean()
        group[f"hl_range_std_5_lag{lag}"] = hl_range.rolling(5).std()

    # 3. Momentum: the first four moments of past returns
    returns = group["close"].pct_change()
    for lag in range(1, 8):
        group[f"return_lag{lag}"] = returns.shift(lag)
    for lag in (1, 2):
        r = returns.shift(lag)
        group[f"return_mean_5_lag{lag}"] = r.rolling(5).mean()
        group[f"return_mean_10_lag{lag}"] = r.rolling(10).mean()
        group[f"return_std_5_lag{lag}"] = r.rolling(5).std()
        group[f"return_std_10_lag{lag}"] = r.rolling(10).std()
        group[f"return_skew_10_lag{lag}"] = r.rolling(10).skew()
        group[f"return_kurt_10_lag{lag}"] = r.rolling(10).kurt()

    # 4. Price/volume ratios
    for lag in range(1, 4):
        close_lag = group["close"].shift(lag)
        vol_lag = group["volumefrom"].shift(lag)
        group[f"close_to_sma5_lag{lag}"] = close_lag / (close_lag.rolling(5).mean() + EPS)
        group[f"close_to_sma10_lag{lag}"] = close_lag / (close_lag.rolling(10).mean() + EPS)
        group[f"vol_price_ratio_lag{lag}"] = vol_lag / (close_lag + EPS)
        group[f"vol_to_sma_lag{lag}"] = vol_lag / (vol_lag.rolling(5).mean() + EPS)

    # 5. RSI(14) / MACD(12,26,9) / Bollinger(20, 2 sd)
    #    Computed on the unshifted close (each bar's indicator uses its own
    #    close, per the standard definitions), then shifted by >= 1 so the
    #    value available when predicting bar t comes from bar t-1 or older.
    indicators = build_indicator_frame(group["close"])
    for lag in range(1, 4):
        shifted = indicators.shift(lag)
        for col in INDICATOR_COLUMNS:
            group[f"{col}_lag{lag}"] = shifted[col]

    # 6. Calendar features (known in advance -> leakage-free)
    time_dt = pd.to_datetime(group["time"])
    group["hour"] = time_dt.dt.hour
    group["day"] = time_dt.dt.day
    group["month"] = time_dt.dt.month
    group["quarter"] = time_dt.dt.quarter
    group["weekday"] = time_dt.dt.weekday
    group["is_weekend"] = (time_dt.dt.weekday >= 5).astype(int)
    group["hour_sin"] = np.sin(2 * np.pi * group["hour"] / 24)
    group["hour_cos"] = np.cos(2 * np.pi * group["hour"] / 24)
    group["month_sin"] = np.sin(2 * np.pi * group["month"] / 12)
    group["month_cos"] = np.cos(2 * np.pi * group["month"] / 12)
    group["weekday_sin"] = np.sin(2 * np.pi * group["weekday"] / 7)
    group["weekday_cos"] = np.cos(2 * np.pi * group["weekday"] / 7)
    return group


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the leakage-free modelling frame from raw OHLCV.

    Returns (frame_with_targets, feature_column_names).

    Every predictor is derived from bars strictly before the bar being
    predicted. The target is the NEXT bar's log return.
    """
    print("Creating log-returns target (next-period)...")
    df_fixed = df.copy()
    # Sort correctness depends on `time` being a real timestamp, not a string.
    df_fixed["time"] = pd.to_datetime(df_fixed["time"], errors="coerce")
    df_fixed = df_fixed.sort_values(["symbol", "time"]).reset_index(drop=True)

    target_parts = []
    for _, group in df_fixed.groupby("symbol"):
        group = group.sort_values("time").reset_index(drop=True)
        group["y_return"] = np.log(group["close"].shift(-1) / group["close"])
        group = group.iloc[:-1]  # last bar has no future
        target_parts.append(group)
    df_ts = pd.concat(target_parts, ignore_index=True)
    print(f"   Rows after target creation: {len(df_ts):,}")

    # Strip every column derived from the CURRENT bar. These are exactly the
    # columns that produced the original leaked R2 = 0.9998.
    leakage_cols = [
        "return_1", "log_return", "hl_range", "candle_body",
        "upper_shadow", "lower_shadow", "body_to_range",
    ]
    present = [c for c in leakage_cols if c in df_ts.columns]
    if present:
        print(f"   Dropping current-bar leakage columns: {present}")
        df_ts = df_ts.drop(columns=present)

    print("Building lagged features (OHLCV, rolling stats, momentum, RSI/MACD/BB, time)...")
    feature_parts = []
    for _symbol, group in df_ts.groupby("symbol"):
        group = group.sort_values("time").reset_index(drop=True)
        feature_parts.append(_engineer_symbol_group(group))

    df_features = pd.concat(feature_parts, ignore_index=True)

    # Drop raw current-bar OHLCV now that all lags are built.
    current_bar = ["open", "high", "low", "close", "volumefrom", "volumeto"]
    df_features = df_features.drop(columns=[c for c in current_bar if c in df_features.columns])

    # Targets are carried inside the frame, so alignment is guaranteed by the
    # index rather than by positional slicing.
    df_features["target_returns"] = df_features["y_return"].replace([np.inf, -np.inf], np.nan)
    df_features["target_direction"] = (df_features["target_returns"] > 0).astype(int)
    df_features = df_features.drop(columns=["y_return"])

    before = len(df_features)
    df_features = df_features[df_features["target_returns"].notna()].copy()
    print(f"   Dropped {before - len(df_features):,} rows with undefined target")

    # Whitelist: only lagged columns and known-in-advance calendar columns.
    feature_cols = [
        c for c in df_features.columns
        if c.endswith(_SAFE_LAG_SUFFIXES) or c in _SAFE_TIME_COLS
    ]

    # Warm-up rows (NaN from the longest rolling window) are dropped, then any
    # residual NaN is zero-filled. Targets are never touched here.
    thresh = int(len(feature_cols) * 0.7)
    df_features = df_features.dropna(subset=feature_cols, thresh=thresh)
    df_features[feature_cols] = df_features[feature_cols].fillna(0.0)
    df_features = df_features.reset_index(drop=True)

    n_ind = sum(1 for c in feature_cols if any(c.startswith(i + "_lag") for i in INDICATOR_COLUMNS))
    print(f"   Final frame: {len(df_features):,} rows x {len(feature_cols)} features "
          f"({n_ind} of them RSI/MACD/Bollinger)")
    return df_features, feature_cols


def create_lagged_features(df: pd.DataFrame, lag_periods: int = 1) -> pd.DataFrame:
    """Backward-compatible thin wrapper around build_feature_frame()."""
    frame, _ = build_feature_frame(df)
    return frame


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(
        2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + EPS)
    ) * 100)


def _fmt(mean: float, std: float, width: int = 7, prec: int = 4) -> str:
    return f"{mean:{width}.{prec}f} +/- {std:.{prec}f}"


@contextlib.contextmanager
def _quiet():
    """
    Silence the per-fit chatter from CryptoPreprocessor.

    A 5-fold x 4-model x 2-task run calls fit/transform 80+ times, and each
    call prints a dozen diagnostic lines. Useful when debugging one fit,
    unreadable across a full CV sweep - so the sweep suppresses it and reports
    aggregated fold statistics instead.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def train_and_save(
    df: pd.DataFrame,
    model_dir: str = "models",
    regressor_name: str = "best_regressor_pipeline.pkl",
    classifier_name: str = "best_classifier_pipeline.pkl",
    k_best: int = 15,
    test_all_models: bool = True,
    lag_periods: int = 1,
    n_splits: int = 5,
    cv_gap: int = 24,
    run_stationarity_test: bool = True,
    run_statsmodels_baseline: bool = True,
) -> Dict[str, Any]:
    """
    Full training run:
      1. ADF stationarity test justifying the returns target
      2. Leakage-free feature construction (incl. RSI / MACD / Bollinger)
      3. 5-fold expanding-window time-series CV, per-symbol chronological
      4. statsmodels OLS / Logit interpretable baselines
      5. Refit the CV winner on all data and persist it
    """
    os.makedirs(model_dir, exist_ok=True)

    # --- Step 1: stationarity evidence ------------------------------------
    adf_results = run_adf_tests(df) if run_stationarity_test else {}

    # --- Step 2: features -------------------------------------------------
    print("\n" + "=" * 78)
    print("FEATURE CONSTRUCTION")
    print("=" * 78)
    frame, feature_cols = build_feature_frame(df)
    if frame.empty:
        raise ValueError("No data remaining after feature engineering")

    X_all = frame[feature_cols]
    y_reg_all = frame["target_returns"].astype(float).to_numpy()
    y_clf_all = frame["target_direction"].astype(int).to_numpy()

    print(f"   Target log-return: mean={y_reg_all.mean():.8f}  std={y_reg_all.std():.6f}")
    print(f"   Direction balance: {y_clf_all.mean():.2%} positive")
    if "return_lag1" in X_all.columns:
        persistence = float(np.corrcoef(y_reg_all, X_all["return_lag1"])[0, 1])
        print(f"   Target vs return_lag1 correlation: {persistence:.4f} "
              f"({'OK' if abs(persistence) < 0.3 else 'SUSPICIOUS'})")

    # --- Step 3: cross-validation ----------------------------------------
    print("\n" + "=" * 78)
    print(f"TIME-SERIES CROSS-VALIDATION ({n_splits} expanding folds, "
          f"per-symbol chronological, {cv_gap}-bar embargo)")
    print("=" * 78)

    folds = per_symbol_time_series_splits(frame, n_splits=n_splits, gap=cv_gap)
    for i, (tr, te) in enumerate(folds, 1):
        print(f"   Fold {i}: train={len(tr):>7,}  test={len(te):>7,}")

    models_to_test = ["random_forest", "decision_tree", "xgboost", "lightgbm"]
    reg_cv: Dict[str, Dict[str, Any]] = {}
    clf_cv: Dict[str, Dict[str, Any]] = {}

    # Naive baseline: for returns the null forecast is zero (random walk).
    baseline_r2, baseline_mae = [], []
    for _, te in folds:
        y_te = y_reg_all[frame.index.get_indexer(te)]
        baseline_r2.append(r2_score(y_te, np.zeros_like(y_te)))
        baseline_mae.append(mean_absolute_error(y_te, np.zeros_like(y_te)))
    print(f"\n   BASELINE (predict zero): R2={np.mean(baseline_r2):.6f}  "
          f"MAE={np.mean(baseline_mae):.6f}")

    print("\n--- REGRESSION: next-period log return ---")
    for model_type in models_to_test:
        scores = {"r2": [], "mae": [], "rmse": [], "smape": []}
        try:
            for tr, te in folds:
                itr = frame.index.get_indexer(tr)
                ite = frame.index.get_indexer(te)
                with _quiet():
                    pipe = build_pipeline_no_leakage(model_type, "regression", k_best)
                    pipe.fit(X_all.iloc[itr], y_reg_all[itr])
                    pred = pipe.predict(X_all.iloc[ite])
                scores["r2"].append(r2_score(y_reg_all[ite], pred))
                scores["mae"].append(mean_absolute_error(y_reg_all[ite], pred))
                scores["rmse"].append(np.sqrt(mean_squared_error(y_reg_all[ite], pred)))
                scores["smape"].append(smape(y_reg_all[ite], pred))
            reg_cv[model_type] = {
                f"{k}_mean": float(np.mean(v)) for k, v in scores.items()
            } | {
                f"{k}_std": float(np.std(v)) for k, v in scores.items()
            } | {"r2_folds": [float(x) for x in scores["r2"]]}
            print(f"   {model_type:<15} R2={_fmt(reg_cv[model_type]['r2_mean'], reg_cv[model_type]['r2_std'])}"
                  f"   MAE={reg_cv[model_type]['mae_mean']:.6f}"
                  f"   RMSE={reg_cv[model_type]['rmse_mean']:.6f}")
        except Exception as exc:
            print(f"   {model_type:<15} FAILED: {exc}")

    print("\n--- CLASSIFICATION: direction ---")
    for model_type in models_to_test:
        scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        try:
            for tr, te in folds:
                itr = frame.index.get_indexer(tr)
                ite = frame.index.get_indexer(te)
                with _quiet():
                    pipe = build_pipeline_no_leakage(model_type, "classification", k_best)
                    pipe.fit(X_all.iloc[itr], y_clf_all[itr])
                    pred = pipe.predict(X_all.iloc[ite])
                yt = y_clf_all[ite]
                scores["accuracy"].append(accuracy_score(yt, pred))
                scores["precision"].append(precision_score(yt, pred, zero_division=0))
                scores["recall"].append(recall_score(yt, pred, zero_division=0))
                scores["f1"].append(f1_score(yt, pred, zero_division=0))
            clf_cv[model_type] = {
                f"{k}_mean": float(np.mean(v)) for k, v in scores.items()
            } | {
                f"{k}_std": float(np.std(v)) for k, v in scores.items()
            } | {"accuracy_folds": [float(x) for x in scores["accuracy"]]}
            print(f"   {model_type:<15} Acc={_fmt(clf_cv[model_type]['accuracy_mean'], clf_cv[model_type]['accuracy_std'])}"
                  f"   F1={clf_cv[model_type]['f1_mean']:.4f}"
                  f"   Prec={clf_cv[model_type]['precision_mean']:.4f}")
        except Exception as exc:
            print(f"   {model_type:<15} FAILED: {exc}")

    # --- Step 4: statsmodels baselines on the final fold ------------------
    sm_results: Dict[str, Any] = {}
    if run_statsmodels_baseline:
        tr, te = folds[-1]
        itr = frame.index.get_indexer(tr)
        ite = frame.index.get_indexer(te)
        sm_results = fit_statsmodels_baselines(
            X_all.iloc[itr], y_reg_all[itr], y_clf_all[itr],
            X_all.iloc[ite], y_reg_all[ite], y_clf_all[ite],
            k_best=k_best, model_dir=model_dir,
        )

    # --- Step 5: pick winners, refit on all data, persist -----------------
    if not reg_cv or not clf_cv:
        raise RuntimeError("Cross-validation produced no usable results.")

    best_reg = max(reg_cv, key=lambda m: reg_cv[m]["r2_mean"])
    best_clf = max(clf_cv, key=lambda m: clf_cv[m]["accuracy_mean"])

    print("\n" + "=" * 78)
    print("REFITTING CV WINNERS ON THE FULL DATASET")
    print("=" * 78)
    print(f"   Regressor : {best_reg}")
    print(f"   Classifier: {best_clf}")

    reg_pipe = build_pipeline_no_leakage(best_reg, "regression", k_best)
    reg_pipe.fit(X_all, y_reg_all)
    clf_pipe = build_pipeline_no_leakage(best_clf, "classification", k_best)
    clf_pipe.fit(X_all, y_clf_all)

    regressor_path = os.path.join(model_dir, regressor_name)
    classifier_path = os.path.join(model_dir, classifier_name)
    dump(reg_pipe, regressor_path)
    dump(clf_pipe, classifier_path)

    regressor_metrics = {
        "cv_r2_mean": reg_cv[best_reg]["r2_mean"],
        "cv_r2_std": reg_cv[best_reg]["r2_std"],
        "cv_mae_mean": reg_cv[best_reg]["mae_mean"],
        "cv_rmse_mean": reg_cv[best_reg]["rmse_mean"],
        "cv_smape_mean": reg_cv[best_reg]["smape_mean"],
        "r2": reg_cv[best_reg]["r2_mean"],          # legacy key
        "mae": reg_cv[best_reg]["mae_mean"],        # legacy key
        "n_features": len(feature_cols),
        "n_rows": int(len(frame)),
        "features_used": feature_cols,
    }
    classifier_metrics = {
        "cv_accuracy_mean": clf_cv[best_clf]["accuracy_mean"],
        "cv_accuracy_std": clf_cv[best_clf]["accuracy_std"],
        "cv_f1_mean": clf_cv[best_clf]["f1_mean"],
        "cv_precision_mean": clf_cv[best_clf]["precision_mean"],
        "cv_recall_mean": clf_cv[best_clf]["recall_mean"],
        "accuracy": clf_cv[best_clf]["accuracy_mean"],  # legacy key
        "f1": clf_cv[best_clf]["f1_mean"],              # legacy key
        "n_features": len(feature_cols),
        "n_rows": int(len(frame)),
        "features_used": feature_cols,
    }

    # --- Final comparison table ------------------------------------------
    print("\n" + "=" * 78)
    print("FINAL MODEL COMPARISON  (5-fold expanding-window time-series CV)")
    print("=" * 78)
    print(f"{'Model':<21}{'Task':<16}{'Metric':<10}{'CV mean +/- std':>24}")
    print("-" * 78)
    for m in models_to_test:
        if m in reg_cv:
            print(f"{m:<21}{'regression':<16}{'R2':<10}"
                  f"{_fmt(reg_cv[m]['r2_mean'], reg_cv[m]['r2_std']):>24}")
    if "ols" in sm_results and "r2" in sm_results["ols"]:
        print(f"{'OLS (statsmodels)':<21}{'regression':<16}{'R2':<10}"
              f"{sm_results['ols']['r2']:>15.4f} (1 fold)")
    print("-" * 78)
    for m in models_to_test:
        if m in clf_cv:
            print(f"{m:<21}{'classification':<16}{'Accuracy':<10}"
                  f"{_fmt(clf_cv[m]['accuracy_mean'], clf_cv[m]['accuracy_std']):>24}")
    if "logit" in sm_results and "accuracy" in sm_results["logit"]:
        print(f"{'Logit (statsmodels)':<21}{'classification':<16}{'Accuracy':<10}"
              f"{sm_results['logit']['accuracy']:>15.4f} (1 fold)")
    print("=" * 78)
    print("Note: the statsmodels rows are single-fold (fitted on the last CV fold),")
    print("      so they are not directly comparable to the 5-fold means above.")

    # --- Leakage tripwire -------------------------------------------------
    if regressor_metrics["cv_r2_mean"] > 0.80 or classifier_metrics["cv_accuracy_mean"] > 0.85:
        print("\n*** WARNING: performance suspiciously high - re-check for data leakage. ***")
    else:
        print("\nPerformance is in the realistic range for hourly crypto returns.")

    # --- Persist a metrics report ----------------------------------------
    report = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "n_rows": int(len(frame)),
        "validation": {
            "method": "per-symbol expanding-window TimeSeriesSplit",
            "n_splits": n_splits,
            "embargo_bars": cv_gap,
        },
        "adf": adf_results,
        "regression_cv": reg_cv,
        "classification_cv": clf_cv,
        "statsmodels_baselines": sm_results,
        "best_regressor": best_reg,
        "best_classifier": best_clf,
        "baseline_zero_forecast": {
            "r2_mean": float(np.mean(baseline_r2)),
            "mae_mean": float(np.mean(baseline_mae)),
        },
    }
    report_path = os.path.join(model_dir, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nMetrics report written to {report_path}")

    # --- Plain-English summary -------------------------------------------
    print_plain_summary(
        feature_cols=feature_cols,
        n_rows=len(frame),
        n_splits=n_splits,
        cv_gap=cv_gap,
        reg_cv=reg_cv,
        clf_cv=clf_cv,
        best_reg=best_reg,
        best_clf=best_clf,
        sm_results=sm_results,
        adf_results=adf_results,
        baseline_r2=float(np.mean(baseline_r2)),
        k_best=k_best,
        symbols=sorted(frame["symbol"].unique().tolist()),
    )

    return {
        "regressor_path": regressor_path,
        "classifier_path": classifier_path,
        "best_regressor_type": best_reg,
        "best_classifier_type": best_clf,
        "regressor_metrics": regressor_metrics,
        "classifier_metrics": classifier_metrics,
        "lag_periods": lag_periods,
        "features_used": feature_cols,
        "all_regressor_results": reg_cv,
        "all_classifier_results": clf_cv,
        "statsmodels_baselines": sm_results,
        "adf_results": adf_results,
        "report_path": report_path,
    }


def print_plain_summary(
    feature_cols, n_rows, n_splits, cv_gap, reg_cv, clf_cv,
    best_reg, best_clf, sm_results, adf_results, baseline_r2, k_best,
    n_symbols=None, symbols=None,
) -> None:
    """Plain-English wrap-up, written to be copied into a CV bullet verbatim."""
    groups = {
        "Lagged OHLCV (t-1..t-5)": lambda c: any(
            c.startswith(p) for p in ("open_lag", "high_lag", "low_lag", "close_lag",
                                      "volumefrom_lag", "volumeto_lag")),
        "RSI (14, Wilder)": lambda c: c.startswith("rsi_"),
        "MACD (12/26/9)": lambda c: c.startswith("macd"),
        "Bollinger Bands (20, 2sd)": lambda c: c.startswith("bb_"),
        "Rolling price/volume stats": lambda c: any(
            c.startswith(p) for p in ("sma_", "price_std_", "vol_sma_", "hl_range_")),
        "Return moments (mean/std/skew/kurt)": lambda c: c.startswith("return_"),
        "Price/volume ratios": lambda c: any(
            c.startswith(p) for p in ("close_to_sma", "vol_price_ratio", "vol_to_sma")),
        "Calendar / cyclical": lambda c: c in _SAFE_TIME_COLS,
    }
    counts, claimed = {}, set()
    for name, pred in groups.items():
        members = [c for c in feature_cols if pred(c) and c not in claimed]
        claimed.update(members)
        counts[name] = len(members)
    leftover = [c for c in feature_cols if c not in claimed]

    print("\n" + "=" * 78)
    print("PLAIN-ENGLISH SUMMARY")
    print("=" * 78)

    if symbols:
        sym_txt = f"{len(symbols)} cryptocurrencies ({', '.join(map(str, symbols))})"
    elif n_symbols:
        sym_txt = f"{n_symbols} cryptocurrencies"
    else:
        sym_txt = "the loaded cryptocurrencies"
    print(f"\nDATA: {n_rows:,} hourly bars across {sym_txt}.")

    print(f"\nFEATURES: {len(feature_cols)} predictors, all built from bars strictly")
    print("before the one being predicted. Breakdown:")
    for name, n in counts.items():
        if n:
            print(f"   - {name}: {n}")
    if leftover:
        print(f"   - Other: {len(leftover)}")
    print(f"   SelectKBest then narrows these to {k_best} inside the pipeline,")
    print("   fit on training folds only.")

    if adf_results.get("conclusion"):
        print(f"\nTARGET JUSTIFICATION (ADF test, statsmodels):")
        print(f"   {adf_results['conclusion']}")

    print(f"\nVALIDATION: {n_splits}-fold expanding-window time-series cross-validation.")
    print("   TimeSeriesSplit is applied within each symbol's own chronology, so every")
    print(f"   training bar precedes every test bar, with a {cv_gap}-bar embargo between")
    print("   them. No shuffling, no random splits, no lookahead.")

    r = reg_cv.get(best_reg, {})
    c = clf_cv.get(best_clf, {})
    print(f"\nRESULTS (mean +/- std across {n_splits} folds):")
    if r:
        print(f"   Best regressor  : {best_reg}")
        print(f"      R2   = {r['r2_mean']:.4f} +/- {r['r2_std']:.4f}   "
              f"(zero-forecast baseline R2 = {baseline_r2:.4f})")
        print(f"      MAE  = {r['mae_mean']:.6f} +/- {r['mae_std']:.6f}")
        print(f"      RMSE = {r['rmse_mean']:.6f}")
    if c:
        print(f"   Best classifier : {best_clf}")
        print(f"      Accuracy  = {c['accuracy_mean']:.4f} +/- {c['accuracy_std']:.4f}")
        print(f"      Precision = {c['precision_mean']:.4f}   "
              f"Recall = {c['recall_mean']:.4f}   F1 = {c['f1_mean']:.4f}")
    if sm_results.get("ols", {}).get("r2") is not None:
        print(f"   OLS baseline    : R2 = {sm_results['ols']['r2']:.4f}, "
              f"{sm_results['ols'].get('n_significant_5pct', 0)} coefficients "
              f"significant at 5%")
    if sm_results.get("logit", {}).get("accuracy") is not None:
        print(f"   Logit baseline  : accuracy = {sm_results['logit']['accuracy']:.4f}, "
              f"pseudo-R2 = {sm_results['logit'].get('pseudo_r2', float('nan')):.5f}")

    print("\nHOW TO READ THIS: hourly crypto returns are near-unpredictable. An R2 a few")
    print("points above zero and a directional accuracy a few points above 50% is the")
    print("honest ceiling. The std across folds is the number that matters - it shows")
    print("whether the edge survives different market regimes or came from one lucky split.")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Inference-time feature construction
# ---------------------------------------------------------------------------
# Longest lookback in the feature set: Bollinger/SMA use a 20-bar rolling window
# read at lag 3 (23 bars), and MACD's slow leg is a 26-period EMA that needs
# roughly its own span again to converge. 50 bars clears both with margin.
MIN_HISTORY_BARS = 50


def build_features_for_inference(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build predictor columns for scoring, using the exact same per-symbol
    engineering as training.

    Differs from build_feature_frame() in two ways, both deliberate:
      - no target is created, so nothing is shifted backwards; and
      - the final bar of each symbol is KEPT, because that is precisely the bar
        we want a forecast from.

    Returns (frame, feature_columns). Take .groupby('symbol').tail(1) for the
    latest forecastable row per symbol.
    """
    work = df.copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work = work.sort_values(["symbol", "time"]).reset_index(drop=True)

    stale = ["return_1", "log_return", "hl_range", "candle_body",
             "upper_shadow", "lower_shadow", "body_to_range"]
    work = work.drop(columns=[c for c in stale if c in work.columns])

    parts = []
    for _symbol, group in work.groupby("symbol"):
        group = group.sort_values("time").reset_index(drop=True)
        parts.append(_engineer_symbol_group(group))
    feats = pd.concat(parts, ignore_index=True)

    current_bar = ["open", "high", "low", "close", "volumefrom", "volumeto"]
    feats = feats.drop(columns=[c for c in current_bar if c in feats.columns])

    feature_cols = [
        c for c in feats.columns
        if c.endswith(_SAFE_LAG_SUFFIXES) or c in _SAFE_TIME_COLS
    ]
    feats[feature_cols] = feats[feature_cols].fillna(0.0)
    return feats, feature_cols
