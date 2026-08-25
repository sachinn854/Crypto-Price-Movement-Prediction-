# unified_pipeline.py
from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional

import pandas as pd
from joblib import load

from model_training import (
    train_and_save,
    build_features_for_inference,
    MIN_HISTORY_BARS,
)

RAW_INPUT_COLS = ["symbol", "time", "open", "high", "low", "close", "volumefrom", "volumeto"]


class InsufficientHistoryError(ValueError):
    """
    Raised when a caller asks for a prediction from fewer bars than the feature
    set needs.

    This is a data requirement, not a bug: the models read a 20-bar Bollinger /
    SMA window at lag 3 and a 26-period MACD EMA, so a single candle cannot
    produce a feature vector at all. The fix is to pass more history, not to
    pad the missing bars - padding would feed the model invented price action.
    """


class CompleteCryptoPipeline:
    """
    Wrapper to:
    - prepare data (ensure target present)
    - train+save both regressor and classifier pipelines
    - load+predict for both tasks
    """

    def __init__(self, models_dir: str = "models",
                 regressor_name: str = "best_regressor_pipeline.pkl",
                 classifier_name: str = "best_classifier_pipeline.pkl"):
        self.models_dir = models_dir
        self.regressor_name = regressor_name
        self.classifier_name = classifier_name
        self.regressor_path = os.path.join(models_dir, regressor_name)
        self.classifier_path = os.path.join(models_dir, classifier_name)
        self.regressor_pipeline = None
        self.classifier_pipeline = None
        self._feature_columns: Optional[List[str]] = None

    # ------------- target creation -------------
    @staticmethod
    def _ensure_target(df: pd.DataFrame) -> pd.DataFrame:
        """
        If 'return_1' missing, create per-symbol % return from previous close
        based on time order.
        """
        out = df.copy()
        if "return_1" not in out.columns:
            if "time" not in out.columns:
                raise ValueError("time column required to create target but not found.")
            out["time"] = pd.to_datetime(out["time"], errors="coerce")
            out = out.sort_values(["symbol", "time"])
            out["prev_close"] = out.groupby("symbol")["close"].shift(1)
            out["return_1"] = (out["close"] - out["prev_close"]) / out["prev_close"]
            out.drop(columns=["prev_close"], inplace=True)
        return out

    # ------------- training -------------
    def fit_and_save(self, df: pd.DataFrame, k_best: int = 20) -> Dict[str, Any]:
        df2 = self._ensure_target(df)
        result = train_and_save(
            df2,
            model_dir=self.models_dir,
            regressor_name=self.regressor_name,
            classifier_name=self.classifier_name,
            k_best=k_best,
        )
        self.regressor_pipeline = load(result["regressor_path"])
        self.classifier_pipeline = load(result["classifier_path"])
        self._feature_columns = result.get("features_used")
        return result

    # ------------- load + predict -------------
    def load_pipelines(self):
        """Load both regressor and classifier pipelines"""
        if not os.path.exists(self.regressor_path):
            raise FileNotFoundError(f"Regressor pipeline not found at {self.regressor_path}")
        if not os.path.exists(self.classifier_path):
            raise FileNotFoundError(f"Classifier pipeline not found at {self.classifier_path}")
        self.regressor_pipeline = load(self.regressor_path)
        self.classifier_pipeline = load(self.classifier_path)
        return self.regressor_pipeline, self.classifier_pipeline

    def feature_columns(self) -> List[str]:
        """
        The exact predictor columns, in the exact order, that the saved
        pipelines were fitted on.

        Order matters: CryptoPreprocessor hands the frame straight to a
        StandardScaler that was fitted on a specific column order, so a
        reordered frame is a different input. training_report.json is the
        source of truth; the preprocessor's own recorded columns are a fallback
        for models saved before that report existed.
        """
        if self._feature_columns:
            return self._feature_columns

        report_path = os.path.join(self.models_dir, "training_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as fh:
                self._feature_columns = json.load(fh)["feature_columns"]
                return self._feature_columns

        if self.regressor_pipeline is None:
            self.load_pipelines()
        pre = self.regressor_pipeline.named_steps["preprocessing"]
        cat = list(getattr(pre, "categorical_columns_", []))
        num = list(getattr(pre, "numerical_columns_", []))
        if not (cat or num):
            raise RuntimeError(
                "Cannot determine the pipeline's feature columns. Retrain with "
                "src/main.py so that models/training_report.json is written."
            )
        self._feature_columns = cat + num
        return self._feature_columns

    # ------------- history-based inference -------------
    def predict_from_history(self, history: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict the next bar's return and direction from a window of recent
        OHLCV bars for ONE symbol.

        `history` needs the raw columns in RAW_INPUT_COLS and at least
        MIN_HISTORY_BARS rows, oldest first. The forecast is made from the most
        recent bar.
        """
        if self.regressor_pipeline is None or self.classifier_pipeline is None:
            self.load_pipelines()

        missing = [c for c in RAW_INPUT_COLS if c not in history.columns]
        if missing:
            raise ValueError(f"history is missing required columns: {missing}")

        n_symbols = history["symbol"].nunique()
        if n_symbols != 1:
            raise ValueError(
                f"predict_from_history expects exactly one symbol, got {n_symbols}."
            )

        if len(history) < MIN_HISTORY_BARS:
            raise InsufficientHistoryError(
                f"Need at least {MIN_HISTORY_BARS} consecutive hourly bars to build "
                f"the feature set (20-bar Bollinger/SMA window read at lag 3, plus a "
                f"26-period MACD EMA); got {len(history)}."
            )

        feats, _ = build_features_for_inference(history[RAW_INPUT_COLS])
        cols = self.feature_columns()
        latest = feats.tail(1).reindex(columns=cols, fill_value=0.0)

        return_pct = float(self.regressor_pipeline.predict(latest)[0])
        direction = int(self.classifier_pipeline.predict(latest)[0])
        return {
            "return_pct": return_pct,
            "direction": direction,
            "direction_label": "Bullish 📈" if direction == 1 else "Bearish 📉",
            "as_of": history["time"].iloc[-1],
            "bars_used": int(len(history)),
        }

    # ------------- legacy single-call API (kept for app.py) -------------
    def predict_return(self, X_row_like: pd.DataFrame) -> float:
        """Predict percentage return. Requires >= MIN_HISTORY_BARS rows."""
        return self.predict_both(X_row_like)["return_pct"]

    def predict_direction(self, X_row_like: pd.DataFrame) -> int:
        """Predict direction: 1 bullish, 0 bearish. Requires >= MIN_HISTORY_BARS rows."""
        return self.predict_both(X_row_like)["direction"]

    def predict_both(self, X_row_like: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict return percentage and direction.

        Signature is unchanged so app.py keeps working, but the input contract
        is now explicit: this needs a window of recent bars, not a single
        candle. Passing one row raises InsufficientHistoryError with an
        actionable message rather than failing deep inside the scaler with a
        KeyError about missing lag columns.
        """
        return self.predict_from_history(X_row_like)
