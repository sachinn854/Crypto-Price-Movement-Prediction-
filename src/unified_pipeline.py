# unified_pipeline.py
from __future__ import annotations
import os
from typing import Dict, Any
import pandas as pd
from joblib import load

from model_training import train_and_save

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

    # ------------- target creation -------------
    @staticmethod
    def _ensure_target(df: pd.DataFrame) -> pd.DataFrame:
        """
        If 'return_1' missing, create per-symbol % return from previous close based on time order.
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
        result = train_and_save(df2, model_dir=self.models_dir, 
                               regressor_name=self.regressor_name,
                               classifier_name=self.classifier_name,
                               k_best=k_best)
        # cache pipelines for this session
        self.regressor_pipeline = load(result["regressor_path"])
        self.classifier_pipeline = load(result["classifier_path"])
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

    def predict_return(self, X_row_like: pd.DataFrame) -> float:
        """
        Predict percentage return using regressor.
        X_row_like must contain: symbol, time, open, high, low, close, volumefrom, volumeto
        returns a single float (predicted return_1)
        """
        if self.regressor_pipeline is None:
            self.load_pipelines()
        raw_cols = ["symbol", "time", "open", "high", "low", "close", "volumefrom", "volumeto"]
        X = X_row_like.reindex(columns=raw_cols)
        pred = float(self.regressor_pipeline.predict(X)[0])
        return pred
    
    def predict_direction(self, X_row_like: pd.DataFrame) -> int:
        """
        Predict market direction using classifier.
        X_row_like must contain: symbol, time, open, high, low, close, volumefrom, volumeto
        returns 1 for bullish (up), 0 for bearish (down)
        """
        if self.classifier_pipeline is None:
            self.load_pipelines()
        raw_cols = ["symbol", "time", "open", "high", "low", "close", "volumefrom", "volumeto"]
        X = X_row_like.reindex(columns=raw_cols)
        pred = int(self.classifier_pipeline.predict(X)[0])
        return pred
    
    def predict_both(self, X_row_like: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict both return percentage and market direction.
        Returns dict with 'return_pct' and 'direction' (and 'direction_label')
        """
        return_pct = self.predict_return(X_row_like)
        direction = self.predict_direction(X_row_like)
        direction_label = "Bullish 📈" if direction == 1 else "Bearish 📉"
        
        return {
            "return_pct": return_pct,
            "direction": direction,
            "direction_label": direction_label
        }
