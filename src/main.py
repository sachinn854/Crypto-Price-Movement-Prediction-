# main.py
from __future__ import annotations
import argparse
import os
import pandas as pd

from unified_pipeline import CompleteCryptoPipeline

def read_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # standardize column names (lowercase)
    df.columns = [c.strip() for c in df.columns]
    # ensure required minimal columns exist
    required = ["symbol", "time", "open", "high", "low", "close", "volumefrom", "volumeto"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../Data/processed/final_cleaned_crypto_zero_removed.csv",
                        help="CSV with columns: symbol,time,open,high,low,close,volumefrom,volumeto[,return_1]")
    parser.add_argument("--models_dir", type=str, default="../models")
    parser.add_argument("--regressor_name", type=str, default="best_regressor_pipeline.pkl")
    parser.add_argument("--classifier_name", type=str, default="best_classifier_pipeline.pkl")
    parser.add_argument("--kbest", type=int, default=20)
    args = parser.parse_args()

    df = read_data(args.data)

    # Train + save both models
    cp = CompleteCryptoPipeline(models_dir=args.models_dir, 
                               regressor_name=args.regressor_name,
                               classifier_name=args.classifier_name)
    result = cp.fit_and_save(df, k_best=args.kbest)

    print("✅ Both pipelines trained and saved.")
    print(f"📦 Regressor: {result['regressor_path']}")
    print(f"📦 Classifier: {result['classifier_path']}")
    print("📈 Regressor Metrics:", result["regressor_metrics"])
    print("📊 Classifier Metrics:", result["classifier_metrics"])

if __name__ == "__main__":
    main()
