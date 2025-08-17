# model_training.py
from __future__ import annotations
import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from joblib import dump

from feature_engineering_module import CryptoFeatureEngineer
from preprocessing_module import CryptoPreprocessor

# ---------------- Time-aware split ----------------
def time_aware_split(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    time_col: str = "time",
    target_col: str = "return_1",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-symbol chronological split to avoid leakage.
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found for time-aware split.")

    parts_train, parts_test = [], []
    for sym, g in df.groupby(symbol_col):
        g = g.sort_values(time_col)
        n = len(g)
        if n < 5:  # too tiny, dump all in train
            parts_train.append(g)
            continue
        cut = int(np.floor(n * (1 - test_size)))
        parts_train.append(g.iloc[:cut])
        parts_test.append(g.iloc[cut:])

    train = pd.concat(parts_train, axis=0).reset_index(drop=True)
    test = pd.concat(parts_test, axis=0).reset_index(drop=True) if parts_test else pd.DataFrame(columns=df.columns)
    # drop rows with missing target
    train = train[train[target_col].notna()]
    test  = test[test[target_col].notna()]
    return train, test


def build_pipeline(model_type: str = "random_forest", task_type: str = "regression", k_best: int = 20, model_params: Dict[str, Any] | None = None) -> Pipeline:
    """
    Build pipeline with specified model type and task type.
    Supported models: random_forest, decision_tree, xgboost, lightgbm
    Supported tasks: regression, classification
    """
    if model_params is None:
        model_params = {}
    
    # Define default parameters for each model type and task
    default_params = {
        "regression": {
            "random_forest": dict(
                n_estimators=300,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            ),
            "decision_tree": dict(
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
            ),
            "xgboost": dict(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ),
            "lightgbm": dict(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        },
        "classification": {
            "random_forest": dict(
                n_estimators=300,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced',
            ),
            "decision_tree": dict(
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
            ),
            "xgboost": dict(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                objective='binary:logistic',
            ),
            "lightgbm": dict(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                objective='binary',
                class_weight='balanced',
            )
        }
    }
    
    # Select model based on task type
    if task_type == "regression":
        if model_type == "random_forest":
            params = {**default_params["regression"]["random_forest"], **model_params}
            model = RandomForestRegressor(**params)
        elif model_type == "decision_tree":
            params = {**default_params["regression"]["decision_tree"], **model_params}
            model = DecisionTreeRegressor(**params)
        elif model_type == "xgboost":
            params = {**default_params["regression"]["xgboost"], **model_params}
            model = xgb.XGBRegressor(**params)
        elif model_type == "lightgbm":
            params = {**default_params["regression"]["lightgbm"], **model_params}
            model = lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f"Unsupported model type for regression: {model_type}")
            
    elif task_type == "classification":
        if model_type == "random_forest":
            params = {**default_params["classification"]["random_forest"], **model_params}
            model = RandomForestClassifier(**params)
        elif model_type == "decision_tree":
            params = {**default_params["classification"]["decision_tree"], **model_params}
            model = DecisionTreeClassifier(**params)
        elif model_type == "xgboost":
            params = {**default_params["classification"]["xgboost"], **model_params}
            model = xgb.XGBClassifier(**params)
        elif model_type == "lightgbm":
            params = {**default_params["classification"]["lightgbm"], **model_params}
            model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type for classification: {model_type}")
    else:
        raise ValueError(f"Unsupported task type: {task_type}. Use: regression, classification")
    
    pipe = Pipeline(steps=[
        ("feature_engineering", CryptoFeatureEngineer()),
        ("preprocessing", CryptoPreprocessor(k=k_best)),
        ("model", model),
    ])
    return pipe


def train_and_save(
    df: pd.DataFrame,
    model_dir: str = "models",
    regressor_name: str = "best_regressor_pipeline.pkl",
    classifier_name: str = "best_classifier_pipeline.pkl",
    k_best: int = 20,
    test_all_models: bool = True,
) -> Dict[str, Any]:
    """
    - Assumes df contains minimal raw columns + target 'return_1'
    - Builds time-aware split
    - Tests all models for both regression and classification tasks
    - Saves best regressor and classifier pipelines separately
    """
    os.makedirs(model_dir, exist_ok=True)

    # split
    train_df, test_df = time_aware_split(df, target_col="return_1", test_size=0.2)

    # minimal inputs required by FE
    raw_cols = ["symbol", "time", "open", "high", "low", "close", "volumefrom", "volumeto"]
    missing_cols = [c for c in raw_cols + ["return_1"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for training: {missing_cols}")

    X_train = train_df[raw_cols].copy()
    y_train_reg = train_df["return_1"].astype(float).values  # For regression
    y_train_clf = (train_df["return_1"] > 0).astype(int).values  # For classification (bullish/bearish)

    X_test = test_df[raw_cols].copy()
    y_test_reg = test_df["return_1"].astype(float).values
    y_test_clf = (test_df["return_1"] > 0).astype(int).values

    if test_all_models:
        # Test all models for both tasks
        print("🎯 Testing all models for REGRESSION and CLASSIFICATION tasks...")
        models_to_test = ["random_forest", "decision_tree", "xgboost", "lightgbm"]
        
        # Results storage
        regressor_results = {}
        classifier_results = {}
        
        best_regressor_type = None
        best_regressor_score = -float('inf')
        best_regressor_metrics = None
        best_regressor_pipeline = None
        
        best_classifier_type = None
        best_classifier_score = -float('inf')
        best_classifier_metrics = None
        best_classifier_pipeline = None
        
        for model_type in models_to_test:
            print(f"\n📈 Testing {model_type}...")
            
            # Test REGRESSION
            try:
                print(f"   🔢 Regression task...")
                pipe_reg = build_pipeline(model_type=model_type, task_type="regression", k_best=k_best)
                pipe_reg.fit(X_train, y_train_reg)
                
                y_pred_reg = pipe_reg.predict(X_test) if len(X_test) else np.array([])
                if len(y_pred_reg) > 0:
                    r2 = float(r2_score(y_test_reg, y_pred_reg))
                    mae = float(mean_absolute_error(y_test_reg, y_pred_reg))
                    
                    regressor_results[model_type] = {
                        "r2": r2,
                        "mae": mae,
                        "pipeline": pipe_reg
                    }
                    
                    print(f"      R² Score: {r2:.4f}")
                    print(f"      MAE: {mae:.4f}")
                    
                    if r2 > best_regressor_score:
                        best_regressor_score = r2
                        best_regressor_type = model_type
                        best_regressor_pipeline = pipe_reg
                        best_regressor_metrics = {
                            "r2": r2,
                            "mae": mae,
                            "n_train": int(len(X_train)),
                            "n_test": int(len(X_test)),
                            "selected_features": getattr(pipe_reg.named_steps["preprocessing"], "selected_feature_names_", []),
                        }
                        
            except Exception as e:
                print(f"      ❌ Error in regression: {str(e)}")
            
            # Test CLASSIFICATION
            try:
                print(f"   📊 Classification task...")
                pipe_clf = build_pipeline(model_type=model_type, task_type="classification", k_best=k_best)
                pipe_clf.fit(X_train, y_train_clf)
                
                y_pred_clf = pipe_clf.predict(X_test) if len(X_test) else np.array([])
                if len(y_pred_clf) > 0:
                    accuracy = float(accuracy_score(y_test_clf, y_pred_clf))
                    
                    classifier_results[model_type] = {
                        "accuracy": accuracy,
                        "pipeline": pipe_clf
                    }
                    
                    print(f"      Accuracy: {accuracy:.4f}")
                    
                    if accuracy > best_classifier_score:
                        best_classifier_score = accuracy
                        best_classifier_type = model_type
                        best_classifier_pipeline = pipe_clf
                        best_classifier_metrics = {
                            "accuracy": accuracy,
                            "n_train": int(len(X_train)),
                            "n_test": int(len(X_test)),
                            "selected_features": getattr(pipe_clf.named_steps["preprocessing"], "selected_feature_names_", []),
                        }
                        
            except Exception as e:
                print(f"      ❌ Error in classification: {str(e)}")
        
        # Fallback to random_forest if no models worked
        if best_regressor_type is None:
            print("❌ No regressors trained successfully, falling back to random_forest")
            best_regressor_type = "random_forest"
            best_regressor_pipeline = build_pipeline(model_type=best_regressor_type, task_type="regression", k_best=k_best)
            best_regressor_pipeline.fit(X_train, y_train_reg)
            
        if best_classifier_type is None:
            print("❌ No classifiers trained successfully, falling back to random_forest")
            best_classifier_type = "random_forest"
            best_classifier_pipeline = build_pipeline(model_type=best_classifier_type, task_type="classification", k_best=k_best)
            best_classifier_pipeline.fit(X_train, y_train_clf)
            
    else:
        # Use only Random Forest for both tasks
        print("🎯 Training Random Forest models...")
        
        best_regressor_type = "random_forest"
        best_regressor_pipeline = build_pipeline(model_type=best_regressor_type, task_type="regression", k_best=k_best)
        best_regressor_pipeline.fit(X_train, y_train_reg)
        
        best_classifier_type = "random_forest"
        best_classifier_pipeline = build_pipeline(model_type=best_classifier_type, task_type="classification", k_best=k_best)
        best_classifier_pipeline.fit(X_train, y_train_clf)

    # Save both models
    regressor_path = os.path.join(model_dir, regressor_name)
    classifier_path = os.path.join(model_dir, classifier_name)
    
    dump(best_regressor_pipeline, regressor_path)
    dump(best_classifier_pipeline, classifier_path)
    
    print(f"\n✅ Models saved:")
    print(f"   📈 Regressor: {regressor_path}")
    print(f"   📊 Classifier: {classifier_path}")
    print(f"🏆 Best regressor: {best_regressor_type}")
    print(f"🏆 Best classifier: {best_classifier_type}")
    
    if best_regressor_metrics and best_regressor_metrics.get("r2") is not None:
        print(f"📊 Regressor R² Score: {best_regressor_metrics['r2']:.4f}")
        print(f"📊 Regressor MAE: {best_regressor_metrics['mae']:.4f}")
        
    if best_classifier_metrics and best_classifier_metrics.get("accuracy") is not None:
        print(f"📊 Classifier Accuracy: {best_classifier_metrics['accuracy']:.4f}")

    return {
        "regressor_path": regressor_path,
        "classifier_path": classifier_path,
        "best_regressor_type": best_regressor_type,
        "best_classifier_type": best_classifier_type,
        "regressor_metrics": best_regressor_metrics,
        "classifier_metrics": best_classifier_metrics,
    }
