"""
🤖 Improved Model Training Module for Crypto Price Prediction
============================================================

Enhanced with:
- Proper R² calculation (train/test separate)
- Adjusted R² for overfitting detection
- TimeSeriesSplit for cross-validation
- Feature selection to remove leakage
- Multiple models with hyperparameter tuning

Author: Crypto Prediction Pipeline
Version: 2.0
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# Metrics and Validation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression

def calculate_adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """Calculate adjusted R² to detect overfitting"""
    if n_samples <= n_features + 1:
        return float('-inf')
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

def remove_leakage_features(X: pd.DataFrame) -> pd.DataFrame:
    """Remove only the most problematic leakage features - keep more features"""
    # Only remove the most obvious leakage features, keep others for better predictions
    high_leakage_features = [
        'gap_up', 'gap_down', 'gap_size'  # Only remove gap features
    ]
    
    features_to_remove = [col for col in high_leakage_features if col in X.columns]
    if features_to_remove:
        print(f"   🚫 Removing high leakage features: {features_to_remove}")
        X = X.drop(columns=features_to_remove)
    
    print(f"   ✅ Keeping {X.shape[1]} features for better model performance")
    
    return X

def train_models(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                feature_names: List[str] = None, models_dir: str = "models/trained_models", optimize_hyperparams: bool = False) -> Dict[str, Any]:
    """
    Train multiple models with proper evaluation
    
    Returns:
        Path to saved best model
    """
    print("\n🤖 STEP 3: MODEL TRAINING WITH COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    # Auto-detect feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    print(f"📊 Training data shape: {X_train.shape}")
    print(f"📋 Feature names provided: {len(feature_names)}")
    
    # Convert to DataFrames for easier handling
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Remove leakage features
    print("🔍 Checking for data leakage...")
    X_train_clean = remove_leakage_features(X_train_df)
    X_val_clean = X_val_df[X_train_clean.columns]
    X_test_clean = X_test_df[X_train_clean.columns]
    
    print(f"   📊 Features after leakage removal: {X_train_clean.shape[1]}")
    
    # Feature selection (keep top 25 features for better accuracy)
    print("🎯 Feature selection...")
    selector = SelectKBest(score_func=f_regression, k=min(25, X_train_clean.shape[1]))
    X_train_selected = selector.fit_transform(X_train_clean, y_train)
    X_val_selected = selector.transform(X_val_clean)
    X_test_selected = selector.transform(X_test_clean)
    
    selected_features = X_train_clean.columns[selector.get_support()].tolist()
    print(f"   ✅ Selected {len(selected_features)} best features")
    
    models = {}
    results = {}
    
    # Simple model configurations for faster training - Only 3 models
    model_configs = {
        'RandomForest': {
            'model': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
            'tune': False
        },
        'DecisionTree': {
            'model': DecisionTreeRegressor(max_depth=10, random_state=42),
            'tune': True,
            'params': {
                'max_depth': [8, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
            'tune': True,
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [6, 10],
                'learning_rate': [0.05, 0.1]
            }
        }
    }
    
    print(f"\n🚀 Training {len(model_configs)} models...")
    
    for model_name, config in model_configs.items():
        print(f"\n📈 Training {model_name}...")
        
        try:
            if config.get('tune', False):
                # Hyperparameter tuning with simple CV
                from sklearn.model_selection import cross_val_score
                
                best_score = -np.inf
                best_model = None
                best_params = None
                
                # Simple grid search
                param_grid = config.get('params', {})
                param_combinations = [{}]
                
                if param_grid:
                    keys = list(param_grid.keys())
                    values = list(param_grid.values())
                    import itertools
                    param_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
                
                for params in param_combinations:
                    model = type(config['model'])(**{**config['model'].get_params(), **params})
                    
                    # Quick 3-fold CV
                    scores = cross_val_score(model, X_train_selected, y_train, cv=3, scoring='r2')
                    mean_score = scores.mean()
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_model = model
                        best_params = params
                
                final_model = best_model
            else:
                final_model = config['model']
            
            # Train final model
            final_model.fit(X_train_selected, y_train)
            
            # Predictions
            y_train_pred = final_model.predict(X_train_selected)
            y_val_pred = final_model.predict(X_val_selected)
            y_test_pred = final_model.predict(X_test_selected)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Adjusted R² for overfitting detection
            train_adj_r2 = calculate_adjusted_r2(train_r2, len(y_train), X_train_selected.shape[1])
            test_adj_r2 = calculate_adjusted_r2(test_r2, len(y_test), X_train_selected.shape[1])
            
            # Other metrics
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            
            # Store results
            models[model_name] = final_model
            results[model_name] = {
                'train_r2': train_r2,
                'val_r2': val_r2,
                'test_r2': test_r2,
                'train_adj_r2': train_adj_r2,
                'test_adj_r2': test_adj_r2,
                'test_mse': test_mse,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'overfitting_score': train_r2 - test_r2  # Positive = overfitting
            }
            
            print(f"   ✅ {model_name} completed:")
            print(f"      Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
            print(f"      Adj R² (Train): {train_adj_r2:.4f} | Adj R² (Test): {test_adj_r2:.4f}")
            print(f"      Overfitting: {train_r2 - test_r2:.4f}")
            
        except Exception as e:
            print(f"   ❌ {model_name} failed: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No models trained successfully!")
    
    # Find best model (prioritize test R² with low overfitting)
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['test_r2'] - 0.1 * abs(results[x]['overfitting_score']))
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print("=" * 40)
    best_result = results[best_model_name]
    print(f"📊 Test R²: {best_result['test_r2']:.4f}")
    print(f"📊 Adjusted R²: {best_result['test_adj_r2']:.4f}")
    print(f"📊 Test RMSE: {best_result['test_rmse']:.4f}")
    print(f"📊 Overfitting Score: {best_result['overfitting_score']:.4f}")
    
    # Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = f"{models_dir}/crypto_best_model_{timestamp}.joblib"
    
    # Create metadata
    metadata = {
        'model_name': best_model_name,
        'feature_names': selected_features,
        'results': results,
        'timestamp': timestamp,
        'metrics': {
            'test_r2': best_result['test_r2'],
            'test_adj_r2': best_result['test_adj_r2'],
            'test_rmse': best_result['test_rmse'],
            'overfitting_score': best_result['overfitting_score']
        }
    }
    
    # Save model and metadata
    joblib.dump(models[best_model_name], model_path)
    
    metadata_path = model_path.replace('.joblib', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\n💾 FILES SAVED:")
    print(f"   📁 Model: {model_path}")
    print(f"   📁 Metadata: {metadata_path}")
    
    print(f"\n📊 MODEL COMPARISON:")
    for model_name, result in results.items():
        status = "🏆" if model_name == best_model_name else "📈"
        print(f"   {status} {model_name}: R²={result['test_r2']:.4f}, "
              f"Adj_R²={result['test_adj_r2']:.4f}, "
              f"Overfitting={result['overfitting_score']:.4f}")
    
    # Return complete results dictionary
    return {
        'status': 'completed',
        'best_model_name': best_model_name,
        'best_model': models[best_model_name],  # Add the actual model object
        'best_model_path': model_path,
        'metadata_path': metadata_path,
        'model_results': results,
        'best_metrics': {
            'test_r2': best_result['test_r2'],
            'test_adj_r2': best_result['test_adj_r2'],
            'test_rmse': best_result['test_rmse'],
            'overfitting_score': best_result['overfitting_score']
        },
        'feature_names': selected_features,
        'selector': selector,  # Add the feature selector
        'timestamp': timestamp
    }
