"""
🚀 Unified Crypto Prediction Pipeline - Fixed Version
====================================================
Complete pipeline with proper 3-step structure:
1. Feature Engineering
2. Preprocessing  
3. Model Training
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

class CryptoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Feature Engineering Transformer"""
    
    def __init__(self):
        self.symbol_encoder = LabelEncoder()
        self.feature_names = None
        
    def fit(self, X, y=None):
        # Fit symbol encoder
        if 'symbol' in X.columns:
            self.symbol_encoder.fit(X['symbol'])
        return self
    
    def transform(self, X):
        """Transform raw data into engineered features"""
        X_copy = X.copy()
        
        # Create features in EXACT order - this is critical for consistency
        features_dict = {}
        
        # 1. First add original numerical features in fixed order
        base_features = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close']
        for col in base_features:
            if col in X_copy.columns:
                features_dict[col] = X_copy[col]
        
        # 2. Add existing engineered features if they exist (in fixed order)
        existing_engineered = ['hl_range', 'candle_body', 'upper_shadow', 'lower_shadow', 'body_to_range', 'return_1', 'log_return']
        for col in existing_engineered:
            if col in X_copy.columns:
                features_dict[col] = X_copy[col]
        
        # 3. Only calculate NEW features if they don't already exist
        if all(col in X_copy.columns for col in ['high', 'low', 'open', 'close']):
            # Technical indicators (only if not already present)
            if 'hl_range' not in features_dict:
                hl_range = X_copy['high'] - X_copy['low']
                features_dict['hl_range'] = hl_range
            else:
                hl_range = features_dict['hl_range']
                
            if 'candle_body' not in features_dict:
                features_dict['candle_body'] = abs(X_copy['close'] - X_copy['open'])
            if 'upper_shadow' not in features_dict:
                features_dict['upper_shadow'] = X_copy['high'] - np.maximum(X_copy['close'], X_copy['open'])
            if 'lower_shadow' not in features_dict:
                features_dict['lower_shadow'] = np.minimum(X_copy['close'], X_copy['open']) - X_copy['low']
            if 'body_to_range' not in features_dict:
                features_dict['body_to_range'] = features_dict['candle_body'] / features_dict['hl_range'].replace(0, 1)
            
            # Add return features (only if not present)
            if 'log_return' not in features_dict:
                prev_close = X_copy['close'] * 0.99  # Simulate previous close
                features_dict['log_return'] = np.log(X_copy['close'] / prev_close.replace(0, 1))
            
            # 4. Add price ratio features (always calculate)
            features_dict.update({
                'close_open_ratio': X_copy['close'] / X_copy['open'].replace(0, 1),
                'high_low_ratio': X_copy['high'] / X_copy['low'].replace(0, 1),
                'typical_price': (X_copy['high'] + X_copy['low'] + X_copy['close']) / 3,
                'hlc_avg': (X_copy['high'] + X_copy['low'] + X_copy['close']) / 3
            })
            
            # 5. Add time-based features
            current_time = datetime.now()
            features_dict.update({
                'hour': current_time.hour,
                'day': current_time.day,
                'month': current_time.month,
                'quarter': (current_time.month - 1) // 3 + 1,
                'weekday': current_time.weekday()
            })
            
            # 6. Add volume features
            features_dict.update({
                'volume_ratio': X_copy['volumefrom'] / X_copy['volumeto'].replace(0, 1),
                'volume_price_ratio': X_copy['volumefrom'] / X_copy['close'].replace(0, 1)
            })
            
            # 7. Add return_1 (if not already present)
            if 'return_1' not in features_dict:
                prev_close = X_copy['close'] * 0.99  # Simulate previous close
                features_dict['return_1'] = (X_copy['close'] - prev_close) / prev_close.replace(0, 1)
            
            # 8. Add additional features
            features_dict.update({
                'log_price': np.log(X_copy['close'].replace(0, 1)),
                'log_volume': np.log(X_copy['volumefrom'].replace(0, 1)),
                'price_volume': X_copy['close'] * X_copy['volumefrom'],
                'range_volume': features_dict['hl_range'] * X_copy['volumefrom']
            })
        
        # 9. Symbol encoding (always last)
        if 'symbol' in X_copy.columns:
            try:
                features_dict['symbol_encoded'] = self.symbol_encoder.transform(X_copy['symbol'])
            except ValueError as e:
                print(f"⚠️  Warning: {e}")
                known_symbols = self.symbol_encoder.classes_
                safe_symbols = X_copy['symbol'].map(lambda x: x if x in known_symbols else known_symbols[0])
                features_dict['symbol_encoded'] = self.symbol_encoder.transform(safe_symbols)
    
        # 10. Create DataFrame with FIXED column order to ensure consistency
        expected_order = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close', 
                         'hl_range', 'candle_body', 'upper_shadow', 'lower_shadow', 'body_to_range', 
                         'log_return', 'close_open_ratio', 'high_low_ratio', 'typical_price', 'hlc_avg', 
                         'hour', 'day', 'month', 'quarter', 'weekday', 'volume_ratio', 'volume_price_ratio', 
                         'return_1', 'log_price', 'log_volume', 'price_volume', 'range_volume', 'symbol_encoded']
        
        result_df = pd.DataFrame(features_dict)
        # Reorder columns to match expected order
        result_df = result_df.reindex(columns=[col for col in expected_order if col in result_df.columns])
        
        # Handle any infinite or NaN values
        result_df = result_df.replace([np.inf, -np.inf], 0)
        result_df = result_df.fillna(0)
        
        # Store feature names
        self.feature_names = result_df.columns.tolist()
        
        print(f"✅ Feature engineering: {X.shape} → {result_df.shape}")
        print(f"✅ Feature names: {list(result_df.columns)}")
        
        return result_df

class CryptoPreprocessor(BaseEstimator, TransformerMixin):
    """Custom Preprocessing Transformer"""
    
    def __init__(self, n_features=25):
        self.scaler = RobustScaler()
        self.selector = SelectKBest(score_func=f_regression, k=n_features)
        self.feature_names = None
        self.n_features = n_features
        
    def fit(self, X, y):
        # Adjust n_features based on available features
        n_available = X.shape[1]
        actual_k = min(self.n_features, n_available)
        
        print(f"✅ Preprocessing fit: {n_available} features available, selecting {actual_k}")
        
        # Recreate selector with correct k
        self.selector = SelectKBest(score_func=f_regression, k=actual_k)
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit feature selector
        X_selected = self.selector.fit_transform(X_scaled, y)
        
        # Store selected feature names
        if hasattr(X, 'columns'):
            selected_features = self.selector.get_support()
            self.feature_names = X.columns[selected_features].tolist()
            print(f"✅ Selected features: {self.feature_names}")
        
        return self
    
    def transform(self, X):
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Select features
        X_selected = self.selector.transform(X_scaled)
        
        print(f"✅ Preprocessing transform: {X.shape} → {X_selected.shape}")
        
        return X_selected

def create_complete_pipeline(data_path):
    """Create complete 3-step pipeline"""
    
    print("🚀 Creating Complete 3-Step Pipeline")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    data = pd.read_csv(data_path)
    print(f"✅ Data loaded: {data.shape}")
    
    # Create target
    if 'return_1' not in data.columns:
        print("🎯 Creating target variable...")
        prev_close = data['close'].shift(1).fillna(data['close'] * 0.99)
        data['return_1'] = (data['close'] - prev_close) / prev_close.replace(0, 1)
    
    # Use ALL available columns (not just basic ones)
    exclude_cols = ['return_1']  # Only exclude target
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols].copy()
    y = data['return_1'].copy()
    
    print(f"✅ Using {len(feature_cols)} feature columns: {feature_cols}")
    
    # Remove any infinite or NaN values
    mask = np.isfinite(y) & ~y.isna()
    X = X[mask]
    y = y[mask]
    
    print(f"✅ Clean data: {X.shape}")
    
    # Fix symbol encoding issue - use stratified split to ensure all symbols in both sets
    if 'symbol' in X.columns:
        # Get symbol distribution
        symbol_counts = X['symbol'].value_counts()
        print(f"📊 Symbol distribution: {symbol_counts.to_dict()}")
        
        # Use stratified split based on symbol
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=X['symbol'], shuffle=True  # Changed: stratify by symbol
        )
    else:
        # Regular split if no symbol column
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
    
    print(f"📚 Training set: {X_train.shape}")
    print(f"🧪 Test set: {X_test.shape}")
    
    # Check symbol distribution in both sets
    if 'symbol' in X.columns:
        train_symbols = set(X_train['symbol'].unique())
        test_symbols = set(X_test['symbol'].unique())
        print(f"🔍 Train symbols: {train_symbols}")
        print(f"🔍 Test symbols: {test_symbols}")
        print(f"✅ All symbols in both sets: {train_symbols == test_symbols}")
    
    # Create pipeline components with flexible settings
    feature_engineer = CryptoFeatureEngineer()
    n_features_to_select = min(15, X.shape[1] - 1)  # Reduced number
    preprocessor = CryptoPreprocessor(n_features=n_features_to_select)

    # Test different models
    models = {
        'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=8),
        'RandomForest': RandomForestRegressor(random_state=42, n_estimators=50, max_depth=8),
        'XGBoost': xgb.XGBRegressor(random_state=42, n_estimators=50, max_depth=6)
    }
    
    best_score = -float('inf')
    best_model_name = None
    best_model = None
    
    print("\n🎯 Training models...")
    
    for name, model in models.items():
        print(f"\n📈 Training {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('feature_engineering', feature_engineer),
            ('preprocessing', preprocessor),
            ('model', model)
        ])
        
        try:
            # Train pipeline
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            train_score = pipeline.score(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)
            
            print(f"   Train R²: {train_score:.4f}")
            print(f"   Test R²: {test_score:.4f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model_name = name
                best_model = model
                
        except Exception as e:
            print(f"   ❌ Error training {name}: {str(e)}")
            continue
    
    if best_model is None:
        raise ValueError("No model could be trained successfully!")
    
    # Create final pipeline with best model
    final_pipeline = Pipeline([
        ('feature_engineering', CryptoFeatureEngineer()),
        ('preprocessing', CryptoPreprocessor(n_features=min(25, X.shape[1]-1))),
        ('model', best_model)
    ])
    
    # Train final pipeline
    print(f"\n🏆 Training final pipeline with {best_model_name}...")
    final_pipeline.fit(X_train, y_train)
    
    # Final evaluation
    train_r2 = final_pipeline.score(X_train, y_train)
    test_r2 = final_pipeline.score(X_test, y_test)
    
    y_pred = final_pipeline.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n🎉 Final Results:")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📈 Train R²: {train_r2:.4f}")
    print(f"📊 Test R²: {test_r2:.4f}")
    print(f"📉 Test RMSE: {test_rmse:.4f}")
    
    return final_pipeline, {
        'best_model': best_model_name,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse
    }

def save_pipeline(pipeline, metadata, models_dir="models"):
    """Save pipeline as best_pipeline.pkl"""
    os.makedirs(models_dir, exist_ok=True)
    
    pipeline_path = f"{models_dir}/best_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    
    print(f"✅ Pipeline saved: {pipeline_path}")
    return pipeline_path

def run_unified_pipeline(data_path):
    """Run complete unified pipeline"""
    print("🚀 Starting Complete Unified Pipeline")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Create complete pipeline
        pipeline, metadata = create_complete_pipeline(data_path)
        
        # Save pipeline
        pipeline_path = save_pipeline(pipeline, metadata)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 Pipeline Completed Successfully!")
        print(f"⏱️  Total Time: {duration}")
        print(f"🏆 Best Model: {metadata['best_model']}")
        print(f"📁 Pipeline Saved: {pipeline_path}")
        print(f"📈 Test R²: {metadata['test_r2']:.4f}")
        print(f"📊 Test RMSE: {metadata['test_rmse']:.4f}")
        
        return True, {
            'pipeline_path': pipeline_path,
            'best_model': metadata['best_model'],
            'test_r2': metadata['test_r2'],
            'test_rmse': metadata['test_rmse']
        }
        
    except Exception as e:
        print(f"❌ Pipeline Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, str(e)

if __name__ == "__main__":
    data_path = "Data/processed/final_cleaned_crypto_zero_removed.csv"
    success, result = run_unified_pipeline(data_path)
    
    if success:
        print("✅ Pipeline completed successfully!")
    else:
        print(f"❌ Pipeline failed: {result}")
