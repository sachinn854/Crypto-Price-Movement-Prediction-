"""
⚙️ Preprocessing Module for Crypto Price Prediction
==================================================

This module handles data preprocessing including:
- Train/validation/test splits with time awareness
- Feature scaling and normalization
- Symbol encoding for multi-crypto modeling
- Handling categorical variables
- Data balancing for classification tasks

Author: Crypto Prediction Pipeline
Version: 1.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class CryptoPreprocessor:
    """
    Advanced preprocessing for cryptocurrency prediction models
    """
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
        """
        Initialize preprocessor with parameters
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of remaining data for validation
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Scalers and encoders
        self.feature_scaler = RobustScaler()  # Robust to outliers
        self.symbol_encoder = LabelEncoder()
        
        # Store preprocessing info
        self.feature_columns = []
        self.categorical_columns = []
        self.numeric_columns = []
        self.preprocessing_info = {}
        
    def time_aware_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data with time awareness to prevent data leakage
        """
        print("📅 Creating time-aware train/validation/test splits...")
        
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(['symbol', 'time'])
        
        # Split by time for each symbol to maintain temporal order
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].reset_index(drop=True)
            
            # Calculate split points
            n = len(symbol_data)
            test_split = int(n * (1 - self.test_size))
            val_split = int(test_split * (1 - self.val_size))
            
            # Split data
            train_data = symbol_data.iloc[:val_split]
            val_data = symbol_data.iloc[val_split:test_split]
            test_data = symbol_data.iloc[test_split:]
            
            train_dfs.append(train_data)
            val_dfs.append(val_data)
            test_dfs.append(test_data)
            
            print(f"   {symbol}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Combine all symbols
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        print(f"\n📊 FINAL SPLITS:")
        print(f"   Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   Validation: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"   Test: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def identify_feature_types(self, df: pd.DataFrame) -> None:
        """
        Identify different types of features for appropriate preprocessing
        """
        print("🔍 Identifying feature types...")
        
        # Exclude target and identifier columns
        exclude_cols = ['time', 'symbol', 'target']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Identify categorical and numeric columns
        self.categorical_columns = []
        self.numeric_columns = []
        
        for col in self.feature_columns:
            if df[col].dtype in ['object', 'category']:
                self.categorical_columns.append(col)
            else:
                self.numeric_columns.append(col)
        
        print(f"   📊 Total features: {len(self.feature_columns)}")
        print(f"   🔢 Numeric features: {len(self.numeric_columns)}")
        print(f"   📝 Categorical features: {len(self.categorical_columns)}")
        
        # Store info
        self.preprocessing_info['total_features'] = len(self.feature_columns)
        self.preprocessing_info['numeric_features'] = len(self.numeric_columns)
        self.preprocessing_info['categorical_features'] = len(self.categorical_columns)
    
    def encode_symbols(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Encode cryptocurrency symbols for modeling
        """
        print("🔤 Encoding cryptocurrency symbols...")
        
        # Fit encoder on training data
        self.symbol_encoder.fit(train_df['symbol'])
        
        # Transform all datasets with unseen label handling
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        # Handle unseen labels in test/val sets
        def safe_transform(df, encoder, column_name):
            """Safely transform labels, handling unseen labels"""
            labels = df[column_name].values
            known_labels = set(encoder.classes_)
            
            # Replace unseen labels with the first known label (or most frequent)
            safe_labels = []
            for label in labels:
                if label in known_labels:
                    safe_labels.append(label)
                else:
                    print(f"   ⚠️ Warning: Unseen label '{label}' replaced with '{encoder.classes_[0]}'")
                    safe_labels.append(encoder.classes_[0])
            
            return encoder.transform(safe_labels)
        
        train_df['symbol_encoded'] = self.symbol_encoder.transform(train_df['symbol'])
        val_df['symbol_encoded'] = safe_transform(val_df, self.symbol_encoder, 'symbol')
        test_df['symbol_encoded'] = safe_transform(test_df, self.symbol_encoder, 'symbol')
        
        # Add symbol_encoded to feature columns
        if 'symbol_encoded' not in self.feature_columns:
            self.feature_columns.append('symbol_encoded')
            self.numeric_columns.append('symbol_encoded')
        
        print(f"   ✅ Encoded {len(self.symbol_encoder.classes_)} symbols: {list(self.symbol_encoder.classes_)}")
        
        return train_df, val_df, test_df
    
    def scale_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale numeric features using RobustScaler
        """
        print("⚖️ Scaling numeric features...")
        
        # Fit scaler on training data only
        train_features = train_df[self.numeric_columns]
        self.feature_scaler.fit(train_features)
        
        # Transform all datasets
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_df[self.numeric_columns] = self.feature_scaler.transform(train_df[self.numeric_columns])
        val_df[self.numeric_columns] = self.feature_scaler.transform(val_df[self.numeric_columns])
        test_df[self.numeric_columns] = self.feature_scaler.transform(test_df[self.numeric_columns])
        
        print(f"   ✅ Scaled {len(self.numeric_columns)} numeric features")
        
        return train_df, val_df, test_df
    
    def handle_categorical_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Handle categorical features (if any)
        """
        if not self.categorical_columns:
            print("✅ No categorical features to process")
            return train_df, val_df, test_df
        
        print(f"🏷️ Processing {len(self.categorical_columns)} categorical features...")
        
        # For simplicity, we'll use label encoding for categorical features
        # In production, you might want to use one-hot encoding or target encoding
        
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        for col in self.categorical_columns:
            encoder = LabelEncoder()
            
            # Fit on training data
            encoder.fit(train_df[col].astype(str))
            
            # Transform all datasets
            train_df[col] = encoder.transform(train_df[col].astype(str))
            val_df[col] = encoder.transform(val_df[col].astype(str))
            test_df[col] = encoder.transform(test_df[col].astype(str))
            
            print(f"   ✅ Encoded {col}: {len(encoder.classes_)} categories")
        
        return train_df, val_df, test_df
    
    def check_class_balance(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check class balance for classification tasks
        """
        # Find target column
        possible_targets = ['target', 'return_1', 'log_return', 'close']
        target_col = None
        
        for col in possible_targets:
            if col in train_df.columns:
                target_col = col
                break
        
        if target_col is None:
            return {}
        
        print("⚖️ Checking target distribution...")
        
        if train_df[target_col].dtype in ['int64', 'int32', 'bool']:
            # Classification target
            class_counts = train_df[target_col].value_counts().sort_index()
            total_samples = len(train_df)
            
            balance_info = {}
            print(f"   Target distribution in training data:")
            
            for class_label, count in class_counts.items():
                percentage = (count / total_samples) * 100
                balance_info[class_label] = {'count': count, 'percentage': percentage}
                print(f"      Class {class_label}: {count:,} samples ({percentage:.1f}%)")
        else:
            # Regression target
            target_stats = train_df[target_col].describe()
            print(f"   Target statistics ({target_col}):")
            print(f"      Mean: {target_stats['mean']:.6f}")
            print(f"      Std: {target_stats['std']:.6f}")
            print(f"      Min: {target_stats['min']:.6f}")
            print(f"      Max: {target_stats['max']:.6f}")
            
            balance_info = {
                'mean': target_stats['mean'],
                'std': target_stats['std'],
                'min': target_stats['min'],
                'max': target_stats['max'],
                'type': 'regression'
            }
        
        return balance_info
        
        if is_balanced:
            print("   ✅ Classes are reasonably balanced")
        else:
            print("   ⚠️ Classes are imbalanced - consider rebalancing techniques")
        
        balance_info['is_balanced'] = is_balanced
        return balance_info
    
    def prepare_model_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare final arrays for model training
        """
        print("🎯 Preparing final model data...")
        
        # Find target column
        possible_targets = ['target', 'return_1', 'log_return', 'close']
        target_col = None
        
        for col in possible_targets:
            if col in train_df.columns:
                target_col = col
                print(f"   🎯 Using '{col}' as target variable")
                break
        
        if target_col is None:
            raise ValueError(f"No target column found. Available columns: {list(train_df.columns)}")
        
        # Extract features and targets
        X_train = train_df[self.feature_columns].values
        y_train = train_df[target_col].values
        
        X_val = val_df[self.feature_columns].values
        y_val = val_df[target_col].values
        
        X_test = test_df[self.feature_columns].values
        y_test = test_df[target_col].values
        
        print(f"   📊 Training features shape: {X_train.shape}")
        print(f"   📊 Validation features shape: {X_val.shape}")
        print(f"   📊 Test features shape: {X_test.shape}")
        
        # Check for any remaining NaN or inf values
        def check_arrays(name, X, y):
            nan_count = np.isnan(X).sum() + np.isnan(y).sum()
            inf_count = np.isinf(X).sum() + np.isinf(y).sum()
            
            if nan_count > 0:
                print(f"   ⚠️ {name}: {nan_count} NaN values found")
            if inf_count > 0:
                print(f"   ⚠️ {name}: {inf_count} Inf values found")
            
            if nan_count == 0 and inf_count == 0:
                print(f"   ✅ {name}: Clean (no NaN/Inf)")
        
        check_arrays("Training", X_train, y_train)
        check_arrays("Validation", X_val, y_val)
        check_arrays("Test", X_test, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def run_preprocessing(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Run complete preprocessing pipeline
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_info
        """
        print("🚀 STARTING PREPROCESSING PIPELINE")
        print("=" * 50)
        
        print(f"Input data shape: {df.shape}")
        
        # Step 1: Time-aware split
        train_df, val_df, test_df = self.time_aware_split(df)
        
        # Step 2: Identify feature types
        self.identify_feature_types(df)
        
        # Step 3: Encode symbols
        train_df, val_df, test_df = self.encode_symbols(train_df, val_df, test_df)
        
        # Step 4: Handle categorical features
        train_df, val_df, test_df = self.handle_categorical_features(train_df, val_df, test_df)
        
        # Step 5: Scale features
        train_df, val_df, test_df = self.scale_features(train_df, val_df, test_df)
        
        # Step 6: Check class balance
        balance_info = self.check_class_balance(train_df)
        
        # Step 7: Prepare final model data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_model_data(train_df, val_df, test_df)
        
        # Store all preprocessing information
        self.preprocessing_info.update({
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'feature_names': self.feature_columns,
            'class_balance': balance_info,
            'symbols': list(self.symbol_encoder.classes_) if hasattr(self.symbol_encoder, 'classes_') else [],
        })
        
        print(f"\n📊 PREPROCESSING COMPLETED!")
        print(f"   Total features for modeling: {len(self.feature_columns)}")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        print(f"   Test samples: {len(X_test):,}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, self.preprocessing_info
    
    def save_preprocessing_objects(self, save_dir: str) -> None:
        """
        Save preprocessing objects for later use
        """
        import joblib
        import os
        
        print(f"💾 Saving preprocessing objects to: {save_dir}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save scalers and encoders
        joblib.dump(self.feature_scaler, os.path.join(save_dir, 'feature_scaler.pkl'))
        joblib.dump(self.symbol_encoder, os.path.join(save_dir, 'symbol_encoder.pkl'))
        
        # Save preprocessing info
        import json
        info_to_save = self.preprocessing_info.copy()
        
        # Convert any numpy arrays to lists for JSON serialization
        for key, value in info_to_save.items():
            if isinstance(value, np.ndarray):
                info_to_save[key] = value.tolist()
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        info_to_save[key][subkey] = subvalue.tolist()
        
        with open(os.path.join(save_dir, 'preprocessing_info.json'), 'w') as f:
            json.dump(info_to_save, f, indent=2)
        
        print(f"   ✅ Saved preprocessing objects successfully")

def preprocess_data(input_path: str, save_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Main function to preprocess featured data for modeling
    
    Args:
        input_path: Path to featured data
        save_dir: Directory to save preprocessing objects
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_info
    """
    print("⚙️ CRYPTO DATA PREPROCESSING")
    print("=" * 40)
    
    # Load featured data
    print(f"📂 Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Loaded shape: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = CryptoPreprocessor()
    
    # Run preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_info = preprocessor.run_preprocessing(df)
    
    # Save preprocessing objects
    preprocessor.save_preprocessing_objects(save_dir)
    
    print(f"\n🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_info

if __name__ == "__main__":
    # Test preprocessing
    input_path = "../Data/processed/crypto_featured_data.csv"
    save_dir = "../models/preprocessing"
    
    X_train, X_val, X_test, y_train, y_val, y_test, info = preprocess_data(input_path, save_dir)
