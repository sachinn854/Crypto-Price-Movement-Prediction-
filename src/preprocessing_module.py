# preprocessing_module.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class CryptoPreprocessor(BaseEstimator, TransformerMixin):
    """
    FIXED: Handles both numerical and categorical features properly.
    Encodes string columns before scaling.
    """
    
    def __init__(self, k=15, task_type='regression'):
        self.k = k
        self.task_type = task_type
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_names_ = None
        self.categorical_columns_ = []
        self.numerical_columns_ = []
        
    def fit(self, X, y=None):
        """Fit the preprocessor - handle categorical and numerical features separately"""
        print(f"🔧 Fitting CryptoPreprocessor...")
        print(f"   Input shape: {X.shape}")
        print(f"   Columns: {list(X.columns)}")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Identify categorical and numerical columns
        self.categorical_columns_ = []
        self.numerical_columns_ = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                self.categorical_columns_.append(col)
            else:
                self.numerical_columns_.append(col)
        
        print(f"   Categorical columns: {self.categorical_columns_}")
        print(f"   Numerical columns: {len(self.numerical_columns_)} columns")
        
        # Encode categorical columns
        X_encoded = X.copy()
        for col in self.categorical_columns_:
            print(f"   Encoding {col}: {X[col].nunique()} unique values")
            self.label_encoders[col] = LabelEncoder()
            # Handle any NaN values in categorical columns
            X_encoded[col] = X_encoded[col].fillna('unknown')
            X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col].astype(str))
        
        # Handle NaN in numerical columns
        X_encoded[self.numerical_columns_] = X_encoded[self.numerical_columns_].fillna(0)
        
        # Ensure all columns are numeric
        for col in X_encoded.columns:
            X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
        
        print(f"   After encoding shape: {X_encoded.shape}")
        
        # Fit scaler on encoded data
        X_scaled = self.scaler.fit_transform(X_encoded)
        print(f"   After scaling shape: {X_scaled.shape}")
        
        # Fit feature selector
        if y is not None and len(X_scaled) > 0:
            # Choose scoring function based on task type
            if self.task_type == 'classification':
                scoring_func = f_classif
            else:
                scoring_func = f_regression
            
            # Ensure k doesn't exceed number of features
            k_actual = min(self.k, X_scaled.shape[1])
            print(f"   Feature selection: {X_scaled.shape[1]} → {k_actual} features")
            
            self.feature_selector = SelectKBest(score_func=scoring_func, k=k_actual)
            self.feature_selector.fit(X_scaled, y)
            
            # Store feature names for selected features
            feature_mask = self.feature_selector.get_support()
            self.feature_names_ = [col for i, col in enumerate(X_encoded.columns) if feature_mask[i]]
            print(f"   Selected features: {len(self.feature_names_)}")
        else:
            print("   No feature selection (no target provided)")
            self.feature_names_ = list(X_encoded.columns)
        
        print(f"✅ CryptoPreprocessor fitted successfully")
        return self
    
    def transform(self, X):
        """Transform the data using fitted encoders and scaler"""
        print(f"🔄 Transforming data...")
        print(f"   Input shape: {X.shape}")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Encode categorical columns using fitted encoders
        X_encoded = X.copy()
        for col in self.categorical_columns_:
            if col in X_encoded.columns:
                # Handle unseen categories
                X_encoded[col] = X_encoded[col].fillna('unknown')
                X_encoded[col] = X_encoded[col].astype(str)
                
                # Transform using fitted encoder, handle unseen values
                try:
                    X_encoded[col] = self.label_encoders[col].transform(X_encoded[col])
                except ValueError:
                    # Handle unseen categories by replacing with most frequent class
                    print(f"   Warning: Unseen categories in {col}, using fallback encoding")
                    known_classes = set(self.label_encoders[col].classes_)
                    X_encoded[col] = X_encoded[col].apply(
                        lambda x: x if x in known_classes else self.label_encoders[col].classes_[0]
                    )
                    X_encoded[col] = self.label_encoders[col].transform(X_encoded[col])
        
        # Handle NaN in numerical columns
        X_encoded[self.numerical_columns_] = X_encoded[self.numerical_columns_].fillna(0)
        
        # Ensure all columns are numeric
        for col in X_encoded.columns:
            X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
        
        # Scale the data
        X_scaled = self.scaler.transform(X_encoded)
        print(f"   After scaling shape: {X_scaled.shape}")
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
            print(f"   After feature selection shape: {X_selected.shape}")
            return X_selected
        else:
            return X_scaled
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Return the names of the selected features"""
        if self.feature_names_ is not None:
            return self.feature_names_
        else:
            return [f"feature_{i}" for i in range(self.k)]
