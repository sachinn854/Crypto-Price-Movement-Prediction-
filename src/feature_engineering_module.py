"""
🔧 Feature Engineering Module for Crypto Price Prediction
=======================================================

This module creates advanced features from the cleaned crypto data including:
- Technical indicators (RSI, SMA, EMA, Bollinger Bands)
- Price patterns and momentum indicators
- Time-based features
- Statistical features
- Lag features for sequence learning

Author: Crypto Prediction Pipeline
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class CryptoFeatureEngineering:
    """
    Advanced feature engineering for cryptocurrency price prediction
    """
    
    def __init__(self, lookback_window: int = 24):
        """
        Initialize feature engineering with parameters
        
        Args:
            lookback_window: Hours to look back for technical indicators
        """
        self.lookback_window = lookback_window
        self.feature_names = []
        
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical analysis indicators
        """
        print("🔧 Creating technical indicators...")
        
        df = df.copy()
        
        # Group by symbol for proper calculation
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            # Sort by time
            symbol_data = symbol_data.sort_values('time')
            
            # 1. Moving Averages
            symbol_data['sma_5'] = symbol_data['close'].rolling(window=5, min_periods=1).mean()
            symbol_data['sma_10'] = symbol_data['close'].rolling(window=10, min_periods=1).mean()
            symbol_data['sma_20'] = symbol_data['close'].rolling(window=20, min_periods=1).mean()
            
            symbol_data['ema_5'] = symbol_data['close'].ewm(span=5, min_periods=1).mean()
            symbol_data['ema_10'] = symbol_data['close'].ewm(span=10, min_periods=1).mean()
            symbol_data['ema_20'] = symbol_data['close'].ewm(span=20, min_periods=1).mean()
            
            # 2. RSI (Relative Strength Index)
            delta = symbol_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            symbol_data['rsi'] = 100 - (100 / (1 + rs))
            
            # 3. Bollinger Bands
            symbol_data['bb_middle'] = symbol_data['close'].rolling(window=20, min_periods=1).mean()
            bb_std = symbol_data['close'].rolling(window=20, min_periods=1).std()
            symbol_data['bb_upper'] = symbol_data['bb_middle'] + (bb_std * 2)
            symbol_data['bb_lower'] = symbol_data['bb_middle'] - (bb_std * 2)
            symbol_data['bb_width'] = symbol_data['bb_upper'] - symbol_data['bb_lower']
            symbol_data['bb_position'] = (symbol_data['close'] - symbol_data['bb_lower']) / (symbol_data['bb_width'] + 1e-10)
            
            # 4. MACD
            ema_12 = symbol_data['close'].ewm(span=12, min_periods=1).mean()
            ema_26 = symbol_data['close'].ewm(span=26, min_periods=1).mean()
            symbol_data['macd'] = ema_12 - ema_26
            symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9, min_periods=1).mean()
            symbol_data['macd_histogram'] = symbol_data['macd'] - symbol_data['macd_signal']
            
            # 5. Stochastic Oscillator
            low_14 = symbol_data['low'].rolling(window=14, min_periods=1).min()
            high_14 = symbol_data['high'].rolling(window=14, min_periods=1).max()
            symbol_data['stoch_k'] = 100 * ((symbol_data['close'] - low_14) / (high_14 - low_14 + 1e-10))
            symbol_data['stoch_d'] = symbol_data['stoch_k'].rolling(window=3, min_periods=1).mean()
            
            # Update main dataframe
            df.loc[mask] = symbol_data
            
        print(f"   ✅ Created technical indicators for {len(df['symbol'].unique())} symbols")
        return df
    
    def create_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price pattern features
        """
        print("📈 Creating price pattern features...")
        
        df = df.copy()
        
        # 1. Price position within range
        df['price_position'] = (df['close'] - df['low']) / (df['hl_range'] + 1e-10)
        
        # 2. Gap analysis
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        df['gap_size'] = df['open'] - df['close'].shift(1)
        
        # 3. Doji patterns (open ≈ close)
        df['is_doji'] = (abs(df['candle_body']) / (df['hl_range'] + 1e-10) < 0.1).astype(int)
        
        # 4. Hammer/Hanging man patterns
        df['is_hammer'] = ((df['lower_shadow'] > 2 * df['candle_body']) & 
                          (df['upper_shadow'] < df['candle_body'])).astype(int)
        
        # 5. Volume patterns
        df['volume_ratio'] = df['volumefrom'] / (df['volumefrom'].rolling(window=20, min_periods=1).mean() + 1e-10)
        df['volume_price_trend'] = df['volumefrom'] * df['return_1']
        
        print(f"   ✅ Created price pattern features")
        return df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum and trend features
        """
        print("⚡ Creating momentum features...")
        
        df = df.copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy().sort_values('time')
            
            # 1. Price momentum (rate of change)
            symbol_data['momentum_3'] = symbol_data['close'].pct_change(periods=3)
            symbol_data['momentum_5'] = symbol_data['close'].pct_change(periods=5)
            symbol_data['momentum_10'] = symbol_data['close'].pct_change(periods=10)
            
            # 2. Acceleration (change in momentum)
            symbol_data['acceleration'] = symbol_data['return_1'].diff()
            
            # 3. Volatility measures
            symbol_data['volatility_5'] = symbol_data['return_1'].rolling(window=5, min_periods=1).std()
            symbol_data['volatility_10'] = symbol_data['return_1'].rolling(window=10, min_periods=1).std()
            symbol_data['volatility_20'] = symbol_data['return_1'].rolling(window=20, min_periods=1).std()
            
            # 4. Trend strength
            symbol_data['trend_strength'] = abs(symbol_data['close'].rolling(window=10, min_periods=1).apply(
                lambda x: np.corrcoef(np.arange(len(x)), x)[0,1] if len(x) > 1 else 0
            ))
            
            # 5. Price efficiency ratio
            symbol_data['efficiency_ratio'] = abs(symbol_data['close'] - symbol_data['close'].shift(10)) / (
                symbol_data['hl_range'].rolling(window=10, min_periods=1).sum() + 1e-10
            )
            
            df.loc[mask] = symbol_data
            
        print(f"   ✅ Created momentum features")
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        """
        print("🕐 Creating time-based features...")
        
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        
        # Time features
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['day_of_month'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['quarter'] = df['time'].dt.quarter
        df['year'] = df['time'].dt.year
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Market timing features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        df['is_quarter_end'] = (df['month'] % 3 == 0).astype(int)
        
        print(f"   ✅ Created time-based features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lag_periods: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create lag features for key indicators
        """
        print(f"🔄 Creating lag features for periods: {lag_periods}...")
        
        df = df.copy()
        
        # Key features to create lags for
        key_features = ['close', 'return_1', 'log_return', 'volumefrom', 'rsi', 'macd']
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy().sort_values('time')
            
            for feature in key_features:
                if feature in symbol_data.columns:
                    for lag in lag_periods:
                        col_name = f'{feature}_lag_{lag}'
                        symbol_data[col_name] = symbol_data[feature].shift(lag)
            
            df.loc[mask] = symbol_data
        
        print(f"   ✅ Created lag features for {len(key_features)} indicators")
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical aggregation features
        """
        print("📊 Creating statistical features...")
        
        df = df.copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy().sort_values('time')
            
            # Rolling statistics for different windows
            for window in [5, 10, 20]:
                # Price statistics
                symbol_data[f'close_mean_{window}'] = symbol_data['close'].rolling(window=window, min_periods=1).mean()
                symbol_data[f'close_std_{window}'] = symbol_data['close'].rolling(window=window, min_periods=1).std()
                symbol_data[f'close_min_{window}'] = symbol_data['close'].rolling(window=window, min_periods=1).min()
                symbol_data[f'close_max_{window}'] = symbol_data['close'].rolling(window=window, min_periods=1).max()
                
                # Return statistics
                symbol_data[f'return_mean_{window}'] = symbol_data['return_1'].rolling(window=window, min_periods=1).mean()
                symbol_data[f'return_std_{window}'] = symbol_data['return_1'].rolling(window=window, min_periods=1).std()
                
                # Volume statistics
                symbol_data[f'volume_mean_{window}'] = symbol_data['volumefrom'].rolling(window=window, min_periods=1).mean()
                symbol_data[f'volume_std_{window}'] = symbol_data['volumefrom'].rolling(window=window, min_periods=1).std()
            
            df.loc[mask] = symbol_data
        
        print(f"   ✅ Created statistical features")
        return df
    
    def create_target_variable(self, df: pd.DataFrame, target_type: str = 'close') -> pd.DataFrame:
        """
        Create target variable for prediction
        
        Args:
            target_type: 'close' for price prediction, 'return' for return prediction, 'log_return' for log returns
        """
        print(f"🎯 Creating target variable: {target_type}...")
        
        df = df.copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy().sort_values('time')
            
            if target_type == 'close':
                # Predict next hour close price
                symbol_data['target'] = symbol_data['close'].shift(-1)
            elif target_type == 'return':
                # Predict next hour return (more stable for ML)
                symbol_data['target'] = symbol_data['close'].pct_change(periods=-1)
            elif target_type == 'log_return':
                # Predict next hour log return (stationary)
                symbol_data['target'] = np.log(symbol_data['close']).diff(periods=-1)
            else:
                raise ValueError("target_type must be 'close', 'return', or 'log_return'")
            
            df.loc[mask] = symbol_data
        
        # Remove last row for each symbol (no future data available)
        df = df.groupby('symbol').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        
        print(f"   ✅ Created target variable: {target_type}")
        return df
    
    def run_feature_engineering(self, df: pd.DataFrame, target_type: str = 'direction') -> pd.DataFrame:
        """
        Run complete feature engineering pipeline
        """
        print("🚀 STARTING FEATURE ENGINEERING PIPELINE")
        print("=" * 50)
        
        print(f"Input data shape: {df.shape}")
        
        # Step 1: Technical indicators
        df = self.create_technical_indicators(df)
        
        # Step 2: Price patterns
        df = self.create_price_patterns(df)
        
        # Step 3: Momentum features
        df = self.create_momentum_features(df)
        
        # Step 4: Time features
        df = self.create_time_features(df)
        
        # Step 5: Lag features
        df = self.create_lag_features(df)
        
        # Step 6: Statistical features
        df = self.create_statistical_features(df)
        
        # Step 7: Target variable
        df = self.create_target_variable(df, target_type)
        
        # Remove rows with missing values (due to lag features)
        initial_shape = df.shape
        df = df.dropna().reset_index(drop=True)
        final_shape = df.shape
        
        print(f"\n📊 FEATURE ENGINEERING COMPLETED!")
        print(f"   Initial shape: {initial_shape}")
        print(f"   Final shape: {final_shape}")
        print(f"   Features created: {final_shape[1] - 16} new features")
        print(f"   Data retention: {(final_shape[0]/initial_shape[0])*100:.1f}%")
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['time', 'symbol', 'target']]
        
        print(f"   Total features for modeling: {len(self.feature_names)}")
        
        return df

def create_features(input_path: str, output_path: str, target_type: str = 'direction') -> str:
    """
    Main function to create features from processed data
    
    Args:
        input_path: Path to processed/cleaned data
        output_path: Path to save featured data
        target_type: Type of target variable ('direction' or 'return')
    
    Returns:
        Path to saved featured data
    """
    print("🔧 CRYPTO FEATURE ENGINEERING")
    print("=" * 40)
    
    # Load processed data
    print(f"📂 Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Loaded shape: {df.shape}")
    
    # Initialize feature engineering
    feature_engineer = CryptoFeatureEngineering()
    
    # Run feature engineering
    df_featured = feature_engineer.run_feature_engineering(df, target_type)
    
    # Save featured data
    print(f"\n💾 Saving featured data to: {output_path}")
    df_featured.to_csv(output_path, index=False)
    
    # Create feature info file
    info_path = output_path.replace('.csv', '_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"FEATURE ENGINEERING SUMMARY\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Input shape: {df.shape}\n")
        f.write(f"Output shape: {df_featured.shape}\n")
        f.write(f"Target type: {target_type}\n")
        f.write(f"Features created: {df_featured.shape[1] - df.shape[1]}\n\n")
        
        f.write(f"ALL FEATURES ({len(feature_engineer.feature_names)}):\n")
        for i, feature in enumerate(feature_engineer.feature_names, 1):
            f.write(f"  {i:3d}. {feature}\n")
    
    print(f"   ✅ Feature info saved to: {info_path}")
    print(f"\n🎉 FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    
    return output_path

if __name__ == "__main__":
    # Test feature engineering
    input_path = "../Data/processed/final_cleaned_crypto_zero_removed.csv"
    output_path = "../Data/processed/crypto_featured_data.csv"
    
    create_features(input_path, output_path, target_type='direction')
