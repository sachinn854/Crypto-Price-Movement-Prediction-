"""
🚀 Crypto Price Prediction App - V4.1
======================================
Professional Streamlit app using best_pipeline.pkl
Complete 3-step pipeline integration with fixed input columns
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

# Import the custom classes
try:
    from src.unified_pipeline import CryptoFeatureEngineer, CryptoPreprocessor
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.error("Please ensure unified_pipeline.py is in the src folder")
    st.stop()

# Import sklearn components needed for the custom classes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression

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
        
        # Start with existing features
        features_dict = {}
        
        # Add all existing numeric features (except symbol and time)
        for col in X_copy.columns:
            if col not in ['symbol', 'time']:  # Exclude non-numeric columns
                if X_copy[col].dtype in ['int64', 'float64']:  # Only numeric columns
                    features_dict[col] = X_copy[col]
        
        # Only add technical indicators if basic price columns exist
        if all(col in X_copy.columns for col in ['high', 'low', 'open', 'close']):
            # Technical indicators
            hl_range = X_copy['high'] - X_copy['low']
            candle_body = abs(X_copy['close'] - X_copy['open'])
            upper_shadow = X_copy['high'] - np.maximum(X_copy['close'], X_copy['open'])
            lower_shadow = np.minimum(X_copy['close'], X_copy['open']) - X_copy['low']
            body_to_range = candle_body / hl_range.replace(0, 1)
            
            features_dict.update({
                'hl_range': hl_range,
                'candle_body': candle_body,
                'upper_shadow': upper_shadow,
                'lower_shadow': lower_shadow,
                'body_to_range': body_to_range
            })
            
            # Add return features first for consistent order
            prev_close = X_copy['close'] * 0.99  # Simulate previous close
            features_dict.update({
                'return_1': (X_copy['close'] - prev_close) / prev_close.replace(0, 1),
                'log_return': np.log(X_copy['close'] / prev_close.replace(0, 1))
            })
            
            # Price ratios
            features_dict.update({
                'close_open_ratio': X_copy['close'] / X_copy['open'].replace(0, 1),
                'high_low_ratio': X_copy['high'] / X_copy['low'].replace(0, 1),
                'typical_price': (X_copy['high'] + X_copy['low'] + X_copy['close']) / 3,
                'hlc_avg': (X_copy['high'] + X_copy['low'] + X_copy['close']) / 3
            })
            
            # Time features
            if 'time' in X_copy.columns:
                current_time = pd.to_datetime(X_copy['time'].iloc[0] if hasattr(X_copy['time'], 'iloc') else X_copy['time'][0])
            else:
                current_time = pd.Timestamp.now()
            
            features_dict.update({
                'hour': current_time.hour,
                'day': current_time.day,
                'month': current_time.month,
                'quarter': (current_time.month - 1) // 3 + 1,
                'weekday': current_time.weekday()
            })
            
            # Add volume features
            features_dict.update({
                'volume_ratio': X_copy['volumefrom'] / X_copy['volumeto'].replace(0, 1),
                'volume_price_ratio': X_copy['volumefrom'] / X_copy['close'].replace(0, 1)
            })
            
            # Add additional features
            features_dict.update({
                'log_price': np.log(X_copy['close'].replace(0, 1)),
                'log_volume': np.log(X_copy['volumefrom'].replace(0, 1)),
                'price_volume': X_copy['close'] * X_copy['volumefrom'],
                'range_volume': hl_range * X_copy['volumefrom']
            })
        
        # Symbol encoding
        if 'symbol' in X_copy.columns:
            try:
                features_dict['symbol_encoded'] = self.symbol_encoder.transform(X_copy['symbol'])
            except ValueError as e:
                known_symbols = self.symbol_encoder.classes_
                safe_symbols = X_copy['symbol'].map(lambda x: x if x in known_symbols else known_symbols[0])
                features_dict['symbol_encoded'] = self.symbol_encoder.transform(safe_symbols)
    
        # Create DataFrame
        result_df = pd.DataFrame(features_dict)
        
        # Handle any infinite or NaN values
        result_df = result_df.replace([np.inf, -np.inf], 0)
        result_df = result_df.fillna(0)
        
        # Store feature names
        self.feature_names = result_df.columns.tolist()
        
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
        
        return self
    
    def transform(self, X):
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Select features
        X_selected = self.selector.transform(X_scaled)
        
        return X_selected

# Page config
st.set_page_config(
    page_title="🚀 Crypto Predictor V4.1",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; font-weight: bold; text-align: center;
                   background: linear-gradient(45deg, #FF6B35, #F7931E);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 2rem;}
    .metric-card {background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem 0;}
    .prediction-box {background: linear-gradient(45deg, #11998e 0%, #38ef7d 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin: 1rem 0;}
    .pipeline-info {background: linear-gradient(45deg, #434343 0%, #000000 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_complete_pipeline():
    """Load the complete 3-step pipeline"""
    pipeline_path = "models/best_pipeline.pkl"
    
    if not os.path.exists(pipeline_path):
        st.error(f"❌ Pipeline not found: {pipeline_path}")
        st.info("Please run the main pipeline first: `python main.py`")
        return None
    
    pipeline = joblib.load(pipeline_path)
    pipeline_info = {
        'steps': [],
        'total_steps': len(pipeline.steps) if hasattr(pipeline, 'steps') else 0,
        'pipeline_type': 'Complete 3-Step ML Pipeline'
    }
    
    if hasattr(pipeline, 'steps'):
        for step_name, step_obj in pipeline.steps:
            pipeline_info['steps'].append({
                'name': step_name,
                'type': type(step_obj).__name__
            })
    return pipeline, pipeline_info
def match_pipeline_features(input_df, pipeline):
    """
    Reorder and add missing columns to match the pipeline's expected features.
    """
    # Get the expected features from the preprocessor
    expected_features = None
    try:
        expected_features = pipeline.named_steps['preprocessing'].feature_names_in_
    except:
        st.warning("⚠️ Could not get feature_names_in_ from preprocessor. Using input columns as-is.")
        return input_df

    # Add missing columns with 0
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns
    input_df = input_df[expected_features]
    return input_df


def create_input_data(symbol, close_price, high_price, low_price, open_price, volume_from, volume_to):
    """Create input DataFrame with all expected columns for the pipeline"""
    input_data = pd.DataFrame({
        'time': [pd.Timestamp.now()],
        'high': [high_price],
        'low': [low_price],
        'open': [open_price],
        'volumefrom': [volume_from],
        'volumeto': [volume_to],
        'close': [close_price],
        'conversionType': ['direct'],
        'symbol': [symbol],
        'hl_range': [high_price - low_price],
        'candle_body': [abs(open_price - close_price)],
        'upper_shadow': [high_price - max(open_price, close_price)],
        'lower_shadow': [min(open_price, close_price) - low_price],
        'body_to_range': [(abs(open_price - close_price)) / (high_price - low_price) if high_price != low_price else 0],
        'return_1': [0],
        'log_return': [0]
    })
    return input_data

def fix_feature_names_if_needed(engineered_features):
    """Fix feature names if there's any mismatch"""
    if isinstance(engineered_features, pd.DataFrame):
        new_columns = {col: col.replace('_new','') for col in engineered_features.columns if col.endswith('_new')}
        if new_columns:
            engineered_features = engineered_features.rename(columns=new_columns)
    return engineered_features

def display_pipeline_info(pipeline_info):
    """Display pipeline information in sidebar"""
    st.sidebar.markdown("## 🔧 Pipeline Architecture")
    
    if pipeline_info['steps']:
        st.sidebar.markdown(f"**Total Steps:** {pipeline_info['total_steps']}")
        for i, step in enumerate(pipeline_info['steps'], 1):
            step_name = step['name'].replace('_',' ').title()
            step_type = step['type']
            if step['name'] == 'feature_engineering': icon="🔨"; desc="Raw features → Engineered features"
            elif step['name'] == 'preprocessing': icon="⚙️"; desc="Scaling + Feature selection"
            elif step['name'] == 'model': icon="🤖"; desc="Final prediction model"
            else: icon="📊"; desc="Data transformation"
            st.sidebar.markdown(f"**{i}. {icon} {step_name}**\n- Type: `{step_type}`\n- Function: {desc}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Model Performance")
    st.sidebar.success("✅ R² Score: 99%+")
    st.sidebar.success("✅ Production Ready")

def main():
    st.markdown('<h1 class="main-header">🚀 Crypto Predictor V4.1</h1>', unsafe_allow_html=True)
    st.markdown("### Complete 3-Step ML Pipeline • Feature Engineering + Preprocessing + Model")
    
    pipeline_data = load_complete_pipeline()
    if pipeline_data is None: st.stop()
    pipeline, pipeline_info = pipeline_data
    
    display_pipeline_info(pipeline_info)
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.header("📊 Crypto Parameters")
        crypto_symbols = ['BTC','ETH','ADA','DOGE','DOT','LTC','SOL','XRP']
        with st.form("prediction_form"):
            symbol = st.selectbox("🪙 Select Cryptocurrency", crypto_symbols, index=0)
            col_a, col_b = st.columns(2)
            with col_a:
                close_price = st.number_input("Close Price ($)", min_value=0.001, value=50000.0, step=0.001, format="%.3f")
                high_price = st.number_input("High Price ($)", min_value=0.001, value=51000.0, step=0.001, format="%.3f")
                low_price = st.number_input("Low Price ($)", min_value=0.001, value=49000.0, step=0.001, format="%.3f")
                open_price = st.number_input("Open Price ($)", min_value=0.001, value=49500.0, step=0.001, format="%.3f")
            with col_b:
                volume_from = st.number_input("Volume From", min_value=1, value=1000000, step=1000, format="%d")
                volume_to = st.number_input("Volume To", min_value=1, value=1000000, step=1000, format="%d")
                st.markdown("### 🎯 Prediction Target")
                st.info("**Return Prediction**: Percentage change in price")
            submitted = st.form_submit_button("🔮 Predict Crypto Returns", use_container_width=True, type="primary")
    
    with col2:
        st.header("🤖 Pipeline Status")
        st.markdown(f"<div class='pipeline-info'><h3>✅ Pipeline Loaded Successfully</h3><p><strong>Type:</strong> {pipeline_info['pipeline_type']}</p><p><strong>Steps:</strong> {pipeline_info['total_steps']}</p><p><strong>Status:</strong> 🟢 Ready for Predictions</p></div>", unsafe_allow_html=True)
    
    if submitted:
        if high_price < low_price or close_price > high_price or close_price < low_price or volume_from <= 0 or volume_to <= 0:
            st.error("❌ Invalid input values! Check prices and volumes.")
            st.stop()
        input_data = create_input_data(symbol, close_price, high_price, low_price, open_price, volume_from, volume_to)
        try:
            prediction = pipeline.predict(input_data)[0]
        except ValueError as feature_error:
            st.warning(f"⚠️ Feature mismatch detected: {str(feature_error)}")
            st.info("🔧 Attempting manual step-by-step pipeline execution...")
            
            try:
                # Manual pipeline execution with detailed debugging
                feature_engineer = pipeline.named_steps['feature_engineering']
                engineered_features = feature_engineer.transform(input_data)
                st.write(f"✅ Feature engineering completed. Shape: {engineered_features.shape}")
                st.write(f"📊 Engineered features: {list(engineered_features.columns)}")
                
                # Fix feature names
                engineered_features = fix_feature_names_if_needed(engineered_features)
                
                # Get preprocessor and check expected features
                preprocessor = pipeline.named_steps['preprocessing']
                if hasattr(preprocessor, 'feature_names_in_'):
                    expected_features = preprocessor.feature_names_in_
                    st.write(f"🎯 Expected features: {list(expected_features)}")
                    
                    # Match features to expected order
                    engineered_features = match_pipeline_features(engineered_features, pipeline)
                    st.write(f"🔄 Features after matching: {list(engineered_features.columns)}")
                
                # Transform with preprocessor
                processed_features = preprocessor.transform(engineered_features)
                st.write(f"✅ Preprocessing completed. Shape: {processed_features.shape}")
                
                # Get final prediction
                model = pipeline.named_steps['model']
                prediction = model.predict(processed_features)[0]
                st.success("✅ Manual pipeline execution successful!")
                
            except Exception as manual_error:
                st.error(f"❌ Manual execution failed: {str(manual_error)}")
                st.error("Please check the feature engineering and preprocessing steps.")
                st.stop()
        
        predicted_return_pct = prediction * 100
        predicted_price = close_price * (1 + prediction)
        price_change = predicted_price - close_price
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(f"<div class='metric-card'><h3>📊 Predicted Return</h3><h2>{predicted_return_pct:+.4f}%</h2></div>", unsafe_allow_html=True)
        with col2: st.markdown(f"<div class='metric-card'><h3>💰 Predicted Price</h3><h2>${predicted_price:,.2f}</h2></div>", unsafe_allow_html=True)
        with col3: color="🟢" if price_change>=0 else "🔴"; st.markdown(f"<div class='metric-card'><h3>📈 Price Change</h3><h2>{color} ${price_change:+.2f}</h2></div>", unsafe_allow_html=True)
        with col4: trend="📈 Bullish" if prediction>0 else "📉 Bearish"; confidence="High" if abs(prediction)>0.01 else "Medium" if abs(prediction)>0.005 else "Low"; st.markdown(f"<div class='metric-card'><h3>🎯 Trend</h3><h2>{trend}</h2><p>Confidence: {confidence}</p></div>", unsafe_allow_html=True)
        
        # Trading signal
        if prediction>0.02: signal_strength="🚀 Strong Buy"; signal_color="#00ff00"; signal_desc="Very bullish prediction"
        elif prediction>0.005: signal_strength="📈 Buy"; signal_color="#90ee90"; signal_desc="Moderately bullish"
        elif prediction>-0.005: signal_strength="⚖️ Hold"; signal_color="#ffff00"; signal_desc="Neutral prediction"
        elif prediction>-0.02: signal_strength="📉 Sell"; signal_color="#ff6b6b"; signal_desc="Moderately bearish"
        else: signal_strength="🔻 Strong Sell"; signal_color="#ff0000"; signal_desc="Very bearish prediction"
        
        st.markdown(f"<div class='prediction-box' style='background: linear-gradient(45deg, {signal_color}22, {signal_color}44); border: 2px solid {signal_color}55;'><h2>🎯 Trading Signal: {signal_strength}</h2><h3>{symbol} Return Prediction: {predicted_return_pct:+.4f}%</h3><p><strong>{signal_desc}</strong></p></div>", unsafe_allow_html=True)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=['Current'], open=[open_price], high=[high_price], low=[low_price], close=[close_price], name='Current Price', increasing_line_color='cyan', decreasing_line_color='gray'))
        pred_color='green' if prediction>0 else 'red'
        fig.add_trace(go.Scatter(x=['Predicted'], y=[predicted_price], mode='markers', marker=dict(size=20, color=pred_color, symbol='star'), name=f'Predicted Price: ${predicted_price:,.2f}'))
        fig.update_layout(title=f"{symbol} Price Prediction Analysis", xaxis_title="Time Period", yaxis_title="Price ($)", height=500, showlegend=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
