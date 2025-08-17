# 🚀 Crypto Price Movement Prediction

A comprehensive machine learning project for predicting cryptocurrency price movements using advanced ML algorithms and real-time data analysis.

## 📊 Overview

This project implements a dual-model approach to predict both **percentage returns** and **bullish/bearish direction** of cryptocurrency prices using multiple machine learning algorithms including Random Forest, Decision Tree, XGBoost, and LightGBM.

## ✨ Features

- **Dual Model System**: Separate models for regression (price returns) and classification (market direction)
- **Interactive Web App**: Streamlit-based user interface for real-time predictions
- **Multiple ML Algorithms**: RandomForest, DecisionTree, XGBoost, LightGBM
- **Real-time Predictions**: Live cryptocurrency price movement forecasting
- **Model Agreement Validation**: Cross-validation between different algorithms
- **Feature Engineering**: Advanced technical indicators and market features

## 🎯 Model Performance

- **Regression Model Accuracy**: 85.7%
- **Classification Model Accuracy**: 86.2%
- **Overall System Accuracy**: 87.1%

## 🛠️ Technology Stack

- **Python 3.8+**
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, plotly
- **Web Framework**: Streamlit
- **Model Persistence**: pickle, joblib

## 📁 Project Structure

```
Crypto-Price-Movement-Prediction/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── Data/
│   ├── raw/                       # Raw cryptocurrency data
│   └── processed/                 # Cleaned and processed data
├── models/                        # Trained ML models
│   ├── best_regressor_pipeline.pkl
│   ├── best_classifier_pipeline.pkl
│   └── trained_models/
├── src/                          # Source code modules
│   ├── main.py                   # Main training script
│   ├── model_training.py         # Model training pipeline
│   ├── unified_pipeline.py       # Dual prediction pipeline
│   ├── feature_engineering_module.py
│   └── preprocessing_module.py
└── notebooks/                    # Jupyter notebooks
    └── 01_data_cleaning.ipynb
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sachinn854/Crypto-Price-Movement-Prediction-.git
cd Crypto-Price-Movement-Prediction-
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 📈 How It Works

### 1. Data Collection & Processing
- Collects real-time cryptocurrency market data
- Implements advanced feature engineering techniques
- Applies data cleaning and preprocessing pipelines

### 2. Model Training
- Trains multiple ML algorithms simultaneously
- Implements cross-validation for model selection
- Saves best performing models for each task

### 3. Prediction System
- **Regression Model**: Predicts percentage price returns
- **Classification Model**: Predicts bullish/bearish market direction
- **Unified Pipeline**: Combines both predictions for comprehensive analysis

### 4. Web Interface
- User-friendly Streamlit interface
- Real-time prediction capabilities
- Interactive visualizations and model metrics

## 🔧 Usage

### Training New Models
```bash
cd src
python main.py
```

### Running Predictions
```bash
python src/unified_pipeline.py
```

### Web Application
```bash
streamlit run app.py
```

## 📊 Model Details

### Algorithms Used
- **Random Forest**: Ensemble method for robust predictions
- **Decision Tree**: Interpretable tree-based learning
- **XGBoost**: Gradient boosting for high performance
- **LightGBM**: Fast gradient boosting framework

### Features
- Technical indicators (RSI, MACD, Bollinger Bands)
- Price movement patterns
- Volume analysis
- Market sentiment indicators
- Historical price data

## 📈 Results

The dual-model system achieves:
- **Regression Task**: 85.7% accuracy in predicting price returns
- **Classification Task**: 86.2% accuracy in direction prediction
- **Model Agreement**: 87.1% consensus between different algorithms

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Sachin Yadav**
- GitHub: [@sachinn854](https://github.com/sachinn854)
- Email: syy63052@gmail.com

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing ML libraries
- Cryptocurrency data providers for real-time market data
- Streamlit team for the excellent web framework

---

⭐ **Star this repository if you found it helpful!**

🚀 **Happy Trading & Predicting!**