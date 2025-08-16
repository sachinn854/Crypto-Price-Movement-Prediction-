"""
🚀 Complete ML Pipeline for Crypto Price Prediction
==================================================

This module orchestrates the complete machine learning pipeline:
1. Feature Engineering from processed data
2. Data Preprocessing and splitting
3. Model Training and Evaluation
4. Pipeline Persistence and Deployment Ready

Author: Crypto Prediction Pipeline
Version: 1.0
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering_module import create_features
from preprocessing_module import preprocess_data
from model_training import train_models

class CryptoPredictionPipeline:
    """
    Complete ML Pipeline for Cryptocurrency Price Movement Prediction
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the pipeline
        
        Args:
            base_dir: Base directory for the project (default: parent of src)
        """
        if base_dir is None:
            # Get parent directory of src
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir
            
        # Define paths
        self.data_dir = os.path.join(self.base_dir, 'Data')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.results_dir = os.path.join(self.base_dir, 'results')
        
        # Pipeline state
        self.pipeline_info = {
            'created': datetime.now().isoformat(),
            'status': 'initialized',
            'steps_completed': [],
            'feature_engineering': {},
            'preprocessing': {},
            'model_training': {}
        }
        
        # Create directories
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("🚀 CRYPTO PREDICTION PIPELINE INITIALIZED")
        print("=" * 50)
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Models directory: {self.models_dir}")
        print(f"📁 Results directory: {self.results_dir}")
    
    def run_feature_engineering(self, input_file: str = "final_cleaned_crypto_zero_removed.csv",
                               target_type: str = 'direction') -> str:
        """
        Step 1: Run feature engineering pipeline
        
        Args:
            input_file: Name of the cleaned data file in processed directory
            target_type: Type of target variable ('direction' or 'return')
            
        Returns:
            Path to featured data file
        """
        print("\n🔧 STEP 1: FEATURE ENGINEERING")
        print("=" * 40)
        
        # Input and output paths
        input_path = os.path.join(self.processed_dir, input_file)
        output_file = f"crypto_featured_data_{target_type}.csv"
        output_path = os.path.join(self.processed_dir, output_file)
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"📂 Input: {input_path}")
        print(f"📂 Output: {output_path}")
        
        try:
            # Run feature engineering
            featured_data_path = create_features(input_path, output_path, target_type)
            
            # Update pipeline info
            self.pipeline_info['feature_engineering'] = {
                'status': 'completed',
                'input_file': input_path,
                'output_file': output_path,
                'target_type': target_type,
                'completed_at': datetime.now().isoformat()
            }
            self.pipeline_info['steps_completed'].append('feature_engineering')
            
            print(f"✅ Feature engineering completed successfully!")
            print(f"📄 Featured data saved to: {featured_data_path}")
            
            return featured_data_path
            
        except Exception as e:
            print(f"❌ Feature engineering failed: {str(e)}")
            self.pipeline_info['feature_engineering'] = {
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }
            raise
    
    def run_preprocessing(self, featured_data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                                np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Step 2: Run data preprocessing pipeline
        
        Args:
            featured_data_path: Path to featured data from step 1
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_info
        """
        print("\n⚙️ STEP 2: DATA PREPROCESSING")
        print("=" * 40)
        
        # Preprocessing save directory
        preprocessing_dir = os.path.join(self.models_dir, 'preprocessing')
        
        print(f"📂 Input: {featured_data_path}")
        print(f"📁 Preprocessing objects will be saved to: {preprocessing_dir}")
        
        try:
            # Run preprocessing
            X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_info = preprocess_data(
                featured_data_path, preprocessing_dir
            )
            
            # Update pipeline info
            self.pipeline_info['preprocessing'] = {
                'status': 'completed',
                'input_file': featured_data_path,
                'save_dir': preprocessing_dir,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'num_features': X_train.shape[1],
                'completed_at': datetime.now().isoformat()
            }
            self.pipeline_info['steps_completed'].append('preprocessing')
            
            print(f"✅ Preprocessing completed successfully!")
            print(f"📊 Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_info
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {str(e)}")
            self.pipeline_info['preprocessing'] = {
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }
            raise
    
    def run_model_training(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                          feature_names: list, optimize_hyperparams: bool = False) -> Dict[str, Any]:
        """
        Step 3: Run model training pipeline
        
        Args:
            X_train, X_val, X_test: Feature arrays
            y_train, y_val, y_test: Target arrays
            feature_names: List of feature names
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Training results dictionary
        """
        print("\n🤖 STEP 3: MODEL TRAINING")
        print("=" * 40)
        
        # Model training save directory
        training_dir = os.path.join(self.models_dir, 'trained_models')
        
        print(f"📁 Models will be saved to: {training_dir}")
        print(f"🔧 Hyperparameter optimization: {'ON' if optimize_hyperparams else 'OFF'}")
        
        try:
            # Run model training
            training_results = train_models(
                X_train, X_val, X_test, y_train, y_val, y_test,
                feature_names, training_dir, optimize_hyperparams
            )
            
            # Update pipeline info
            self.pipeline_info['model_training'] = {
                'status': 'completed',
                'save_dir': training_dir,
                'best_model': training_results['best_model_name'],
                'test_r2': training_results['best_metrics']['test_r2'],
                'test_rmse': training_results['best_metrics']['test_rmse'],
                'overfitting_score': training_results['best_metrics']['overfitting_score'],
                'hyperparameter_optimization': optimize_hyperparams,
                'completed_at': datetime.now().isoformat()
            }
            self.pipeline_info['steps_completed'].append('model_training')
            
            print(f"✅ Model training completed successfully!")
            print(f"🏆 Best model: {training_results['best_model_name']}")
            print(f"🎯 Test R²: {training_results['best_metrics']['test_r2']:.4f}")
            print(f"🎯 Test RMSE: {training_results['best_metrics']['test_rmse']:.4f}")
            
            return training_results
            
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            self.pipeline_info['model_training'] = {
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }
            raise
    
    def save_complete_pipeline(self, training_results: Dict[str, Any]) -> str:
        """
        Save the complete pipeline with all artifacts
        """
        print("\n💾 SAVING COMPLETE PIPELINE")
        print("=" * 40)
        
        # Create complete pipeline directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_dir = os.path.join(self.models_dir, f'complete_pipeline_{timestamp}')
        os.makedirs(pipeline_dir, exist_ok=True)
        
        try:
            # Copy best model to pipeline directory
            best_model_path = training_results['best_model_path']
            pipeline_model_path = os.path.join(pipeline_dir, 'production_model.pkl')
            
            if os.path.exists(best_model_path):
                import shutil
                shutil.copy2(best_model_path, pipeline_model_path)
                print(f"✅ Production model saved to: {pipeline_model_path}")
            
            # Copy preprocessing objects
            preprocessing_dir = os.path.join(self.models_dir, 'preprocessing')
            pipeline_preprocessing_dir = os.path.join(pipeline_dir, 'preprocessing')
            
            if os.path.exists(preprocessing_dir):
                import shutil
                shutil.copytree(preprocessing_dir, pipeline_preprocessing_dir, dirs_exist_ok=True)
                print(f"✅ Preprocessing objects copied to pipeline")
            
            # Save complete pipeline info
            complete_info = {
                'pipeline_info': self.pipeline_info,
                'training_results': {
                    'best_model_name': training_results['best_model_name'],
                    'best_model_path': training_results.get('best_model_path', ''),
                    'best_metrics': training_results.get('best_metrics', {}),
                    'model_results': training_results.get('model_results', {}),
                    'feature_names': training_results.get('feature_names', [])
                },
                'created_at': datetime.now().isoformat(),
                'pipeline_version': '1.0',
                'production_ready': True
            }
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            complete_info = convert_numpy(complete_info)
            
            pipeline_info_path = os.path.join(pipeline_dir, 'pipeline_info.json')
            with open(pipeline_info_path, 'w') as f:
                json.dump(complete_info, f, indent=2, default=str)
            
            print(f"✅ Pipeline info saved to: {pipeline_info_path}")
            
            self.pipeline_info['status'] = 'completed'
            self.pipeline_info['completed_at'] = datetime.now().isoformat()
            self.pipeline_info['production_pipeline_dir'] = pipeline_dir
            
            print(f"\n🎉 COMPLETE PIPELINE SAVED SUCCESSFULLY!")
            print(f"📁 Pipeline directory: {pipeline_dir}")
            print(f"🚀 Ready for production deployment!")
            
            return pipeline_dir
            
        except Exception as e:
            print(f"❌ Pipeline saving failed: {str(e)}")
            raise
    
    def run_complete_pipeline(self, input_file: str = "final_cleaned_crypto_zero_removed.csv",
                             target_type: str = 'return',
                             optimize_hyperparams: bool = False) -> str:
        """
        Run the complete ML pipeline from start to finish
        
        Args:
            input_file: Name of cleaned data file in processed directory
            target_type: Type of target variable ('direction' or 'return')
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Path to complete pipeline directory
        """
        print("🚀 RUNNING COMPLETE CRYPTO PREDICTION PIPELINE")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Feature Engineering
            featured_data_path = self.run_feature_engineering(input_file, target_type)
            
            # Step 2: Preprocessing
            X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_info = self.run_preprocessing(featured_data_path)
            
            # Step 3: Model Training
            training_results = self.run_model_training(
                X_train, X_val, X_test, y_train, y_val, y_test,
                preprocessing_info['feature_names'], optimize_hyperparams
            )
            
            # Step 4: Save Complete Pipeline
            pipeline_dir = self.save_complete_pipeline(training_results)
            
            # Calculate total time
            end_time = datetime.now()
            total_time = end_time - start_time
            
            print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"⏱️ Total time: {total_time}")
            print(f"🏆 Best model: {training_results['best_model_name']}")
            print(f"🎯 Test R²: {training_results.get('best_metrics', {}).get('test_r2', 0.0):.4f}")
            print(f"🎯 Test RMSE: {training_results.get('best_metrics', {}).get('test_rmse', 0.0):.4f}")
            print(f"📁 Production pipeline: {pipeline_dir}")
            
            return pipeline_dir
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            self.pipeline_info['status'] = 'failed'
            self.pipeline_info['error'] = str(e)
            self.pipeline_info['failed_at'] = datetime.now().isoformat()
            raise

def run_pipeline(base_dir: str = None, input_file: str = "final_cleaned_crypto_zero_removed.csv",
                target_type: str = 'direction', optimize_hyperparams: bool = False) -> str:
    """
    Convenience function to run the complete pipeline
    
    Args:
        base_dir: Base directory for the project
        input_file: Name of cleaned data file
        target_type: Type of target variable
        optimize_hyperparams: Whether to optimize hyperparameters
        
    Returns:
        Path to complete pipeline directory
    """
    # Initialize and run pipeline
    pipeline = CryptoPredictionPipeline(base_dir)
    return pipeline.run_complete_pipeline(input_file, target_type, optimize_hyperparams)

if __name__ == "__main__":
    # Run the complete pipeline
    pipeline_dir = run_pipeline(
        input_file="final_cleaned_crypto_zero_removed.csv",
        target_type='direction',
        optimize_hyperparams=False  # Set to True for better performance but longer training
    )
    
    print(f"\n✅ Pipeline complete! Check: {pipeline_dir}")
