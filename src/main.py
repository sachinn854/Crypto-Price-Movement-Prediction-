"""
🚀 Crypto Price Movement Prediction - Main Entry Point
=====================================================
Enhanced with unified pipeline approach
"""

import os
import sys
from datetime import datetime

# Add current directory to the Python path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# Import the unified pipeline
from unified_pipeline import run_unified_pipeline

def main():
    """Main execution function"""
    print("🚀 CRYPTO PRICE MOVEMENT PREDICTION")
    print("=" * 50)
    print(f"📅 Started at: {datetime.now()}")
    print()
    
    # Data path - corrected to use processed data (go up one level from src)
    data_path = os.path.join(parent_dir, "Data", "processed", "final_cleaned_crypto_zero_removed.csv")
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("   Please ensure the data file exists.")
        return False
    
    print(f"📊 Using data: {data_path}")
    print()
    
    # Run unified pipeline
    success, result = run_unified_pipeline(data_path)
    
    if success:
        print("\n🎉 SUCCESS: Unified pipeline completed!")
        print(f"📁 Pipeline saved: {result['pipeline_path']}")
        print(f"🏆 Best model: {result['best_model']}")
        return True
    else:
        print(f"\n❌ FAILED: {result}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Pipeline execution completed successfully!")
        else:
            print("\n❌ Pipeline execution failed!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
