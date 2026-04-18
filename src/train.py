import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
import sys
import os

# 1. Bulletproof Pathing
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from preprocessing import prepare_silver_layer
except ImportError as e:
    print(f"❌ Error importing preprocessing: {e}")
    sys.exit(1)

def run_training_pipeline(data_path):
    print(f"📂 Checking for data at: {data_path}")
    if not os.path.exists(data_path):
        print(f"❌ DATA NOT FOUND! Current directory: {os.getcwd()}")
        sys.exit(1)

    # 2. Local MLflow Setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Dubai_Real_Estate_Valuation")
    
    with mlflow.start_run(run_name="Production_Run"):
        # 3. Load & Process
        df = pd.read_csv(data_path)
        print("✅ Data Loaded Successfully")
        
        df_processed = prepare_silver_layer(df)
        print("✅ Preprocessing Complete")
        
        # 4. Train with Safety Checks
        X = df_processed[['size']].values.reshape(-1, 1) # Ensure 2D array
        y = df_processed['price'].values
        
        model = RandomForestRegressor(n_estimators=10)
        model.fit(X, y)
        print("✅ Model Trained")
        
        # 5. Log
        mlflow.log_metric("sample_size", len(df))
        mlflow.sklearn.log_model(model, "valuation_model")
        print("🚀 Successfully Logged to MLflow!")

if __name__ == "__main__":
    # Ensure this matches your folder: data/raw_data.csv
    run_training_pipeline("data/raw_data.csv")
