import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
import sys
import os

# 1. Pathing setup to ensure internal modules are found
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from preprocessing import prepare_silver_layer
except ImportError as e:
    print(f"❌ Error importing preprocessing: {e}")
    sys.exit(1)

def run_training_pipeline(data_path):
    print(f"📂 Accessing data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"❌ DATA NOT FOUND! Path checked: {os.path.abspath(data_path)}")
        sys.exit(1)

    # 2. MLflow Configuration
    # Note: Tracking URI is now handled by the GitHub Action environment variables
    mlflow.set_experiment("Dubai_Real_Estate_Valuation")
    
    with mlflow.start_run(run_name="Production_Run"):
        # 3. Load & Process (Silver Layer)
        df = pd.read_csv(data_path)
        print("✅ Data Loaded Successfully")
        
        df_processed = prepare_silver_layer(df)
        print("✅ Silver Layer Transformation Complete")
        
        # 4. Feature Engineering & Training
        # Using 'size' to predict 'price' for this MLOps validation
        X = df_processed[['size']].values.reshape(-1, 1) 
        y = df_processed['price'].values
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        print("✅ Model Training Complete")
        
        # 5. Generate Predictions for Result Validation
        predictions = model.predict(X)
        results_df = pd.DataFrame({
            'Actual_Price': y, 
            'Predicted_Price': predictions,
            'Error': y - predictions
        })
        
        # 6. Log Metrics, Model, and Artifacts
        mlflow.log_metric("sample_size", len(df))
        # Calculate a simple MAPE for the log
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        mlflow.log_metric("MAPE", mape)
        
        # Log the model object
        mlflow.sklearn.log_model(model, "valuation_model")
        
        # Save and log the predictions table (This is what you want to see!)
        results_df.to_csv("predictions_comparison.csv", index=False)
        mlflow.log_artifact("predictions_comparison.csv")
        
        print(f"🚀 Successfully Logged to DagsHub! MAPE: {mape:.2f}%")

if __name__ == "__main__":
    # Pointing to the data folder created in the repo
    run_training_pipeline("data/raw_data.csv")
