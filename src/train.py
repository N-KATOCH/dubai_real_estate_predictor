import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
import sys
import os

# 1. Setup Remote Tracking WITHOUT the popup window
if os.getenv("MLFLOW_TRACKING_USERNAME"):
    # We use the Environment Variables directly instead of dagshub.init()
    tracking_uri = f"https://dagshub.com/{os.getenv('MLFLOW_TRACKING_USERNAME')}/dubai_real_estate_predictor.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"🔗 Remote tracking set to: {tracking_uri}")
else:
    mlflow.set_tracking_uri("file:./mlruns")
    print("🏠 Running locally.")

# 2. Pathing setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from preprocessing import prepare_silver_layer
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

def run_training_pipeline(data_path):
    if not os.path.exists(data_path):
        sys.exit(1)

    mlflow.set_experiment("Dubai_Real_Estate_Valuation")
    
    with mlflow.start_run(run_name="Production_Run"):
        df = pd.read_csv(data_path)
        df_processed = prepare_silver_layer(df)
        
        # Training
        X = df_processed[['size']].values.reshape(-1, 1) 
        y = df_processed['price'].values
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Predictions & Logging
        predictions = model.predict(X)
        results_df = pd.DataFrame({'Actual_Price': y, 'Predicted_Price': predictions})
        
        mlflow.log_metric("sample_size", len(df))
        mlflow.sklearn.log_model(model, "valuation_model")
        
        results_df.to_csv("predictions_comparison.csv", index=False)
        mlflow.log_artifact("predictions_comparison.csv")
        
        print("🚀 Success! Results sent to DagsHub.")

if __name__ == "__main__":
    run_training_pipeline("data/raw_data.csv")
