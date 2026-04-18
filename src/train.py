import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
import sys
import os

# 1. Handle DagsHub Connection
# We only initialize DagsHub if we are running in the GitHub Action environment
if os.getenv("MLFLOW_TRACKING_USERNAME"):
    try:
        import dagshub
        dagshub.init(repo_owner='N-KATOCH', repo_name='dubai_real_estate_predictor', mlflow=True)
        print("🔗 Remote tracking initialized: Connected to DagsHub")
    except Exception as e:
        print(f"⚠️ DagsHub initialization failed: {e}")
else:
    mlflow.set_tracking_uri("file:./mlruns")
    print("🏠 Local tracking initialized: Saving to ./mlruns")

# 2. Pathing setup for modular code
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

    # 3. MLflow Experiment Setup
    mlflow.set_experiment("Dubai_Real_Estate_Valuation")
    
    with mlflow.start_run(run_name="Production_Run"):
        # 4. Load & Process (Silver Layer)
        df = pd.read_csv(data_path)
        df_processed = prepare_silver_layer(df)
        print("✅ Silver Layer Transformation Complete")
        
        # 5. Training
        X = df_processed[['size']].values.reshape(-1, 1) 
        y = df_processed['price'].values
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        print("✅ Model Training Complete")
        
        # 6. Generate Predictions for Results Table
        predictions = model.predict(X)
        results_df = pd.DataFrame({
            'Actual_Price': y, 
            'Predicted_Price': predictions,
            'Error_Margin': y - predictions
        })
        
        # 7. Log Metrics and Artifacts
        mlflow.log_metric("sample_size", len(df))
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        mlflow.log_metric("MAPE", mape)
        
        mlflow.sklearn.log_model(model, "valuation_model")
        
        # Save results table as a CSV so you can see it in DagsHub
        results_df.to_csv("predictions_comparison.csv", index=False)
        mlflow.log_artifact("predictions_comparison.csv")
        
        print(f"🚀 Success! MAPE: {mape:.2f}% | Results logged to DagsHub.")

if __name__ == "__main__":
    run_training_pipeline("data/raw_data.csv")
