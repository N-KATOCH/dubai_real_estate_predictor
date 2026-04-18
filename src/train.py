import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
import sys
import os

# 1. Force DagsHub Connection if running in CI/CD
if os.getenv("MLFLOW_TRACKING_USERNAME"):
    import dagshub
    # This initializes the connection using environment variables from GitHub
    dagshub.init(repo_owner='N-KATOCH', repo_name='dubai_real_estate_predictor', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/N-KATOCH/dubai_real_estate_predictor.mlflow")
    print("🔗 Connected to DagsHub MLflow Remote")
else:
    mlflow.set_tracking_uri("file:./mlruns")
    print("🏠 Running locally - saving to ./mlruns")

# 2. Pathing setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from preprocessing import prepare_silver_layer
except ImportError as e:
    print(f"❌ Error importing preprocessing: {e}")
    sys.exit(1)

def run_training_pipeline(data_path):
    if not os.path.exists(data_path):
        print(f"❌ DATA NOT FOUND at {data_path}")
        sys.exit(1)

    mlflow.set_experiment("Dubai_Real_Estate_Valuation")
    
    with mlflow.start_run(run_name="Production_Run"):
        # 3. Process Data
        df = pd.read_csv(data_path)
        df_processed = prepare_silver_layer(df)
        
        # 4. Train Model
        X = df_processed[['size']].values.reshape(-1, 1) 
        y = df_processed['price'].values
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # 5. Generate and Log Predictions (Actual vs Predicted)
        predictions = model.predict(X)
        results_df = pd.DataFrame({
            'Actual_Price': y, 
            'Predicted_Price': predictions,
            'Error': y - predictions
        })
        
        # 6. Logging
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("sample_size", len(df))
        
        # Log the actual model
        mlflow.sklearn.log_model(model, "valuation_model")
        
        # Log the side-by-side comparison table
        results_df.to_csv("predictions_comparison.csv", index=False)
        mlflow.log_artifact("predictions_comparison.csv")
        
        print(f"🚀 Success! MAPE: {mape:.2f}% | Logged to DagsHub")

if __name__ == "__main__":
    run_training_pipeline("data/raw_data.csv")
