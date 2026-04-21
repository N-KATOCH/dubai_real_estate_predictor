import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import sys
import os

# MLflow Tracking Setup
if os.getenv("MLFLOW_TRACKING_USERNAME"):
    tracking_uri = f"https://dagshub.com/{os.getenv('MLFLOW_TRACKING_USERNAME')}/dubai_real_estate_predictor.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
else:
    mlflow.set_tracking_uri("file:./mlruns")

def run_training_pipeline(data_path):
    mlflow.set_experiment("Dubai_Real_Estate_Valuation")
    
    with mlflow.start_run(run_name="Production_Run_1000_Rows"):
        df = pd.read_csv(data_path)
        
        # 1. Feature Engineering: Neighborhood Encoding
        # This turns 'Dubai Marina' into 1, 'Downtown' into 2, etc.
        le = LabelEncoder()
        df['neighborhood_enc'] = le.fit_transform(df['neighborhood'])
        
        # 2. Define Features (Size + Neighborhood)
        X = df[['size', 'neighborhood_enc']]
        y = df['price']
        
        # 3. Train a more robust model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 4. Predictions & Metrics
        predictions = model.predict(X)
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        
        # 5. Log Everything
        mlflow.log_param("features", "size, neighborhood")
        mlflow.log_metric("MAPE", mape)
        mlflow.sklearn.log_model(model, "valuation_model")
        
        # Save results for DagsHub
        results_df = pd.DataFrame({
            'Neighborhood': df['neighborhood'],
            'Size': df['size'],
            'Actual_Price': y, 
            'Predicted_Price': predictions
        })
        results_df.to_csv("predictions_comparison.csv", index=False)
        mlflow.log_artifact("predictions_comparison.csv")
        
        print(f"🚀 Model Trained on {len(df)} rows. Accuracy (MAPE): {100-mape:.2f}%")

if __name__ == "__main__":
    run_training_pipeline("data/raw_data.csv")
