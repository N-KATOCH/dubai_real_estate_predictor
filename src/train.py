import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap


import sys
import os
# Add the 'src' directory to the path so Python can find preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from preprocessing import prepare_silver_layer

def run_training_pipeline(data_path):
    # 1. Start MLflow Experiment
    mlflow.set_experiment("Dubai_Real_Estate_Valuation")
    
    with mlflow.start_run(run_name="Production_LightGBM_Run"):
        # 2. Load and Preprocess (Silver Layer)
        df = pd.read_csv(data_path)
        df = prepare_silver_layer(df)
        
        # 3. Dummy Setup for Demonstration 
        # (In reality, you'd insert your random_search.best_estimator_ logic here)
        print("🚀 Training LightGBM Model and calculating SHAP...")
        
        # 4. Log Metrics
        mlflow.log_metric("accuracy_baseline", 0.96)
        
        # 5. Log the Model
        # This registers the model in the MLflow 'Vault'
        mlflow.sklearn.log_model(
            sk_model="YOUR_MODEL_OBJECT", 
            artifact_path="model",
            registered_model_name="Dubai_Property_Pricer"
        )
        
        print("✅ Pipeline execution complete. Metrics and Model logged to MLflow.")

if __name__ == "__main__":
    # In production, this would be triggered by your CI/CD pipeline
    # run_training_pipeline("data/raw_data.csv")
    pass
