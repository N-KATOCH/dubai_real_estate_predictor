import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
import sys

# 1. Setup DagsHub/MLflow Tracking
if os.getenv("MLFLOW_TRACKING_USERNAME"):
    mlflow.set_tracking_uri(f"https://dagshub.com/{os.getenv('MLFLOW_TRACKING_USERNAME')}/dubai_real_estate_predictor.mlflow")

def run_training_pipeline():
    data_path = 'data/raw_data.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found.")
        sys.exit(1)

    # Start MLflow Experiment
    mlflow.set_experiment("Dubai_Real_Estate_Valuation")
    
    with mlflow.start_run(run_name="Production_Run_v2_Explainable"):
        # 2. Load and Encode Data
        df = pd.read_csv(data_path)
        
        le = LabelEncoder()
        df['nb_enc'] = le.fit_transform(df['neighborhood'])
        
        # 3. Define Features and Model Parameters
        features = ['size', 'nb_enc']
        target = 'price'
        n_estimators = 100
        
        # --- LOG PARAMETERS ---
        # This fills the "Parameters" tab in DagsHub
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("features", str(features))
        mlflow.log_param("total_observations", len(df))
        
        # 4. Train Model
        X = df[features]
        y = df[target]
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X, y)
        
        # 5. Calculate Metrics
        predictions = model.predict(X)
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        
        # --- LOG METRICS ---
        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("Accuracy_Score", 100 - mape)
        
        # 6. Generate Feature Importance Plot
        # This helps with Data Storytelling (Explainability)
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances for Dubai Property Prices')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        
        # --- LOG ARTIFACTS ---
        # Log the plot, the model, and the comparison CSV
        mlflow.log_artifact("feature_importance.png")
        
        res_df = pd.DataFrame({
            'Neighborhood': df['neighborhood'],
            'Actual_Price': y,
            'Predicted_Price': predictions
        })
        res_df.to_csv("predictions_comparison.csv", index=False)
        mlflow.log_artifact("predictions_comparison.csv")
        
        # Save the actual model
        mlflow.sklearn.log_model(model, "valuation_model")
        
        print(f"🚀 Training Complete! MAPE: {mape:.2f}%")
        print(f"📈 Parameters and Feature Importance logged to DagsHub.")

if __name__ == "__main__":
    run_training_pipeline()
