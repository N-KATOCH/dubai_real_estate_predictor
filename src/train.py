import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os, sys

# Tracking URI setup
if os.getenv("MLFLOW_TRACKING_USERNAME"):
    uri = f"https://dagshub.com/{os.getenv('MLFLOW_TRACKING_USERNAME')}/dubai_real_estate_predictor.mlflow"
    mlflow.set_tracking_uri(uri)

def run_training_pipeline():
    # Direct relative path
    path = 'data/raw_data.csv'
    
    if not os.path.exists(path):
        print(f"❌ ERROR: Cannot find {path}")
        sys.exit(1)

    mlflow.set_experiment("Dubai_Real_Estate_Valuation")
    with mlflow.start_run(run_name="Production_Run_1000_Rows"):
        df = pd.read_csv(path)
        
        le = LabelEncoder()
        df['nb_enc'] = le.fit_transform(df['neighborhood'])
        
        X = df[['size', 'nb_enc']]
        y = df['price']
        
        model = RandomForestRegressor(n_estimators=50).fit(X, y)
        
        mape = np.mean(np.abs((y - model.predict(X)) / y)) * 100
        mlflow.log_metric("MAPE", mape)
        mlflow.sklearn.log_model(model, "valuation_model")
        
        # Log comparison artifact
        res = pd.DataFrame({'Neighborhood': df['neighborhood'], 'Actual': y, 'Predicted': model.predict(X)})
        res.to_csv("predictions_comparison.csv", index=False)
        mlflow.log_artifact("predictions_comparison.csv")
        
        print(f"🚀 SUCCESS! Accuracy: {100-mape:.2f}%")

if __name__ == "__main__":
    run_training_pipeline()
