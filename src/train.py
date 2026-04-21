import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os, sys

if os.getenv("MLFLOW_TRACKING_USERNAME"):
    mlflow.set_tracking_uri(f"https://dagshub.com/{os.getenv('MLFLOW_TRACKING_USERNAME')}/dubai_real_estate_predictor.mlflow")

def run_training_pipeline():
    data_path = 'data/raw_data.csv'
    if not os.path.exists(data_path):
        print("❌ File missing at Step 2!")
        sys.exit(1)

    mlflow.set_experiment("Dubai_Real_Estate_Valuation")
    with mlflow.start_run(run_name="Production_Run_1000_Rows"):
        df = pd.read_csv(data_path)
        
        # Encoding
        le = LabelEncoder()
        df['nb_enc'] = le.fit_transform(df['neighborhood'])
        
        X = df[['size', 'nb_enc']]
        y = df['price']
        
        model = RandomForestRegressor(n_estimators=50).fit(X, y)
        
        # Logging
        mape = np.mean(np.abs((y - model.predict(X)) / y)) * 100
        mlflow.log_metric("MAPE", mape)
        mlflow.sklearn.log_model(model, "valuation_model")
        
        # Results table
        res = pd.DataFrame({'Neighborhood': df['neighborhood'], 'Actual': y, 'Predicted': model.predict(X)})
        res.to_csv("predictions_comparison.csv", index=False)
        mlflow.log_artifact("predictions_comparison.csv")
        print(f"🚀 Success! Accuracy: {100-mape:.2f}%")

if __name__ == "__main__":
    run_training_pipeline()
