import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import os
import sys

# Connect to DagsHub
if os.getenv("MLFLOW_TRACKING_USERNAME"):
    mlflow.set_tracking_uri(f"https://dagshub.com/{os.getenv('MLFLOW_TRACKING_USERNAME')}/dubai_real_estate_predictor.mlflow")

def run_training_pipeline():
    data_path = 'data/raw_data.csv'
    if not os.path.exists(data_path):
        print("❌ Data missing!")
        sys.exit(1)

    mlflow.set_experiment("Dubai_Real_Estate_Valuation")
    
    with mlflow.start_run(run_name="LightGBM_Log_Production"):
        df = pd.read_csv(data_path)
        
        # 1. Feature Engineering
        le = LabelEncoder()
        df['nb_enc'] = le.fit_transform(df['neighborhood'])
        
        X = df[['size', 'nb_enc']]
        # 🧪 KAGGLE LOGIC: Log Transformation to handle price skew
        y = np.log1p(df['price']) 
        
        # 2. Hyperparameters (The ones you tuned)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'verbose': -1
        }
        
        # Log params to the "Parameters" tab
        mlflow.log_params(params)
        mlflow.log_param("target_transform", "np.log1p")

        # 3. Training
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_data, num_boost_round=200)
        
        # 4. Predictions & Back-Transformation
        log_preds = model.predict(X)
        actual_preds = np.expm1(log_preds) # Back to AED
        actual_y = df['price']
        
        # 5. Metrics
        mape = np.mean(np.abs((actual_y - actual_preds) / actual_y)) * 100
        mlflow.log_metric("MAPE", mape)

        # 6. Feature Importance Plot
        lgb.plot_importance(model, importance_type='gain')
        plt.title("Dubai Real Estate - Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        # 7. Save Results
        res = pd.DataFrame({
            'Neighborhood': df['neighborhood'],
            'Actual': actual_y,
            'Predicted': actual_preds
        })
        res.to_csv("predictions_comparison.csv", index=False)
        mlflow.log_artifact("predictions_comparison.csv")
        
        # Register the LightGBM model
        mlflow.lightgbm.log_model(model, "valuation_model")
        
        print(f"🚀 SUCCESS! LightGBM (Log) trained. MAPE: {mape:.2f}%")

if __name__ == "__main__":
    run_training_pipeline()
