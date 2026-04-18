import pandas as pd
import os
import sys
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

def run_drift_report(reference_path, current_path):
    print(f"📊 Running Drift Analysis...")
    
    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        print("⚠️ Data files missing. Skipping monitoring.")
        return

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)
    
    # Initialize the report with presets
    report = Report(metrics=[
        DataDriftPreset(), 
        TargetDriftPreset()
    ])
    
    report.run(reference_data=reference_df, current_data=current_df)
    
    # Ensure the reports folder exists
    os.makedirs("reports", exist_ok=True)
    report.save_html("reports/data_drift_report.html")
    print("✅ Drift Report saved to reports/data_drift_report.html")

if __name__ == "__main__":
    run_drift_report("data/raw_data.csv", "data/raw_data.csv")
