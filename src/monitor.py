import pandas as pd
# Updated imports for Evidently 0.4.0+
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import os

def run_drift_report(reference_path, current_path):
    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        print("⚠️ Data files missing for monitoring. skipping...")
        return

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)
    
    # Generate the report
    report = Report(metrics=[
        DataDriftPreset(), 
        TargetDriftPreset()
    ])
    
    report.run(reference_data=reference_df, current_data=current_df)
    
    # Create reports folder if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    report.save_html("reports/data_drift_report.html")
    print("✅ Drift Report generated successfully.")

if __name__ == "__main__":
    # Pointing to your sample data
    run_drift_report("data/raw_data.csv", "data/raw_data.csv")
