import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

def run_drift_report(reference_path, current_path):
    # Load past 'Gold' data and new 'Incoming' data
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)
    
    # Create the report
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    
    # Save as an artifact for the team to review
    report.save_html("reports/data_drift_report.html")
    print("🔍 Drift Report generated in /reports folder.")

if __name__ == "__main__":
    # run_drift_report("data/training_snapshot.csv", "data/new_listings.csv")
    pass
