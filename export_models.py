"""
Export MLflow models to pickle files for Streamlit deployment.

Run this once locally after training to create portable model files.

Usage:
    python export_models.py
"""

import mlflow
import joblib
from pathlib import Path

def export_models():
    """Export models from MLflow registry to pickle files."""
    
    # Create output directory
    output_dir = Path("streamlit_app/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading models from MLflow registry...")
    
    # Load from registry
    lrc = mlflow.sklearn.load_model("models:/credit-risk-lrc/latest")
    rfc = mlflow.sklearn.load_model("models:/credit-risk-rfc/latest")
    svc = mlflow.sklearn.load_model("models:/credit-risk-svc/latest")
    
    print("Exporting to pickle files...")
    
    # Export to files
    joblib.dump(lrc, output_dir / "lrc_pipeline.pkl")
    print(f"  ✓ Saved: {output_dir / 'lrc_pipeline.pkl'}")
    
    joblib.dump(rfc, output_dir / "rfc_pipeline.pkl")
    print(f"  ✓ Saved: {output_dir / 'rfc_pipeline.pkl'}")
    
    joblib.dump(svc, output_dir / "svc_pipeline.pkl")
    print(f"  ✓ Saved: {output_dir / 'svc_pipeline.pkl'}")
    
    print("\nDone! Models exported to streamlit_app/models/")


def export_sample_data():
    """Export sample data for the Streamlit demo."""
    import pandas as pd
    
    # Load test data
    test_data = pd.read_csv("data/processed/test_data.csv")
    
    # Get samples of each class (more samples for variety)
    good_samples = test_data[test_data["class"] == 0].head(50)
    bad_samples = test_data[test_data["class"] == 1].head(50)
    
    # Combine and save
    sample_data = pd.concat([good_samples, bad_samples], ignore_index=True)
    
    output_dir = Path("streamlit_app/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_data.to_csv(output_dir / "sample_data.csv", index=False)
    print(f"  ✓ Saved: {output_dir / 'sample_data.csv'} ({len(sample_data)} rows)")


if __name__ == "__main__":
    export_models()
    export_sample_data()
