#!/usr/bin/env python3
"""Main training script: runs complete EDA, trains and compares 3 models,
evaluates performance metrics, and saves the best model pipeline and artifacts.
"""

import os
import sys
from pathlib import Path
import argparse
import pandas as pd

# Ensure writable matplotlib cache on macOS sandbox
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eda import run_eda
from src.models import train_and_compare_all_models


def parse_args():
    parser = argparse.ArgumentParser(description="Employee Attrition Analysis & Modeling Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        help="Path to the IBM HR dataset CSV",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test split fraction (default: 0.20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-eda-figures",
        action="store_true",
        help="Skip regenerating EDA charts",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("       EMPLOYEE ATTRITION ANALYSIS & PREDICTIVE MODELING PIPELINE       ")
    print("=" * 80)
    print(f"Dataset path:      {args.data}")
    print(f"Test size:         {args.test_size * 100:.0f}%")
    print(f"Random seed:       {args.seed}")
    print("-" * 80)

    # 1. Exploratory Data Analysis
    print("\n[STEP 1/3] Executing Exploratory Data Analysis (EDA)...")
    eda_results = run_eda(
        data_path=args.data,
        save_figures=not args.skip_eda_figures,
        output_dir="reports/figures",
    )

    print(f"✓ Total Employee Records:  {eda_results['total_employees']}")
    print(f"✓ Attrition Count:         {eda_results['left_count']} left, {eda_results['stayed_count']} stayed")
    print(f"✓ Overall Attrition Rate:   {eda_results['attrition_rate']:.2f}%")
    print("\nKey Exploratory Findings:")
    for finding in eda_results["findings"]:
        print(f"  • {finding}")

    # 2. Model Training & Comparison
    print("\n[STEP 2/3] Training and Comparing Machine Learning Models...")
    print("  Models: Logistic Regression, Decision Tree, Random Forest")
    print("  Class weighting: 'balanced' (addressing 5.2:1 class imbalance)")
    
    ml_results = train_and_compare_all_models(
        data_path=args.data,
        test_size=args.test_size,
        random_state=args.seed,
        save_figures=True,
        output_fig_dir="reports/figures",
        output_model_dir="models",
    )

    # 3. Model Comparison Table
    print("\n[STEP 3/3] Final Test Set Evaluation Results:")
    print("-" * 80)
    table_rows = []
    for model_name, metrics in ml_results["evaluation_results"].items():
        table_rows.append({
            "Model": model_name,
            "CV ROC-AUC": f"{ml_results['cv_scores'][model_name]:.4f}",
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1-Score": f"{metrics['f1']:.4f}",
            "ROC-AUC": f"{metrics['roc_auc']:.4f}",
        })

    results_df = pd.DataFrame(table_rows)
    print(results_df.to_string(index=False))
    print("-" * 80)

    best_model = ml_results["best_model_name"]
    print(f"\n★ SELECTED MODEL: {best_model}")
    print(f"✓ Saved fitted pipeline to: {ml_results['best_model_path']}")
    print(f"✓ Saved metadata to:        {ml_results['metadata_path']}")
    
    print("\nTop 5 Most Influential Predictive Features:")
    for item in ml_results["top_features"][:5]:
        print(f"  • {item['feature']}: {item['importance']:.4f}")

    print("\n" + "=" * 80)
    print("Training pipeline completed successfully.")
    print("To launch the Streamlit dashboard, run:")
    print("  streamlit run app.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
