"""Data loading, validation, and schema definitions for Employee Attrition."""

from pathlib import Path
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

# Column definitions
TARGET_COL = "Attrition"

EXCLUDED_COLS = [
    "EmployeeNumber",  # Identifier, no predictive value
    "EmployeeCount",   # Constant value (1), zero variance
    "StandardHours",   # Constant value (80), zero variance
    "Over18",          # Constant value ('Y'), zero variance
]

CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

NUMERIC_FEATURES = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]


def load_raw_data(filepath: str | Path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv") -> pd.DataFrame:
    """Load raw dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file is not found at filepath.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please ensure the IBM HR dataset "
            f"'WA_Fn-UseC_-HR-Employee-Attrition.csv' exists in the 'data/' directory."
        )
    df = pd.read_csv(path)
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate dataset structure, types, missing values and duplicates.
    
    Args:
        df: Raw DataFrame to validate.
        
    Returns:
        dict: Summary validation report.
    """
    missing = df.isnull().sum().to_dict()
    total_missing = sum(missing.values())
    duplicates = int(df.duplicated().sum())
    
    # Check expected columns
    expected_cols = set([TARGET_COL] + EXCLUDED_COLS + CATEGORICAL_FEATURES + NUMERIC_FEATURES)
    actual_cols = set(df.columns)
    missing_cols = list(expected_cols - actual_cols)
    
    attrition_counts = df[TARGET_COL].value_counts().to_dict() if TARGET_COL in df.columns else {}
    attrition_rate = (
        float(attrition_counts.get("Yes", 0) / len(df)) if len(df) > 0 and "Yes" in attrition_counts else 0.0
    )
    
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "total_missing": total_missing,
        "n_duplicates": duplicates,
        "missing_cols": missing_cols,
        "attrition_counts": attrition_counts,
        "attrition_rate": attrition_rate,
        "is_valid": total_missing == 0 and len(missing_cols) == 0,
    }


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features (X) and binary target (y).
    
    Excludes non-informative identifiers and zero-variance columns.
    Maps target 'Yes' -> 1, 'No' -> 0.
    
    Args:
        df: Input DataFrame containing features and target.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    cols_to_drop = [c for c in EXCLUDED_COLS if c in df.columns]
    
    if TARGET_COL in df.columns:
        cols_to_drop.append(TARGET_COL)
        y = (df[TARGET_COL] == "Yes").astype(int)
    else:
        y = None
        
    X = df.drop(columns=cols_to_drop)
    
    # Ensure remaining columns match expected feature set
    expected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = X[expected_features]
    
    return X, y
