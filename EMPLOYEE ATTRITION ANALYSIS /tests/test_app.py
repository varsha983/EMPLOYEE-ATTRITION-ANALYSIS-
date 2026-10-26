"""Tests for Streamlit application components and inference consistency."""

from pathlib import Path
import pytest
import pandas as pd
import joblib

from src.data_loader import (
    load_raw_data,
    prepare_features_and_target,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.models import predict_employee_attrition


def test_app_file_exists():
    app_path = Path("app.py")
    assert app_path.is_file(), "app.py does not exist"


def test_app_syntax_and_compilation():
    with open("app.py", "r") as f:
        code = f.read()
    compile(code, "app.py", "exec")


def test_metadata_consistency_with_saved_model():
    model_path = Path("models/best_model.joblib")
    meta_path = Path("models/model_metadata.json")
    
    assert model_path.exists()
    assert meta_path.exists()

    pipeline = joblib.load(model_path)
    import json
    with open(meta_path) as f:
        meta = json.load(f)

    assert "best_model_name" in meta
    assert "evaluation_summary" in meta
    assert "top_features" in meta
    assert len(meta["top_features"]) > 0


def test_preset_simulations_inference():
    pipeline = joblib.load("models/best_model.joblib")
    
    # High risk profile simulation
    high_risk_dict = {
        "Age": 22,
        "BusinessTravel": "Travel_Frequently",
        "DailyRate": 300,
        "Department": "Sales",
        "DistanceFromHome": 25,
        "Education": 1,
        "EducationField": "Marketing",
        "EnvironmentSatisfaction": 1,
        "Gender": "Male",
        "HourlyRate": 40,
        "JobInvolvement": 1,
        "JobLevel": 1,
        "JobRole": "Sales Representative",
        "JobSatisfaction": 1,
        "MaritalStatus": "Single",
        "MonthlyIncome": 2100,
        "MonthlyRate": 4000,
        "NumCompaniesWorked": 4,
        "OverTime": "Yes",
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 1,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 2,
        "TrainingTimesLastYear": 1,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 1,
        "YearsInCurrentRole": 0,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0,
    }
    input_df = pd.DataFrame([high_risk_dict])
    res = predict_employee_attrition(pipeline, input_df)

    assert res["predicted_class"] == 1
    assert res["attrition_probability"] > 0.50
    assert res["risk_tier"] == "Elevated Risk"
