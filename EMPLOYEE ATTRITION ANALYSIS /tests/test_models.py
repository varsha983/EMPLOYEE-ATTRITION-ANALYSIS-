"""Tests for model training, evaluation, comparison, and serialization."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

from src.data_loader import load_raw_data, prepare_features_and_target
from src.models import (
    get_model_definitions,
    build_pipeline,
    split_data,
    evaluate_pipeline,
    extract_feature_importances,
    predict_employee_attrition,
)


@pytest.fixture(scope="module")
def prepared_data():
    df = load_raw_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    X, y = prepare_features_and_target(df)
    return split_data(X, y, test_size=0.20, random_state=42)


def test_get_model_definitions():
    models = get_model_definitions()
    assert "Logistic Regression" in models
    assert "Decision Tree" in models
    assert "Random Forest" in models


def test_stratified_split_preserves_ratios():
    df = load_raw_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)

    total_ratio = y.mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    # Stratification must keep ratio within tight tolerance (< 1%)
    assert abs(train_ratio - total_ratio) < 0.01
    assert abs(test_ratio - total_ratio) < 0.01
    assert len(X_train) == 1176
    assert len(X_test) == 294


def test_train_and_evaluate_all_three_models(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    models = get_model_definitions(random_state=42)

    for name, clf in models.items():
        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_pipeline(pipeline, X_test, y_test)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.5 <= metrics["roc_auc"] <= 1.0  # Must beat random guess

        # Confusion matrix checks
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (2, 2)
        assert cm.sum() == len(X_test)


def test_saved_model_joblib_loading():
    model_path = Path("models/best_model.joblib")
    assert model_path.is_file(), "models/best_model.joblib does not exist. Run train.py first."

    loaded_pipeline = joblib.load(model_path)
    assert hasattr(loaded_pipeline, "predict")
    assert hasattr(loaded_pipeline, "predict_proba")


def test_predict_employee_attrition_inference():
    model_path = Path("models/best_model.joblib")
    loaded_pipeline = joblib.load(model_path)

    # Use first row of dataset as test input
    df = load_raw_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    X, _ = prepare_features_and_target(df)
    sample_input = X.iloc[[0]]

    result = predict_employee_attrition(loaded_pipeline, sample_input)

    assert "predicted_class" in result
    assert result["predicted_class"] in [0, 1]
    assert "attrition_probability" in result
    assert 0.0 <= result["attrition_probability"] <= 1.0
    assert 0.0 <= result["retention_probability"] <= 1.0
    assert abs((result["attrition_probability"] + result["retention_probability"]) - 1.0) < 0.001
    assert result["risk_tier"] in ["Low Risk", "Moderate Risk", "Elevated Risk"]
