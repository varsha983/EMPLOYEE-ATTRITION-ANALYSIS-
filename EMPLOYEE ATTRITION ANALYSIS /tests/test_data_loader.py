"""Tests for dataset loading, validation, and schema definitions."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_raw_data,
    validate_dataset,
    prepare_features_and_target,
    TARGET_COL,
    EXCLUDED_COLS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

DATA_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


def test_dataset_file_exists():
    assert DATA_PATH.is_file(), f"Expected dataset file at {DATA_PATH} not found"


def test_load_raw_data_shape():
    df = load_raw_data(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1470, 35)


def test_validate_dataset_integrity():
    df = load_raw_data(DATA_PATH)
    val = validate_dataset(df)
    assert val["is_valid"] is True
    assert val["total_missing"] == 0
    assert val["n_duplicates"] == 0
    assert val["n_rows"] == 1470
    assert val["n_cols"] == 35
    assert len(val["missing_cols"]) == 0
    assert val["attrition_counts"]["Yes"] == 237
    assert val["attrition_counts"]["No"] == 1233


def test_prepare_features_and_target():
    df = load_raw_data(DATA_PATH)
    X, y = prepare_features_and_target(df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(df)
    assert len(y) == len(df)

    # Excluded and target columns must be removed from X
    for col in EXCLUDED_COLS:
        assert col not in X.columns
    assert TARGET_COL not in X.columns

    # X must have 30 features (23 numeric + 7 categorical)
    assert X.shape[1] == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    assert set(X.columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    # Target y must be binary 0 and 1
    assert set(y.unique()) == {0, 1}
    assert int(y.sum()) == 237
