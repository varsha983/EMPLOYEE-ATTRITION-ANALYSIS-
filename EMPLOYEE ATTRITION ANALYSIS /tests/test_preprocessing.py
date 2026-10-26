"""Tests for preprocessing pipeline and feature encoders."""

import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.data_loader import (
    load_raw_data,
    prepare_features_and_target,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.preprocessing import create_preprocessor, get_feature_names


@pytest.fixture
def sample_data():
    df = load_raw_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    X, y = prepare_features_and_target(df)
    return X.head(100), y.head(100)


def test_create_preprocessor_structure():
    preprocessor = create_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_preprocessor_fit_transform(sample_data):
    X, y = sample_data
    preprocessor = create_preprocessor()
    X_trans = preprocessor.fit_transform(X)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(X)
    # Total transformed columns: 23 numeric + one-hot encoded categories (> 30)
    assert X_trans.shape[1] > len(NUMERIC_FEATURES)

    feature_names = get_feature_names(preprocessor)
    assert len(feature_names) == X_trans.shape[1]
    assert all(isinstance(f, str) for f in feature_names)


def test_preprocessor_handles_unseen_categories(sample_data):
    X, y = sample_data
    preprocessor = create_preprocessor()
    preprocessor.fit(X)

    # Modify categorical value to an unseen category
    X_novel = X.copy()
    X_novel.loc[0, "Department"] = "CompletelyUnseenDepartment"
    X_novel.loc[0, "BusinessTravel"] = "UnknownTravelFrequency"

    # Must transform without raising an exception due to handle_unknown='ignore'
    X_trans = preprocessor.transform(X_novel)
    assert X_trans.shape[0] == len(X_novel)
    assert not np.isnan(X_trans).any()
