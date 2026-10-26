"""Preprocessing pipeline creation and feature transformation."""

from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.data_loader import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def create_preprocessor(
    numeric_features: List[str] = NUMERIC_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Create a ColumnTransformer pipeline for preprocessing.
    
    Numeric features are standardized using StandardScaler.
    Categorical features are one-hot encoded using OneHotEncoder with
    handle_unknown='ignore' to gracefully handle unseen categories at test time.
    
    Args:
        numeric_features: List of column names for numeric features.
        categorical_features: List of column names for categorical features.
        
    Returns:
        ColumnTransformer: Unfitted preprocessor.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(
    preprocessor: ColumnTransformer,
    numeric_features: List[str] = NUMERIC_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
) -> List[str]:
    """Retrieve output feature names from a fitted preprocessor.
    
    Args:
        preprocessor: Fitted ColumnTransformer instance.
        numeric_features: Original numeric feature names.
        categorical_features: Original categorical feature names.
        
    Returns:
        List[str]: Combined list of output feature names.
    """
    feature_names = []
    
    # Check if get_feature_names_out is available (scikit-learn >= 1.0)
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            pass

    # Fallback feature name construction
    feature_names.extend(numeric_features)
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(list(encoded_cat_names))
    except Exception:
        feature_names.extend(categorical_features)

    return feature_names
