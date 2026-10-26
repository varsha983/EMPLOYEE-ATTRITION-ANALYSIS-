"""Machine Learning model training, evaluation, comparison, and serialization."""

from pathlib import Path
import json
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)

from src.data_loader import (
    load_raw_data,
    prepare_features_and_target,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.preprocessing import create_preprocessor, get_feature_names
from src.visualization import (
    ensure_fig_dir,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_feature_importances,
)


def get_model_definitions(random_state: int = 42) -> Dict[str, Any]:
    """Define classifiers to train and compare.
    
    All models use class_weight='balanced' to handle the 5:1 class imbalance
    inherent in employee turnover datasets, ensuring high sensitivity (recall)
    for at-risk employees rather than simply predicting the majority class.
    
    Args:
        random_state: Seed for reproducibility.
        
    Returns:
        Dict[str, Any]: Dictionary of classifier instances.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.5,
            solver="lbfgs",
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=5,
            min_samples_split=12,
            min_samples_leaf=6,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=150,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def build_pipeline(classifier: Any) -> Pipeline:
    """Build a complete Scikit-Learn Pipeline combining preprocessor and classifier.
    
    Args:
        classifier: Estimator instance.
        
    Returns:
        Pipeline: Combined preprocessing and modeling pipeline.
    """
    preprocessor = create_preprocessor()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a reproducible, stratified train/test split.
    
    Stratification preserves the original class ratio (~16.1% Attrition)
    in both training and test partitions.
    
    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Fraction of data allocated to test set.
        random_state: Random seed.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics for a fitted pipeline.
    
    Args:
        pipeline: Fitted Pipeline.
        X_test: Test features.
        y_test: True test labels.
        
    Returns:
        dict: Performance metrics, confusion matrix, ROC curve coordinates.
    """
    y_pred = pipeline.predict(X_test)
    
    # Check if classifier supports predict_proba
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline.named_steps["classifier"], "decision_function"):
        y_prob = pipeline.decision_function(X_test)
    else:
        y_prob = y_pred

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist(),
    }


def extract_feature_importances(
    pipeline: Pipeline,
) -> Tuple[List[str], np.ndarray]:
    """Extract feature importance or absolute coefficient values from fitted pipeline.
    
    Args:
        pipeline: Fitted Pipeline containing preprocessor and classifier.
        
    Returns:
        Tuple of (feature_names, importance_values).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    feature_names = get_feature_names(preprocessor)
    
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        # Absolute coefficients indicate feature influence magnitude
        importances = np.abs(classifier.coef_[0])
    else:
        importances = np.zeros(len(feature_names))
        
    return feature_names, importances


def train_and_compare_all_models(
    data_path: str = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv",
    test_size: float = 0.20,
    random_state: int = 42,
    save_figures: bool = True,
    output_fig_dir: str = "reports/figures",
    output_model_dir: str = "models",
) -> Dict[str, Any]:
    """Train, evaluate, compare, and serialize all three machine learning models.
    
    Models evaluated:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. Random Forest Classifier
    
    Returns structured results including cross-validation scores, test set metrics,
    and saves the best model pipeline via joblib.
    """
    df = load_raw_data(data_path)
    X, y = prepare_features_and_target(df)
    
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    models_dict = get_model_definitions(random_state=random_state)
    fitted_pipelines: Dict[str, Pipeline] = {}
    evaluation_results: Dict[str, Dict[str, Any]] = {}
    cv_scores: Dict[str, float] = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, clf in models_dict.items():
        pipe = build_pipeline(clf)
        
        # 5-fold cross validation on training data for ROC-AUC
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_scores[name] = round(float(np.mean(scores)), 4)
        
        # Fit strictly on training set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        # Evaluate on test set
        eval_metrics = evaluate_pipeline(pipe, X_test, y_test)
        evaluation_results[name] = eval_metrics

    # Model selection based on ROC-AUC and F1-score on held-out test data
    # (High ROC-AUC ensures balanced discriminative ability; F1 ensures minority recall)
    ranked_models = sorted(
        evaluation_results.items(),
        key=lambda item: (item[1]["roc_auc"], item[1]["f1"]),
        reverse=True,
    )
    best_model_name = ranked_models[0][0]
    best_pipeline = fitted_pipelines[best_model_name]

    # Save figures
    fig_dir = ensure_fig_dir(Path(output_fig_dir))
    if save_figures:
        # 1. ROC curves
        roc_plot_data = {
            name: {
                "fpr": np.array(res["fpr"]),
                "tpr": np.array(res["tpr"]),
                "auc": res["roc_auc"],
            }
            for name, res in evaluation_results.items()
        }
        plot_roc_curves(roc_plot_data, save_path=fig_dir / "07_model_roc_curves.png")

        # 2. Confusion matrices
        cm_dict = {
            name: np.array(res["confusion_matrix"])
            for name, res in evaluation_results.items()
        }
        plot_confusion_matrices(cm_dict, save_path=fig_dir / "08_model_confusion_matrices.png")

        # 3. Feature importance of best model
        feat_names, importances = extract_feature_importances(best_pipeline)
        plot_feature_importances(
            feat_names,
            importances,
            model_name=best_model_name,
            top_n=15,
            save_path=fig_dir / "09_feature_importances.png",
        )

    # Save best model pipeline
    model_dir = Path(output_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_model.joblib"
    joblib.dump(best_pipeline, best_model_path)

    # Save metadata
    feat_names, importances = extract_feature_importances(best_pipeline)
    top_features = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:15]
    ]

    metadata = {
        "best_model_name": best_model_name,
        "best_model_path": str(best_model_path),
        "random_state": random_state,
        "test_size": test_size,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "evaluation_summary": {
            name: {
                "cv_roc_auc_mean": cv_scores[name],
                "test_accuracy": res["accuracy"],
                "test_precision": res["precision"],
                "test_recall": res["recall"],
                "test_f1": res["f1"],
                "test_roc_auc": res["roc_auc"],
            }
            for name, res in evaluation_results.items()
        },
        "top_features": top_features,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }

    metadata_path = model_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "best_model_name": best_model_name,
        "best_model_path": str(best_model_path),
        "metadata_path": str(metadata_path),
        "evaluation_results": evaluation_results,
        "cv_scores": cv_scores,
        "top_features": top_features,
        "metadata": metadata,
    }


def predict_employee_attrition(
    pipeline: Pipeline,
    input_data: pd.DataFrame,
) -> Dict[str, Any]:
    """Run inference on an individual employee record or DataFrame.
    
    Args:
        pipeline: Fitted pipeline from joblib.
        input_data: DataFrame containing required raw features.
        
    Returns:
        dict: Predicted class (0 or 1), label, probability of attrition, and risk tier.
    """
    pred_class = int(pipeline.predict(input_data)[0])
    
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        prob_arr = pipeline.predict_proba(input_data)[0]
        prob_attrition = float(prob_arr[1])
        prob_retention = float(prob_arr[0])
    else:
        prob_attrition = float(pred_class)
        prob_retention = float(1.0 - pred_class)
        
    # Risk categorization
    if prob_attrition < 0.30:
        risk_tier = "Low Risk"
    elif prob_attrition < 0.60:
        risk_tier = "Moderate Risk"
    else:
        risk_tier = "Elevated Risk"

    return {
        "predicted_class": pred_class,
        "predicted_label": "Likely to Leave (Attrition Risk)" if pred_class == 1 else "Likely to Stay (Low Attrition Risk)",
        "attrition_probability": round(prob_attrition, 4),
        "retention_probability": round(prob_retention, 4),
        "risk_tier": risk_tier,
    }
