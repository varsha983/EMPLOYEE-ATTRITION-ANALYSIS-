"""Plotting and visualization utilities for EDA and Model Evaluation."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless/scripts
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Consistent aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {"No": "#3470a3", "Yes": "#d9534f", 0: "#3470a3", 1: "#d9534f"}
FIG_DIR = Path("reports/figures")


def ensure_fig_dir(output_dir: Path = FIG_DIR) -> Path:
    """Ensure output directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_attrition_distribution(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot overall employee attrition breakdown and percentage.
    
    Args:
        df: DataFrame containing 'Attrition' column.
        save_path: Optional path to save figure.
        
    Returns:
        plt.Figure: Created figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    counts = df["Attrition"].value_counts()
    percentages = df["Attrition"].value_counts(normalize=True) * 100
    labels = ["Stayed (No)", "Left (Yes)"]
    colors = [PALETTE["No"], PALETTE["Yes"]]
    
    # Bar plot
    bars = ax1.bar(labels, [counts.get("No", 0), counts.get("Yes", 0)], color=colors, width=0.55, edgecolor="black", alpha=0.9)
    ax1.set_title("Employee Count by Attrition Status", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Number of Employees", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height:,}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(counts) * 1.15)
    
    # Donut pie chart
    wedges, texts, autotexts = ax2.pie(
        [counts.get("No", 0), counts.get("Yes", 0)],
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        explode=(0, 0.08),
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    plt.setp(autotexts, size=11, weight="bold", color="white")
    plt.setp(texts, size=11)
    ax2.set_title(f"Attrition Proportion ({percentages.get('Yes', 0):.1f}% Attrition Rate)", fontsize=13, fontweight="bold", pad=12)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_attrition_by_department_and_role(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot attrition rates across Departments and Job Roles.
    
    Args:
        df: DataFrame with 'Department', 'JobRole', and 'Attrition'.
        save_path: Optional path to save figure.
        
    Returns:
        plt.Figure: Created figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # By Department
    dept_stats = (
        df.groupby("Department")["Attrition"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )
    bars1 = ax1.barh(dept_stats.index, dept_stats.values, color="#e67e22", edgecolor="black", alpha=0.88)
    ax1.set_title("Attrition Rate by Department (%)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Attrition Rate (%)", fontsize=11)
    ax1.set_xlim(0, max(dept_stats.values) * 1.25)
    for bar in bars1:
        width = bar.get_width()
        ax1.annotate(f"{width:.1f}%",
                     xy=(width, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=10, fontweight="bold")
    
    # By Job Role
    role_stats = (
        df.groupby("JobRole")["Attrition"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=True)
    )
    bars2 = ax2.barh(role_stats.index, role_stats.values, color="#2980b9", edgecolor="black", alpha=0.88)
    ax2.set_title("Attrition Rate by Job Role (%)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Attrition Rate (%)", fontsize=11)
    ax2.set_xlim(0, max(role_stats.values) * 1.25)
    for bar in bars2:
        width = bar.get_width()
        ax2.annotate(f"{width:.1f}%",
                     xy=(width, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_attrition_by_age_and_income(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot distribution of Age and Monthly Income by Attrition status.
    
    Args:
        df: DataFrame with 'Age', 'MonthlyIncome', and 'Attrition'.
        save_path: Optional path to save figure.
        
    Returns:
        plt.Figure: Created figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Age boxplot/kde
    sns.kdeplot(data=df[df["Attrition"] == "No"]["Age"], ax=ax1, label="Stayed (No)", color=PALETTE["No"], fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(data=df[df["Attrition"] == "Yes"]["Age"], ax=ax1, label="Left (Yes)", color=PALETTE["Yes"], fill=True, alpha=0.35, linewidth=2)
    ax1.set_title("Age Distribution by Attrition Status", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Employee Age (Years)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.legend(title="Attrition", frameon=True)
    
    # Monthly income boxplot
    sns.boxplot(x="Attrition", y="MonthlyIncome", hue="Attrition", data=df, ax=ax2, palette=[PALETTE["No"], PALETTE["Yes"]], width=0.45, boxprops=dict(alpha=0.85), legend=False)
    ax2.set_title("Monthly Income by Attrition Status", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Attrition Status", fontsize=11)
    ax2.set_ylabel("Monthly Income ($)", fontsize=11)
    
    # Annotate medians
    medians = df.groupby("Attrition")["MonthlyIncome"].median()
    for i, (status, median) in enumerate(medians.items()):
        ax2.text(i, median + 400, f"Median: ${median:,.0f}", ha="center", fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8))
        
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_attrition_by_overtime(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot attrition comparison for Overtime vs Non-overtime employees.
    
    Args:
        df: DataFrame with 'OverTime' and 'Attrition'.
        save_path: Optional path to save figure.
        
    Returns:
        plt.Figure: Created figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # OverTime Attrition Rate
    ot_stats = (
        df.groupby("OverTime")["Attrition"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reindex(["No", "Yes"])
    )
    labels = ["No Overtime", "Works Overtime"]
    bars1 = ax1.bar(labels, ot_stats.values, color=["#27ae60", "#c0392b"], width=0.5, edgecolor="black", alpha=0.88)
    ax1.set_title("Attrition Rate by Overtime Status", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Attrition Rate (%)", fontsize=11)
    ax1.set_ylim(0, max(ot_stats.values) * 1.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    # Stacked count breakdown
    ct = pd.crosstab(df["OverTime"], df["Attrition"])
    ct_norm = ct.div(ct.sum(1), axis=0) * 100
    ct_norm = ct_norm[["No", "Yes"]]
    ct_norm.index = ["No Overtime", "Works Overtime"]
    
    ct_norm.plot(kind="bar", stacked=True, color=[PALETTE["No"], PALETTE["Yes"]], ax=ax2, edgecolor="black", alpha=0.88, width=0.5)
    ax2.set_title("Proportional Breakdown by Overtime", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.set_ylim(0, 100)
    ax2.legend(["Stayed (No)", "Left (Yes)"], frameon=True)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_attrition_by_satisfaction_and_wlb(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot attrition rates across Job Satisfaction and Work-Life Balance levels.
    
    Args:
        df: DataFrame with 'JobSatisfaction', 'WorkLifeBalance', and 'Attrition'.
        save_path: Optional path to save figure.
        
    Returns:
        plt.Figure: Created figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sat_map = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
    wlb_map = {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}
    
    # Job satisfaction
    sat_stats = df.groupby("JobSatisfaction")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
    sat_labels = [sat_map.get(i, str(i)) for i in sat_stats.index]
    bars1 = ax1.bar(sat_labels, sat_stats.values, color="#8e44ad", width=0.55, edgecolor="black", alpha=0.88)
    ax1.set_title("Attrition Rate by Job Satisfaction", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Attrition Rate (%)", fontsize=11)
    ax1.set_ylim(0, max(sat_stats.values) * 1.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    # Work-Life balance
    wlb_stats = df.groupby("WorkLifeBalance")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
    wlb_labels = [wlb_map.get(i, str(i)) for i in wlb_stats.index]
    bars2 = ax2.bar(wlb_labels, wlb_stats.values, color="#16a085", width=0.55, edgecolor="black", alpha=0.88)
    ax2.set_title("Attrition Rate by Work-Life Balance", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylabel("Attrition Rate (%)", fontsize=11)
    ax2.set_ylim(0, max(wlb_stats.values) * 1.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_attrition_by_tenure(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot attrition relationship with Years at Company and Years in Current Role.
    
    Args:
        df: DataFrame with 'YearsAtCompany', 'YearsInCurrentRole', and 'Attrition'.
        save_path: Optional path to save figure.
        
    Returns:
        plt.Figure: Created figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Tenure bins
    tenure_bins = [0, 1, 3, 5, 10, 40]
    tenure_labels = ["<=1 yr", "2-3 yrs", "4-5 yrs", "6-10 yrs", ">10 yrs"]
    df_copy = df.copy()
    df_copy["TenureGroup"] = pd.cut(df_copy["YearsAtCompany"], bins=tenure_bins, labels=tenure_labels, include_lowest=True)
    
    tenure_stats = df_copy.groupby("TenureGroup", observed=False)["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
    bars1 = ax1.bar(tenure_labels, tenure_stats.values, color="#34495e", width=0.55, edgecolor="black", alpha=0.88)
    ax1.set_title("Attrition Rate by Years at Company", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Attrition Rate (%)", fontsize=11)
    ax1.set_ylim(0, max(tenure_stats.values) * 1.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    # Years in current role bins
    role_bins = [0, 1, 3, 7, 20]
    role_labels = ["<=1 yr", "2-3 yrs", "4-7 yrs", ">7 yrs"]
    df_copy["RoleGroup"] = pd.cut(df_copy["YearsInCurrentRole"], bins=role_bins, labels=role_labels, include_lowest=True)
    
    role_group_stats = df_copy.groupby("RoleGroup", observed=False)["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
    bars2 = ax2.bar(role_labels, role_group_stats.values, color="#d35400", width=0.55, edgecolor="black", alpha=0.88)
    ax2.set_title("Attrition Rate by Years in Current Role", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylabel("Attrition Rate (%)", fontsize=11)
    ax2.set_ylim(0, max(role_group_stats.values) * 1.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_roc_curves(
    roc_data: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot ROC curves comparison for multiple models.
    
    Args:
        roc_data: Dict mapping model name to dict containing 'fpr', 'tpr', 'auc'.
        save_path: Optional path to save figure.
        
    Returns:
        plt.Figure: Created figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22"]
    for i, (name, data) in enumerate(roc_data.items()):
        color = colors[i % len(colors)]
        ax.plot(data["fpr"], data["tpr"], color=color, lw=2.2,
                label=f"{name} (ROC-AUC = {data['auc']:.3f})")
        
    # Baseline chance diagonal
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot side-by-side confusion matrices for evaluated models.
    
    Args:
        confusion_matrices: Dict mapping model name to 2x2 confusion matrix.
        save_path: Optional path to save figure.
        
    Returns:
        plt.Figure: Created figure object.
    """
    n_models = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.8))
    if n_models == 1:
        axes = [axes]
        
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Stayed (0)", "Left (1)"],
            yticklabels=["Stayed (0)", "Left (1)"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = "Random Forest Classifier",
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot top N feature importances or model coefficients.
    
    Args:
        feature_names: Names of features.
        importances: Array of importance values (or absolute coefficients).
        model_name: Name of model for title.
        top_n: Number of top features to plot.
        save_path: Optional path to save figure.
        
    Returns:
        plt.Figure: Created figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    s = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    s = s.sort_values(ascending=True)  # ascending for horizontal bar chart
    
    bars = ax.barh(s.index, s.values, color="#1f77b4", edgecolor="black", alpha=0.88)
    ax.set_title(f"Top {top_n} Most Influential Features ({model_name})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance / Relative Weight", fontsize=11)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.3f}",
                     xy=(width, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=9, fontweight="bold")
        
    ax.set_xlim(0, max(s.values) * 1.2)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
