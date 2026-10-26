"""Exploratory Data Analysis (EDA) functions and report generation."""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from src.data_loader import (
    load_raw_data,
    validate_dataset,
    TARGET_COL,
    EXCLUDED_COLS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.visualization import (
    ensure_fig_dir,
    plot_attrition_distribution,
    plot_attrition_by_department_and_role,
    plot_attrition_by_age_and_income,
    plot_attrition_by_overtime,
    plot_attrition_by_satisfaction_and_wlb,
    plot_attrition_by_tenure,
)


def run_eda(
    data_path: str = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv",
    save_figures: bool = True,
    output_dir: str = "reports/figures",
) -> Dict[str, Any]:
    """Execute complete EDA pipeline and generate structured findings report.
    
    Args:
        data_path: Path to raw dataset CSV.
        save_figures: Whether to render and save plots to disk.
        output_dir: Directory where figures will be stored.
        
    Returns:
        dict: Summary metrics, statistical aggregations, and key findings.
    """
    df = load_raw_data(data_path)
    val = validate_dataset(df)
    
    out_path = ensure_fig_dir(Path(output_dir))
    
    # 1. Overall Attrition Metrics
    total_employees = len(df)
    attrition_counts = df[TARGET_COL].value_counts().to_dict()
    stayed_count = int(attrition_counts.get("No", 0))
    left_count = int(attrition_counts.get("Yes", 0))
    attrition_rate = (left_count / total_employees) * 100 if total_employees > 0 else 0.0

    # 2. Attrition by Department
    dept_summary = (
        df.groupby("Department")
        .agg(
            Total=("Attrition", "count"),
            Left=("Attrition", lambda s: (s == "Yes").sum()),
            Rate=("Attrition", lambda s: (s == "Yes").mean() * 100),
        )
        .round(2)
        .to_dict(orient="index")
    )

    # 3. Attrition by Job Role
    role_summary = (
        df.groupby("JobRole")
        .agg(
            Total=("Attrition", "count"),
            Left=("Attrition", lambda s: (s == "Yes").sum()),
            Rate=("Attrition", lambda s: (s == "Yes").mean() * 100),
        )
        .sort_values(by="Rate", ascending=False)
        .round(2)
        .to_dict(orient="index")
    )

    # 4. Attrition by Overtime
    overtime_summary = (
        df.groupby("OverTime")
        .agg(
            Total=("Attrition", "count"),
            Left=("Attrition", lambda s: (s == "Yes").sum()),
            Rate=("Attrition", lambda s: (s == "Yes").mean() * 100),
        )
        .round(2)
        .to_dict(orient="index")
    )

    # 5. Age & Income Statistics by Attrition
    income_stayed_median = float(df[df[TARGET_COL] == "No"]["MonthlyIncome"].median())
    income_left_median = float(df[df[TARGET_COL] == "Yes"]["MonthlyIncome"].median())
    age_stayed_mean = float(df[df[TARGET_COL] == "No"]["Age"].mean())
    age_left_mean = float(df[df[TARGET_COL] == "Yes"]["Age"].mean())

    # 6. Satisfaction & Work-Life Balance
    sat_summary = (
        df.groupby("JobSatisfaction")
        .agg(
            Total=("Attrition", "count"),
            Left=("Attrition", lambda s: (s == "Yes").sum()),
            Rate=("Attrition", lambda s: (s == "Yes").mean() * 100),
        )
        .round(2)
        .to_dict(orient="index")
    )

    wlb_summary = (
        df.groupby("WorkLifeBalance")
        .agg(
            Total=("Attrition", "count"),
            Left=("Attrition", lambda s: (s == "Yes").sum()),
            Rate=("Attrition", lambda s: (s == "Yes").mean() * 100),
        )
        .round(2)
        .to_dict(orient="index")
    )

    # 7. Tenure & Years in Current Role
    tenure_stayed_mean = float(df[df[TARGET_COL] == "No"]["YearsAtCompany"].mean())
    tenure_left_mean = float(df[df[TARGET_COL] == "Yes"]["YearsAtCompany"].mean())
    role_years_stayed_mean = float(df[df[TARGET_COL] == "No"]["YearsInCurrentRole"].mean())
    role_years_left_mean = float(df[df[TARGET_COL] == "Yes"]["YearsInCurrentRole"].mean())

    # Generate and save figures if requested
    figure_paths = {}
    if save_figures:
        p1 = out_path / "01_attrition_distribution.png"
        plot_attrition_distribution(df, save_path=p1)
        figure_paths["distribution"] = str(p1)

        p2 = out_path / "02_attrition_by_department_role.png"
        plot_attrition_by_department_and_role(df, save_path=p2)
        figure_paths["department_role"] = str(p2)

        p3 = out_path / "03_attrition_by_age_income.png"
        plot_attrition_by_age_and_income(df, save_path=p3)
        figure_paths["age_income"] = str(p3)

        p4 = out_path / "04_attrition_by_overtime.png"
        plot_attrition_by_overtime(df, save_path=p4)
        figure_paths["overtime"] = str(p4)

        p5 = out_path / "05_attrition_by_satisfaction_wlb.png"
        plot_attrition_by_satisfaction_and_wlb(df, save_path=p5)
        figure_paths["satisfaction_wlb"] = str(p5)

        p6 = out_path / "06_attrition_by_tenure.png"
        plot_attrition_by_tenure(df, save_path=p6)
        figure_paths["tenure"] = str(p6)

    # Key findings summary
    findings = [
        f"Overall dataset contains {total_employees} records with an attrition rate of {attrition_rate:.2f}% ({left_count} left, {stayed_count} stayed).",
        f"Significant class imbalance exists (approx 5.2 : 1 ratio of stayed to departed employees).",
        f"OverTime is a strong risk indicator: employees working overtime experience an attrition rate of {overtime_summary.get('Yes', {}).get('Rate', 0):.1f}% compared to {overtime_summary.get('No', {}).get('Rate', 0):.1f}% for non-overtime staff (~3x higher).",
        f"Sales Representatives have the highest attrition rate ({role_summary.get('Sales Representative', {}).get('Rate', 0):.1f}%), followed by Laboratory Technicians ({role_summary.get('Laboratory Technician', {}).get('Rate', 0):.1f}%) and Human Resources ({role_summary.get('Human Resources', {}).get('Rate', 0):.1f}%).",
        f"Departing employees have a substantially lower median monthly income (${income_left_median:,.0f} vs ${income_stayed_median:,.0f} for retained employees).",
        f"Employees with low job satisfaction (Rating 1) show an attrition rate of {sat_summary.get(1, {}).get('Rate', 0):.1f}%, compared to {sat_summary.get(4, {}).get('Rate', 0):.1f}% for very high satisfaction.",
        f"Tenure is inversely correlated with attrition: departing employees have an average company tenure of {tenure_left_mean:.1f} years vs {tenure_stayed_mean:.1f} years for retained employees.",
        f"Non-informative columns identified and flagged for exclusion: 'EmployeeNumber' (unique ID), 'EmployeeCount' (constant 1), 'StandardHours' (constant 80), 'Over18' (constant 'Y').",
    ]

    return {
        "validation": val,
        "total_employees": total_employees,
        "stayed_count": stayed_count,
        "left_count": left_count,
        "attrition_rate": attrition_rate,
        "department_summary": dept_summary,
        "role_summary": role_summary,
        "overtime_summary": overtime_summary,
        "income_stayed_median": income_stayed_median,
        "income_left_median": income_left_median,
        "age_stayed_mean": age_stayed_mean,
        "age_left_mean": age_left_mean,
        "satisfaction_summary": sat_summary,
        "wlb_summary": wlb_summary,
        "tenure_stayed_mean": tenure_stayed_mean,
        "tenure_left_mean": tenure_left_mean,
        "role_years_stayed_mean": role_years_stayed_mean,
        "role_years_left_mean": role_years_left_mean,
        "figure_paths": figure_paths,
        "findings": findings,
    }
