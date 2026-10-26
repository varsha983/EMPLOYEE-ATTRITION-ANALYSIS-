"""Streamlit HR Analytics & Employee Attrition Prediction Dashboard."""

import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Ensure matplotlib uses safe headless backend
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Add root folder to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import (
    load_raw_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLS,
    TARGET_COL,
)
from src.models import predict_employee_attrition

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Analysis & Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #2980b9;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-card-risk {
        background-color: #fdf3f2;
        border-left: 5px solid #d9534f;
        padding: 1.2rem;
        border-radius: 8px;
    }
    .metric-card-safe {
        background-color: #f4fbf7;
        border-left: 5px solid #27ae60;
        padding: 1.2rem;
        border-radius: 8px;
    }
    .disclaimer-box {
        background-color: #fff8e1;
        border-left: 5px solid #ffa000;
        padding: 1rem;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_dataset():
    """Load and cache the raw dataset."""
    return load_raw_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


@st.cache_resource
def get_model_and_metadata():
    """Load and cache the trained model pipeline and metadata."""
    model_path = Path("models/best_model.joblib")
    meta_path = Path("models/model_metadata.json")
    
    pipeline = joblib.load(model_path) if model_path.exists() else None
    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    return pipeline, metadata


df = get_dataset()
pipeline, metadata = get_model_and_metadata()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/group.png", width=64)
st.sidebar.title("HR Analytics Hub")
st.sidebar.markdown("**Employee Attrition Intelligence**")

menu = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive Overview",
        "👥 Workforce Analytics",
        "🤖 Machine Learning Models",
        "🔮 Interactive Risk Predictor",
        "⚖️ Ethics & Data Governance",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Info**  
    • Dataset: IBM HR Analytics (Fictional)  
    • Records: 1,470 employees  
    • Primary Model: Logistic Regression  
    • Environment: Python 3.12 (venv)  
    """
)


# ==============================================================================
# SECTION 1: EXECUTIVE OVERVIEW
# ==============================================================================
if menu == "📊 Executive Overview":
    st.markdown('<div class="main-header">Employee Attrition Analysis & Predictive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive summary of organizational retention metrics, turnover drivers, and proactive retention models.</div>', unsafe_allow_html=True)

    # Core Metrics
    total_employees = len(df)
    left_count = int((df[TARGET_COL] == "Yes").sum())
    stayed_count = total_employees - left_count
    attrition_rate = (left_count / total_employees) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Workforce", f"{total_employees:,}", help="Total number of employee records evaluated")
    with col2:
        st.metric("Retained Employees", f"{stayed_count:,}", f"{(stayed_count/total_employees)*100:.1f}% Retained")
    with col3:
        st.metric("Departed Employees", f"{left_count:,}", delta=f"-{attrition_rate:.1f}% Attrition", delta_color="inverse")
    with col4:
        st.metric("Baseline Attrition Rate", f"{attrition_rate:.2f}%", help="Industry average typically ranges 12-18%")

    st.markdown("---")

    # High-level overview columns
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.subheader("Workforce Status Breakdown")
        fig_path = Path("reports/figures/01_attrition_distribution.png")
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
        else:
            st.info("Run `python train.py` to generate report figures.")

    with c2:
        st.subheader("Key Business Insights")
        st.markdown(
            f"""
            - **Workforce Health**: The overall turnover rate stands at **{attrition_rate:.1f}%**, reflecting an imbalance ratio of **5.2 : 1** between retained and departing personnel.
            - **Overtime Risk Surge**: Employees who work overtime exhibit an attrition rate of **30.5%**, compared to just **10.4%** for those who do not—a **3x multiplier** on departure risk.
            - **Critical Job Roles**: **Sales Representatives** experience the highest role turnover (**39.8%**), followed by **Laboratory Technicians** (**23.9%**) and **Human Resources** (**23.1%**).
            - **Compensation Gap**: Departed personnel had a median monthly compensation of **$3,202**, compared to **$5,204** for retained colleagues (a 38.5% lower compensation profile).
            - **Tenure Vulnerability**: Employees in their first 2 years demonstrate significantly higher departure rates than veteran staff.
            """
        )

        st.info("💡 **Actionable Recommendation**: Targeted retention strategies should prioritize early-tenure Sales Representatives and Laboratory Technicians, while actively mitigating excessive uncompensated overtime.")


# ==============================================================================
# SECTION 2: WORKFORCE ANALYTICS
# ==============================================================================
elif menu == "👥 Workforce Analytics":
    st.markdown('<div class="main-header">Workforce Distribution & Attrition Factors</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep-dive multidimensional analysis into departments, roles, demographics, satisfaction, and tenure.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏢 Department & Job Roles",
        "💰 Age & Compensation",
        "⏱️ OverTime & Work-Life Balance",
        "📈 Tenure & Retention Dynamics",
    ])

    with tab1:
        st.subheader("Turnover Across Business Functions")
        p2 = Path("reports/figures/02_attrition_by_department_role.png")
        if p2.exists():
            st.image(str(p2), use_container_width=True)
        
        st.markdown("#### Departmental Summary Data")
        dept_df = (
            df.groupby("Department")
            .agg(
                Total_Headcount=("Attrition", "count"),
                Departures=("Attrition", lambda s: (s == "Yes").sum()),
                Attrition_Rate=("Attrition", lambda s: f"{(s == 'Yes').mean() * 100:.2f}%"),
            )
            .reset_index()
        )
        st.dataframe(dept_df, use_container_width=True)

    with tab2:
        st.subheader("Age Distribution & Compensation Profiles")
        p3 = Path("reports/figures/03_attrition_by_age_income.png")
        if p3.exists():
            st.image(str(p3), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                """
                **Age Insights:**
                - Younger employees (under 30) exhibit higher mobility and turnover.
                - Average age of departed employees is **33.6 years**, compared to **37.6 years** for retained staff.
                """
            )
        with col_b:
            st.markdown(
                """
                **Income Insights:**
                - Low monthly income correlates strongly with turnover in entry-level positions.
                - Median income for departed employees is **$3,202/month** vs. **$5,204/month** for stayers.
                """
            )

    with tab3:
        st.subheader("Overtime Impact & Engagement Ratings")
        col_c, col_d = st.columns(2)
        with col_c:
            p4 = Path("reports/figures/04_attrition_by_overtime.png")
            if p4.exists():
                st.image(str(p4), use_container_width=True)
        with col_d:
            p5 = Path("reports/figures/05_attrition_by_satisfaction_wlb.png")
            if p5.exists():
                st.image(str(p5), use_container_width=True)

        st.markdown(
            """
            - **Overtime Disparity**: 127 out of 416 employees working OverTime left the organization (30.5%), while only 110 of 1054 non-overtime staff left (10.4%).
            - **Satisfaction**: Employees rating Job Satisfaction as 1 (Low) exhibit an attrition rate of 22.8%, twice as high as those rating 4 (Very High, 11.3%).
            """
        )

    with tab4:
        st.subheader("Tenure & Promotion Trajectories")
        p6 = Path("reports/figures/06_attrition_by_tenure.png")
        if p6.exists():
            st.image(str(p6), use_container_width=True)

        st.markdown(
            """
            - **The "First-Year Flight" Phenomenon**: Highest attrition occurs within the first 12–24 months of joining.
            - **Stagnation Indicator**: Extended periods without promotion combined with low manager continuity increases flight probability.
            """
        )


# ==============================================================================
# SECTION 3: MACHINE LEARNING MODELS
# ==============================================================================
elif menu == "🤖 Machine Learning Models":
    st.markdown('<div class="main-header">Predictive Model Comparison & Interpretability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating Logistic Regression, Decision Tree, and Random Forest on held-out test data.</div>', unsafe_allow_html=True)

    if not metadata:
        st.warning("Model metadata not found. Please run `python train.py` first.")
    else:
        st.subheader("Model Evaluation on Stratified Test Data (294 Employees)")
        
        # Build comparison dataframe from actual metadata
        summary = metadata.get("evaluation_summary", {})
        table_data = []
        for model_name, metrics in summary.items():
            table_data.append({
                "Model Architecture": model_name,
                "5-Fold CV ROC-AUC": f"{metrics.get('cv_roc_auc_mean', 0):.4f}",
                "Test Accuracy": f"{metrics.get('test_accuracy', 0):.4f}",
                "Precision": f"{metrics.get('test_precision', 0):.4f}",
                "Recall (Sensitivity)": f"{metrics.get('test_recall', 0):.4f}",
                "F1-Score": f"{metrics.get('test_f1', 0):.4f}",
                "ROC-AUC": f"{metrics.get('test_roc_auc', 0):.4f}",
            })
            
        comp_df = pd.DataFrame(table_data)
        st.dataframe(comp_df, use_container_width=True)

        st.markdown(
            f"""
            > **Champion Model Selected: `{metadata.get('best_model_name', 'Logistic Regression')}`**  
            > In human resource retention, a **False Negative** (failing to recognize an at-risk employee who then leaves) is substantially more costly than a **False Positive** (reaching out with retention support to someone who was planning to stay).  
            > By applying `class_weight='balanced'`, **Logistic Regression achieves a Recall of 63.8%** and **ROC-AUC of 0.8070**, identifying nearly two-thirds of all departing personnel on unseen test data.
            """
        )

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Receiver Operating Characteristic (ROC)")
            p7 = Path("reports/figures/07_model_roc_curves.png")
            if p7.exists():
                st.image(str(p7), use_container_width=True)

        with col2:
            st.subheader("Test Set Confusion Matrices")
            p8 = Path("reports/figures/08_model_confusion_matrices.png")
            if p8.exists():
                st.image(str(p8), use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Importance & Predictive Drivers")
        p9 = Path("reports/figures/09_feature_importances.png")
        if p9.exists():
            st.image(str(p9), use_container_width=True)


# ==============================================================================
# SECTION 4: INTERACTIVE PREDICTOR
# ==============================================================================
elif menu == "🔮 Interactive Risk Predictor":
    st.markdown('<div class="main-header">Interactive Employee Attrition Risk Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate employee profiles to evaluate estimated flight risk and review contributing factors.</div>', unsafe_allow_html=True)

    if pipeline is None:
        st.error("Model pipeline file `models/best_model.joblib` not found. Please execute `python train.py` first.")
    else:
        # Presets selector
        preset = st.selectbox(
            "Select a Simulation Scenario Preset:",
            [
                "Custom Input",
                "High-Risk Profile (Overtime, Entry Role, Low Income)",
                "Average Retention Profile (Mid-Level Developer)",
                "Senior Veteran Profile (Long Tenure, High Satisfaction)",
            ],
            help="Choose a pre-filled profile or customize every parameter below."
        )

        # Set default values based on preset
        if preset == "High-Risk Profile (Overtime, Entry Role, Low Income)":
            def_age = 25
            def_dept = "Sales"
            def_role = "Sales Representative"
            def_income = 2300
            def_ot = "Yes"
            def_travel = "Travel_Frequently"
            def_dist = 22
            def_sat = 1
            def_wlb = 1
            def_tenure = 1
            def_role_yrs = 1
            def_tot_exp = 3
            def_mgr_yrs = 0
            def_promo_yrs = 0
            def_stock = 0
            def_marital = "Single"
            def_gender = "Female"
            def_num_comp = 3
        elif preset == "Senior Veteran Profile (Long Tenure, High Satisfaction)":
            def_age = 48
            def_dept = "Research & Development"
            def_role = "Research Director"
            def_income = 14500
            def_ot = "No"
            def_travel = "Non-Travel"
            def_dist = 3
            def_sat = 4
            def_wlb = 3
            def_tenure = 14
            def_role_yrs = 8
            def_tot_exp = 24
            def_mgr_yrs = 7
            def_promo_yrs = 2
            def_stock = 2
            def_marital = "Married"
            def_gender = "Male"
            def_num_comp = 1
        elif preset == "Average Retention Profile (Mid-Level Developer)":
            def_age = 35
            def_dept = "Research & Development"
            def_role = "Research Scientist"
            def_income = 5800
            def_ot = "No"
            def_travel = "Travel_Rarely"
            def_dist = 8
            def_sat = 3
            def_wlb = 3
            def_tenure = 6
            def_role_yrs = 4
            def_tot_exp = 10
            def_mgr_yrs = 4
            def_promo_yrs = 1
            def_stock = 1
            def_marital = "Married"
            def_gender = "Male"
            def_num_comp = 2
        else:
            def_age = 32
            def_dept = "Research & Development"
            def_role = "Laboratory Technician"
            def_income = 4000
            def_ot = "No"
            def_travel = "Travel_Rarely"
            def_dist = 5
            def_sat = 2
            def_wlb = 2
            def_tenure = 3
            def_role_yrs = 2
            def_tot_exp = 6
            def_mgr_yrs = 2
            def_promo_yrs = 0
            def_stock = 0
            def_marital = "Single"
            def_gender = "Female"
            def_num_comp = 1

        with st.form("employee_prediction_form"):
            st.markdown("#### 1. Personal & Demographics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                age = st.slider("Age", 18, 65, def_age)
                gender = st.selectbox("Gender", ["Female", "Male"], index=0 if def_gender == "Female" else 1)
            with col2:
                marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=["Single", "Married", "Divorced"].index(def_marital))
                distance = st.slider("Distance From Home (miles)", 1, 30, def_dist)
            with col3:
                education = st.select_slider(
                    "Education Level",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    format_func=lambda x: {1: "1-Below College", 2: "2-College", 3: "3-Bachelor", 4: "4-Master", 5: "5-Doctor"}[x]
                )
                edu_field = st.selectbox(
                    "Education Field",
                    ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
                    index=0
                )
            with col4:
                num_companies = st.slider("Num Companies Worked", 0, 9, def_num_comp)

            st.markdown("#### 2. Position & Compensation")
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                dept_options = ["Research & Development", "Sales", "Human Resources"]
                department = st.selectbox("Department", dept_options, index=dept_options.index(def_dept))
                role_options = [
                    "Sales Executive", "Research Scientist", "Laboratory Technician",
                    "Manufacturing Director", "Healthcare Representative", "Manager",
                    "Sales Representative", "Research Director", "Human Resources"
                ]
                job_role = st.selectbox("Job Role", role_options, index=role_options.index(def_role))
            with col6:
                job_level = st.slider("Job Level (Hierarchy 1-5)", 1, 5, 2 if def_income < 6000 else 4)
                monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=def_income, step=250)
            with col7:
                stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=def_stock)
                salary_hike = st.slider("Percent Salary Hike (%)", 11, 25, 14)
            with col8:
                daily_rate = st.number_input("Daily Rate ($)", 100, 1500, 800)
                hourly_rate = st.number_input("Hourly Rate ($)", 30, 100, 65)
                monthly_rate = st.number_input("Monthly Rate ($)", 2000, 27000, 14000)

            st.markdown("#### 3. Work Environment & Engagement")
            col9, col10, col11, col12 = st.columns(4)
            with col9:
                overtime = st.selectbox("Works OverTime?", ["No", "Yes"], index=0 if def_ot == "No" else 1)
                travel_options = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
                bus_travel = st.selectbox("Business Travel", travel_options, index=travel_options.index(def_travel))
            with col10:
                job_sat = st.select_slider(
                    "Job Satisfaction",
                    options=[1, 2, 3, 4],
                    value=def_sat,
                    format_func=lambda x: {1: "1-Low", 2: "2-Medium", 3: "3-High", 4: "4-Very High"}[x]
                )
                env_sat = st.select_slider(
                    "Environment Satisfaction",
                    options=[1, 2, 3, 4],
                    value=def_sat,
                    format_func=lambda x: {1: "1-Low", 2: "2-Medium", 3: "3-High", 4: "4-Very High"}[x]
                )
            with col11:
                wlb = st.select_slider(
                    "Work-Life Balance",
                    options=[1, 2, 3, 4],
                    value=def_wlb,
                    format_func=lambda x: {1: "1-Bad", 2: "2-Good", 3: "3-Better", 4: "4-Best"}[x]
                )
                job_inv = st.select_slider(
                    "Job Involvement",
                    options=[1, 2, 3, 4],
                    value=3,
                    format_func=lambda x: {1: "1-Low", 2: "2-Medium", 3: "3-High", 4: "4-Very High"}[x]
                )
            with col12:
                rel_sat = st.select_slider(
                    "Relationship Satisfaction",
                    options=[1, 2, 3, 4],
                    value=3,
                    format_func=lambda x: {1: "1-Low", 2: "2-Medium", 3: "3-High", 4: "4-Very High"}[x]
                )
                perf_rating = st.selectbox("Performance Rating", [3, 4], index=0, format_func=lambda x: f"{x}-{'Excellent' if x==3 else 'Outstanding'}")

            st.markdown("#### 4. Tenure & Experience")
            col13, col14, col15, col16 = st.columns(4)
            with col13:
                tot_working_years = st.slider("Total Working Years", 0, 40, def_tot_exp)
                years_at_company = st.slider("Years At Company", 0, 40, def_tenure)
            with col14:
                years_in_role = st.slider("Years In Current Role", 0, 18, def_role_yrs)
                years_with_mgr = st.slider("Years With Current Manager", 0, 17, def_mgr_yrs)
            with col15:
                years_since_promo = st.slider("Years Since Last Promotion", 0, 15, def_promo_yrs)
                trainings = st.slider("Training Times Last Year", 0, 6, 2)
            with col16:
                st.write("")  # Spacing
                st.write("")
                submit_button = st.form_submit_button("Run Risk Prediction", use_container_width=True, type="primary")

        if submit_button:
            # Construct row with exact feature names matching training pipeline
            input_dict = {
                "Age": age,
                "BusinessTravel": bus_travel,
                "DailyRate": daily_rate,
                "Department": department,
                "DistanceFromHome": distance,
                "Education": education,
                "EducationField": edu_field,
                "EnvironmentSatisfaction": env_sat,
                "Gender": gender,
                "HourlyRate": hourly_rate,
                "JobInvolvement": job_inv,
                "JobLevel": job_level,
                "JobRole": job_role,
                "JobSatisfaction": job_sat,
                "MaritalStatus": marital,
                "MonthlyIncome": monthly_income,
                "MonthlyRate": monthly_rate,
                "NumCompaniesWorked": num_companies,
                "OverTime": overtime,
                "PercentSalaryHike": salary_hike,
                "PerformanceRating": perf_rating,
                "RelationshipSatisfaction": rel_sat,
                "StockOptionLevel": stock_option,
                "TotalWorkingYears": tot_working_years,
                "TrainingTimesLastYear": trainings,
                "WorkLifeBalance": wlb,
                "YearsAtCompany": years_at_company,
                "YearsInCurrentRole": years_in_role,
                "YearsSinceLastPromotion": years_since_promo,
                "YearsWithCurrManager": years_with_mgr,
            }
            input_df = pd.DataFrame([input_dict])
            
            # Run inference
            res = predict_employee_attrition(pipeline, input_df)
            prob_pct = res["attrition_probability"] * 100

            st.markdown("---")
            st.subheader("Model Prediction Output")

            c_out1, c_out2, c_out3 = st.columns([1, 1, 1])
            with c_out1:
                if res["predicted_class"] == 1:
                    st.error(f"**Predicted Status: {res['predicted_label']}**")
                else:
                    st.success(f"**Predicted Status: {res['predicted_label']}**")

            with c_out2:
                st.metric(
                    "Estimated Attrition Probability",
                    f"{prob_pct:.1f}%",
                    help="Model estimated probability that this employee profile departs"
                )

            with c_out3:
                if res["risk_tier"] == "Elevated Risk":
                    st.warning(f"**Risk Level: {res['risk_tier']}**")
                elif res["risk_tier"] == "Moderate Risk":
                    st.info(f"**Risk Level: {res['risk_tier']}**")
                else:
                    st.success(f"**Risk Level: {res['risk_tier']}**")

            st.progress(res["attrition_probability"])

            # Explain contributing factors
            st.markdown("#### Primary Contributing Drivers for this Profile:")
            drivers = []
            if overtime == "Yes":
                drivers.append("⚠️ **OverTime Active**: Employee works overtime, historically associated with a 3x higher attrition rate.")
            if monthly_income < 3500:
                drivers.append(f"⚠️ **Below Median Compensation**: Monthly salary (${monthly_income:,}) is below company median ($5,204).")
            if job_sat <= 2:
                drivers.append("⚠️ **Low Job Satisfaction**: Rating of 1 or 2 significantly elevates flight likelihood.")
            if wlb <= 2:
                drivers.append("⚠️ **Suboptimal Work-Life Balance**: Strained work-life balance correlates with burnout.")
            if years_at_company <= 2:
                drivers.append("⚠️ **Early Tenure Stage**: First 2 years represent the highest vulnerability window.")
            if stock_option == 0:
                drivers.append("ℹ️ **Zero Stock Equity**: Employee holds no equity or vesting incentives.")
            
            if not drivers:
                drivers.append("✅ **Positive Retention Factors**: High satisfaction, competitive tenure, reasonable work-life balance, and no overtime.")

            for d in drivers:
                st.markdown(d)

            st.markdown(
                """
                <div class="disclaimer-box">
                <b>Interpretability Note:</b> Probabilities reflect statistical patterns derived from historical synthetic IBM HR data. 
                They do not represent deterministic outcomes or evaluate individual character or loyalty.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# SECTION 5: ETHICS & DATA GOVERNANCE
# ==============================================================================
elif menu == "⚖️ Ethics & Data Governance":
    st.markdown('<div class="main-header">Ethical HR Analytics & Governance Framework</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Responsible AI guidelines, privacy boundaries, and model limitations in workplace analytics.</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### 1. Educational Scope & Synthetic Data Disclaimer
        - **Dataset Source**: This application utilizes the **IBM HR Analytics Employee Attrition & Performance** dataset, an open synthetic benchmark released for data science education and algorithm evaluation.
        - **Non-Representativeness**: The data is **fictional** and does not model or identify any real organization, employer, or employee.
        - **Strict Prohibited Use**: **This project is strictly for academic and portfolio demonstration**. Under no circumstances should this system be used to inform, automate, or influence consequential employment actions—including hiring, salary adjustments, performance reviews, promotions, reassignments, or terminations.

        ### 2. Fair Employment & Anti-Bias Principles
        - **Protected Demographic Attributes**: While demographic attributes such as `Gender` and `Age` are included in the underlying benchmark dataset, real-world deployment requires strict audits under Title VII of the Civil Rights Act, the Equal Pay Act, the Age Discrimination in Employment Act (ADEA), and GDPR.
        - **Preventing Self-Fulfilling Disinvestment**: Flagging an employee as "likely to leave" can cause management to withhold training, promotions, or key projects, directly triggering the attrition the model predicted.
        - **Human-in-the-Loop Retention**: Predictive signals must only be used by HR leadership to diagnose macro-level organizational health (e.g. departmental burnout, compensation benchmarks) rather than to monitor individuals.

        ### 3. Technical Limitations & Data Leakage Prevention
        - **Correlation vs. Causation**: Feature importance scores indicate historical statistical associations, not causal links. For example, low tenure is associated with attrition, but passing time does not itself cause retention.
        - **Class Imbalance & Thresholding**: Employee turnover is naturally skewed (~16% positive rate). Precision-recall trade-offs must be calibrated to organizational resources.
        - **Concept Drift**: Workplace dynamics, remote work norms, and macroeconomic conditions evolve over time, necessitating continuous model monitoring and re-calibration.
        """
    )
