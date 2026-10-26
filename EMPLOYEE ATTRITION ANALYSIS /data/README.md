# IBM HR Analytics Employee Attrition & Performance Dataset

## 1. Overview & Problem Context
Employee attrition (the rate at which employees leave an organization) presents significant financial, operational, and cultural challenges for organizations. The cost of replacing an employee includes recruiting expenses, onboarding, lost productivity, and domain knowledge drain.

This dataset is used to analyze demographic, compensation, engagement, and work-environment factors associated with voluntary and involuntary employee turnover, and to develop supervised machine learning models to estimate attrition risk.

---

## 2. Dataset Provenance & Fictional Nature Notice

> **IMPORTANT NOTICE:**  
> The **IBM HR Analytics Employee Attrition & Performance** dataset is a **synthetic / fictional dataset** created by IBM data scientists for use with IBM Watson Analytics and machine learning education.  
> - It does **not** represent any real individual, workforce, or specific company.  
> - All names, numbers, and employee records are entirely fictitious.  
> - This project is designed strictly for **educational and portfolio demonstration purposes**. It must **not** be used for real-world consequential employment decisions (hiring, promotion, compensation, or termination).

- **Original Creator:** IBM Data Scientists  
- **Popular Public Source:** [Kaggle: IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
- **Canonical Filename:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`  
- **License / Terms:** Open Database License (ODbL) / Public Community Data. Fictional sample data released for educational evaluation.

---

## 3. Dataset Characteristics

- **Total Observations:** 1,470 employee records
- **Total Features:** 35 columns
- **Target Variable:** `Attrition` (`'Yes'` = employee left, `'No'` = employee stayed)
- **Class Balance:**
  - `'No'` (Stayed): 1,233 (83.88%)
  - `'Yes'` (Left): 237 (16.12%)
  - Imbalance Ratio: ~ 5.2 : 1
- **Missing Values:** 0 null values across all columns
- **Duplicate Records:** 0 duplicated rows

---

## 4. Feature Dictionary

### Identifiers & Non-Informative Columns (Excluded from Modeling)
1. `EmployeeNumber`: Unique numeric identifier for each employee record (1 to 2068).
2. `EmployeeCount`: Constant value (`1`) for all rows. Zero variance.
3. `StandardHours`: Constant value (`80`) for all rows (bi-weekly standard hours). Zero variance.
4. `Over18`: Constant value (`'Y'`) for all rows. Zero variance.

### Target Feature
- `Attrition`: Categorical (`'Yes'`, `'No'`). Encoded as binary (`1` for Yes, `0` for No).

### Demographic & Personal Characteristics
- `Age`: Numeric (18 – 60). Employee chronological age.
- `Gender`: Categorical (`'Female'`, `'Male'`).
- `MaritalStatus`: Categorical (`'Single'`, `'Married'`, `'Divorced'`).
- `Education`: Ordinal rating (1: Below College, 2: College, 3: Bachelor, 4: Master, 5: Doctor).
- `EducationField`: Categorical (`'Life Sciences'`, `'Medical'`, `'Marketing'`, `'Technical Degree'`, `'Human Resources'`, `'Other'`).
- `DistanceFromHome`: Numeric (1 – 29 km/miles). Commute distance.

### Job Role & Department
- `Department`: Categorical (`'Sales'`, `'Research & Development'`, `'Human Resources'`).
- `JobRole`: Categorical (9 roles: Healthcare Representative, Human Resources, Laboratory Technician, Manager, Manufacturing Director, Research Director, Research Scientist, Sales Executive, Sales Representative).
- `JobLevel`: Ordinal hierarchy level (1 to 5).
- `BusinessTravel`: Categorical (`'Non-Travel'`, `'Travel_Rarely'`, `'Travel_Frequently'`).

### Compensation & Equity
- `DailyRate`: Numeric ($102 – $1,499).
- `HourlyRate`: Numeric ($30 – $100).
- `MonthlyIncome`: Numeric ($1,009 – $19,999). Monthly base salary.
- `MonthlyRate`: Numeric ($2,094 – $26,999).
- `PercentSalaryHike`: Numeric (11% – 25%). Percentage increase in last salary review.
- `StockOptionLevel`: Ordinal (0 to 3). Stock option grant level.

### Employee Engagement, Ratings & Work Environment
- `EnvironmentSatisfaction`: Ordinal rating (1: Low, 2: Medium, 3: High, 4: Very High).
- `JobInvolvement`: Ordinal rating (1: Low, 2: Medium, 3: High, 4: Very High).
- `JobSatisfaction`: Ordinal rating (1: Low, 2: Medium, 3: High, 4: Very High).
- `PerformanceRating`: Ordinal rating (3: Excellent, 4: Outstanding).
- `RelationshipSatisfaction`: Ordinal rating (1: Low, 2: Medium, 3: High, 4: Very High).
- `WorkLifeBalance`: Ordinal rating (1: Bad, 2: Good, 3: Better, 4: Best).
- `OverTime`: Categorical (`'Yes'`, `'No'`). Whether the employee works overtime.

### Work History & Tenure
- `NumCompaniesWorked`: Numeric (0 – 9). Number of previous employers.
- `TotalWorkingYears`: Numeric (0 – 40). Total professional experience.
- `TrainingTimesLastYear`: Numeric (0 – 6). Number of training sessions attended last year.
- `YearsAtCompany`: Numeric (0 – 40). Tenure at current company.
- `YearsInCurrentRole`: Numeric (0 – 18). Years in current position.
- `YearsSinceLastPromotion`: Numeric (0 – 15). Years since last job promotion.
- `YearsWithCurrManager`: Numeric (0 – 17). Years reporting to current direct manager.

---

## 5. How to Download Manually (If Needed)

If downloading the file afresh via Kaggle:
1. Navigate to [Kaggle Dataset Page](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
2. Click **Download (50 kB)**. Log in if prompted by Kaggle.
3. Unzip the downloaded archive `archive.zip`.
4. Move `WA_Fn-UseC_-HR-Employee-Attrition.csv` into this `data/` directory.
