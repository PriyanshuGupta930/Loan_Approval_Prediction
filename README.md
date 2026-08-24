# Loan Approval Prediction

## Objective

- Develop a machine learning model to predict whether a loan application will be **approved or rejected**.
- Use applicant demographic, financial, credit-history, and property-related information for prediction.
- Compare multiple classification algorithms and identify their performance using appropriate evaluation metrics.
- Build a preprocessing and modeling pipeline that handles missing values, numerical transformations, categorical variables, and class imbalance.

## Dataset Information

- Dataset: `Loan_dataset.csv`
- Total observations: **614**
- Total variables: **13**
- Target variable: `Loan_Status`
- Target classes:
  - `1` → Loan Approved
  - `0` → Loan Rejected
- Features include:
  - Gender
  - Married
  - Dependents
  - Education
  - Self-Employed
  - Applicant Income
  - Coapplicant Income
  - Loan Amount
  - Loan Amount Term
  - Credit History
  - Property Area
- `Loan_ID` was removed because it is only an identifier and does not provide meaningful predictive information.

## Exploratory Data Analysis

- Performed initial inspection of the dataset to understand its structure, data types, and basic statistics.
- Analyzed the **target variable distribution** to identify class imbalance.
- Visualized the distribution of the target variable using a bar chart.
- Analyzed missing values across all variables.
- Visualized missing-value patterns to identify variables requiring treatment.
- Examined numerical-variable distributions using histograms.
- Used boxplots to identify the presence of skewness and potential outliers in numerical variables.
- Analyzed categorical variables using count plots.
- Examined the relationship between important categorical variables and loan approval using categorical distribution plots.
- Compared loan approval rates across relevant applicant characteristics.
- Investigated the relationship between numerical variables and the target variable through appropriate visualizations.
- Identified skewed numerical variables that required transformation before model training.

## Feature Engineering

- Converted the `Dependents` category `3+` into the numerical category `3`.
- Encoded the target variable:
  - `Y` → `1`
  - `N` → `0`
- Created `Total_Income` by combining applicant and co-applicant income:
  - `Total_Income = ApplicantIncome + CoapplicantIncome`
- Created `EMI` as a simplified measure of loan repayment burden:
  - `EMI = LoanAmount / Loan_Amount_Term`
- Created `Balance_Income` to represent income remaining after the estimated loan repayment burden:
  - `Balance_Income = Total_Income - (EMI × 1000)`

## Data Preprocessing

- Applied median imputation to missing numerical values.
- Applied most-frequent imputation to missing categorical values.
- Applied `log1p` transformation to skewed numerical variables.
- Standardized numerical variables using `StandardScaler`.
- Applied one-hot encoding to categorical variables.
- Used `drop='first'` to avoid redundant dummy variables.
- Used `handle_unknown='ignore'` to safely handle unseen categorical levels.
- Implemented preprocessing using a `ColumnTransformer` and integrated it with the modeling pipeline.

## Model Training

The following classification models were trained and tuned:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Support Vector Machine (SVM)**

### Hyperparameter Tuning

- Used **5-fold Stratified Cross-Validation** for model tuning.
- Used `GridSearchCV` for:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
- Used `RandomizedSearchCV` for XGBoost.
- ROC-AUC was used as the scoring metric during hyperparameter tuning.

## Class Imbalance Handling

- The training data showed a majority-to-minority class ratio of approximately **2.19:1**.
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to address class imbalance.
- SMOTE was placed inside the modeling pipeline so that oversampling was performed only on the training portion of each cross-validation fold.
- The test set was kept untouched during model evaluation.

## Model Evaluation

The trained models were evaluated using:

- **Accuracy** — overall proportion of correctly classified applications.
- **Precision** — proportion of predicted approvals that were actually approved.
- **Recall** — proportion of actual approved applications correctly identified.
- **F1-Score** — harmonic mean of precision and recall.
- **ROC-AUC** — measures the model's ability to distinguish between approved and rejected applications.
- **Confusion Matrix** — analyzes true positives, true negatives, false positives, and false negatives.
- **ROC Curve** — compares the classification performance of the models across different classification thresholds.
