# Product Requirements Document (PRD)
| **Project Name** | Medical Cost Prediction |
| :--- | :--- |
| **Status** | Project scoping |
| **Date** | December 5, 2025 |
| **Data Source** | Medical Expenditure Panel Survey (MEPS) |


## Executive Summary
The **Medical Cost Prediction App** is a consumer-facing web application that uses machine learning trained on the Medical Expenditure Panel Survey (MEPS) to predict annual healthcare costs.

**The Problem:** Healthcare pricing is a "black box." While insurance portals show *unit prices* for individual treatments (e.g., "cost of an MRI"), consumers struggle to predict their total expected costs for the entire year. Fixed calculators (e.g., "add $500 per child") are inaccurate because they ignore health status, and insurance tools require specific procedure codes that often users don't know.

**Our Solution:** This tool enables users to predict their expected healthcare costs based on easily available information instead of complex medical records. Users can simply enter their age, sex, insurance status, self-rated health, and pre-existing health conditions to receive a personalized forecast for expected healthcare costs for the upcoming year. This forecast can be used for FSA/HSA contributions and emergency fund planning.

**How it works:** The web app is powered by a machine learning model trained on the Medical Expenditure Panel Survey (MEPS)—the gold standard for U.S. healthcare cost data. MEPS captures what people with specific demographic and health profiles *actually* spent on healthcare. Our ML model learns these complex, often nonlinear patterns and makes them accessible through a simple web app, providing a free, data-driven financial planning tool for everyone.


## User Personas
| Persona | Description | Primary Need |
| :--- | :--- | :--- |
| **The Open Enrollment Planner** | Employees deciding how much to contribute to FSAs/HSAs during open enrollment. | "Should I contribute $1,000 or $3,000?" |
| **The Budgeter** | Individuals with tight budgets needing to anticipate potential medical expenses. | "What's the worst-case scenario I should prepare for?" |
| **The Newly Diagnosed** | Users recently diagnosed with a chronic condition (e.g., Diabetes). | "How does this diagnosis impact my financial bottom line?" |
| **The Gig Worker** | Uninsured or underinsured individuals weighing coverage options. | "What's the financial risk of skipping coverage vs. buying a plan?" |
| **The Caregiver** | The "sandwich generation" estimating costs for an elderly parent. | "How much should I budget for my parent's healthcare?" |


## Competitive Positioning

### Market Gap
No existing tool combines **annual total cost prediction** with **simple, consumer-friendly inputs**. Current healthcare cost tools fall into three categories, none of which solve our users' core problem:

1. **Procedure-Specific Cost Estimators** (FAIR Health, Healthcare Bluebook, hospital-based tools): Require users to know CPT codes or specific procedures. Solve "How much will THIS procedure cost?" not "How much should I budget for the year?"
2. **Insurance Premium Calculators** (KFF Marketplace, state exchanges): Only estimate cost of insurance, not actual healthcare spending.
3. **FSA/HSA Contribution Calculators** (FSAFEDS, HSAstore): Require users to already know their expected annual costs, they don't predict it.

### Positive Differentiation
This tool is the **only ML-powered annual healthcare cost predictor** designed for consumers who lack specialized medical or insurance knowledge. Key advantages:

| **Our Approach** | **Competitor Tools** |
| :--- | :--- |
| Predicts total annual spending | Estimates per-procedure costs |
| 10 simple inputs (< 1 min) | Requires CPT codes, deductibles, provider selection |
| ML-trained on actual spending (MEPS) | Insurance rate tables or manual user input |
| Free, no login required | Often gated behind insurance portals |

### UX-First Rationale
The 10-input constraint is a feature rather than a limitation. Our personas (Open Enrollment Planners, Budgeters) need directional accuracy for financial planning (e.g., "Should I contribute $1,000 or $3,000 to my FSA?"), not clinical precision. Being within $500 (our MAE target) is sufficient for these decisions. Sacrificing 2% accuracy to achieve 80%+ completion rate is a reasonable trade-off.


## Functional Requirements

### User Input 
The UI must be a simple form with no more than 10 inputs on a single page. Inputs are mapped to MEPS variables.

| ID | UI Label | UI Element | Value Range | MEPS Variable |
| :--- | :--- | :--- | :--- | :--- |
| **IN-01** | Age | `gr.Number` | [18, 85] | `AGE23X` |
| **IN-02** | Sex | `gr.Radio` | ["Male", "Female"] | `SEX` |
| **IN-03** | Region | `gr.Dropdown` | ["Northeast", "Midwest", "South", "West"] | `REGION23` |
| **IN-04** | Income | `gr.Dropdown` | ["Low (<$30k)", "Middle", "High (>$100k)"] | `POVCAT23` |
| **IN-05** | Insurance Status | `gr.Dropdown` | ["Private", "Public (Medicare/Medicaid)", "Uninsured"] | `INSCOV23` |
| **IN-06** | Physical Health | `gr.Radio` | ["(1) Poor", "(2) Fair", "(3) Good", "(4) Very Good", "(5) Excellent"] | `RTHLTH31` |
| **IN-07** | Mental Health | `gr.Radio` | ["(1) Poor", "(2) Fair", "(3) Good", "(4) Very Good", "(5) Excellent"] | `MNHLTH31` |
| **IN-08** | Diabetes | `gr.Checkbox` | [True, False] | `DIABDX_M18` |
| **IN-09** | High Blood Pressure | `gr.Checkbox` | [True, False] | `HIBPDX` |
| **IN-10** | Smoker | `gr.Checkbox` | [True, False] | `ADSMOK42` |

### Prediction Engine
| ID | Requirement | Details |
| :--- | :--- | :--- |
| **FR-01** | **Imputation** | If optional fields are skipped, default to the mode for categorical features and the median for numerical features. |
| **FR-02** | **Inflation Adjustment** | Model predicts in 2023 dollars. Apply medical inflation multiplier: `Final_Prediction = Model_Output × (1 + Medical_Inflation_Rate)^(CurrentYear - 2023)` |
| **FR-03** | **Cost Range** | Generate 25th–75th percentile range to communicate prediction uncertainty. Never output a single point estimate. |
| **FR-04** | **Cost Drivers** | Compute SHAP values for each prediction to explain feature contributions as dollar impacts. |
| **FR-05** | **Comparison Benchmarks** | Compare user's prediction to: (1) National average, (2) Average for their age group. Pre-compute benchmarks from MEPS data. |

### Result Display
| ID | Component | Description | UI Element | Example |
| :--- | :--- | :--- | :--- | :--- |
| **UI-01** | **Cost Range** | Large, prominent display of cost prediction as a range. | `gr.Markdown` | "Estimated Healthcare Cost for Next Year: **$1,450 – $2,100**" |
| **UI-02** | **Cost Drivers** | Explanation of key cost drivers and their dollar impact (SHAP). | `gr.Markdown` | "Your Diabetes Diagnosis (+$1,200), your Age (+$400), but your "Excellent" self-reported health lowered the estimate by (-$300)" |
| **UI-03** | **Comparison Benchmarks** | Bar chart comparing user vs. national and age group benchmarks. | `gr.Plot` | "Typical American (median): $4,800 vs. Typical for Age 45–54 (median): $3,200" |


## Data & Machine Learning Specifications

### Dataset
*   **Source:** MEPS-HC 2023 Full Year Consolidated Data File (H251).
*   **Documentation:** [H251 Codebook](https://meps.ahrq.gov/data_stats/download_data_files_codebook.jsp?PUFId=H251).

### Feature Selection 
The primary goal is a fast, frictionless user experience. We prioritize usability over  predictive power if it requires complex inputs. 

**Feature Selection Principles**:
1.  **UX-First Constraint**: Maximum of 10 inputs to ensure user completion in under 1 minute.
2.  **Consumer Accessibility**: Inputs must be information users know offhand (e.g., age, self-rated health). The user doesn't need to leave their chair to find an insurance card, past bill, or medical record. No asking for specifics like "deductible amount" or "ICD-10 codes" that require mental effort or looking up technical terms.
3.  **Optimize Within Constraints**: Among the pool of "accessible" inputs, select the variables with the highest feature importance to maximize predictive power within the specific UX constraints.

**Feature Selection Process**:
1.  **Candidate Screening**: Identify all MEPS variables that a layperson can answer easily without having to look something up or think too hard.
2.  **Feature Importance Ranking**: Train preliminary models on these candidate features to obtain feature importance scores.
3.  **Final Feature Selection**: Select the top 10 or less features that maximize predictive power within the strict user experience constraints.

### Feature Mapping (MEPS to UI)
The model will utilize the following features mapping to user inputs:

| MEPS Variable | Data Type | UI Input |
| :--- | :--- | :--- |
| `AGE23X` | Continuous | Slider |
| `SEX` | Binary | Toggle |
| `REGION23` | Categorical | Dropdown (Map from Zip) |
| `POVCAT23` | Ordinal (1-5) | Dropdown (Income Brackets) |
| `INSCOV23` | Categorical (1-3) | Dropdown |
| `RTHLTH31` | Ordinal (1-5) | Radio |
| `MNHLTH31` | Ordinal (1-5) | Radio |
| `DIABDX_M18` | Binary | Checkbox |
| `HIBPDX` | Binary | Checkbox |
| `ADSMOK42` | Binary | Checkbox |

### Model Architecture
*   **Algorithm:** Gradient Boosting Regressor (XGBoost or LightGBM).
*   **Objective:** `TOTEXP23` (Total Health Care Expenditures).
*   **Handling Zeros:** Use a Two-Part Model (Hurdle Model) or Tweedie Objective function to handle zero-inflated cost data.
*   **Weighting:** Training must utilize `PERWT23F` (Person Weight) to ensure national representation.
*   **Preprocessing:** Log-transformation of target variable `log(TOTEXP23 + 1)` recommended for training stability.


## Non-Functional Requirements

### Privacy & Security
*   **NFR-01: Stateless Operation.** The app must NOT store any user input in a database. Data exists only in RAM during the session (Ephemeral).
*   **NFR-02: No PII.** No names, emails, exact addresses, or SSNs shall be requested.

### Performance
*   **NFR-03: Latency.** Inference prediction must be returned in < 200ms.
*   **NFR-04: Mobile responsive.** The UI must adapt to smartphone screens.

### Reliability
*   **NFR-05: Fallback Mode.** If the user doesn't provide an input, the app must display an informative message or impute the value.


## UI/UX Guidelines
*   **Tone:** Helpful, calm, non-judgmental. Avoid complex medical jargon (e.g. "High Blood Pressure" instead of "Hypertension").
*   **Visuals:** Use trust-building colors (Blues/Greens).
*   **Disclaimer:** A permanent footer stating: *"This tool is for educational purposes only. It is a statistical estimate based on 2023 national data, not a medical billing quote."*


## Technical Stack Recommendation
*   **Model Training:** Python (Scikit-Learn, XGBoost). Model serialized as `.joblib`.
*   **Web App:** 
    *   **Frontend**: Gradio or Streamlit (for demo).
    *   **Backend**: FastAPI (for production).
*   **Hosting:** 
    *   **Source Code**: GitHub.
    *   **Model/Pipeline**: Hugging Face Hub.
    *   **Web App**: Hugging Face Spaces.


## Success Metrics
*   **Model Accuracy:** Mean Absolute Error (MAE) on the test set is within $500 of the baseline MEPS benchmarks for the median patient.
*   **Completion Rate:** > 80% of users who start the questionnaire complete it.
*   **User Satisfaction:** Positive sentiment on optional "Was this helpful?" feedback.


## Risk Assessment
| Risk | Example | Mitigation Strategy |
| :--- | :--- | :--- |
| **Outlier Prediction** | Model predicts extreme costs ($100k+) for a standard user. | Consider implementing "Guardrails" in the code to cap displayed predictions at the 95th percentile with a "High Cost Risk" label instead of a raw number. |
| **Bias/Fairness** | Model consistently under-predicts needs for low-income users due to historical access barriers. | Perform a Fairness Audit. Include income as a feature so the user sees that income impacts the prediction. |
| **Data Aging** | 2023 data becomes outdated. | Inform user about limitations of the model. Consider applying a "Medical Inflation Factor" adjustment to the final output. |
