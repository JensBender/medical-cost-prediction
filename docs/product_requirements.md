# Product Requirements Document (PRD)
| **Project Name** | Medical Cost Prediction |
| :--- | :--- |
| **Status** | Project scoping |
| **Date** | December 5, 2025 |
| **Data Source** | Medical Expenditure Panel Survey (MEPS) |


## Executive Summary
The **Medical Cost Planner** is a consumer-facing web application that uses machine learning trained on the Medical Expenditure Panel Survey (MEPS) to predict annual **out-of-pocket** healthcare costs.

**The Problem:** Healthcare pricing is a "black box." While insurance portals show unit prices for individual treatments (e.g., cost of an MRI), consumers struggle to predict their total expected costs for the entire year. Fixed calculators (e.g., "add $500 per child") are inaccurate because they ignore important factors such as a person's health conditions, and insurance tools require specific procedure codes that often users don't know.

**Our Solution:** This tool enables users to predict their expected healthcare costs based on easily available information instead of complex medical records. Users can simply enter their age, sex, insurance status, self-rated health, and pre-existing health conditions to receive a personalized forecast for expected out-of-pocket healthcare costs for the upcoming year. This forecast can be used for FSA/HSA contributions and emergency fund planning.

**How it works:** The web app is powered by a machine learning model trained on the Medical Expenditure Panel Survey (MEPS), which is considered the gold standard for U.S. healthcare cost data. MEPS captures what people with specific demographic and health profiles actually spent on healthcare. Our ML model learns these complex, often nonlinear patterns and makes them accessible through a simple web app, providing a free, data-driven financial planning tool for everyone.


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
No existing tool combines **ML-powered personalized predictions** with **simple, consumer-friendly inputs** for annual healthcare costs. Current tools fall into three categories, none of which solve our users' core problem:

1. **Procedure-Specific Cost Estimators** (FAIR Health, Healthcare Bluebook, hospital-based tools): Require users to know CPT codes or specific procedures. Solve "How much will THIS procedure cost?" not "How much should I budget for the year?"
2. **Insurance Premium Calculators** (KFF Marketplace, state exchanges): Only estimate cost of insurance, not actual healthcare spending.
3. **FSA/HSA Contribution Calculators** (FSAFEDS, HSAstore): Require users to already know their expected annual costs—they don't predict it.

**Closest Competitor**: The KFF (Kaiser Family Foundation) Household Health Spending Calculator provides annual cost estimates but uses **demographic subgroup averaging** rather than personalized ML predictions. It categorizes health status as simply "good" or "worse" and returns the average spending for broad demographic buckets (e.g., "single person, $50k income, employer coverage, good health"). This approach cannot capture individual-level variations (e.g., diabetes vs. hypertension) or complex non-linear interactions between features that ML models learn from individual-level data.

### Our Differentiation
This tool is the **only ML-powered out-of-pocket healthcare cost planner** with easily accessible inputs, designed for consumers who lack specialized medical or insurance knowledge.

| **Our ML Approach** | **Competitor Approaches** |
| :--- | :--- |
| **Personalized predictions** from individual-level MEPS data (28k+ records) | Demographic subgroup averages or insurance rate tables |
| **Granular health inputs**: 5-point scales for physical/mental health + specific chronic conditions (diabetes, hypertension, smoking) | Broad health categories ("good" vs. "worse") or no health inputs |
| **Explainable predictions**: SHAP values show cost drivers (e.g., "Diabetes +$1,200") | Black box averages with no explanation |
| **Uncertainty ranges**: 25th–75th percentile for planning worst-case scenarios | Single point estimates |
| **10 accessible inputs** (< 1 min completion) | Either requires CPT codes/deductibles OR oversimplified 4-5 broad buckets |
| **Free, no login required** | Often gated behind insurance portals |

### UX-First Rationale
The 10-input constraint is a feature rather than a limitation. Our personas (Open Enrollment Planners, Budgeters) need ballpark estimates for financial planning (e.g., "Should I contribute $1,000 or $3,000 to my FSA?"), not clinical precision. Being within $500 (our MAE target) is sufficient for these decisions. The ML model captures complex patterns from individual-level data while keeping inputs simple and accessible. Sacrificing 2% absolute accuracy to achieve 80%+ completion rate is the right trade-off for this use case.


## Functional Requirements

### User Input 
> **Status:** *Preliminary. Features shown below are the most promising candidates based on initial analysis. Final feature selection pending.*

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
| **FR-03** | **Cost Range** | Generate 25th–75th percentile range (typical range) and 90th percentile (budget-safe estimate) to communicate prediction uncertainty. Never output a single point estimate. |
| **FR-04** | **Cost Drivers** | Compute SHAP values for each prediction to explain feature contributions as dollar impacts. |
| **FR-05** | **Comparison Benchmarks** | Compare user's prediction to (1) national average and (2) average for their age group. Pre-compute benchmarks from MEPS data. |
| **FR-06** | **Confidence Indicator** | Flag predictions in the top 10% of the cost distribution (>90th percentile) as "High Uncertainty" to signal reduced model accuracy for extreme costs. |

### Result Display
| ID | Component | Description | UI Element | Example |
| :--- | :--- | :--- | :--- | :--- |
| **UI-01** | **Cost Range** | Large, prominent display of out-of-pocket cost prediction as a typical range, plus a budget-safe estimate for worst-case planning. | `gr.Markdown` | "Estimated Out-of-Pocket Healthcare Cost for Next Year: **$1,450 – $2,100** (typical range)<br>To be safe, budget up to: **$3,200**" |
| **UI-02** | **Cost Drivers** | Explanation of key cost drivers and their dollar impact (SHAP). | `gr.Markdown` | "Your Diabetes Diagnosis (+$1,200), your Age (+$400), but your "Excellent" self-reported health lowered the estimate by (-$300)" |
| **UI-03** | **Comparison Benchmarks** | Bar chart comparing user vs. national and age group benchmarks. | `gr.Plot` | "Typical American (median): $4,800 vs. Typical for Age 45–54 (median): $3,200" |
| **UI-04** | **Limitations Notice** | Contextual guidance to help users interpret their prediction. | `gr.Markdown` | "**ℹ️ About This Estimate**<br>• Based on 2023 national survey data; recent policy changes may affect actual costs.<br>• Does not include insurance premiums or over-the-counter medications.<br>• This is a statistical estimate. Actual costs depend on your specific plan, providers, and health events." |
| **UI-05** | **High-Cost Disclaimer** | Dynamic warning displayed when predicted cost exceeds the 90th percentile threshold. | `gr.Markdown` | "**⚠️ Note on This Estimate**<br>Your predicted cost is in the top 10% of healthcare spending. Estimates in this range have higher uncertainty because high costs are often driven by unpredictable events. Consider this a rough guideline rather than a precise forecast." |


## Data Specifications

### Dataset
*   **Source:** [MEPS-HC 2023 Full Year Consolidated Data File (H251)](https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-251).
*   **Documentation:** [MEPS-HC 2023 Documentation](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.shtml).
*   **Codebook:** [MEPS-HC 2023 Codebook](https://meps.ahrq.gov/data_stats/download_data_files_codebook.jsp?PUFId=H251).

### Target Variable
*   **Variable:** `TOTSLF23` — Total amount paid out-of-pocket by the person or their family for all medical events in the year 2023.
*   **Rationale:** Our primary personas (Open Enrollment Planners, Budgeters) need to know what **they will personally pay**, not the total cost shared across insurance and government payers. Out-of-pocket costs directly answer: "How much should I contribute to my FSA/HSA?" and "What's my financial exposure?"
*   **Note:** For uninsured users, out-of-pocket ≈ total cost, so this target remains appropriate across all insurance statuses.

<details>
<summary><strong>Detailed Breakdown of TOTSLF23</strong> (click to expand)</summary>

`TOTSLF23` is the aggregate sum of all out-of-pocket payments made by the person or their family for healthcare services received in 2023. It sums the "Self/Family" share of costs across all medical event categories. It includes the following service types:
*   **Office-Based Visits (`OBSLF23`):** Co-pays and deductibles for doctor's appointments, check-ups, and specialist visits.
*   **Prescribed Medicines (`RXSLF23`):** Out-of-pocket costs for filled prescriptions (does **not** include over-the-counter drugs).
*   **Hospital Inpatient Stays (`IPSLF23`):** Direct payments for overnight hospitalizations (room & board, treatments).
*   **Emergency Room Visits (`ERSLF23`):** Co-pays and bills for ER visits that did not result in an admission.
*   **Outpatient Department Visits (`OPSLF23`):** Costs for same-day surgeries, scans, or therapies at a hospital.
*   **Dental Care (`DVSLF23`):** Payments for cleanings, fillings, orthodontia, etc.
*   **Vision Services (`VISLF23`):** Costs for eye exams, glasses, and contact lenses.
*   **Home Health Care (`HHSLF23` + `HNSLF23`):** Payments for agency or independent home health providers.
*   **Other Medical (`OMSLF23`):** Equipment (crutches, hearing aids), ambulance services, and other miscellaneous supplies.

Notes:  
*   **Excludes Premiums:** Monthly insurance premiums (e.g., deducted from a paycheck) are not included. `TOTSLF23` only tracks payments for services received.
*   **Excludes Over-the-Counter (OTC) Drugs:** Expenses for non-prescription medications (e.g., Tylenol, vitamins) are not included.
*   **For uninsured users**: `TOTSLF23` will typically equal the **Total Expenditure** (`TOTTCH23`), as they bear the full cost unless they received charity care (which is not counted as an expenditure in MEPS).

</details>

### Feature Selection 
The primary goal is a fast, frictionless user experience. We prioritize usability over predictive power if it requires complex inputs. 

**Feature Selection Principles**:
1.  **UX-First Constraint**: Maximum of 10 inputs to ensure user completion in under 1 minute.
2.  **Consumer Accessibility**: Inputs must be information users know offhand (e.g., age, self-rated health). The user doesn't need to leave their chair to find an insurance card, past bill, or medical record. No asking for specifics like "deductible amount" or "ICD-10 codes" that require mental effort or looking up technical terms.
3.  **Optimize Within Constraints**: Among the pool of "accessible" inputs, select the variables with the highest feature importance to maximize predictive power within the UX constraints. 

**Feature Selection Process**:
1.  **Candidate Screening**: Identify all MEPS variables that a layperson can answer easily without having to look something up or think too hard.
2.  **Feature Importance Ranking**: Train preliminary models on these candidate features to obtain feature importance scores.
3.  **Final Feature Selection**: Select the top 10 or less features that maximize predictive power within the UX constraints.


## Machine Learning Specifications

### Data Preprocessing
All preprocessing steps are implemented as scikit-learn pipelines to ensure consistency between training and inference, prevent data leakage, and simplify deployment.

**Data Cleaning**  
Perform once before pipeline:

| Action | Rationale |
| :--- | :--- |
| Drop rows where `PERWT23F = 0` | Zero-weight respondents don't represent the population. |
| Handle MEPS Negative Codes | Convert `-1` (Inapplicable), `-8` (DK) to: `0` for binary conditions (assume "No"), `NaN` for ordinal health status (then impute). |

**Feature Preprocessing**  
Implemented via `ColumnTransformer`:

| Feature Type | Columns | Transformer | Notes |
| :--- | :--- | :--- | :--- |
| Numerical | `AGE23X`, `RTHLTH31`, `MNHLTH31` | `StandardScaler` | Health scales (1–5) treated as numerical |
| Ordinal | `POVCAT23` | `OrdinalEncoder` | Low < Middle < High; preserve ordering |
| Nominal | `SEX`, `REGION23`, `INSCOV23` | `OneHotEncoder` | Drop first to avoid multicollinearity |
| Binary | `DIABDX_M18`, `HIBPDX`, `ADSMOK42` | passthrough | Already 0/1 encoded |
| Engineered | — | `FunctionTransformer` | Create `COMORBIDITY_COUNT` = sum of 3 binary flags |

**Target Preprocessing**  
Different model families require different target variable handling. Implemented via `TransformedTargetRegressor` for log-transform branch.

| Model Family | Target Transform | Models | Rationale |
| :--- | :--- | :--- | :--- |
| Regression-based | `log(y + 1)` → inverse on prediction | Linear Regression, Elastic Net, KNN, MLP | Stabilizes variance for models assuming homoscedastic errors |
| Tree-based | None | Decision Tree, Random Forest, XGBoost | Non-parametric models; MAE/Tweedie criteria handle skew natively |

### Model Training
**Training Strategy**  
1.  **Baseline Models**: Train all candidate models with (mostly) default hyperparameters.
2.  **Model Selection**: Select best 2–4 models based on MdAE on validation set.
3.  **Hyperparameter Tuning**: Tune selected models via randomized search, evaluated on validation set.
4.  **Final Model Selection**: Select best-performing model based on validation set performance.
5.  **Final Evaluation**: Evaluate the selected final model on held-out test set ONCE for unbiased performance reporting.

**Baseline Models**  
All models use MAE-based training loss (instead of MSE) for robustness to the right-skewed, zero-inflated target variable distribution.

| Model | Training Loss | Preprocessing | Notes |
| :--- | :--- | :--- | :--- |
| Linear Regression | MSE (default) | Log-transform | Inverse-transform predictions |
| Elastic Net | MSE (default) | Log-transform | Inverse-transform predictions |
| K-Nearest Neighbors | N/A (distance-based) | Log-transform | Inverse-transform predictions |
| MLP (sklearn) | MSE (default) | Log-transform | Inverse-transform predictions |
| Decision Tree | `criterion='absolute_error'` | Native target | MAE-based splits |
| Random Forest | `criterion='absolute_error'` | Native target | MAE-based splits |
| XGBoost | `objective='reg:tweedie'` | Native target | Tweedie handles skew natively |

> **Note:** A Two-Part (Hurdle) Model, where part 1 predicts P(cost > 0) and part 2 predicts E[cost \| cost > 0], is a valid alternative for explicitly modeling zero-inflation. Consider as a future enhancement if single-stage models underperform on zero-cost users.

**Sample Weights**  
*   All models must use `PERWT23F` (person weight) via `sample_weight` parameter to ensure national representativeness.
*   Models without native `sample_weight` support should be excluded.

### Model Evaluation
Evaluate predictive performance of model and perform error analysis.

| ID | Evaluation Task | Details |
| :--- | :--- | :--- |
| **EV-01** | **Overall Performance** | Report overall MdAE on the full test set as the primary success metric. |
| **EV-02** | **Stratified Error Analysis** | Report MdAE separately for low (0–50th percentile), medium (50th–90th percentile), and high (90th+ percentile) cost tiers. This diagnoses where the model underperforms and quantifies heteroskedasticity. |
| **EV-03** | **Interval Calibration** | Report what % of actual costs fall within the predicted 25th–75th percentile range, both overall and for each cost tier. This diagnoses how accurate the prediction interval is across cost levels. Performance is expected to degrade for the high-cost tier due to (1) inherent unpredictability of high-cost events and (2) less training data in that range. |


## Non-Functional Requirements

### Privacy & Security
| ID | Requirement | Details |
| :--- | :--- | :--- |
| **NFR-01** | Ephemeral Sessions | No user data written to disk or database. All inputs remain in browser/RAM session state only. |
| **NFR-02** | No PII Collection | No names, emails, exact addresses, or SSNs shall be requested. |

### Performance & Usability
| ID | Requirement | Details |
| :--- | :--- | :--- |
| **NFR-03** | Latency | Inference prediction must return in < 200ms (server-side) or < 2 seconds (end-to-end including network). |
| **NFR-04** | Responsive Design | Expect ~65% desktop, ~35% mobile (typical for Hugging Face Spaces). Gradio handles responsive layouts natively. Ensure form inputs remain usable on smaller screens. |
| **NFR-05** | Fallback Mode | If user skips an input, display informative message or impute value. |


## UI/UX Guidelines
*   **Tone:** Helpful, calm, non-judgmental. Avoid complex medical jargon (e.g. "High Blood Pressure" instead of "Hypertension").
*   **Visuals:** Use trust-building colors (Blues/Greens).
*   **Footer Disclaimer:** A permanent footer on all pages: *"Estimates based on 2023 U.S. national data. Not intended as medical or financial advice."*


## Technical Stack Recommendation
*   **Core:** Python 3.13.
*   **Preprocessing:** NumPy, Pandas, Scikit-Learn Pipeline (imputation, scaling, encoding).
*   **EDA:** Pandas, Seaborn, Matplotlib, JupyterLab.
*   **Modeling:** Scikit-Learn, XGBoost, Joblib (for serialization).
*   **Web App:** 
    *   **Frontend**: Gradio or Streamlit (for demo).
    *   **Backend**: FastAPI (for production) and Pydantic (for API data validation).
*   **Testing:** Pytest.
*   **Hosting:** 
    *   **Source Code**: GitHub.
    *   **Model/Pipeline**: Hugging Face Hub.
    *   **Web App**: Hugging Face Spaces.


## Success Metrics
*   **Predictive Performance:** Median Absolute Error (MdAE) on the test set is < $500 (i.e., for the typical user, the prediction is within $500 of the actual cost).
*   **Interval Coverage:** ≥ 50% of actual costs fall within the predicted 25th–75th percentile range.
*   **Completion Rate:** > 80% of users who start the questionnaire complete it.
*   **User Satisfaction:** Positive sentiment on optional "Was this helpful?" feedback (optional).

### Metric Selection Rationale
Healthcare cost data has unique characteristics that influence metric selection: 
* **Zero-inflated**: Many users have $0 out-of-pocket costs. 
* **Right-skewed**: Few users have extremely high costs. 
* **Heteroskedastic**: Prediction performance is typically lower for high costs due to less training data in this range and costs being driven by unpredictable events (e.g., accidents, sudden diagnoses).

| Metric | Pros | Cons | Verdict |
| :--- | :--- | :--- | :--- |
| **MdAE** (Median Absolute Error) | Robust to outliers; represents "typical" error; directly interpretable in dollars | Ignores performance on high-cost outliers; median can hide bimodal error distributions | ✅ **Selected** |
| **MAE** (Mean Absolute Error) | Intuitive; interpretable in dollars; standard metric | Sensitive to outliers, a few extreme errors inflate the metric | ⚠️ Use as secondary metric |
| **RMSE** (Root Mean Squared Error) | Penalizes large errors more heavily | Very sensitive to outliers; less interpretable for skewed data | ❌ Not recommended |
| **MAPE** (Mean Absolute Percentage Error) | Scale-independent; good for comparing across cost levels | Undefined when actual cost = $0 (common in healthcare data) | ❌ Not recommended |
| **R²** (Coefficient of Determination) | Shows variance explained; standard in regression | Not in dollar terms; can be misleading on skewed/zero-inflated data | ⚠️ Use as secondary metric |

**Why MdAE?**
*  **Robust to zero-inflation and outliers**: Many users have $0 costs; some have $50k+. MdAE captures "typical" performance without being distorted.
*  **Aligns with user story**: Our goal is accuracy "for the typical user", the median explicitly measures this.
*  **Actionable threshold**: "$500 error" is intuitive for stakeholders and directly maps to FSA/HSA contribution decisions.
*  **Interval Coverage as complement**: Since we output 25th–75th percentile ranges, we also measure calibration of uncertainty estimates.


## Risk Assessment & Mitigation
| Risk | Example | Mitigation Strategy |
| :--- | :--- | :--- |
| **Outlier Prediction** | Model predicts extreme costs ($100k+) for a standard user. | Consider implementing "Guardrails" in the code to cap displayed predictions at the 95th percentile with a "High Cost Risk" label instead of a raw number. |
| **Bias/Fairness** | Model consistently under-predicts needs for low-income users due to historical access barriers. | Perform a Fairness Audit. Include income as a feature so the user sees that income impacts the prediction. |
| **Data Aging** | 2023 data becomes outdated. | Inform user about limitations of the model. Apply a "Medical Inflation Factor" adjustment (FR-02) to account for cost increases from 2023 to current prediction year. |
| **Policy Changes** | Policy changes enacted after 2023 data collection (e.g., Medicare Part D $2k cap, ACA marketplace adjustments) create systemic over/under-prediction for specific insurance groups. | Display general disclaimer about model limitations. For Medicare/Medicaid users, add contextual note: "Recent policy changes (2024-2026) may lower actual costs compared to this estimate." |


## Policy Changes (2024-2026)
Since the collection of the 2023 MEPS data, key policy changes have been enacted that affect out-of-pocket costs. This section documents these changes for awareness and transparency.

**Note**: The app predicts **total out-of-pocket costs** (`TOTSLF23`), not costs itemized by category. This limits our ability to apply policy-specific hard caps (e.g., prescription drug cap). The existing Medical Inflation Factor (FR-02) already accounts for general cost changes.

**1. Inflation Reduction Act (IRA) - Medicare Part D**
*   **2025 Change**: Annual out-of-pocket cap of **$2,000** for prescription drugs for Medicare Part D beneficiaries.
*   **Impact**: Model may over-predict costs for Medicare users with high prescription spending, as 2023 training data includes seniors who spent >$2,000 on drugs.
*   **Mitigation**: Display contextual disclaimer for "Public (Medicare/Medicaid)" users: *"Note: The 2025 Inflation Reduction Act caps prescription drug costs at $2,000/year for Medicare beneficiaries. Your actual costs may be lower than this estimate."*

**2. ACA Marketplace Adjustments**
*   **Cost Sharing Limits**: For 2025, the maximum out-of-pocket limit is **$9,200** (individual) for ACA marketplace plans.
*   **Subsidy Expiration Risk**: Enhanced premium tax credits expire December 31, 2025. If not renewed, unsubsidized premiums could increase significantly in 2026.
*   **Mitigation**: Display general disclaimer about model limitations. Cannot apply hard cap because app does not differentiate marketplace from employer insurance within the "Private" insurance category.

**3. Medicare Part B Increases**
*   **Deductibles**: Increased to $257 in 2025 (up from $226 in 2023).
*   **Premiums**: Standard premium rose to $185.00/month in 2025 (up from $164.90 in 2023).
*   **Mitigation**: The Medical Inflation Factor (FR-02) already accounts for these cost increases. No additional adjustment needed.

