# Product Requirements Document (PRD)
| **Project Name** | Medical Cost Prediction |
| :--- | :--- |
| **Status** | Project scoping |
| **Date** | December 5, 2025 |
| **Data Source** | Medical Expenditure Panel Survey (MEPS) |


## Executive Summary
The **Medical Cost Prediction App** is a consumer-facing web application that uses machine learning trained on the Medical Expenditure Panel Survey (MEPS) to predict annual **out-of-pocket** healthcare costs.

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
No existing tool combines **ML-powered personalized predictions** with **simple, consumer-friendly inputs** for annual healthcare costs. Current tools fall into three categories, none of which solve our users' core problem:

1. **Procedure-Specific Cost Estimators** (FAIR Health, Healthcare Bluebook, hospital-based tools): Require users to know CPT codes or specific procedures. Solve "How much will THIS procedure cost?" not "How much should I budget for the year?"
2. **Insurance Premium Calculators** (KFF Marketplace, state exchanges): Only estimate cost of insurance, not actual healthcare spending.
3. **FSA/HSA Contribution Calculators** (FSAFEDS, HSAstore): Require users to already know their expected annual costs—they don't predict it.

**Closest Competitor**: The KFF (Kaiser Family Foundation) Household Health Spending Calculator provides annual cost estimates but uses **demographic subgroup averaging** rather than personalized ML predictions. It categorizes health status as simply "good" or "worse" and returns the average spending for broad demographic buckets (e.g., "single person, $50k income, employer coverage, good health"). This approach cannot capture individual-level variations (e.g., diabetes vs. hypertension) or complex non-linear interactions between features that ML models learn from individual-level data.

### Our Differentiation
This tool is the **only ML-powered annual healthcare cost predictor** with easily accessible inputs, designed for consumers who lack specialized medical or insurance knowledge.

| **Our ML Approach** | **Competitor Approaches** |
| :--- | :--- |
| **Personalized predictions** from individual-level MEPS data (30k+ records) | Demographic subgroup averages or insurance rate tables |
| **Granular health inputs**: 5-point scales for physical/mental health + specific chronic conditions (diabetes, hypertension, smoking) | Broad health categories ("good" vs. "worse") or no health inputs |
| **Explainable predictions**: SHAP values show cost drivers (e.g., "Diabetes +$1,200") | Black box averages with no explanation |
| **Uncertainty ranges**: 25th–75th percentile for planning worst-case scenarios | Single point estimates |
| **10 accessible inputs** (< 1 min completion) | Either requires CPT codes/deductibles OR oversimplified 4-5 broad buckets |
| **Free, no login required** | Often gated behind insurance portals |

### UX-First Rationale
The 10-input constraint is a feature rather than a limitation. Our personas (Open Enrollment Planners, Budgeters) need ballpark estimates for financial planning (e.g., "Should I contribute $1,000 or $3,000 to my FSA?"), not clinical precision. Being within $500 (our MAE target) is sufficient for these decisions. The ML model captures complex patterns from individual-level data while keeping inputs simple and accessible. Sacrificing 2% absolute accuracy to achieve 80%+ completion rate is the right trade-off for this use case.


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
| **UI-01** | **Cost Range** | Large, prominent display of out-of-pocket cost prediction as a range. | `gr.Markdown` | "Estimated Out-of-Pocket Healthcare Cost for Next Year: **$1,450 – $2,100**" |
| **UI-02** | **Cost Drivers** | Explanation of key cost drivers and their dollar impact (SHAP). | `gr.Markdown` | "Your Diabetes Diagnosis (+$1,200), your Age (+$400), but your "Excellent" self-reported health lowered the estimate by (-$300)" |
| **UI-03** | **Comparison Benchmarks** | Bar chart comparing user vs. national and age group benchmarks. | `gr.Plot` | "Typical American (median): $4,800 vs. Typical for Age 45–54 (median): $3,200" |


## Data & Machine Learning Specifications

### Dataset
*   **Source:** MEPS-HC 2023 Full Year Consolidated Data File (H251).
*   **Documentation:** [H251 Codebook](https://meps.ahrq.gov/data_stats/download_data_files_codebook.jsp?PUFId=H251).

### Target Variable
*   **Variable:** `TOTSLF23` — Total amount paid out-of-pocket by the person or their family for all medical events in the year 2023.
*   **Rationale:** Our primary personas (Open Enrollment Planners, Budgeters) need to know what **they will personally pay**, not the total cost shared across insurance and government payers. Out-of-pocket costs directly answer: "How much should I contribute to my FSA/HSA?" and "What's my financial exposure?"
*   **Note:** For uninsured users, out-of-pocket ≈ total cost, so this target remains appropriate across all insurance statuses.
*   **Details:** See [Target Variable Details](#target-variable-details) in Appendix.

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
*   **Objective:** `TOTSLF23` (Out-of-Pocket Health Care Expenditures).
*   **Handling Zeros:** Use a Two-Part Model (Hurdle Model) or Tweedie Objective function to handle zero-inflated cost data.
*   **Weighting:** Training must utilize `PERWT23F` (Person Weight) to ensure national representation.
*   **Preprocessing:** Log-transformation of target variable `log(TOTSLF23 + 1)` recommended for training stability.


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


## Appendix

### Target Variable Details
*   **Variable:** `TOTSLF23` — **Total Amount Paid by Self/Family (2023)**
*   **Definition:** This variable is the aggregate sum of all out-of-pocket payments made by the person or their family for healthcare services received in 2023. It sums the "Self/Family" share of costs across all medical event categories.
*   **Rationale:** Our primary personas (Open Enrollment Planners, Budgeters) need to know their personal financial liability, not the total cost of care. `TOTSLF23` directly answers, "How much cash did I have to pay?"—which determines FSA/HSA funding needs and maximum financial exposure.
*   **Components (What is included):**
    `TOTSLF23` is calculated by summing the out-of-pocket expenses for the following specific service types:
    *   **Office-Based Visits (`OBSLF23`):** Co-pays and deductibles for doctor's appointments, check-ups, and specialist visits.
    *   **Prescribed Medicines (`RXSLF23`):** Out-of-pocket costs for filled prescriptions (does **not** include over-the-counter drugs).
    *   **Hospital Inpatient Stays (`IPSLF23`):** Direct payments for overnight hospitalizations (room & board, treatments).
    *   **Emergency Room Visits (`ERSLF23`):** Co-pays and bills for ER visits that did not result in an admission.
    *   **Outpatient Department Visits (`OPSLF23`):** Costs for same-day surgeries, scans, or therapies at a hospital.
    *   **Dental Care (`DVSLF23`):** Payments for cleanings, fillings, orthodontia, etc.
    *   **Vision Services (`VISLF23`):** Costs for eye exams, glasses, and contact lenses.
    *   **Home Health Care (`HHSLF23` + `HNSLF23`):** Payments for agency or independent home health providers.
    *   **Other Medical (`OMSLF23`):** Equipment (crutches, hearing aids), ambulance services, and other miscellaneous supplies.

*   **Important Exclusions:**
    *   **Premiums:** This variable does **not** include monthly insurance premiums (e.g., deducted from a paycheck). It only tracks payments for *services* received.
    *   **Over-the-Counter (OTC) Drugs:** Expenses for non-prescription medications (e.g., Tylenol, vitamins) are not included.

*   **User Note:** For uninsured users, `TOTSLF23` will typically equal the **Total Expenditure** (`TOTTCH23`), as they bear the full cost unless they received charity care (which is not counted as an expenditure in MEPS).