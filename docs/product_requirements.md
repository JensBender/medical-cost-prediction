# Product Requirements Document (PRD)
| **Project Name** | Medical Cost Prediction |
| :--- | :--- |
| **Status** | Project scoping |
| **Date** | December 5, 2025 |
| **Data Source** | Medical Expenditure Panel Survey (MEPS) |


## Executive Summary
The **Medical Cost Prediction App** is a consumer-facing web application that uses machine learning trained on the Medical Expenditure Panel Survey (MEPS) to predict annual healthcare costs.

**The Problem:** Healthcare pricing is a "black box." While insurance portals show *unit prices* for individual treatments (e.g., "cost of an MRI"), consumers struggle to predict their total expected costs for the entire year. Fixed calculators (e.g., "add $500 per child") are inaccurate because they ignore health status, and insurance tools require specific procedure codes that often users don't know.

**Our Solution:** This tool predicts expected healthcare costs based on simple information that users know about themselves. It translates complex epidemiological data from MEPS into a simple financial planning tool, enabling users to input basic information such as age, sex, insurance status, self-rated health, and pre-existing health conditions to receive a personalized forecast for expected healthcare costs for the upcoming year. This forecast can be used for FSA/HSA contributions and emergency fund planning.

## Problem Statement
*   **User Pain Point:** Consumers do not know how much to set aside for healthcare. Fixed calculators ("add $500 per child") are inaccurate because they ignore health status. Insurance tools require specific procedure codes (CPT) that users don't know.
*   **The Opportunity:** MEPS data contains the ground truth of what people with specific profiles *actually* spent. By exposing this via an easy-to-use web app that is available to everyone, we can provide data-driven financial guidance without requiring complex, detailed medical records.

## Target Audience
1.  **The Open Enrollment Planner:** Employees deciding how much to contribute to FSAs/HSAs during open enrollment.
2.  **The Budgeter:** Individuals with tight budgets needing to anticipate potential medical shocks.
3.  **The Newly Diagnosed:** Users recently diagnosed with a chronic condition (e.g., Diabetes) wondering how it impacts their financial bottom line.


## Functional Requirements

### User Input 
The app shall present a simple form on a single page, with no more than 20 inputs. 
*   **Age**: Numeric field (18-85).
*   **Sex**: Dropdown list (Male/Female).
*   **Region**: Dropdown list (Northeast, Midwest, South, West).
*   **Income Tier**: Dropdown list (Income range Low/Middle/High).
*   **Insurance Type**: Dropdown list (Private/Public/Uninsured).
*   **Physical Health**: Dropdown list (Excellent to Poor).
*   **Mental Health**: Dropdown list (Excellent to Poor).
*   **Diabetes**: Checkbox.
*   **High Blood Pressure**: Checkbox.

### Prediction
*   System shall calculate the **Predicted Total Expenditure** using the pre-trained ML model.
*   System shall calculate a **Prediction Interval** (e.g., 25th to 75th percentile) to communicate uncertainty.
*   System shall handle missing optional inputs (e.g., impute based on user profile or run a reduced-feature model).

### Output
*   System shall display the "Likely Annual Cost" as a dollar range.  
    ```
    Example: "Your predicted healthcare cost for the upcoming year is between $1,200 and $2,500."
    ```
*   System shall display a "Key Cost Drivers" section (Explainability via SHAP values).  
    ```
    Example: The main cost drivers for this prediction are:
    Your Diabetes Diagnosis (+ $1,200)
    Your Age (+ $400)
    But your 'Excellent' self-reported health lowered the estimate by (- $300)
    ```
*   System shall display a Comparison Benchmark.
    ```
    Example: "You are projected to spend 15% less than the national average for your age group."
    ```

## Data & Machine Learning Specifications

### Dataset
*   **Source:** MEPS-HC 2023 Full Year Consolidated Data File (H251).
*   **Documentation:** [H251 Codebook](https://meps.ahrq.gov/data_stats/download_data_files_codebook.jsp?PUFId=H251).

### Feature Mapping (MEPS to UI)
The model will utilize the following features mapping to user inputs:

| MEPS Variable | Data Type | UI Input |
| :--- | :--- | :--- |
| `AGE23X` | Continuous | Slider |
| `SEX` | Binary | Toggle |
| `REGION23` | Categorical | Dropdown (Map from Zip) |
| `POVCAT23` | Ordinal (1-5) | Dropdown (Income Brackets) |
| `INSCOV23` | Categorical (1-3) | Dropdown |
| `RTHLTH31` | Ordinal (1-5) | Dropdown |
| `MNHLTH31` | Ordinal (1-5) | Dropdown |
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
