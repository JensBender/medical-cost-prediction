# Technical Specifications
| **Project Name** | Medical Cost Prediction |
| :--- | :--- |
| **Status** | Project scoping |
| **Created** | December 12, 2025 |
| **Last Updated** | December 18, 2025 |

**Note:** This document details the technical implementation for the [Product Requirements Document (PRD)](./product_requirements.md).


## Table of Contents
1. [Data Specifications](#data-specifications)
   - [Dataset](#dataset)
   - [Target Variable](#target-variable)
   - [Feature Selection](#feature-selection)
   - [Candidate Features](#candidate-features)
2. [Machine Learning Specifications](#machine-learning-specifications)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Training](#model-training)
   - [Model Evaluation](#model-evaluation)
3. [Deployment Specifications](#deployment-specifications)
   - [API Contract](#api-contract)
   - [Inference Pipeline](#inference-pipeline)
4. [Testing Strategy](#testing-strategy)
5. [Technical Stack Recommendation](#technical-stack-recommendation)
6. [References](#references)


## Data Specifications

### Dataset
*   **Data File:** [MEPS-HC 2023 Full Year Consolidated Data File (H251)](https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-251).
*   **Documentation:** [MEPS-HC 2023 Documentation](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.shtml).
*   **Codebook:** [MEPS-HC 2023 Codebook](https://meps.ahrq.gov/data_stats/download_data_files_codebook.jsp?PUFId=H251).

### Target Variable
*   **`TOTSLF23`:** Total amount paid out-of-pocket by the person or their family for all medical events in the year 2023.
*   **Rationale:** Our primary personas (Open Enrollment Planners, Budgeters) need to know what they will personally pay, not the total cost shared across insurance and government payers. Out-of-pocket costs directly answer: "How much should I contribute to my FSA/HSA?" and "What's my financial exposure?"
*   **Note:** For uninsured users, out-of-pocket ≈ total cost, so this target remains appropriate across all insurance statuses.

<details>
<summary>ℹ️ <strong>Detailed Breakdown of TOTSLF23</strong> (click to expand)</summary>

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
The primary goal is a fast, frictionless user experience. We prioritize usability over predictive performance if it requires complex inputs. 

**Feature Selection Rationale**:
1.  **UX-First Constraint**: Target form completion in **under 90 seconds**. This is a soft guideline. Cognitive load and completion time matter more than a strict input count. As a ballpark, aim for ~10–12 discrete UI interactions, noting that a multi-select checklist (e.g., chronic conditions) counts as one interaction even with many options.
2.  **Consumer Accessibility**: Inputs must be information users know offhand (e.g., age, self-rated health). The user doesn't need to leave their chair to find an insurance card, past bill, or medical record. No asking for specifics like "deductible amount" or "ICD-10 codes" that require mental effort or looking up technical terms.
3.  **Optimize Within Constraints**: Among the pool of "accessible" inputs, select the variables with the highest feature importance to maximize predictive power within the UX constraints. 

**Feature Selection Process**:
1.  **Candidate Screening**: Identify all MEPS variables that a layperson can answer easily without having to look something up or think too hard.
2.  **Feature Importance Ranking**: Train preliminary models on these candidate features to obtain feature importance scores.
3.  **Final Feature Selection**: Select the top-performing features that maximize predictive power while keeping form completion under ~90 seconds.

### Candidate Features
The following MEPS variables have been identified as candidate features for the model. All candidates satisfy the UX-first constraint: users can answer from memory without looking up documents or technical terms. The final feature set will be selected based on empirical feature importance ranking, targeting form completion in under 90 seconds.

**Demographics & Socioeconomic**
| UI Label | MEPS Variable | Data Type | Description | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Age** | `AGE23X` | Numerical | Age at end of year (18–85). | ✅ Universal predictor of healthcare utilization; strongly correlated with chronic conditions and costs. |
| **Sex** | `SEX` | Nominal | Male or Female. | ✅ Biologically relevant (e.g., pregnancy, gender-specific conditions); easy to answer. |
| **Region** | `REGION23` | Nominal | Census region (Northeast, Midwest, South, West). | ⚠️ May have low predictive power for out-of-pocket costs specifically; consider dropping if feature importance is weak. |
| **Income** | `POVCAT23` | Ordinal | Family income as % of poverty line (Poor/Near Poor/Low/Middle/High). | ✅ Correlated with insurance type, access to care, and ability to pay out-of-pocket. |
| **Insurance** | `INSCOV23` | Nominal | Insurance coverage status (Private, Public, Uninsured). | ✅ **Critical.** Directly determines cost-sharing structure; strongest predictor of out-of-pocket vs. total cost. |

**Perceived Health**
| UI Label | MEPS Variable | Data Type | Description | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Phys. Health** | `RTHLTH31` | Ordinal | Self-reported physical health (1=Excellent to 5=Poor). | ✅ Strong predictor of utilization; captures overall health burden in one question. |
| **Ment. Health** | `MNHLTH31` | Ordinal | Self-reported mental health (1=Excellent to 5=Poor). | ✅ Complements physical health; captures behavioral health costs (therapy, Rx). |

**Chronic Conditions**
| UI Label | MEPS Variable | Data Type | Description | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Diabetes** | `DIABDX_M18` | Binary | Ever diagnosed with diabetes. | ✅ High-cost chronic condition with ongoing Rx, monitoring, and complication costs. Very common (~11% prevalence). |
| **Hypertension** | `HIBPDX` | Binary | Ever diagnosed with high blood pressure. | ✅ Very common (~30% prevalence); drives ongoing Rx costs and cardiovascular risk. |
| **Heart Disease** | `CHDDX` | Binary | Ever diagnosed with coronary heart disease. | ✅ **Major cost driver** with high downstream costs (procedures, Rx, monitoring). Lower prevalence but high impact. |
| **High Cholesterol** | `CHOLDX` | Binary | Ever diagnosed with high cholesterol. | ⚠️ Very common (~28% prevalence); drives statin Rx costs. May be partially redundant with hypertension. |
| **Arthritis** | `ARTHDX` | Binary | Ever diagnosed with arthritis. | ⚠️ Very common (~25% prevalence); high Rx/therapy costs. May overlap with age effects. |
| **Cancer** | `CANCERDX` | Binary | Ever diagnosed with any cancer. | ⚠️ Major cost driver if active; easy to answer. May capture historical rather than current-year costs. |
| **Asthma** | `ASTHDX` | Binary | Ever diagnosed with asthma. | ⚠️ Chronic condition with ongoing costs (inhalers, visits). Lower prevalence (~8%). |
| **Stroke** | `STRKDX` | Binary | Ever diagnosed with stroke. | ⚠️ High downstream costs (rehab, Rx). Lower prevalence (~3%). |

**Behavioral**
| UI Label | MEPS Variable | Data Type | Description | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Smoker** | `ADSMOK42` | Binary | Currently smokes cigarettes. | ⚠️ Known health risk factor but may have **lower predictive power** for out-of-pocket costs than chronic conditions. Consider dropping if outperformed by condition flags. |

> **Note:** The final feature set targets form completion in **under 90 seconds** (soft goal). Chronic conditions may be grouped into a single multi-select checklist to minimize cognitive load.


## Machine Learning Specifications

### Data Preprocessing
All preprocessing steps are implemented as scikit-learn pipelines to ensure consistency between training and inference, prevent data leakage, and simplify deployment.

**Data Cleaning**  
Perform once before pipeline:

| Action | Rationale | Details |
| :--- | :--- | :--- |
| Drop rows where `PERWT23F = 0` | Respondents with a person weight of zero don't represent the population. | Removes ~456 rows (N=18,919 total, 18,463 with positive weight). |
| Handle MEPS Negative Codes | Standardize missing/inapplicable values for modeling. | Convert `-1` (Inapplicable), `-7` (Refused), `-8` (Don't know), `-15` (Cannot be computed) to `NaN`.<br>Treating survey non-response and missing inputs from web app users identically (as `NaN` → Imputed Mode/Median) to align data handling between training and inference. |

**Feature Preprocessing**  
Implemented via `ColumnTransformer`:

| Feature Type | Columns | Transformer | Notes |
| :--- | :--- | :--- | :--- |
| Numerical | `AGE23X`, `RTHLTH31`, `MNHLTH31` | `StandardScaler` | Health scales (1–5) treated as numerical |
| Ordinal | `POVCAT23` | `OrdinalEncoder` | Low < Middle < High; preserve ordering |
| Nominal | `SEX`, `REGION23`, `INSCOV23` | `OneHotEncoder` | Drop first to avoid multicollinearity |
| Binary | `DIABDX_M18`, `HIBPDX`, `ADSMOK42` | passthrough | Already 0/1 encoded |
| Engineered | — | `FunctionTransformer` | Create `COMORBIDITY_COUNT` = sum of 3 binary flags |

**The Heteroscedasticity Problem**  
Medical cost data is inherently **heteroscedastic**, meaning the "noise" or variance in errors is not constant but grows with the target value. 
*   **Example:** A standard office visit costing **$100** might vary by **±$20**. A complex surgery costing **$100,000** might vary by **±$20,000**. 
*   **Impact:** If we treat raw dollars as the target, models that assume constant variance (homoscedasticity) will be overwhelmingly driven by the massive errors in the high-cost patients, ignoring the smaller, consistent patterns in the low-cost majority.

**Target Preprocessing**  
We apply transformations selectively based on how each model handles this variance. 

| Models | Target Transform | Rationale |
| :--- | :--- | :--- |
| **Linear Regression, Elastic Net, MLP** | `log(y + 1)` | **Strict Assumption.** These models assume constant error variance. `log` stabilizes variance by compressing high values; `+1` handles zero-cost cases where `log(0)` is undefined. |
| **K-Nearest Neighbors (KNN)** | `log(y + 1)` | **Outlier Sensitive.** KNN averages the values of neighbors. Without `log`, a single high-cost neighbor can skew the prediction massively. |
| **Decision Tree, Random Forest, XGBoost** | None | **Robust.** These models are non-parametric and partition data into local regions. They can learn to accept low variance in one node and high variance in another without global transformation. |

### Model Training
**Procedure**  
1.  **Train-Validation-Test Split**: Split the data into 70% train, 15% validation, and 15% test (~2,700 test samples).
2.  **Baseline Models**: Train all candidate models with (mostly) default hyperparameters. Evaluate all models based on MdAE on the validation set. Select the best 2–4 models for optimization.
3.  **Hyperparameter Tuning**: Tune selected models via randomized search. Select the best-performing model based on validation set performance.
4.  **Final Model**: Evaluate the selected model on the hold-out test set to assess real-world performance on unseen data.

**Baseline Models**  
We evaluate all models using MdAE (Median Absolute Error). For training, tree-based models use absolute error criteria where possible. Linear models default to MSE but are evaluated on their ability to minimize median error on the validation set.

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
| **EV-01** | **Overall Performance** | Report overall MdAE on the full test set as the primary success metric. **Target: < $500.** |
| **EV-02** | **Stratified Error Analysis** | Report MdAE separately for low (0–50th percentile), medium (50th–90th percentile), and high (90th+ percentile) cost tiers. This diagnoses where the model underperforms and quantifies heteroskedasticity. |
| **EV-03** | **Interval Calibration** | Report what % of actual costs fall within the predicted 25th–75th percentile range. **Target: ≥ 50% coverage.** This diagnoses how accurate the prediction interval is across cost levels. Performance is expected to degrade for the high-cost tier due to (1) inherent unpredictability of high-cost events and (2) less training data in that range. |

**Metric Selection Rationale**  
Healthcare cost data has unique characteristics that influence evaluation metric selection: 
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


## Deployment Specifications

### API Contract
The model will be exposed via a Python API (internal to the web app process) or a REST endpoint if decoupled.
*   **Input Schema:** `features: Dict[str, Union[str, int, float]]`
*   **Output Schema:**
    ```json
    {
      "prediction_median": float,
      "prediction_25th": float,
      "prediction_75th": float,
      "prediction_90th": float,
      "shap_values": Dict[str, float],
      "warning_flags": List[str]
    }
    ```

### Inference Pipeline
1.  **Validation:** Ensure inputs are within valid ranges (e.g., Age 18-85).
2.  **Imputation:** Fill missing values (Mode for categorical, Median for numerical).
3.  **Transformation:** Apply `ColumnTransformer` (Scaling/Encoding).
4.  **Prediction:** Run model inference.
5.  **Explainability:** Run TreeExplainer/KernelExplainer.
6.  **Post-Processing:** Apply inflation adjustment + inverse log-transform (if applicable).


## Testing Strategy
**Unit Tests**  
*   **Preprocessing Pipelines:** Validate encoding, scaling, imputation, and handling of edge cases (e.g., unseen categories).
*   **Data Validation:** Ensure Pydantic models correctly reject invalid inputs.

**Integration Tests**  
*   **Serving:** Verify serialized pipeline loading and output structure.
*   **Endpoints:** Validate JSON responses and HTTP status codes (200/422).

**End-to-End Tests**  
*   **User Journey:** Simulate full flow: user input → processing → cost prediction.


## Technical Stack Recommendation
*   **Core:** Python 3.13.
*   **Preprocessing:** NumPy, Pandas, Scikit-Learn Pipeline (imputation, scaling, encoding).
*   **EDA:** Pandas, Seaborn, Matplotlib, JupyterLab.
*   **Modeling:** Scikit-Learn, XGBoost, SHAP (for explainability), Joblib (for serialization).
*   **Web App:** 
    *   **Frontend**: Gradio or Streamlit (for demo).
    *   **Backend**: FastAPI (for production) and Pydantic (for API data validation).
*   **Testing:** Pytest.
*   **Hosting:** 
    *   **Source Code**: GitHub.
    *   **Model/Pipeline**: Hugging Face Hub.
    *   **Web App**: Hugging Face Spaces.


## References
*   [Product Requirements Document (PRD)](./product_requirements.md): The high-level product vision, user personas, requirements, and success metrics that guide this technical implementation.
*   [U.S. Healthcare Costs Guide](./us_healthcare_costs_guide.md): Primer on U.S. healthcare payment structures, terminology, and why Americans need to predict out-of-pocket costs.
*   [ML with MEPS](./ml_with_meps.md): Literature review of existing machine learning projects using MEPS data, informing competitive positioning.
