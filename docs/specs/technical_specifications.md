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
*   **Population Scope:** The dataset represents the **U.S. civilian noninstitutionalized (CSN) population**.
    *   **Included:** People living in households and non-institutional group quarters (e.g., college dorms).
    *   **Excluded:** Active-duty military, people in institutions like nursing homes or prisons, and people who moved abroad during the survey year.

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
3.  **Temporal Validity**: Variables must reflect **beginning-of-year status** to enable prospective prediction without data leakage (see box below).
4.  **Optimize Within Constraints**: Among the pool of "accessible" inputs, select the variables with the highest feature importance to maximize predictive power within the UX constraints. 

<details>
<summary>ℹ️ <strong>Temporal Alignment: Why Variable Suffixes Matter</strong> (click to expand)</summary>

**Use Case:** Users access the app during Open Enrollment (Nov–Dec) to predict costs for the **upcoming year**. We train on MEPS data where features predict that same year's total costs.

**MEPS Variable Suffixes:**
| Suffix | Timing | Example |
|:---|:---|:---|
| `31` | Beginning of year (Rounds 3/1) | `RTHLTH31` |
| `42` | Mid-year (Rounds 4/2) | `ADSMOK42` |
| `53` | End of year (Rounds 5/3) | `RTHLTH53` |
| `23` / `23X` | Full-year summary or year-end point | `AGE23X`, `INSCOV23` |

**The Leakage Risk:** Using `RTHLTH53` (end-of-year health) to predict `TOTSLF23` (full-year costs) would mean using information collected *after* many costs already occurred. A person diagnosed mid-year would report worse health at year-end, but that health decline was caused by events we're trying to predict.

**The Rule:** For time-varying self-reported status (health, limitations), use **`31` suffix** (beginning of year). For "ever diagnosed" chronic conditions, timing is less critical since they're stable. For utilization counts (`ERTOT23`, `ADAPPT42`), **exclude entirely** — these are accumulated during the year and unavailable at prediction time.

**Note on Training-Serving Skew:** While MEPS '31' interviews occur in the first half of the year (Jan–July), most app usage may occur in Nov–Dec (pre-year). This creates a slight discrepancy: training data may contain minor "leakage" where a Feb illness influences a June interview score. Consequently, real-world predictive performance may be slightly lower than during training stage, as the app performs a more strictly prospective prediction.

</details>

**Feature Selection Process**:
1.  **Candidate Screening**: Identify all MEPS variables that a layperson can answer easily without having to look something up or think too hard. Ensure temporal validity (no end-of-year or in-year utilization variables).
2.  **Feature Importance Ranking**: Train preliminary models on these candidate features to obtain feature importance scores.
3.  **Final Feature Selection**: Select the top-performing features that maximize predictive power while keeping form completion under ~90 seconds.

### Candidate Features
The following MEPS variables have been identified as candidate features for the model. All candidates satisfy three constraints: (1) users can answer from memory, (2) variables reflect beginning-of-year status (temporal validity), and (3) established predictive power in literature. The final feature set will be selected based on empirical feature importance ranking, targeting form completion in under 90 seconds.

**Full Details:** See [Candidate Features Research](../research/candidate_features.md) document.

**Demographics**
| UI Label | MEPS Variable | Data Type | Description | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Birth Year** | `AGE23X` | Numerical (Int) | In what year were you born? Used to calculate age at end of year (18–85). Top-coded at 85 per MEPS privacy protocol. | ✅ Primary driver of utilization; costs follow a U-curve with age. |
| **Sex** | `SEX` | Binary (Int) | Male or Female. | ✅ Biologically relevant; easy to answer. |
| **Region** | `REGION23` | Nominal (Int) | Census region (Northeast, Midwest, South, West). | ⚠️ May have low predictive power; consider dropping if low feature importance. |
| **Marital Status** | `MARRY31X` | Nominal (Int) | Marital status at beginning of year. | ⚠️ Proxy for social support and income stability. |

**Socioeconomic**
| UI Label | MEPS Variable | Data Type | Description | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Family Income** | `POVCAT23` | Ordinal (Int) | Family income mapped to poverty category. | ✅ Correlated with insurance type and ability to pay OOP. |
| **Family Size** | `FAMSZE23` | Numerical (Int) | Number of related persons residing together (CPS definition). | ✅ Required to derive Poverty Category; captures household resource sharing. |
| **Education** | `HIDEG` | Ordinal (Int) | Highest degree attained. Maps UI labels to MEPS `HIDEG` categories. | ✅ Correlates with health literacy; includes Professional Degrees in Doctorate bucket. |
| **Employment** | `EMPST31` | Nominal (Int) | Employment status at beginning of year. | ⚠️ Strong proxy for insurance type. |

**Insurance & Access**
| UI Label | MEPS Variable | Data Type | Description | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Insurance** | `INSCOV23` | Nominal (Int) | Coverage status (Private, Public Only, Uninsured). | ✅ **Critical.** Determines OOP vs. total cost split. |
| **Usual Source of Care** | `HAVEUS42` | Binary (Int) | Has regular doctor or clinic | ✅ Strong predictor of access and preventive care. |

**Perceived Health & Lifestyle** 
| UI Label | MEPS Variable | Data Type | Description | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Physical Health** | `RTHLTH31` | Numerical (Int) | Self-rated physical health (1=Excellent to 5=Poor). | ✅ Strongest subjective predictor of utilization. |
| **Mental Health** | `MNHLTH31` | Numerical (Int) | Self-rated mental health (1=Excellent to 5=Poor). | ✅ Significant cost multiplier via treatment adherence. |
| **Smoker** | `ADSMOK42` | Binary (Int) | Currently smokes cigarettes. | ⚠️ Stable behavioral risk factor. |

**Limitations & Symptoms**
| UI Label | MEPS Variable | Data Type | Description | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **ADL Help** | `ADLHLP31` | Binary (Int) | Needs help with personal care (bathing, dressing). | ⚠️ High-cost functional indicator. |
| **IADL Help** | `IADLHP31` | Binary (Int) | Needs help with bills, meds, shopping, etc. | ⚠️ Signals high-cost care requirements. |
| **Walking Limit** | `WLKLIM31` | Binary (Int) | Difficulty walking or climbing stairs. | ⚠️ Captures mobility impairment. |
| **Cognitive Limit** | `COGLIM31` | Binary (Int) | Confusion or memory loss. | ⚠️ Correlates with specialized care needs. |
| **Joint Pain** | `JTPAIN31_M18` | Binary (Int) | Joint pain/stiffness in past 12 months. | ⚠️ Captures undiagnosed musculoskeletal issues. |

**Chronic Conditions** *("Ever diagnosed" — stable over time)*
| UI Label | MEPS Variable | Data Type | Description | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Hypertension** | `HIBPDX` | Binary (Int) | Diagnosed with high blood pressure. | ✅ Common; drives Rx costs. |
| **High Cholesterol** | `CHOLDX` | Binary (Int) | Diagnosed with high cholesterol. | ⚠️ Common; drives Rx costs. |
| **Diabetes** | `DIABDX_M18` | Binary (Int) | Diagnosed with diabetes. | ✅ High-cost condition. |
| **Heart Disease** | `CHDDX` | Binary (Int) | Diagnosed with heart disease (CHD). | ✅ Major cost driver. |
| **Stroke** | `STRKDX` | Binary (Int) | Diagnosed with stroke. | ⚠️ High downstream costs. |
| **Cancer** | `CANCERDX` | Binary (Int) | Diagnosed with cancer. | ⚠️ Primary driver of tail costs. |
| **Arthritis** | `ARTHDX` | Binary (Int) | Diagnosed with arthritis. | ⚠️ Drives Rx/therapy costs. |
| **Asthma** | `ASTHDX` | Binary (Int) | Diagnosed with asthma. | ⚠️ Chronic condition with ongoing costs. |
| **Depression** | `DEPRDX` | Binary (Int) | Diagnosed with depression. | ✅ Significant cost multiplier. |

**Note:** The final feature set targets form completion in **under 90 seconds** (soft goal). Chronic conditions and limitations/symptoms should be grouped into multi-select checklists to minimize cognitive load (~13–14 total UI interactions).

#### Income Mapping Table (POVCAT23)
To ensure stigma-free and accurate income reporting, the UI displays dynamic income ranges based on the user's reported family size. These ranges map directly to the `POVCAT23` categories used in training, based on 2023 Federal Poverty Level (FPL) thresholds.

| Family Size | Poor (<100% FPL) | Near Poor (100–124%) | Low Income (125–199%) | Middle Income (200–399%) | High Income (≥400%) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | Under $14,600 | $14,600 – $18,100 | $18,200 – $29,000 | $29,100 – $58,300 | Over $58,300 |
| 2 | Under $19,700 | $19,700 – $24,400 | $24,500 – $39,200 | $39,300 – $78,800 | Over $78,800 |
| 3 | Under $24,900 | $24,900 – $30,800 | $30,900 – $49,500 | $49,600 – $99,400 | Over $99,400 |
| 4 | Under $30,000 | $30,000 – $37,200 | $37,300 – $59,700 | $59,800 – $119,900 | Over $119,900 |
| 5 | Under $35,100 | $35,100 – $43,500 | $43,600 – $69,900 | $70,000 – $140,500 | Over $140,500 |
| 6 | Under $40,300 | $40,300 – $49,900 | $50,000 – $80,100 | $80,200 – $161,000 | Over $161,000 |
| 7 | Under $45,400 | $45,400 – $56,200 | $56,300 – $90,300 | $90,400 – $181,600 | Over $181,600 |
| 8+ | Under $50,600 | $50,600 – $62,600 | $62,700 – $100,600 | $100,700 – $202,100 | Over $202,100 |

*Note: Thresholds derived from 2023 HHS Federal Poverty Guidelines. Values rounded for clean UI display.*

#### Education Mapping Table (HIDEG)
To ensure high-quality data while maintaining a clean UI, the "Education" dropdown options are mapped to the consolidated `HIDEG` categories used during model training.

| UI Label | HIDEG Code | Mapping Rationale |
| :--- | :---: | :--- |
| No Degree | 1 | Standard MEPS mapping. |
| GED | 2 | Standard MEPS mapping. |
| High School Diploma | 3 | Standard MEPS mapping. |
| Associate's Degree | 7 | Maps to "Other Degree" to avoid "No Degree" or "Bachelor's" misclassification. |
| Bachelor's Degree | 4 | Standard MEPS mapping. |
| Master's Degree | 5 | Standard MEPS mapping. |
| Doctorate or Professional (MD, JD, etc.) | 6 | Ensures high-income professional degrees are correctly attributed to the top tier. |
| Other | 7 | Standard MEPS mapping for vocational/trade degrees. |


## Machine Learning Specifications

### Data Preprocessing
All preprocessing steps are implemented as scikit-learn pipelines to ensure consistency between training and inference, prevent data leakage, and simplify deployment.

**Data Cleaning**  
Perform once before pipeline:

| Action | Rationale | Details |
| :--- | :--- | :--- |
| Drop rows where `PERWT23F = 0` | Respondents with a person weight of zero are "out-of-scope" for the full-year population (e.g., they joined the military, were institutionalized, or moved abroad during the year). | Removes ~456 rows (N=18,919 total, 18,463 with positive weight). |
| Handle MEPS Negative Codes | Standardize missing/inapplicable values for modeling. | Convert `-1` (Inapplicable), `-7` (Refused), `-8` (Don't know), `-15` (Cannot be computed) to `NaN`.<br>Treating survey non-response and missing inputs from web app users identically (as `NaN` → Imputed Mode/Median) to align data handling between training and inference. |

**Feature Preprocessing**  
Implemented via `ColumnTransformer`. Exact columns depend on final feature selection.

| Feature Type | Example Columns | Transformer | Notes |
| :--- | :--- | :--- | :--- |
| Numerical | `AGE23X` | `StandardScaler` | Age in years |
| Ordinal | `POVCAT23`, `HIDEG` | `OrdinalEncoder` | Preserve ordering (Low < Middle < High) |
| Nominal | `SEX`, `REGION23`, `INSCOV23`, `MARRY31X`, `EMPST31` | `OneHotEncoder` | Drop first to avoid multicollinearity |
| Binary | `DIABDX_M18`, `HIBPDX`, `CHDDX`, etc. | passthrough | Already 0/1 encoded |

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

**Training Procedure**  
A 4-phase approach where each model is evaluated with its own optimal feature set.

| Phase | Goal | Details |
| :--- | :--- | :--- |
| **1. Baseline Models** | Compare baseline models on full features | Train all models on ~30 features with (mostly) default hyperparameters. Evaluate MdAE on validation set. Select top 3–4 models. |
| **2. Feature Selection** | Find optimal features per model | Each model uses its own selection method (see below). Target ~10–12 features for 90 sec UX. |
| **3. Hyperparameter Tuning** | Optimize hyperparameters | Tune each model via randomized search on its own reduced feature set. Select best tuned model + features combination. |
| **4. Final Model** | Evaluate & deploy | Evaluate final model + features on hold-out test set. |

**Candidate Models**
| Model | Loss Function | Target Transform | Notes |
| :--- | :--- | :--- | :--- |
| Linear Regression | MSE | `log(y+1)` | Inverse-transform predictions |
| Elastic Net | MSE + L1/L2 | `log(y+1)` | Built-in feature selection |
| K-Nearest Neighbors | Distance-based | `log(y+1)` | Sensitive to outliers without log |
| MLP (sklearn) | MSE | `log(y+1)` | Inverse-transform predictions |
| Decision Tree | `absolute_error` | None | MAE-based splits; robust to skew |
| Random Forest | `absolute_error` | None | MAE-based splits; robust to skew |
| XGBoost | `reg:tweedie` | None | Tweedie handles skew natively |

**Feature Selection by Model Type**
| Model | Method | Notes |
| :--- | :--- | :--- |
| Elastic Net | L1 regularization | Non-zero coefficients = selected features |
| Random Forest | `feature_importances_` | Built-in feature importance scores |
| XGBoost | `feature_importances_` | Built-in feature importance scores |
| KNN, MLP | Recursive Feature Elimination (RFE) | Wrapper method with CV |

> **Sample Weights:** All models must use `PERWT23F` as `sample_weight` to ensure national representativeness.

> **Future Enhancement:** A Two-Part (Hurdle) Model—P(cost > 0) then E[cost | cost > 0]—may improve predictions for zero-cost users if single-stage models underperform.

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
