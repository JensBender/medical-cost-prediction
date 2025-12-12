# Technical Specifications

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

### Metric Selection Rationale
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


## References
*   [ML with MEPS: Prior Work](./ml_with_meps.md): Literature review of existing ML projects using MEPS data, informing competitive positioning.
*   [U.S. Healthcare Costs Guide](./us_healthcare_costs_guide.md): Primer on U.S. healthcare payment structures, terminology, and why Americans need to predict out-of-pocket costs.
