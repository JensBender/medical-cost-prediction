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


## User Personas
| Persona | Description | Primary Need |
| :--- | :--- | :--- |
| **The Open Enrollment Planner** | Employees deciding how much to contribute to FSAs/HSAs during open enrollment. | "Should I contribute $1,000 or $3,000?" |
| **The Budgeter** | Individuals with tight budgets needing to anticipate potential medical expenses. | "What's the worst-case scenario I should prepare for?" |
| **The Newly Diagnosed** | Users recently diagnosed with a chronic condition (e.g., Diabetes). | "How does this diagnosis impact my financial bottom line?" |
| **The Gig Worker** | Uninsured or underinsured individuals weighing coverage options. | "What's the financial risk of skipping coverage vs. buying a plan?" |
| **The Caregiver** | The "sandwich generation" estimating costs for an elderly parent. | "How much should I budget for my parent's healthcare?" |


## Functional Requirements

### Input Module (The "1-Minute" Form)
The UI must be minimal friction—a single page with no more than 10 core inputs. Inputs are mapped to specific MEPS variables.

| ID | Label (UI) | Input Type | Options / Range | MEPS Variable |
| :--- | :--- | :--- | :--- | :--- |
| **IN-01** | Age | Slider | 18 – 85 | `AGE23X` |
| **IN-02** | Biological Sex | Toggle | Male / Female | `SEX` |
| **IN-03** | Region | Dropdown | Northeast, Midwest, South, West | `REGION23` |
| **IN-04** | Household Income | Dropdown | Low (<$30k), Middle, High (>$100k) | `POVCAT23` |
| **IN-05** | Insurance Status | Dropdown | Private, Public (Medicare/Medicaid), Uninsured | `INSCOV23` |
| **IN-06** | Physical Health | 5-Star Rating | Poor (1) → Excellent (5) | `RTHLTH31` |
| **IN-07** | Mental Health | 5-Star Rating | Poor (1) → Excellent (5) | `MNHLTH31` |
| **IN-08** | Diabetes? | Checkbox | Yes / No | `DIABDX_M18` |
| **IN-09** | High Blood Pressure? | Checkbox | Yes / No | `HIBPDX` |
| **IN-10** | Current Smoker? | Checkbox | Yes / No | `ADSMOK42` |

### Processing Engine
| ID | Requirement | Details |
| :--- | :--- | :--- |
| **FR-01** | **Imputation** | If optional fields (Health Status) are skipped, default to the *mode* (most common value) for the user's Age/Sex bracket. |
| **FR-02** | **Inflation Adjustment** | Model predicts in 2023 dollars. Apply CPI-Medical multiplier: `Final_Prediction = Model_Output × (1 + Medical_Inflation_Rate)^(CurrentYear - 2023)` |
| **FR-03** | **Prediction Intervals** | Generate 25th–75th percentile range to communicate uncertainty. Never output a single point estimate. |
| **FR-04** | **Cost Toggle** | Allow users to view "Total Cost to Healthcare System" (`TOTEXP23`) OR "Out-of-Pocket Cost to You" (`TOTSLF23`). |

### Visualization Dashboard
| ID | Component | Description | Example |
| :--- | :--- | :--- | :--- |
| **UI-01** | **Hero Metric** | Large, prominent display of the estimated cost range. | "Estimated Annual Cost: **$1,450 – $2,100**" |
| **UI-02** | **Cost Drivers (SHAP)** | Dynamic text explaining key factors with dollar impact. | "Your Diabetes Diagnosis (+$1,200), Your Age (+$400), Your 'Excellent' physical health (-$300)" |
| **UI-03** | **Benchmark Comparison** | Bar chart comparing user vs. national average for their age group. | "You are projected to spend 15% less than the national average." |
| **UI-04** | **Cost Toggle** | Switch between "Total System Cost" and "Your Out-of-Pocket Cost". | Button/Toggle in UI |

## Data & Machine Learning Specifications

### Dataset
- **Source:** MEPS-HC 2023 Full Year Consolidated Data File (H251)
- **Documentation:** [H251 Codebook](https://meps.ahrq.gov/data_stats/download_data_files_codebook.jsp?PUFId=H251)

### Preprocessing Pipeline
| Step | Action | Rationale |
| :--- | :--- | :--- |
| 1 | Drop rows where `PERWT23F` = 0 | Zero-weight respondents don't represent the population. |
| 2 | Handle MEPS Negative Codes | Convert `-1` (Inapplicable), `-8` (DK) to: `0` for binary conditions (assume "No"), `NaN` for health status (then impute). |
| 3 | Log-Transform Target | `log(TOTEXP23 + 1)` for training stability with highly skewed cost data. |
| 4 | Feature Engineering | Create `COMORBIDITY_COUNT` = sum of `DIABDX` + `HIBPDX` + `ADSMOK42` to capture interaction effects. |

### Model Architecture
| Aspect | Specification | Rationale |
| :--- | :--- | :--- |
| **Algorithm** | Quantile Gradient Boosting (XGBoost/LightGBM) | Predicts median (50th) and spread (25th/75th). Standard MSE is too sensitive to US healthcare outliers. |
| **Zero Handling** | Consider Two-Part Hurdle Model or Tweedie objective | MEPS data is zero-inflated (many people have $0 expenditure). |
| **Sample Weighting** | Use `PERWT23F` as `sample_weight` | Ensures model represents the US population, not just the survey sample. |
| **Target Variables** | `TOTEXP23` (Total) AND `TOTSLF23` (Out-of-Pocket) | Enables the "Cost Toggle" feature. |

## Non-Functional Requirements

### Privacy & Security
| ID | Requirement | Details |
| :--- | :--- | :--- |
| **NFR-01** | Ephemeral Sessions | No user data written to disk or database. All inputs remain in browser/RAM session state only. |
| **NFR-02** | No PII Collection | No names, emails, exact addresses, or SSNs shall be requested. |
| **NFR-03** | Client-Side Option | Consider converting model to ONNX/TensorFlow.js to run entirely in the user's browser for maximum privacy. |

### Performance & Usability
| ID | Requirement | Details |
| :--- | :--- | :--- |
| **NFR-04** | Latency | Inference prediction must return in < 200ms (server-side) or < 2 seconds (client-side). |
| **NFR-05** | Mobile-First | 70%+ traffic expected from mobile. Use native OS pickers, large tap targets. |
| **NFR-06** | Fallback Mode | If user skips an input, display informative message or impute value gracefully. |


## UI/UX Guidelines
| Aspect | Guideline |
| :--- | :--- |
| **Tone** | Helpful, calm, non-judgmental. Avoid medical jargon (e.g., "High Blood Pressure" not "Hypertension"). |
| **Visuals** | Trust-building colors (Blues/Greens). Clean, modern design. |
| **Disclaimer** | Permanent footer: *"This tool is for educational purposes only. It is a statistical estimate based on 2023 national data, not a medical billing quote or guarantee."* |
| **Startup Modal** | First-time users see a disclaimer modal before accessing the tool. |

## Technical Stack
| Layer | Technology | Notes |
| :--- | :--- | :--- |
| **Model Training** | Python (Scikit-Learn, XGBoost) | Model serialized as `.joblib` or `.pkl` |
| **Frontend (MVP)** | Streamlit | Rapid prototyping and deployment |
| **Frontend (Production)** | Gradio or custom React/Next.js | Enhanced UX and customization |
| **Backend (Production)** | FastAPI | For scalable API serving |
| **Hosting - Source** | GitHub | Version control and collaboration |
| **Hosting - Model** | Hugging Face Hub | Model versioning and distribution |
| **Hosting - App** | Hugging Face Spaces (Free Tier) | Or AWS Lambda + API Gateway |


## Success Metrics
| Metric | Target | Measurement Method |
| :--- | :--- | :--- |
| **Model Accuracy** | MAE within $500 of MEPS benchmarks for median patient | Test set evaluation |
| **Completion Rate** | > 80% of users who start the form complete it | Analytics tracking |
| **User Satisfaction** | Positive sentiment majority | Optional "Was this helpful?" feedback |
| **Latency** | P95 < 200ms | Server monitoring |


## Risk Assessment & Mitigation
| Risk | Probability | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **"Zero Cost" Prediction** | High | User is healthy, model predicts $0—unrealistic. | Implement a "Floor" value (~$200) for OTC meds and unexpected urgent care. |
| **Outlier Shock** | Medium | User sees terrifyingly high number ($100k+). | Cap visual display at 90th percentile; label as "High Complexity Case" instead of raw number. |
| **Misinterpretation** | High | User treats prediction as a billing quote. | Hard-coded Disclaimer Modal on startup + permanent footer. |
| **Bias/Fairness** | Medium | Model under-predicts for low-income users due to historical access barriers. | Perform Fairness Audit. Include income as visible feature so user sees its impact. |
| **Data Aging** | High | 2023 data becomes outdated. | Apply Medical Inflation Factor (FR-02). Inform users of data limitations. |


## Implementation Roadmap
### Phase 1: MVP (Weeks 1–2)
- [ ] Clean H251 data and extract the "Golden 10" features
- [ ] Handle MEPS negative codes and missing values
- [ ] Train XGBoost Regressor on `TOTEXP23` with sample weights
- [ ] Build basic Streamlit interface with all inputs
- [ ] Deploy to internal server / Hugging Face Spaces for testing

### Phase 2: Refinement (Weeks 3–4)
- [ ] Implement Quantile Regression for prediction intervals (25th/75th)
- [ ] Add Inflation Adjustment logic (CPI-Medical)
- [ ] Train secondary model on `TOTSLF23` for out-of-pocket toggle
- [ ] Integrate SHAP for "Cost Drivers" explainability
- [ ] Style UI with custom CSS (trust-building colors, mobile-responsive)

### Phase 3: Release (Week 5)
- [ ] Add disclaimers (startup modal + footer)
- [ ] Implement guardrails (floor value, outlier cap)
- [ ] QA on edge cases (e.g., 18yo with all conditions, 85yo healthy)
- [ ] Public launch on GitHub portfolio site


## Appendix: Feature Mapping Reference
| MEPS Variable | Description | Data Type | UI Mapping |
| :--- | :--- | :--- | :--- |
| `AGE23X` | Age as of 12/31/2023 | Continuous | Slider (18–85) |
| `SEX` | Biological Sex | Binary (1=M, 2=F) | Toggle |
| `REGION23` | Census Region | Categorical (1–4) | Dropdown |
| `POVCAT23` | Poverty Category | Ordinal (1–5) | Dropdown (Income Brackets) |
| `INSCOV23` | Insurance Coverage | Categorical (1–3) | Dropdown |
| `RTHLTH31` | Physical Health Status | Ordinal (1–5) | 5-Star Rating (inverted) |
| `MNHLTH31` | Mental Health Status | Ordinal (1–5) | 5-Star Rating (inverted) |
| `DIABDX_M18` | Diabetes Diagnosis | Binary | Checkbox |
| `HIBPDX` | High Blood Pressure Diagnosis | Binary | Checkbox |
| `ADSMOK42` | Current Smoker Status | Binary | Checkbox |
| `PERWT23F` | Person Weight (for training) | Continuous | N/A (internal) |
| `TOTEXP23` | Total Healthcare Expenditure | Continuous ($) | Target Variable 1 |
| `TOTSLF23` | Total Out-of-Pocket Expenditure | Continuous ($) | Target Variable 2 |
