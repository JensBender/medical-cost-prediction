# Product Requirements Document (PRD)
| **Project Name** | Medical Cost Prediction |
| :--- | :--- |
| **Status** | Model Development |
| **Created** | 2025-12-05 |
| **Last Updated** | 2026-03-20 |
| **Data Source** | Medical Expenditure Panel Survey (MEPS) |


## Executive Summary
The **Medical Cost Planner** is a consumer-facing web application that uses machine learning to predict annual out-of-pocket healthcare costs.

**The Problem:** Healthcare pricing is a "black box." While insurance portals show prices for individual treatments (e.g., an MRI), consumers lack tools to predict their total expected costs for the year. Existing calculators are often too generic (ignoring health conditions) or too complex (requiring specific procedure codes).

**Our Solution:** A personalized forecasting tool based on accessible inputs. Users simply enter demographic and health details such as age, insurance status, and chronic conditions to receive a cost estimate for the upcoming year. This empowers users to make data-driven decisions for FSA/HSA contributions and emergency planning.

**How it works:** The web app is powered by a machine learning model trained on the Medical Expenditure Panel Survey (MEPS), the gold standard for U.S. healthcare data. By analyzing what people with similar demographic and health profiles actually spent, our model learns real-world cost patterns and translates them into actionable financial insights without requiring complex medical records.


## User Personas
| Persona | Description | Primary Need |
| :--- | :--- | :--- |
| **The Open Enrollment Planner** | Employees deciding how much to contribute to FSAs/HSAs during open enrollment. | "Should I contribute $1,000 or $3,000?" |
| **The Budgeter** | Individuals with tight budgets needing to anticipate potential medical expenses. | "How much cushion should I set aside for a higher-cost year?" |
| **The Newly Diagnosed** | Users recently diagnosed with a chronic condition (e.g., Diabetes). | "How does this diagnosis impact my financial bottom line?" |
| **The Gig Worker** | Uninsured or underinsured individuals weighing coverage options. | "What's the financial risk of skipping coverage vs. buying a plan?" |
| **The Caregiver** | The "sandwich generation" estimating costs for an elderly parent. | "How much should I budget for my parent's healthcare?" |


## User Flow
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  1. LAND ON  │      │ 2. FILL OUT  │      │  3. VIEW     │
│     PAGE     │─────▶│     FORM     │────▶│   RESULTS    │
└──────────────┘      └──────────────┘      └──────────────┘
       │                     │                     │
  See headline,        Answer a few          See cost range,
  understand value     quick questions       cost drivers,
  proposition          (< 90 seconds)        benchmarks
```

**Happy Path:** User lands → fills out form → sees prediction → leaves with actionable number for budgeting.

**Edge Cases:**
| Scenario | System Response |
| :--- | :--- |
| User skips optional inputs | Imputation applied (median/mode); prediction proceeds; missing-input note displayed |
| High predicted uncertainty | Dynamic high-uncertainty guidance displayed when predicted tail risk or interval width crosses model-version thresholds |
| Predicted cost = $0 | Valid result; display "minimal expected costs" messaging |
| Uninsured user | Note that OOP ≈ total cost and that the safety cushion is more useful than the typical range when costs are volatile |
| Server error / timeout | Display friendly error message; suggest retry |


## Competitive Positioning

### Market Gap
Current tools are either too complex or their predictions are too generic. No existing solution is easy to use, delivers personalized predictions, and is powered by machine learning (ML).

1.  **Procedure Estimators** (FAIR Health, Bluebook): Too specific. Great for checking the price of one MRI, useless for budgeting a whole year.
2.  **Premium Calculators** (KFF, Exchanges): Too broad. They estimate insurance bills, not medical spending.
3.  **FSA/HSA Calculators** (FSAFEDS): Circular logic. They ask you to input your expected costs, the very number you are trying to find.

**Closest Competitor:** The KFF Household Health Spending Calculator uses broad demographic averages (e.g., "Health: Good vs. Bad"). It ignores important details like specific health conditions and produces generic estimates that fail to capture the complex interactions of multiple risk factors in individual users.

### Our Differentiation
We stand apart as the first consumer-centric planner for out-of-pocket healthcare costs. By leveraging machine learning, we prioritize ease of use without compromising predictive performance.

| **Our ML Approach** | **Competitor Approaches** |
| :--- | :--- |
| **Personalized Model**: Trained on 28k+ real profiles (MEPS). | **Averages**: Broad demographic buckets. |
| **Granular Health Inputs**: Specific conditions (Diabetes, Hypertension) & 5-point health scales. | **Binary**: Often just "Good" vs. "Poor". |
| **Explainable**: SHAP values reveal *why* costs are high (e.g., "Tobacco use: +$500"). | **Black Box**: No context provided. |
| **Probabilistic**: Output includes a 25th-75th percentile typical range and a 90th percentile budget-safe estimate. | **Deterministic**: Single point estimates that imply false precision. |
| **Frictionless**: Quick, easy, and free to use, no login required. | **High Friction**: Requires CPT codes, logins, or deep insurance knowledge. |

### UX-First Rationale
**Simplicity > Perfection**: Users need a "ballpark" estimate for decision making (e.g., $1k vs $3k FSA contribution), not a medical bill audit. By targeting form completion in **under 90 seconds** with ~10–12 high-impact inputs, we achieve user friendliness while keeping the median/q50 estimate within a useful error range for personal finance. The input count is a soft guideline; cognitive load and completion time are the true UX goals.


## Out of Scope
The following are explicitly **not** part of this project:

| Category | What's Excluded | Rationale |
| :--- | :--- | :--- |
| **Population** | Children (< 18 years) | MEPS pediatric data has different cost drivers; adult-focused MVP product release. See [Future Consideration](#future-considerations). |
| **Population** | Family/household aggregation | Predicts individual costs only; users can run multiple times |
| **Features** | Specific procedure predictions | We predict annual totals, not "How much will my MRI cost?" |
| **Features** | Insurance plan comparison | We don't recommend plans; users input their current plan |
| **Features** | Historical trends / user tracking | No user accounts, no longitudinal user records, and no default collection of follow-up actual spend |
| **Integrations** | Insurance portal integration | No API connections to external systems |
| **Integrations** | Medical record imports | Users self-report; no clinical data ingestion |
| **Data** | Real-time pricing data | Model uses MEPS survey data, not live cost databases |
| **Compliance** | HIPAA-regulated data handling | Not applicable — no PHI collected; inputs are self-reported, anonymous, and not stored |


## Functional Requirements

### User Input 
> **Status:** *Preliminary (Candidate Features). Final feature selection pending. Reduce number of features based on empirical feature importance ranking to meet the UX budget below.*

**UX Budget:**
* Completion Time Goal: < 90 seconds
* UI Interactions Target: ~12–14 
* UI Interactions Current: 15 

The UI must be a simple form on a single page. A multi-select checklist (e.g., chronic conditions) counts as one UI interaction. Inputs are mapped to MEPS variables with correct temporal alignment (beginning-of-year status for prospective prediction).

**Full Details:** See [Technical Specifications: Candidate Features](./technical_specifications.md#candidate-features) and [Candidate Features Research](../research/candidate_features.md).

**Single-Selection Interactions** (13 UI interactions, 13 inputs)
| ID | UI Label | UI Question | UI Element | Value Range | MEPS Variable |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IN-01** | Birth Year | In what year were you born? | `gr.Number` | [1940, 2007] | `AGE23X` |
| **IN-02** | Sex | Are you male or female? | `gr.Radio` | ["Male", "Female"] | `SEX` |
| **IN-03** | State | In which state do you live? | `gr.Dropdown` | ["Alabama", "Alaska", "Arizona", ...] (50 States + DC) | `REGION23` (Mapped from State) |
| **IN-04** | Marital Status | Are you now married, widowed, divorced, separated, or never married? | `gr.Dropdown` | ["Married", "Widowed", "Divorced", "Separated", "Never Married"] | `MARRY31X` |
| **IN-05a** | Family Size | How many people (including yourself) live in your home who are related to you by blood, marriage, or adoption? | `gr.Dropdown` | [1, 2, 3, 4, 5, 6, 7, 8+] | `FAMSZE23` |
| **IN-05b** | Family Income | What is your family's total annual income from all sources? | `gr.Dropdown` | Dynamic ranges based on family size (IN-05a). Map to 2023 poverty categories. | `POVCAT23` |
| **IN-06** | Education | What is the highest degree you have obtained? | `gr.Dropdown` | ["No Degree", "GED", "High School Diploma", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctorate or Professional (MD, JD, etc.)", "Other"] | `HIDEG` |
| **IN-07** | Employment Status | Do you currently have a job for pay or own a business? | `gr.Dropdown` | ["Employed", "On leave / Job to return to", "Worked earlier this year, but not now", "Not employed this year"] | `EMPST31` |
| **IN-08** | Insurance Status | Are you covered by any type of health insurance? | `gr.Dropdown` | ["Private", "Public Only", "Uninsured"] | `INSCOV23` |
| **IN-09** | Physical Health | In general, would you say your health is...? | `gr.Radio` | ["Excellent", "Very Good", "Good", "Fair", "Poor"] | `RTHLTH31` |
| **IN-10** | Mental Health | In general, would you say your mental health is...? | `gr.Radio` | ["Excellent", "Very Good", "Good", "Fair", "Poor"] | `MNHLTH31` |
| **IN-11** | Usual Source of Care | Is there a particular doctor's office, clinic, health center, or other place you usually go if you are sick or need medical advice? | `gr.Radio` | ["Yes", "No"] | `HAVEUS42` |
| **IN-12** | Smoker | Do you currently smoke cigarettes? | `gr.Radio` | ["Yes", "No"] | `ADSMOK42` |

**Multi-Selection Interactions** (2 UI interactions, 13 inputs)
| ID | UI Label | UI Question | UI Element | Options | MEPS Variables |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IN-13** | Chronic Conditions Checklist (8 inputs) | Have you ever been told by a doctor or other health professional that you: | `gr.CheckboxGroup` | (1) have hypertension, also called high blood pressure, (2) have high cholesterol, (3) have diabetes or sugar diabetes, (4) have coronary heart disease, (5) had a stroke, (6) had cancer or a malignancy of any kind, (7) have arthritis, (8) have asthma | `HIBPDX`, `CHOLDX`, `DIABDX_M18`, `CHDDX`, `STRKDX`, `CANCERDX`, `ARTHDX`, `ASTHDX` |
| **IN-14** | Limitations & Symptoms Checklist (5 inputs) | Do any of the following apply to you? | `gr.CheckboxGroup` | (1) I receive help or supervision with personal care such as bathing, dressing, or getting around the house, (2) I receive help or supervision with using the telephone, paying bills, taking medications, preparing light meals, doing laundry, or going shopping, (3) I have serious difficulty walking or climbing stairs, (4) I have serious difficulty concentrating, remembering, or making decisions, (5) During the past 12 months, I had pain, aching, stiffness, or swelling in or around a joint | `ADLHLP31`, `IADLHP31`, `WLKLIM31`, `COGLIM31`, `JTPAIN31_M18` |


### Prediction Engine
| ID | Requirement | Details |
| :--- | :--- | :--- |
| **FR-01** | **Imputation** | If optional fields are skipped, default to the mode for categorical features and the median for numerical features. |
| **FR-02** | **Inflation Adjustment** | Model predicts in 2023 dollars. Apply medical inflation multiplier: `Final_Prediction = Model_Output × (1 + Medical_Inflation_Rate)^(CurrentYear - 2023)` |
| **FR-03** | **Cost Range** | Generate 25th–75th percentile range (typical range) and 90th percentile (budget-safe estimate) to communicate prediction uncertainty. Never output a single point estimate. |
| **FR-04** | **Cost Drivers** | Compute SHAP values for each prediction to explain feature contributions as dollar impacts. |
| **FR-05** | **Comparison Benchmarks** | Compare user's prediction to (1) national average and (2) average for their age group. Pre-compute benchmarks from MEPS data. |
| **FR-06** | **Prediction Warning Policy** | Generate neutral, actionable warning flags for high predicted uncertainty, wide prediction intervals, uninsured-user volatility, missing optional inputs, and public-coverage policy changes. Threshold-based prediction flags must be defined from the validation prediction distribution for the active model version, not from the locked test set. |

### Result Display
| ID | Component | Description | UI Element | Example |
| :--- | :--- | :--- | :--- | :--- |
| **UI-01** | **Cost Range** | Large, prominent display of out-of-pocket cost prediction as a typical range, plus a budget-safe estimate for higher-cost-year planning. | `gr.Markdown` | "Estimated Out-of-Pocket Healthcare Cost for Next Year: **$1,450 – $2,100** (typical range)<br>For a higher-cost year, consider budgeting up to: **$3,200**" |
| **UI-02** | **Cost Drivers** | Explanation of key cost drivers and their dollar impact (SHAP). | `gr.Markdown` | "Your Diabetes Diagnosis (+$1,200), your Age (+$400), but your "Excellent" self-reported health lowered the estimate by (-$300)" |
| **UI-03** | **Comparison Benchmarks** | Bar chart comparing user vs. national and age group benchmarks. | `gr.Plot` | "Typical American (median): $4,800 vs. Typical for Age 45–54 (median): $3,200" |
| **UI-04** | **Limitations Notice** | Always-on guidance to help users interpret their prediction and understand the rare-tail limitation. | `gr.Markdown` | "**About This Estimate**<br>• Based on 2023 national survey data; recent policy changes may affect actual costs.<br>• Does not include insurance premiums or over-the-counter medications.<br>• Some high-cost years are driven by new diagnoses, accidents, hospitalizations, or plan-specific billing details this form cannot know in advance." |
| **UI-05** | **Contextual Warning Notes** | Dynamic warning notes displayed only when they are actionable and based on the current prediction or user-selected inputs. | `gr.Markdown` | "**Planning Note**<br>This estimate has a wider uncertainty range because people with similar profiles had more variable costs. For planning, pay more attention to the safety cushion than the plan-around estimate." |
| **UI-06** | **Permanent Footer** | Always-visible disclaimer at the bottom of the page. Covers legal liability and data aging limitations. | `gr.Markdown` | *"Not intended as medical, financial, or legal advice. Based on 2023 U.S. national survey data."* |

### Prediction Warning Policy
Warning copy must be concise, neutral, and tied to a concrete user action. The app should not display stigmatizing per-user subgroup banners simply because the user belongs to a subgroup that was flagged during the offline audit. In particular, the low income, poor mental health, doctorate degree, and near-poor subgroup findings should be handled through documentation, monitoring, and future validation rather than direct user-facing warnings.

| Warning Flag | Trigger | User-Facing Guidance |
| :--- | :--- | :--- |
| `HIGH_PREDICTED_UNCERTAINTY` | Predicted safety cushion (`q90`) falls in the top 20% of validation predicted `q90` values for the active model version | Explain that similar profiles had more variable costs and emphasize the safety cushion |
| `WIDE_PREDICTION_INTERVAL` | Safety-cushion width (`q90 - q50`) or typical-range width (`q75 - q25`) exceeds validation-derived width thresholds | Explain that the range is intentionally wide and should be treated as a planning band |
| `UNINSURED_UNCERTAINTY` | User selects uninsured status | Explain that out-of-pocket costs can be more volatile and that the safety cushion is the more useful planning number |
| `MISSING_OPTIONAL_INPUTS` | One or more optional inputs are skipped and imputed | Explain that typical training values were used and that more complete inputs may make the estimate more tailored |
| `PUBLIC_COVERAGE_POLICY_CHANGE` | User selects public-only coverage | Explain that policy changes after 2023, especially Medicare drug-cost caps, may lower actual costs compared with estimates based on 2023 survey data |

The always-on limitations notice remains the primary way to communicate that rare future high-cost events cannot always be identified from pre-year user inputs. The dynamic high-uncertainty flags should be based on prediction behavior, not on the unknowable actual future cost tier.


## Non-Functional Requirements

### Privacy & Security
| ID | Requirement | Details |
| :--- | :--- | :--- |
| **NFR-01** | Ephemeral Sessions | No user data written to disk or database. All inputs remain in browser/RAM session state only. |
| **NFR-02** | No PII Collection | No names, emails, addresses, or social security numbers shall be requested. |
| **NFR-03** | Aggregate Monitoring Only | Monitoring for the MVP product release may use aggregate, non-identifying counters for app health, completion rate, input drift, and prediction drift. By default, do not persist user-level input rows, predictions, SHAP explanations, app-level IP logs, user-agent logs, or session identifiers. |

**Hosting Boundary:** The planned Hugging Face Hub and Hugging Face Spaces deployment may involve provider-level infrastructure logging under Hugging Face's own policies. The product privacy promise is that the Medical Cost Planner app does not intentionally collect, persist, expose, or join IP addresses, user agents, session identifiers, raw inputs, per-user predictions, or SHAP explanations for model monitoring.

### Performance & Usability
| ID | Requirement | Details |
| :--- | :--- | :--- |
| **NFR-04** | Latency | Inference prediction (including SHAP generation) must return in < 1 second (server-side) to ensure a responsive UX (~3s end-to-end). |
| **NFR-05** | Responsive Design | Expect ~65% desktop, ~35% mobile (typical for Hugging Face Spaces). Gradio handles responsive layouts natively. Ensure form inputs remain usable on smaller screens. |
| **NFR-06** | Fallback Mode | If user skips an input, display informative message or impute value. |


## UI/UX Guidelines

| Aspect | Guideline | Gradio Implementation |
| :--- | :--- | :--- |
| **Tone** | Helpful, calm, non-judgmental. Avoid medical jargon (e.g., "High Blood Pressure" not "Hypertension"). | Use plain language in all UI text elements. |
| **Visuals** | Trust-building colors (Blues/Greens). Clean, modern design. | Use `gr.themes.Soft()` or custom CSS via `gr.Blocks(css=...)`. |


## Technical Approach
For technical implementation details such as data preprocessing, machine learning modeling, and web app deployment, see [Technical Specifications](./technical_specifications.md).


## Success Metrics
*   **Predictive Performance:** Median Absolute Error (MdAE) for the median/q50 estimate on the unseen test set is < $500. This is the hard release gate for the MVP; the MVP product launch performance target is < $350 on test.
*   **Interval Calibration:** Predicted ranges must demonstrate high reliability:
    *   **Typical Range (25th–75th%):** 50% ± 5% of actual costs fall within the predicted range.
    *   **Safety Cushion (90th%):** 90% ± 5% of actual costs fall below the predicted budget-safe estimate.
*   **Interval Width:** Prediction intervals should be narrow enough to support decisions. The model should not meet coverage targets by returning overly wide ranges for most users.
*   **Stratified Reliability:** Report MdAE and interval coverage by cost tier (e.g., low, middle, high) and key user groups before launch to confirm that strong overall metrics are not hiding weak subgroup performance.
*   **Privacy-Preserving Monitoring:** After launch, monitor aggregate app health, completion rate, input drift, and prediction drift without retaining user-level records, app-level IP addresses, user agents, or session identifiers. Aggregate counters should be created during prediction handling and stored only as bucketed counts, not by saving individual rows for later aggregation. True post-launch calibration requires observed annual out-of-pocket spending and is out of scope for the privacy-first MVP product release.
*   **Completion Rate:** > 70% of users who enter at least one value (e.g., select an age) successfully generate a cost prediction.
*   **User Satisfaction:** Positive sentiment on "Was this helpful?" feedback (optional).


## Risk Assessment & Mitigation
| Risk | Example | Mitigation Strategy |
| :--- | :--- | :--- |
| **Outlier Prediction** | Model predicts extreme costs ($100k+) for a standard user. | Consider implementing display guardrails such as a model-version cap plus `HIGH_PREDICTED_UNCERTAINTY`, so the user sees a planning warning rather than a falsely precise extreme number. |
| **Bias/Fairness** | Model consistently under-predicts needs for low-income users due to historical access barriers. | Perform a fairness audit, document subgroup caveats, and use neutral user-facing warnings only when they are actionable and based on the current prediction or user-selected inputs. Do not display subgroup-only warnings if they are not actionable. |
| **Data Aging** | 2023 data becomes outdated. | Display permanent footer (UI-06) and limitations notice (UI-04). Apply Medical Inflation Factor (FR-02) to adjust for cost increases. |
| **Policy Changes** | Policy changes enacted after 2023 data collection (e.g., Medicare Part D $2k cap, ACA marketplace adjustments) create systemic over/under-prediction for specific insurance groups. | Covered by permanent footer (UI-06). For Medicare/Medicaid users, add contextual note: *"Recent policy changes (2024-2026) may lower actual costs compared to this estimate."* |
| **Unobserved Outcomes** | App users usually will not return one year later with reliable actual out-of-pocket spending, and collecting linked follow-up outcomes would weaken the anonymous, zero-retention privacy promise. | Do not claim production calibration from default app telemetry. Use aggregate drift monitoring for the MVP product release. Evaluate true calibration by testing the deployed model on future MEPS survey years when available, or through a separately approved opt-in study. |


## Future Considerations  
The following features and improvements are planned for future releases beyond the MVP product release:

**Two-Part (Hurdle) Modeling**  
*   **Goal**: Improve accuracy for users with zero or very low expected medical spending.
*   **Technical Strategy**: Implement a **Two-Part Model** to handle zero-inflation in healthcare costs:
    *   **Part 1 (Classifier)**: Predict the probability that a user will have *any* out-of-pocket costs (Cost > $0 vs. Cost = $0).
    *   **Part 2 (Regressor)**: For those predicted to have costs, predict the specific dollar amount.
*   **Value**: This approach prevents the model from "averaging" zero-cost and high-cost users together, which can lead to biased estimates for healthy individuals.

**Optional Outcome Study for True Calibration Monitoring**  
*   **Goal**: Measure real-world calibration against actual annual out-of-pocket spending if the product strategy ever changes to support labeled outcome collection.
*   **Privacy Constraint**: This is not part of the MVP product release. It would require explicit opt-in, clear consent language, data minimization, retention limits, and a review of whether the product still satisfies the anonymous/no-account positioning.
*   **Feasibility Caveat**: User-reported annual spend is likely noisy because MEPS derives `TOTSLF23` through detailed event and payment questions. A lightweight recall question may be useful for directional research but should not be treated as MEPS-equivalent ground truth.
*   **Default Alternative**: Use future MEPS releases to test the deployed model on newer survey years. Treat retraining or recalibration as a separate model-refresh decision.

**Support for Under 18 Population**  
*   **Rationale for 18+ in current model**:
    *   **Distinct Cost Drivers**: Pediatric costs are driven by development, vaccinations, and acute illness, whereas adult costs are driven by chronic disease and aging.
    *   **Feature Incompatibility**: Key adult predictors (e.g., Marital Status, Employment, Education) are inapplicable to children, necessitating a different feature set.
    *   **Data Constraints**: Many MEPS chronic condition variables are only recorded for respondents aged 18+.
    *   **Model Precision**: A unified 0–85 model risks lower accuracy by conflating two very different healthcare utilization patterns. 
*   **Goal**: Expand the app to support users under the age of 18.
*   **Technical Strategy**: Implement a **Two-Model Architecture**.
    *   **Adult Model (18+)**: Continue using the current model specialized for chronic disease and aging drivers.
    *   **Pediatric Model (< 18)**: Train a separate model using child-specific features (e.g., asthma, ADHD, well-child visits, and birth-related costs).
*   **UX/UI Impact**: Add an initial age selection that dynamically adjusts the rest of the form questions (e.g., hiding "Employment", "Education", and "Marital Status" for children).


## Appendix

### Policy Changes (2024-2026)
Since the collection of the 2023 MEPS data, key policy changes have been enacted that affect out-of-pocket costs. This section documents these changes for awareness and transparency.

**Note**: The app predicts **total out-of-pocket costs** (`TOTSLF23`), not costs itemized by category. This limits our ability to apply policy-specific hard caps (e.g., prescription drug cap). The existing Medical Inflation Factor (FR-02) already accounts for general cost changes.

**1. Inflation Reduction Act (IRA) - Medicare Part D**
*   **2025 Change**: Annual out-of-pocket cap of **$2,000** for prescription drugs for Medicare Part D beneficiaries.
*   **Impact**: Model may over-predict costs for Medicare users with high prescription spending, as 2023 training data includes seniors who spent >$2,000 on drugs.
*   **Mitigation**: Display contextual disclaimer for "Public (Medicare/Medicaid)" users: *"Note: The 2025 Inflation Reduction Act caps prescription drug costs at $2,000/year for Medicare beneficiaries. Your actual costs may be lower than this estimate."*

**2. ACA Marketplace Adjustments**
*   **Cost Sharing Limits**: For 2025, the maximum out-of-pocket limit is **$9,200** (individual) for ACA marketplace plans.
*   **Subsidy Expiration Risk**: Enhanced premium tax credits expire December 31, 2025. If not renewed, unsubsidized premiums could increase significantly in 2026.
*   **Mitigation**: Covered by permanent footer (UI-06). Cannot apply hard cap because app does not differentiate marketplace from employer insurance within the "Private" insurance category.

**3. Medicare Part B Increases**
*   **Deductibles**: Increased to $257 in 2025 (up from $226 in 2023).
*   **Premiums**: Standard premium rose to $185.00/month in 2025 (up from $164.90 in 2023).
*   **Mitigation**: The Medical Inflation Factor (FR-02) already accounts for these cost increases. No additional adjustment needed.

### Glossary
| Term | Definition |
| :--- | :--- |
| **FSA** | Flexible Spending Account. Employer-sponsored account for tax-free healthcare savings. "Use it or lose it" annually. |
| **HSA** | Health Savings Account. Tax-advantaged account for healthcare expenses. Funds roll over year to year. |
| **MdAE** | Median Absolute Error. Primary evaluation metric measuring the median difference between predicted and actual costs. |
| **MEPS** | Medical Expenditure Panel Survey. Federal survey by the Agency for Healthcare Research and Quality tracking U.S. healthcare utilization and expenditures. |
| **Model Artifact** | The serialized trained model/pipeline used for inference. It is not the product by itself. |
| **MVP Product Release** | The first end-to-end releasable version of the Medical Cost Planner, including the selected model artifact, prediction service, user-facing app, privacy behavior, disclaimers, and deployment configuration. |
| **Noninstitutionalized Civilian Population** | The specific U.S. population segment represented by MEPS. Includes people in households and dorms; **excludes** active-duty military and people in institutions (nursing homes, prisons). |
| **Out-of-Pocket (OOP)** | Healthcare costs paid directly by the patient (deductibles, copays, coinsurance). Excludes premiums. |
| **Prediction Service** | The inference/API layer that validates user inputs, calls the model artifact, applies post-processing, and returns prediction outputs to the app. |
| **Product** | The user-facing Medical Cost Planner experience and promise, not just the machine-learning model. |
| **SHAP** | SHapley Additive exPlanations. Method for explaining ML predictions by showing each feature's contribution. |
| **TOTSLF23** | MEPS variable representing total out-of-pocket healthcare expenditure for 2023. |

### Related Documentation
*   [U.S. Healthcare Costs Guide](./us_healthcare_costs_guide.md): Primer on U.S. healthcare payment structures, terminology, and why Americans need to predict out-of-pocket costs.
*   [ML with MEPS](./ml_with_meps.md): Literature review of existing machine learning projects using MEPS data, informing competitive positioning.
*   [Technical Specifications](./technical_specifications.md): Implementation details for data preprocessing, machine learning modeling, and web app deployment.
