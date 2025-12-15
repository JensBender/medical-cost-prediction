# Product Requirements Document (PRD)
| **Project Name** | Medical Cost Prediction |
| :--- | :--- |
| **Status** | Project scoping |
| **Created** | December 5, 2025 |
| **Last Updated** | December 15, 2025 |
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
| **The Budgeter** | Individuals with tight budgets needing to anticipate potential medical expenses. | "What's the worst-case scenario I should prepare for?" |
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
  See headline,        Enter 10 inputs       See cost range,
  understand value     (< 1 minute)          cost drivers,
  proposition                                benchmarks
```

**Happy Path:** User lands → fills form → sees prediction → leaves with actionable number for budgeting.

**Edge Cases:**
| Scenario | System Response |
| :--- | :--- |
| User skips optional inputs | Imputation applied (median/mode); prediction proceeds |
| Predicted cost > 90th percentile | High-uncertainty disclaimer displayed (UI-05) |
| Predicted cost = $0 | Valid result; display "minimal expected costs" messaging |
| Uninsured user | Note that OOP ≈ total cost; prediction may be higher than insured peers |
| Server error / timeout | Display friendly error message; suggest retry |


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


## Out of Scope
The following are explicitly **not** part of this project:

| Category | What's Excluded | Rationale |
| :--- | :--- | :--- |
| **Population** | Children (< 18 years) | MEPS pediatric data has different cost drivers; adult-focused MVP |
| **Population** | Family/household aggregation | Predicts individual costs only; users can run multiple times |
| **Features** | Specific procedure predictions | We predict annual totals, not "How much will my MRI cost?" |
| **Features** | Insurance plan comparison | We don't recommend plans; users input their current plan |
| **Features** | Historical trends / user tracking | No user accounts; stateless predictions |
| **Integrations** | Insurance portal integration | No API connections to external systems |
| **Integrations** | Medical record imports | Users self-report; no clinical data ingestion |
| **Data** | Real-time pricing data | Model uses MEPS survey data, not live cost databases |
| **Compliance** | HIPAA-regulated data handling | Not applicable — no PHI collected; inputs are self-reported, anonymous, and not stored |


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
| **UI-06** | **Permanent Footer** | Always-visible disclaimer at the bottom of the page. Covers legal liability and data aging limitations. | `gr.Markdown` | *"Not intended as medical, financial, or legal advice. Based on 2023 U.S. national survey data."* |


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

| Aspect | Guideline | Gradio Implementation |
| :--- | :--- | :--- |
| **Tone** | Helpful, calm, non-judgmental. Avoid medical jargon (e.g., "High Blood Pressure" not "Hypertension"). | Use plain language in all `label` and `info` parameters. |
| **Visuals** | Trust-building colors (Blues/Greens). Clean, modern design. | Use `gr.themes.Soft()` or custom CSS via `gr.Blocks(css=...)`. |
| **Footer** | Permanent disclaimer visible on all pages (define text in Result Display section). | Add `gr.Markdown(footer_text)` at the end of the layout. |


## Technical Approach
> For technical implementation details (data preprocessing pipeline, machine learning model architecture, web app deployment), see [Technical Specifications](./technical_specifications.md).


## Success Metrics
*   **Predictive Performance:** Median Absolute Error (MdAE) on the test set is < $500 (i.e., for the typical user, the prediction is within $500 of the actual cost).
*   **Interval Coverage:** ≥ 50% of actual costs fall within the predicted 25th–75th percentile range.
*   **Completion Rate:** > 80% of users who start the questionnaire complete it.
*   **User Satisfaction:** Positive sentiment on optional "Was this helpful?" feedback (optional).


## Risk Assessment & Mitigation
| Risk | Example | Mitigation Strategy |
| :--- | :--- | :--- |
| **Outlier Prediction** | Model predicts extreme costs ($100k+) for a standard user. | Consider implementing "Guardrails" in the code to cap displayed predictions at the 95th percentile with a "High Cost Risk" label instead of a raw number. |
| **Bias/Fairness** | Model consistently under-predicts needs for low-income users due to historical access barriers. | Perform a Fairness Audit. Include income as a feature so the user sees that income impacts the prediction. |
| **Data Aging** | 2023 data becomes outdated. | Display permanent footer (UI-06) and limitations notice (UI-04). Apply Medical Inflation Factor (FR-02) to adjust for cost increases. |
| **Policy Changes** | Policy changes enacted after 2023 data collection (e.g., Medicare Part D $2k cap, ACA marketplace adjustments) create systemic over/under-prediction for specific insurance groups. | Covered by permanent footer (UI-06). For Medicare/Medicaid users, add contextual note: *"Recent policy changes (2024-2026) may lower actual costs compared to this estimate."* |


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
| **Out-of-Pocket (OOP)** | Healthcare costs paid directly by the patient (deductibles, copays, coinsurance). Excludes premiums. |
| **SHAP** | SHapley Additive exPlanations. Method for explaining ML predictions by showing each feature's contribution. |
| **TOTSLF23** | MEPS variable representing total out-of-pocket healthcare expenditure for 2023. |

### Related Documentation
*   [U.S. Healthcare Costs Guide](./us_healthcare_costs_guide.md): Primer on U.S. healthcare payment structures, terminology, and why Americans need to predict out-of-pocket costs.
*   [ML with MEPS: Prior Work](./ml_with_meps.md): Literature review of existing ML projects using MEPS data, informing competitive positioning.
*   [Technical Specifications](./technical_specifications.md): Implementation details for data preprocessing pipeline, machine learning model architecture, and web app deployment.
