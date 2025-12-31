# Candidate Features for Medical Cost Prediction

> **Sources:** Gemini Deep Research (Dec 2025), MEPS H251 Codebook, and domain literature review.

Candidate feature list for predicting annual out-of-pocket healthcare costs (`TOTSLF23`) using MEPS-HC 2023 data. Features are selected based on three criteria:

1. **Consumer Accessibility:** Users can answer from memory without looking up documents or records.
2. **Beginning-of-Year Data:** Variables were measured at the beginning of the year to enable cost prediction for the upcoming year without data leakage.
3. **Predictive Power:** Features have established significance in the healthcare cost prediction literature.

**Beginning-of-Year Rationale**  
MEPS variable suffixes indicate when data was collected during the survey year:

| Suffix | Timing | Example |
|:---|:---|:---|
| `31` | Beginning of year (Rounds 3/1) | `RTHLTH31` |
| `42` | Mid-year (Rounds 4/2) | `ADSMOK42` |
| `53` | End of year (Rounds 5/3) | `RTHLTH53` |
| `23` or `23X` | Full-year summary or year-end point | `AGE23X`, `INSCOV23` |

The Medical Cost Planner app is designed for use during Open Enrollment (Nov–Dec) to forecast healthcare costs for the **upcoming calendar year**. To mirror this prospective use case and prevent data leakage (using year-end health status to predict costs that have already occurred), we train specifically on **beginning-of-year (`31`) variables** for all time-varying features like perceived health, limitations, and symptoms. Chronic conditions are treated as temporally stable and measures from mid-year or end-of-year may be used if no beginning-of-year variable is available.

**Training-Serving Temporal Skew**  
MEPS `31` variables are collected Jan–June of the survey year, meaning some "beginning of year" values may reflect events that occurred after costs started accumulating. App users providing data in Nov–Dec for the upcoming year perform a strictly prospective prediction. This may cause slight overestimation of costs in training metrics compared to real-world performance. The effect is expected to be small since most high-importance features (chronic conditions, demographics) are temporally stable. 

Note: For predictions made in Oct–Dec, consider using the upcoming year as the target year for age calculation to align with open enrollment planning.


## Feature Categories

### 1. Demographics
Primary drivers of healthcare utilization based on biological and geographic factors.

| Variable | Label | Description | Survey Question | Response Categories | Rationale |
|:---|:---|:---|:---|:---|:---|
| `AGE23X` | Age | Age as of Dec 31, 2023 | "What is your date of birth?" | Num (DOB) | Primary driver of utilization; costs follow a U-curve with most spending at high age. [[1]](#ref1) |
| `SEX` | Sex | Biological sex | "Are you male or female?" | Male, Female | Influences utilization via gender-specific conditions. [[2]](#ref2) |
| `REGION23` | Region | Census region | Derived from address | Northeast, Midwest, South, West | Captures geographic pricing variations. [[2]](#ref2) |
| `MARRY31X` | Marital Status | Status at beginning of year | "Are you now married, widowed, divorced, separated, or never married?" | Married, Widowed, Divorced, Separated, Never Married | Proxy for social support and income stability. [[3]](#ref3) |

### 2. Socioeconomic
Proxies for healthcare access, literacy, insurance quality, and ability to pay.

| Variable | Label | Description | Survey Question | Response Categories | Rationale |
|:---|:---|:---|:---|:---|:---|
| `POVCAT23` | Poverty Category | Family income % of poverty | Derived from family income + size | Poor, Near Poor, Low, Middle, High | Determines subsidy eligibility and insurance quality. [[4]](#ref4) |
| `FAMSZE23` | Family Size | Number of related persons residing together | "How many people (including yourself) live in your home who are related to you by birth, marriage, or adoption?" | Numerical (1–8+) | Based on CPS definition; excludes roommates and family living elsewhere. |
| `HIDEG` | Education | Highest degree attained | Derived from years of education, high school diploma status, and degree type | No Degree, GED, HS Diploma, Bachelor's, Master's, Doctorate, Other | Correlates with health literacy and preventive care use. [[4]](#ref4) |
| `EMPST31` | Employment Status | Status at beginning of year | Derived from a series of questions covering current job, job to return to, and work earlier in the round | Employed, Job to Return to, Job During Ref Period, Not Employed | Strong proxy for insurance type and income |

**UI Recommendation for POVCAT23:** Users should not be asked to self-identify with poverty labels (e.g., "Poor", "Near Poor"), which causes stigma and inaccuracy. Instead, use a two-step approach: (1) ask **Family Size** first, then (2) display income ranges that dynamically map to POVCAT categories based on 2023 FPL thresholds. Note: For accuracy, "Family" should follow the CPS definition (people related by blood, marriage, or adoption living together).

### 3. Insurance & Access
Variables defining cost-sharing structure and healthcare access patterns.

| Variable | Label | Description | Survey Question | Response Categories | Rationale |
|:---|:---|:---|:---|:---|:---|
| `INSCOV23` | Insurance Coverage | Coverage status | "Are you covered by any type of health insurance?" | Private, Public Only, Uninsured | Critical as it determines OOP vs. total cost split. [[1]](#ref1) |
| `HAVEUS42` | Usual Source of Care | Regular doctor/clinic | "Is there a particular place you usually go if you are sick or need advice about your health?" | Yes, No | Strong predictor of access and preventive care |

**UI Recommendation for HAVEUS42:**
The MEPS variable `HAVEUS42` is derived from a two-step process in the original survey:
1. **AC01:** "Is there a particular doctor's office, clinic, health center, or other place that you usually go if you are sick or need advice about health?" (Options: Yes, No, More than one place)
2. **AC02:** (If "More than one place") "Would you go to one of these places first or most often if you are sick?"

To optimize for UX and minimize cognitive load, we condense this into a single binary question:
> **"Is there a particular doctor's office, clinic, health center, or other place you usually go if you are sick or need medical advice?"**

**Rationale for Single UI Input:**
* **Mapping Accuracy:** The `HAVEUS42` variable in the training data already collapses "More than one place" into a binary "Yes" (if a primary place exists) or "No." By asking about a "regular" place, we naturally guide users to this same logic without requiring a multi-step skip pattern.
* **Inclusion of "Medical Advice":** Including "need medical advice" is critical. It ensures users who visit for routine care or wellness (not just acute illness) correctly identify as having a usual source of care, matching the MEPS definition and improving feature accuracy.
* **Reduced Friction:** A single radio-button interaction aligns with the "under 90 seconds" form completion goal.


### 4. Perceived Health & Lifestyle
Subjective indicators of overall health burden and behavioral risk factors that drive healthcare utilization.

| Variable | Label | Description | Survey Question | Response Categories | Rationale |
|:---|:---|:---|:---|:---|:---|
| `RTHLTH31` | Physical Health | Self-rated physical health (1–5) | "In general, would you say your health is...?" | Excellent, Very good, Good, Fair, Poor | Strongest subjective predictor of utilization. [[5]](#ref5) |
| `MNHLTH31` | Mental Health | Self-rated mental health | "In general, would you say your mental health is...?" | Excellent, Very good, Good, Fair, Poor | Significant cost multiplier via treatment adherence. [[5]](#ref5) |
| `ADSMOK42` | Smoker | Currently smokes | "Do you currently smoke cigarettes?" | Yes, No | Stable behavioral risk factor. [[2]](#ref2) |


### 5. Limitations & Symptoms
Screener questions identifying individuals requiring more frequent care due to physical or cognitive impairments.

| Variable | Label | Description | Survey Question | Response Categories | Rationale |
|:---|:---|:---|:---|:---|:---|
| `ADLHLP31` | ADL Help | Needs help with personal care | "Do you receive help or supervision with personal care such as bathing, dressing, or getting around the house?" | Yes, No | High-cost functional indicator. [[7]](#ref7) |
| `IADLHP31` | IADL Help | Needs help with IADLs | "Do you receive help or supervision with using the telephone, paying bills, taking medications, preparing light meals, doing laundry, or going shopping?" | Yes, No | Signals high-cost care requirements. [[7]](#ref7) |
| `WLKLIM31` | Walking Limitation | Difficulty walking or climbing stairs | "Do you have serious difficulty walking or climbing stairs?" | Yes, No | Captures mobility impairment |
| `COGLIM31` | Cognitive Limitation | Confusion or memory loss | "Do you have serious difficulty concentrating, remembering, or making decisions?" | Yes, No | Correlates with specialized care needs. [[3]](#ref3) |
| `JTPAIN31_M18` | Joint Pain | Pain/stiffness in past year | "During the past 12 months, did you have pain, aching, stiffness, or swelling in or around a joint?" | Yes, No | Captures undiagnosed musculoskeletal issues |

**UI Recommendation:** Present as a single checklist with the prompt "Do any of the following apply to you?" and options as first-person statements (e.g., "I receive help or supervision...").

### 6. Chronic Conditions
The "cost engine" driving sustained medical expenditures. 

| Variable | Label | Description | Survey Question | Response Categories | Rationale |
|:---|:---|:---|:---|:---|:---|
| `HIBPDX` | Hypertension | Diagnosed with high BP | "Have you ever been told by a doctor or other health professional that you have hypertension, also called high blood pressure?" | Yes, No | Common; drives Rx costs. [[6]](#ref6) |
| `CHOLDX` | High Cholesterol | Diagnosed with high chol | "Have you ever been told by a doctor or other health professional that you have high cholesterol?" | Yes, No | Common; drives Rx costs. [[6]](#ref6) |
| `DIABDX_M18` | Diabetes | Diagnosed with diabetes | "Have you ever been told by a doctor or other health professional that you have diabetes or sugar diabetes?" | Yes, No | High-cost condition. [[1]](#ref1) |
| `CHDDX` | Heart Disease | Diagnosed with heart disease | "Have you ever been told by a doctor or other health professional that you have coronary heart disease?" | Yes, No | Major cost driver. [[6]](#ref6) |
| `STRKDX` | Stroke | Diagnosed with stroke | "Have you ever been told by a doctor or other health professional that you had a stroke?" | Yes, No | High downstream costs. [[7]](#ref7) |
| `CANCERDX` | Cancer | Diagnosed with cancer | "Have you ever been told by a doctor or other health professional that you had cancer or a malignancy of any kind?" | Yes, No | Primary driver of tail costs. [[7]](#ref7) |
| `ARTHDX` | Arthritis | Diagnosed with arthritis | "Have you ever been told by a doctor or other health professional that you have arthritis?" | Yes, No | Drives Rx/therapy costs. [[7]](#ref7) |
| `ASTHDX` | Asthma | Diagnosed with asthma | "Have you ever been told by a doctor or other health professional that you have asthma?" | Yes, No | Chronic condition with ongoing costs. [[1]](#ref1) |
| `DEPRDX` | Depression | Diagnosed with depression | "Have you ever been told by a doctor or other health professional that you have depression?" | Yes, No | Significant cost multiplier. [[7]](#ref7) |

**UI Recommendation:** Present chronic conditions as a multi-select checklist with the prompt "Have you ever been told by a doctor or other health professional that you:" and with wording as close to original survey question as possible.


## Excluded Variables

| Variable | Reason for Exclusion |
|:---|:---|
| `ADAPPT42` | Doctor visit count accumulated during year; not known at prediction time. |
| `ERTOT23` | ER visit count for full year; creates data leakage. |
| `DDNWRK23` | Work days missed during year; outcome-adjacent variable. |
| `RACETHX` | Ethically sensitive; likely redundant with income/region proxies. |
| `RUSIZE23` | Household size; potentially redundant with FAMSZE23. |
| `HOUR31`, `HOUR53` | Hours worked per week; low marginal value over employment status; adds cognitive load. |
| `BMINDX53` | BMI; requires 2 inputs (height/weight); end-of-year suffix; effects captured by chronic conditions. |


## Summary

| Category | # Features | # UI Interactions |
|:---|:---|:---|
| Demographics | 4 | 4 |
| Socioeconomic | 2–3 | 2–3 |
| Insurance & Access | 2 | 2 |
| Perceived Health & Lifestyle | 3 | 3 |
| Limitations & Symptoms | 5 | 1 (checklist) |
| Chronic Conditions | 9 | 1 (checklist) |
| **Total** | **~25–26 features** | **~13–14 interactions** |

Final feature selection will be based on empirical feature importance ranking, targeting form completion in **under 90 seconds**.


## Works Cited

1. <a id="ref1"></a>Who determines United States Healthcare out-of-pocket costs..., accessed December 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8184979/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8184979/)
2. <a id="ref2"></a>Predictive Modelling of Healthcare Insurance Costs Using Machine..., accessed December 19, 2025, [https://www.preprints.org/manuscript/202502.1873](https://www.preprints.org/manuscript/202502.1873)
3. <a id="ref3"></a>AIX360/examples/tutorials/MEPS.ipynb at master \- GitHub, accessed December 19, 2025, [https://github.com/Trusted-AI/AIX360/blob/master/examples/tutorials/MEPS.ipynb](https://github.com/Trusted-AI/AIX360/blob/master/examples/tutorials/MEPS.ipynb)
4. <a id="ref4"></a>MEPS HC 251 2023 Full Year Consolidated Data File August 2025, accessed December 19, 2025, [https://meps.ahrq.gov/data\_stats/download\_data/pufs/h251/h251doc.pdf](https://meps.ahrq.gov/data\_stats/download\_data/pufs/h251/h251doc.pdf)
5. <a id="ref5"></a>Supervised Learning Methods for Predicting Healthcare Costs \- NIH, accessed December 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5977561/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5977561/)
6. <a id="ref6"></a>MEPS HC 251 2023 Full Year Consolidated Data File, accessed December 19, 2025, [https://meps.ahrq.gov/data\_stats/download_data/pufs/h251/h251doc.shtml](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.shtml)
7. <a id="ref7"></a>Dataset: Medical Expenditure Panel Survey (MEPS), accessed December 19, 2025, [https://www.disabilitystatistics.org/dataset-directory/dataset/70](https://www.disabilitystatistics.org/dataset-directory/dataset/70)