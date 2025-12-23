# Candidate Features for Medical Cost Prediction

Candidate feature list for predicting annual out-of-pocket healthcare costs (`TOTSLF23`) using MEPS-HC 2023 data. Features are selected based on three criteria:

1. **Consumer Accessibility:** Users can answer from memory without looking up documents or records.
2. **Temporal Validity:** Variables were measured at the beginning of the year to enable cost prediction for the upcoming year without data leakage.
3. **Predictive Power:** Features have established significance in the healthcare cost prediction literature.

**Sources:** Gemini Deep Research (Dec 2025), MEPS H251 Codebook, and domain literature review.


## Temporal Alignment Rationale

MEPS variable suffixes indicate when data was collected during the survey year:

| Suffix | Timing | Example |
|:---|:---|:---|
| `31` | Beginning of year (Rounds 3/1) | `RTHLTH31` |
| `42` | Mid-year (Rounds 4/2) | `ADSMOK42` |
| `53` | End of year (Rounds 5/3) | `RTHLTH53` |
| `23` or `23X` | Full-year summary or year-end point | `AGE23X`, `INSCOV23` |

The Medical Cost Planner app is designed for use during Open Enrollment (Nov–Dec) to forecast healthcare costs for the **upcoming calendar year**. To mirror this prospective use case and prevent data leakage (using year-end health status to predict costs that have already occurred), we train specifically on **beginning-of-year (`31`) variables** for all time-varying metrics like self-rated health and functional limitations. Diagnostic flags for chronic conditions are treated as temporally stable and do not require such adjustment.

## Feature Categories

### 1. Demographics
Primary drivers of healthcare utilization based on biological and geographic factors.

| Variable | Label | Type | Description | Survey Question | Rationale |
|:---|:---|:---|:---|:---|:---|
| `AGE23X` | Age | Numerical | Age as of Dec 31, 2023 (18–85). | "What is your date of birth?" | Primary driver of utilization; costs follow a U-curve with most spending at high age. [[1]](#ref1) |
| `SEX` | Sex | Nominal | Biological sex (Male/Female). | "Are you male or female?" | Influences utilization via gender-specific conditions. [[2]](#ref2) |
| `REGION23` | Region | Nominal | Census region. | Derived from primary residence address | Captures geographic pricing variations. [[2]](#ref2) |
| `MARRY31X` | Marital Status | Nominal | Marital status at beginning of year. | "Are you now married, widowed, divorced, separated, or never married?" | Proxy for social support and income stability. [[3]](#ref3) |

### 2. Socioeconomic
Proxies for healthcare access, literacy, insurance quality, and ability to pay.

| Variable | Label | Type | Description | Survey Question | Rationale |
|:---|:---|:---|:---|:---|:---|
| `POVCAT23` | Income Category | Ordinal | Family income as % of poverty line. | Derived from total family income and household size | Determines subsidy eligibility and insurance quality. [[4]](#ref4) |
| `HIDEG` | Education | Ordinal | Highest degree attained. | "What is the highest degree you have received?" | Correlates with health literacy and preventive care use. [[4]](#ref4) |
| `EMPST31` | Employment Status | Nominal | Employment status at beginning of year. | "Are you currently working at a job or business?" | Strong proxy for insurance type and income. |

### 3. Insurance & Access
Variables defining cost-sharing structure and healthcare access patterns.

| Variable | Label | Type | Description | Survey Question | Rationale |
|:---|:---|:---|:---|:---|:---|
| `INSCOV23` | Insurance Coverage | Nominal | Coverage status (Private, Public, Uninsured). | "Are you covered by any type of health insurance?" | **Critical.** Directly determines OOP vs. total cost split. [[1]](#ref1) |
| `HAVEUS42` | Usual Source of Care | Binary | Whether person has a regular doctor/clinic. | "Is there a particular place you usually go if you are sick?" | Strong predictor of access and preventive care. |

### 4. Perceived Health & Lifestyle
Subjective indicators of overall health burden and behavioral risk factors that drive healthcare utilization.

| Variable | Label | Type | Description | Survey Question | Rationale |
|:---|:---|:---|:---|:---|:---|
| `RTHLTH31` | Physical Health | Numerical | Self-rated physical health (1–5). | "In general, would you say your health is excellent, very good, good, fair, or poor?" | Strongest subjective predictor of utilization. [[5]](#ref5) |
| `MNHLTH31` | Mental Health | Numerical | Self-rated mental health (1–5). | "In general, would you say your mental health is excellent, very good, good, fair, or poor?" | Significant cost multiplier via treatment adherence. [[5]](#ref5) |
| `ADSMOK42` | Current Smoker | Binary | Currently smokes cigarettes. | "Do you currently smoke cigarettes?" | Stable behavioral risk factor. [[2]](#ref2) |


### 5. Limitations & Symptoms
Screener questions identifying individuals requiring more frequent care due to physical or cognitive impairments.

| Variable | Label | Type | Description | Survey Question | Rationale |
|:---|:---|:---|:---|:---|:---|
| `ADLHLP31` | ADL Help Needed | Binary | Help with personal care. | "Do you receive help with personal care such as bathing, dressing, or getting around?" | High-cost functional indicator. [[7]](#ref7) |
| `IADLHP31` | IADL Help Needed | Binary | Help with instrumental ADLs. | "Do you receive help using the phone, paying bills, or shopping?" | Signals high-cost care requirements. [[7]](#ref7) |
| `WLKLIM31` | Walking Limitation | Binary | Physical limitation. | "Do you have any difficulty walking, climbing stairs, or lifting?" | Captures mobility impairment. |
| `COGLIM31` | Cognitive Limitation | Binary | Confusion or memory loss. | "Do you experience confusion, memory loss, or problems making decisions?" | Correlates with specialized care needs. [[3]](#ref3) |
| `JTPAIN31_M18` | Joint Pain | Binary | Joint pain/stiffness in past year. | "In the past 12 months, have you had pain, aching, or stiffness in or around a joint?" | Captures undiagnosed musculoskeletal issues. |

**UI Recommendation:** Present as a single checklist ("Do you have difficulty with any of the following?").

### 6. Chronic Conditions
The "cost engine" driving sustained medical expenditures. 

| Variable | Label | Type | Description | Survey Question | Rationale |
|:---|:---|:---|:---|:---|:---|
| `HIBPDX` | Hypertension | Binary | Ever diagnosed with high BP. | "Have you ever been told by a doctor that you have high blood pressure?" | Common; drives Rx costs. [[6]](#ref6) |
| `CHOLDX` | High Cholesterol | Binary | Ever diagnosed with high chol. | "Have you ever been told by a doctor that you have high cholesterol?" | Common; drives Rx costs. [[6]](#ref6) |
| `DIABDX_M18` | Diabetes | Binary | Ever diagnosed with diabetes. | "Have you ever been told by a doctor that you have diabetes?" | High-cost condition. [[1]](#ref1) |
| `CHDDX` | Heart Disease | Binary | Ever diagnosed with heart dis. | "Have you ever been told by a doctor that you have coronary heart disease?" | Major cost driver. [[6]](#ref6) |
| `STRKDX` | Stroke | Binary | Ever diagnosed with stroke. | "Have you ever been told by a doctor that you had a stroke?" | High downstream costs. [[7]](#ref7) |
| `CANCERDX` | Cancer | Binary | Ever diagnosed with cancer. | "Have you ever been told by a doctor that you had cancer?" | Primary driver of tail costs. [[7]](#ref7) |
| `ARTHDX` | Arthritis | Binary | Ever diagnosed with arthritis. | "Have you ever been told by a doctor that you have arthritis?" | Drives Rx/therapy costs. [[7]](#ref7) |
| `ASTHDX` | Asthma | Binary | Ever diagnosed with asthma. | "Have you ever been told by a doctor that you have asthma?" | Chronic condition with ongoing costs. [[1]](#ref1) |
| `DEPRDX` | Depression | Binary | Ever diagnosed with depression. | "Have you ever been told by a doctor that you have depression?" | Significant cost multiplier. [[7]](#ref7) |

**UI Recommendation:** Present chronic conditions as a multi-select checklist ("Have you ever been told by a doctor that you have any of the following?").


## Excluded Variables

| Variable | Reason for Exclusion |
|:---|:---|
| `ADAPPT42` | Doctor visit count accumulated during year; not known at prediction time. |
| `ERTOT23` | ER visit count for full year; creates data leakage. |
| `DDNWRK23` | Work days missed during year; outcome-adjacent variable. |
| `RACETHX` | Ethically sensitive; likely redundant with income/region proxies. |
| `RUSIZE23`, `FAMSZE23` | Household size; we predict individual costs, not family costs. |
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