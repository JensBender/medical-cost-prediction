# MEPS-HC 2023 Data Dictionary

**Survey:** Medical Expenditure Panel Survey (MEPS)  
**Component**: Household Component (HC)  
**Year:** 2023  
**Dataset:** Full Year Consolidated Data File (HC-251)  
**Level:** Person-Level  

## Missing Value Codes
Most MEPS variables use the following codes for missing or non-applicable data:
*   **-1 INAPPLICABLE**: Variable does not apply to this person.
*   **-7 REFUSED**: Person refused to answer the question.
*   **-8 DON'T KNOW**: Person did not know the answer.
*   **-9 NOT ASCERTAINED**: Data not collected (e.g., due to skip patterns or interview termination).
*   **-15 CANNOT BE COMPUTED**: Used for some complex derived variables.


## Target Population
MEPS data represents the **U.S. civilian noninstitutionalized population**. Understanding this scope is critical for interpreting weights and results.

*   **Included (In-Scope):** People living in households (houses, apartments, etc.) and non-institutional group quarters (e.g., college dormitories).
*   **Excluded (Out-of-Scope):**
    *   **Institutionalized Individuals:** People in nursing homes, prisons, jails, or long-term psychiatric hospitals.
    *   **Active-Duty Military:** Members of the U.S. Armed Forces (they typically use the military's own healthcare system).
    *   **Non-U.S. Residents:** Individuals who moved abroad during the survey year (they are considered "out-of-scope" for the period they reside abroad).


## 1. Identifiers
| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **DUPERSID** | PERSON ID (DUID + PID) | Char(8) | Unique ID | **Primary Key.** Unique identifier for each person. |
| **DUID** | DWELLING UNIT ID | Num | 30001–68884 | Identifies the dwelling unit (household). |
| **PID** | PERSON NUMBER | Num | 101–503 | Identifies person within the dwelling unit. |
| **PANEL** | PANEL NUMBER | Num | 27, 28 | Panel number associated with the round. |

## 2. Survey Weights
*CRITICAL: Use these for population estimates.*

| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **PERWT23F** | FINAL PERSON WEIGHT, 2023 | Num | 0.0 – 98,252.0 | Weight to represent the US population. |
| **VARSTR** | VARIANCE ESTIMATION STRATUM | Num | 2001–2117 | Use for Taylor Series variance estimation. |
| **VARPSU** | VARIANCE ESTIMATION PSU | Num | 1–3 | Use for Taylor Series variance estimation. |

## 3. Demographics
| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **AGE23X** | AGE - 12/31/23 (EDITED/IMPUTED) | Num | 0–85 | Age as of end of year. Top-coded at 85 for privacy (85+ is coded as 85). |
| **SEX** | SEX | Enum | 1=Male, 2=Female | Biological sex. No missing codes. |
| **REGION23** | CENSUS REGION AS OF 12/31/23 | Enum | 1=Northeast, 2=Midwest, 3=South, 4=West | Census region based on address. |
| **MARRY31X** | MARITAL STATUS - R3/1 (EDITED/IMPUTED) | Enum | 1=Married, 2=Widowed, 3=Divorced, 4=Separated, 5=Never Married, 6=Under 16, 7-10=Status changed in round | Status at beginning of year. |

## 4. Socioeconomic
| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **POVCAT23** | FAMILY INCOME AS % OF POVERTY LINE | Enum | 1=Poor/Negative, 2=Near Poor, 3=Low Income, 4=Middle Income, 5=High Income | Derived variable. Usually no missing codes. |
| **FAMSZE23** | TOTAL NUMBER OF PERSONS IN FAMILY | Num | 1–14 | Count of related persons in the reporting unit. |
| **HIDEG** | HIGHEST DEGREE WHEN FIRST ENTERED | Enum | 1=No Degree, 2=GED, 3=HS Diploma, 4=Bachelor's, 5=Master's, 6=Doctorate, 7=Other, 8=Under 16 | Highest degree attained. Codes -1, -7, -8, -9 apply. 8 is Under 16. |
| **EMPST31** | EMPLOYMENT STATUS - R3/1 | Enum | 1=Employed, 2=Job to return to, 3=Job during ref period, 4=Not employed | Status at beginning of year. Codes -1, -7, -8, -9 apply. |

## 5. Insurance & Access
| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **INSCOV23** | HEALTH INSURANCE COVERAGE INDICATOR | Enum | 1=Any Private, 2=Public Only, 3=Uninsured | Summary coverage status for 2023. No missing codes. |
| **HAVEUS42** | AC05 DOES PERSON HAVE USC PROVIDER | Enum | 1=Yes, 2=No | Has a usual source of care. Codes -1, -7, -8, -9 apply. |

## 6. Perceived Health & Lifestyle
| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **RTHLTH31** | PERCEIVED HEALTH STATUS - RD 3/1 | Enum | 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor | Codes -1, -7, -8, -9 apply. |
| **MNHLTH31** | PERCEIVED MENTAL HEALTH - RD 3/1 | Enum | 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor | Codes -1, -7, -8, -9 apply. |
| **ADSMOK42** | CURRENTLY SMOKE (>17) | Enum | 1=Yes, 2=No | Asked of adults 18+. Codes -1, -7, -8, -9 apply. |

## 7. Limitations & Symptoms
| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **ADLHLP31** | ADL HELP - RD 3/1 | Enum | 1=Yes, 2=No | Needs help with personal care. Codes -1, -7, -8, -9 apply. |
| **IADLHP31** | IADL HELP - RD 3/1 | Enum | 1=Yes, 2=No | Needs help with routine needs. Codes -1, -7, -8, -9 apply. |
| **WLKLIM31** | LIMITATION IN PHYSICAL FUNCTIONING | Enum | 1=Yes, 2=No | Difficulty walking/climbing stairs. Codes -1, -7, -8, -9 apply. |
| **COGLIM31** | COGNITIVE LIMITATIONS | Enum | 1=Yes, 2=No | Confusion/memory loss. Codes -1, -7, -8, -9 apply. |
| **JTPAIN31_M18**| JOINT PAIN LAST 12 MONTHS (>17) | Enum | 1=Yes, 2=No | Pain/stiffness in joints. Codes -1, -7, -8, -9 apply. |

## 8. Chronic Conditions
*Note: Chronic conditions (DX) are "ever diagnosed". The (>17) tag indicates variables only collected for adults due to MEPS skip patterns; children (under 18) are coded as -1 (Inapplicable) for these.*

| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **HIBPDX** | HIGH BLOOD PRESSURE DIAG (>17) | Enum | 1=Yes, 2=No | Codes -1, -7, -8, -9 apply. |
| **CHOLDX** | HIGH CHOLESTEROL DIAGNOSIS (>17) | Enum | 1=Yes, 2=No | Codes -1, -7, -8, -9 apply. |
| **DIABDX_M18**| DIABETES DIAGNOSIS | Enum | 1=Yes, 2=No | Asked for all ages. Codes -1, -7, -8, -9 apply. |
| **CHDDX** | CORONARY HEART DISEASE DIAG (>17) | Enum | 1=Yes, 2=No | Codes -1, -7, -8, -9 apply. |
| **STRKDX** | STROKE DIAGNOSIS (>17) | Enum | 1=Yes, 2=No | Codes -1, -7, -8, -9 apply. |
| **CANCERDX** | CANCER DIAGNOSIS (>17) | Enum | 1=Yes, 2=No | Codes -1, -7, -8, -9 apply. |
| **ARTHDX** | ARTHRITIS DIAGNOSIS (>17) | Enum | 1=Yes, 2=No | Codes -1, -7, -8, -9 apply. |
| **ASTHDX** | ASTHMA DIAGNOSIS | Enum | 1=Yes, 2=No | Asked for all ages. Codes -1, -7, -8, -9 apply. |
| **DEPRDX** | DEPRESSION DIAGNOSIS | Enum | 1=Yes, 2=No | Asked for all ages. Codes -1, -7, -8, -9 apply. |

## 9. Healthcare Expenditure (Targets)
*Suffix is '23' for the year 2023.*

| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **TOTSLF23** | TOTAL AMT PAID BY SELF/FAMILY 23 | Num | $0.00 – $176,550.00 | **Target Variable.** Out-of-pocket costs paid by patient/family. |
| **TOTEXP23** | TOTAL HEALTH CARE EXP 23 | Num | $0.00 – $2,301,675.00 | Total cost across all payers (insurance, public, self). |
