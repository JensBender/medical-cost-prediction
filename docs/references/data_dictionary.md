# MEPS-HC 2023 Data Dictionary
**Survey:** Medical Expenditure Panel Survey (MEPS)  
**Component**: Household Component (HC)  
**Year:** 2023  
**Dataset:** Full Year Consolidated Data File (HC-251)  
**Level:** Person-Level  

## 1. Identifiers (Keys)
| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **DUPERSID** | PERSON ID (DUID + PID) | Char(8) | Unique ID | **Primary Key.** Unique identifier for each person. |
| **DUID** | DWELLING UNIT ID | Num | 30001–68884 | Identifies the household. |
| **PID** | PERSON ID | Num | 101–503 | Identifies person within the household. |
| **PANEL** | PANEL NUMBER | Num | 27, 28 | Panel number associated with the round. |

## 2. Survey Design & Weights
*CRITICAL: Use these for population estimates.*

| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **PERWT23F** | FINAL PERSON WEIGHT, 2023 | Num | 0.0 – 98,252.0 | Weight to represent the US population. |
| **VARSTR** | VARIANCE ESTIMATION STRATUM | Num | 2001–2117 | Use for Taylor Series variance estimation. |
| **VARPSU** | VARIANCE ESTIMATION PSU | Num | 1–3 | Use for Taylor Series variance estimation. |

## 3. Demographics & Socioeconomic
| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **AGE23X** | AGE - 12/31/23 (EDITED/IMPUTED) | Num | 0–85 | Age as of end of year. Top-coded at 85. |
| **SEX** | SEX | Enum | 1=Male, 2=Female | Biological sex. |
| **REGION23** | CENSUS REGION AS OF 12/31/23 | Enum | 1=Northeast, 2=Midwest, 3=South, 4=West | Census region based on address. |
| **MARRY31X** | MARITAL STATUS - R3/1 | Enum | 1=Married, 2=Widowed, 3=Divorced, 4=Separated, 5=Never Married, 6=Under 16 | Status at beginning of year. |
| **POVCAT23** | FAMILY INC AS % OF POVERTY LINE | Enum | 1=Poor/Negative, 2=Near Poor, 3=Low Income, 4=Middle Income, 5=High Income | Derived variable based on family income and size. |
| **FAMSZE23** | TOTAL NUMBER OF PERSONS IN FAMILY | Num | 1–14 | Count of related persons in the reporting unit. |
| **HIDEG** | HIGHEST DEGREE WHEN FIRST ENTERED | Enum | 1=No Degree, 2=GED, 3=High School Diploma, 4=Bachelor's Degree, 5=Master's Degree, 6=Doctorate Degree, 7=Other Degree, 8=Under 16 | Highest degree attained at time of entry. |
| **EMPST31** | EMPLOYMENT STATUS - R3/1 | Enum | 1=Employed, 2=Job to return to, 3=Job during ref period, 4=Not employed | Status at beginning of year. |

## 4. Insurance & Access
| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **INSCOV23** | HEALTH INSURANCE COVERAGE INDICATOR | Enum | 1=Any Private, 2=Public Only, 3=Uninsured | Summary coverage status for 2023. |
| **HAVEUS42** | AC01 HAS USUAL 3RD PARTY SRC PROVIDER | Enum | 1=Yes, 2=No | Has a particular doctor or clinic they usually go to. |

## 5. Perceived Health & Lifestyle
| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **RTHLTH31** | PERCEIVED HEALTH STATUS - RD 3/1 | Enum | 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor | Self-rated physical health at start of year. |
| **MNHLTH31** | PERCEIVED MENTAL HEALTH - RD 3/1 | Enum | 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor | Self-rated mental health at start of year. |
| **ADSMOK42** | CURRENTLY SMOKE | Enum | 1=Yes, 2=No | Round 4/2 is the standard asking round for health behaviors. |

## 6. Limitations & Symptoms
| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **ADLHLP31** | ADL HELP - RD 3/1 | Enum | 1=Yes, 2=No | Needs help with personal care (bathing, dressing, etc.). |
| **IADLHP31** | IADL HELP - RD 3/1 | Enum | 1=Yes, 2=No | Needs help with routine needs (bills, shopping, etc.). |
| **WLKLIM31** | LIMITATION IN PHYSICAL FUNCTIONING | Enum | 1=Yes, 2=No | Difficulty walking or climbing stairs. |
| **COGLIM31** | COGNITIVE LIMITATIONS | Enum | 1=Yes, 2=No | Confusion or memory loss. |
| **JTPAIN31_M18**| JOINT PAIN LAST 12 MONTHS | Enum | 1=Yes, 2=No | Pain, swelling, or stiffness in joints. |

## 7. Chronic Conditions
*Note: Chronic conditions (DX) are "ever diagnosed".*

| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **HIBPDX** | HIGH BLOOD PRESSURE DIAG (>17) | Enum | 1=Yes, 2=No | Diagnosed hypertension. |
| **CHOLDX** | HIGH CHOLESTEROL DIAGNOSIS (>17) | Enum | 1=Yes, 2=No | Diagnosed high cholesterol. |
| **DIABDX_M18**| DIABETES DIAGNOSIS (>17) | Enum | 1=Yes, 2=No | Diagnosed diabetes. |
| **CHDDX** | CORONARY HEART DISEASE DIAG (>17) | Enum | 1=Yes, 2=No | Diagnosed coronary heart disease. |
| **STRKDX** | STROKE DIAGNOSIS (>17) | Enum | 1=Yes, 2=No | Diagnosed stroke. |
| **CANCERDX** | CANCER DIAGNOSIS (>17) | Enum | 1=Yes, 2=No | Diagnosed cancer/malignancy. |
| **ARTHDX** | ARTHRITIS DIAGNOSIS (>17) | Enum | 1=Yes, 2=No | Diagnosed arthritis. |
| **ASTHDX** | ASTHMA DIAGNOSIS | Enum | 1=Yes, 2=No | Diagnosed asthma. |
| **DEPRDX** | DEPRESSION DIAGNOSIS | Enum | 1=Yes, 2=No | Diagnosed depression. |

## 8. Healthcare Utilization & Expenditure (Targets)
*Suffix is '23' for the year 2023.*

| Variable | Label | Type | Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **TOTSLF23** | TOTAL AMT PAID BY SELF/FAMILY 23 | Num | $0.00 – $176,550.00 | **Target Variable.** Out-of-pocket costs paid by patient/family. |
| **TOTEXP23** | TOTAL HEALTH CARE EXP 23 | Num | $0.00 – $2,301,675.00 | Total cost across all payers (insurance, public, self). |