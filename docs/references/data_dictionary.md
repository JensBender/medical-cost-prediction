# MEPS-HC 2023 Data Dictionary
**Survey:** Medical Expenditure Panel Survey (MEPS)  
**Component**: Household Component (HC)  
**Year:** 2023  
**Dataset:** Full Year Consolidated Data File (HC-251)  
**Level:** Person-Level  

## 1. Identifiers (Keys)
| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **DUPERSID** | PERSON ID (DUID + PID) | Char(8) | **Primary Key.** Unique identifier for each person. |
| **DUID** | DWELLING UNIT ID | Num | Identifies the household. |
| **PID** | PERSON ID | Num | Identifies person within the household. |
| **PANEL** | PANEL NUMBER | Num | Panel 27 or 28. |

## 2. Survey Design & Weights
*CRITICAL: Use these for population estimates.*

| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **PERWT23F** | FINAL PERSON WEIGHT, 2023 | Num | Weight to represent the US population. |
| **VARSTR** | VARIANCE ESTIMATION STRATUM | Num | Use for Taylor Series variance estimation. |
| **VARPSU** | VARIANCE ESTIMATION PSU | Num | Use for Taylor Series variance estimation. |

## 3. Demographics
| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **AGE23X** | AGE - 12/31/23 (EDITED/IMPUTED) | Num | Age as of end of year. |
| **SEX** | SEX | Enum | 1=Male, 2=Female. |
| **REGION23** | CENSUS REGION AS OF 12/31/23 | Enum | 1=Northeast, 2=Midwest, 3=South, 4=West. |
| **MARRY31X** | MARITAL STATUS - R3/1 | Enum | Status at beginning of year. 1=Married, 2=Widowed, 3=Divorced, 4=Separated, 5=Never Married, 6=Under 16. |

## 4. Socioeconomic Status
| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **POVCAT23** | FAMILY INC AS % OF POVERTY LINE | Enum | Derived variable. 1=Poor/Negative, 2=Near Poor, 3=Low Income, 4=Middle Income, 5=High Income. |
| **FAMSZE23** | TOTAL NUMBER OF PERSONS IN FAMILY | Num | Count of persons in the family (Range: 1-14). |
| **HIDEG** | HIGHEST DEGREE WHEN FIRST ENTERED | Enum | 1=No Degree, 2=GED, 3=High School Diploma, 4=Bachelor's Degree, 5=Master's Degree, 6=Doctorate Degree, 7=Other Degree, 8=Under 16. |
| **EMPST31** | EMPLOYMENT STATUS - R3/1 | Enum | Status at beginning of year. 1=Employed, 2=Job to return to, 3=Job during ref period, 4=Not employed. |

## 5. Health Status & Conditions
*Note: Conditions ending in `31` reflect beginning-of-year status. Chronic conditions (DX) are "ever diagnosed".*

| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **RTHLTH31** | PERCEIVED HEALTH STATUS - RD 3/1 | Enum | 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor. |
| **MNHLTH31** | PERCEIVED MENTAL HEALTH - RD 3/1 | Enum | 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor. |
| **ADSMOK42** | CURRENTLY SMOKE | Enum | 1=Yes, 2=No. (Round 4/2 is standard for health behaviors). |
| **HIBPDX** | HIGH BLOOD PRESSURE DIAG (>17) | Enum | 1=Yes, 2=No. |
| **CHOLDX** | HIGH CHOLESTEROL DIAGNOSIS (>17) | Enum | 1=Yes, 2=No. |
| **DIABDX_M18**| DIABETES DIAGNOSIS (>17) | Enum | 1=Yes, 2=No. |
| **CHDDX** | CORONARY HEART DISEASE DIAG (>17) | Enum | 1=Yes, 2=No. |
| **STRKDX** | STROKE DIAGNOSIS (>17) | Enum | 1=Yes, 2=No. |
| **CANCERDX** | CANCER DIAGNOSIS (>17) | Enum | 1=Yes, 2=No. |
| **ARTHDX** | ARTHRITIS DIAGNOSIS (>17) | Enum | 1=Yes, 2=No. |
| **ASTHDX** | ASTHMA DIAGNOSIS | Enum | 1=Yes, 2=No. |
| **DEPRDX** | DEPRESSION DIAGNOSIS | Enum | 1=Yes, 2=No. |

## 6. Functional Limitations
| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **ADLHLP31** | ADL HELP - RD 3/1 | Enum | 1=Yes, 2=No. Help with personal care. |
| **IADLHP31** | IADL HELP - RD 3/1 | Enum | 1=Yes, 2=No. Help with routine needs. |
| **WLKLIM31** | LIMITATION IN PHYSICAL FUNCTIONING | Enum | 1=Yes, 2=No. Difficulty walking/climbing. |
| **COGLIM31** | COGNITIVE LIMITATIONS | Enum | 1=Yes, 2=No. Confusion/memory loss. |
| **JTPAIN31_M18**| JOINT PAIN LAST 12 MONTHS | Enum | 1=Yes, 2=No. |

## 7. Insurance & Access
| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **INSCOV23** | HEALTH INSURANCE COVERAGE INDICATOR | Enum | 1=Any Private, 2=Public Only, 3=Uninsured. |
| **HAVEUS42** | AC01 HAS USUAL 3RD PARTY SRC PROVIDER | Enum | 1=Yes, 2=No. (Has regular doctor/clinic). |

## 8. Healthcare Utilization & Expenditure (Targets)
*Suffix is '23' for the year 2023.*

| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **TOTSLF23** | TOTAL AMT PAID BY SELF/FAMILY 23 | Num | **Target Variable.** Out-of-pocket costs. |
| **TOTEXP23** | TOTAL HEALTH CARE EXP 23 | Num | Total cost (all payers). |