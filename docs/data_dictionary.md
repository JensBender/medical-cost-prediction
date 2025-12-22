# MEPS-HC 2023 Data Dictionary
**Survey:** Medical Expenditure Panel Survey (MEPS)  
**Component**: Household Component (HC)  
**Year:** 2023  
**Dataset:** Full Year Consolidated Data File (HC-251)  
**Level:** Person-Level  

## 1. Identifiers (Keys)
| Variable | Label | Format | Notes |
| :--- | :--- | :--- | :--- |
| **DUPERSID** | PERSON ID (DUID + PID) | Char(8) | **Primary Key.** Unique identifier for each person. |
| **DUID** | DWELLING UNIT ID | Num | Identifies the household. |
| **PID** | PERSON NUMBER | Num | Identifies person within the household. |
| **PANEL** | PANEL NUMBER | Num | Panel 27 or 28. |

## 2. Survey Design & Weights
*CRITICAL: Use these for population estimates.*

| Variable | Label | Type | Notes |
| :--- | :--- | :--- | :--- |
| **PERWT23F** | FINAL PERSON WEIGHT, 2023 | Num | Weight to represent the US population. |
| **VARSTR** | VARIANCE ESTIMATION STRATUM | Num | Use for Taylor Series variance estimation. |
| **VARPSU** | VARIANCE ESTIMATION PSU | Num | Use for Taylor Series variance estimation. |

## 3. Demographics
| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **AGE23X** | AGE - 12/31/23 (EDITED/IMPUTED) | Num | Age as of end of year. |
| **SEX** | SEX | Enum | 1=Male, 2=Female. |
| **RACEV1X** | RACE (IMPUTED/EDITED/RECODED) | Enum | 1=White, 2=Black, 3=Amer Ind, 4=Asian/Nat. Hawaiian/Pac. Isl., 6=Multiple. |
| **HISPANX** | HISPANIC ETHNICITY (IMPUTED/EDITED) | Enum | 1=Hispanic, 2=Not Hispanic. |
| **MARRY23X** | MARITAL STATUS-12/31/23 (EDITED/IMPUTED) | Enum | 1=Married, 2=Widowed, 3=Divorced, 4=Separated, 5=Never Married, 6=Under 16. |
| **REGION23** | CENSUS REGION AS OF 12/31/23 | Enum | 1=Northeast, 2=Midwest, 3=South, 4=West. |

## 4. Socioeconomic Status
| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **POVCAT23** | FAMILY INC AS % OF POVERTY LINE | Enum | 1=Poor/Negative, 2=Near Poor, 3=Low Inc, 4=Middle Inc, 5=High Inc. |
| **TTLP23X** | PERSON'S TOTAL INCOME | Num | Sum of all income sources for 2023. |
| **FAMINC23** | FAMILY'S TOTAL INCOME | Num | Total income for the reporting unit. |
| **EDUCYR** | YEARS OF EDUC WHEN FIRST ENTERED | Num | Years of education (0-17). |
| **HIDEG** | HIGHEST DEGREE WHEN FIRST ENTERED | Enum | 1=No Degree, 2=GED, 3=HS Diploma, 4=Bachelor, 5=Master, 6=Doctorate, 7=Other, 8=Under 16. |

## 5. Health Status & Conditions
*Note: Conditions often do not have a year suffix.*

| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **RTHLTH53** | PERCEIVED HEALTH STATUS - RD 5/3 | Enum | 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor. |
| **MNHLTH53** | PERCEIVED MENTAL HEALTH - RD 5/3 | Enum | 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor. |
| **HIBPDX** | HIGH BLOOD PRESSURE DIAG (>17) | Enum | 1=Yes, 2=No. |
| **CHDDX** | CORONARY HEART DISEASE DIAG (>17) | Enum | 1=Yes, 2=No. |
| **ANGIDX** | ANGINA DIAGNOSIS (>17) | Enum | 1=Yes, 2=No. |
| **MIDX** | HEART ATTACK (MI) DIAG (>17) | Enum | 1=Yes, 2=No. |
| **STRKDX** | STROKE DIAGNOSIS (>17) | Enum | 1=Yes, 2=No. |
| **DIABDX_M18** | DIABETES DIAGNOSIS | Enum | 1=Yes, 2=No. |
| **ASTHDX** | ASTHMA DIAGNOSIS | Enum | 1=Yes, 2=No. |
| **CANCERDX** | CANCER DIAGNOSIS (>17) | Enum | 1=Yes, 2=No. |

## 6. Insurance Coverage (Summary)
| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **INSCOV23** | HEALTH INSURANCE COVERAGE INDICATOR | Enum | 1=Any Private, 2=Public Only, 3=Uninsured. |
| **MCDEV23** | EVER HAVE MEDICAID/SCHIP DURING 23 | Enum | 1=Yes, 2=No. |
| **MCREV23** | EVER HAVE MEDICARE DURING 23 | Enum | 1=Yes, 2=No. |
| **PRVEV23** | EVER HAVE PRIVATE INSURANCE DURING 23 | Enum | 1=Yes, 2=No. |

## 7. Healthcare Utilization & Expenditure (Targets)
*Suffix is '23' for the year 2023.*

| Variable | Label | Type | Description |
| :--- | :--- | :--- | :--- |
| **TOTEXP23** | TOTAL HEALTH CARE EXP 23 | Num | **Total Cost.** Sum of all payments for all events. |
| **TOTSLF23** | TOTAL AMT PAID BY SELF/FAMILY 23 | Num | **Out of Pocket.** Amount paid by patient. |
| **TOTMCR23** | TOTAL AMT PAID BY MEDICARE 23 | Num | Amount paid by Medicare. |
| **TOTMCD23** | TOTAL AMT PAID BY MEDICAID 23 | Num | Amount paid by Medicaid. |
| **TOTPRV23** | TOTAL AMT PAID BY PRIVATE INS 23 | Num | Amount paid by Private Insurance. |
| **OBTOTV23** | OFFICE BASED PROVIDER VISITS 23 | Num | Total number of office visits. |
| **OPTOTV23** | OUTPATIENT DEPT PROVIDER VISITS 23 | Num | Total number of outpatient hospital visits. |
| **ERTOT23** | EMERGENCY ROOM VISITS 23 | Num | Total number of ER visits. |
| **IPDIS23** | HOSPITAL DISCHARGES 23 | Num | Total number of inpatient stays. |
| **RXTOT23** | # PRESC MEDS INCL REFILLS 23 | Num | Total number of prescription fills. |