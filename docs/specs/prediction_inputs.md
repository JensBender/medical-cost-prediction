# Prediction Inputs

This document describes the inputs for the web form, API, prediction service,
and direct Python use. The planned app builds on the 27-column input format
already used for direct prediction in Python.

## How the inputs fit together

- **Web form:** converts answers such as birth year and state into a shared
  prediction profile: a set of named fields such as `age` and `region`.
- **API:** accepts that same profile as JSON. Both interfaces must use the
  same field definitions and validation rules.
- **Prediction service:** maps a validated profile to the 27 preprocessor
  input columns. Both interfaces must use this same mapping.
- **Direct Python use:** supplies those 27 columns in a pandas DataFrame.

The service and direct Python callers both use `CostPredictor`. It runs the
fitted preprocessor and passes the model-ready features to the fitted model.

The [product requirements](product_requirements.md#user-inputs) describe the
form questions. The [technical specifications](technical_specifications.md#prediction-architecture)
describe how the software is organized.

## Input rules

- In the shared profile, required fields must have a value. Optional fields
  may be omitted or `null`.
- Missing optional answers become `NaN` in the preprocessor inputs. The
  preprocessor fills these gaps with values learned during training.
  `false` means No. Optional form controls must start unanswered.
- Use the best available answers for the start of the forecast year, except
  age, which uses the end of that year. Conditions cover diagnoses ever
  received; joint pain covers the prior 12 months.

## Required inputs

The service sends the numeric codes below. Direct Python callers can also
use the listed labels. The preprocessor matches those labels regardless of
case. Codes and labels follow the same order as the profile values.

| Profile field | Allowed values | Preprocessor input |
| --- | --- | --- |
| `age` | Integer 18–85; `85` means 85 or older | `AGE23X`: same integer |
| `sex` | `female`, `male` | `SEX`: `0` or `Female`, `1` or `Male` |
| `region` | `northeast`, `midwest`, `south`, `west` | `REGION23`: `1` or `Northeast`, `2` or `Midwest`, `3` or `South`, `4` or `West` |
| `insurance_status` | `any_private`, `public_only`, `uninsured` | `INSCOV23`: `1` or `Any Private`, `2` or `Public Only`, `3` or `Uninsured` |
| `physical_health` | [Health rating](#health-ratings) | `RTHLTH31`: `1`–`5` |

The forecast year is the year the estimate covers. The form calculates age as
forecast year minus birth year, rejects ages below
18, and caps age at 85. It converts state (50 states or DC) to a
[Census region](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf).
API callers supply age and region directly. Choose `any_private` if the person
has any private coverage, even alongside public coverage.

## Optional inputs

| Profile field | Allowed values | Preprocessor input |
| --- | --- | --- |
| `marital_status` | `married`, `widowed`, `divorced`, `separated`, `never_married` | `MARRY31X_GRP`: `1` or `Married`, `2` or `Widowed`, `3` or `Divorced`, `4` or `Separated`, `5` or `Never Married` |
| `recent_life_transition` | Proposed boolean for a [recent marital or employment change](#recent-life-transition) | `RECENT_LIFE_TRANSITION`: `false` → `0` or `No`, `true` → `1` or `Yes` |
| `family_size` | Integer 1–14: related people living together, including the person answering | `FAMSZE23`: same integer; the form's “8+” sends `8` |
| `income_bracket` | `poor`, `near_poor`, `low_income`, `middle_income`, `high_income` | `POVCAT23`: `1`, `2`, `3`, `4`, `5` |
| `education` | [Education choice](#education) | `HIDEG`: code or label below |
| `employment_status` | `employed`, `not_employed` | `EMPST31_GRP`: `1` or `Employed`, `0` or `Not Employed` |
| `mental_health` | [Health rating](#health-ratings) | `MNHLTH31`: `1`–`5` |
| `usual_source_of_care` | Boolean | `HAVEUS42`: `false` → `0` or `No`, `true` → `1` or `Yes` |
| `smoker` | Boolean: currently smokes cigarettes | `ADSMOK42`: `false` → `0` or `No`, `true` → `1` or `Yes` |
| `chronic_conditions` | List of [condition values](#checklists) | Eight flags below |
| `limitations` | List of [limitation values](#checklists) | Five flags below |

### Family income ranges

The five income categories mean below 100%, 100% to under 125%, 125% to under
200%, 200% to under 400%, and at least 400% of the applicable poverty line.
The form shows dollar ranges based on family size; changing family size must
clear the income selection. API callers may supply a category without family
size. The form approximates families larger than eight as eight.

### Health ratings

For both health fields: `excellent` → `1`, `very_good` → `2`, `good` → `3`,
`fair` → `4`, `poor` → `5`.

### Education

| Profile value | Preprocessor `HIDEG` |
| --- | --- |
| `no_degree` | `1` or `No Degree` |
| `ged` | `2` or `GED` |
| `high_school_diploma` | `3` or `HS Diploma` |
| `associate_degree` | `7` or `Other` |
| `bachelors_degree` | `4` or `Bachelor's` |
| `masters_degree` | `5` or `Master's` |
| `doctorate_or_professional` | `6` or `Doctorate` |
| `other` | `7` or `Other` |

### Recent life transition

The form plans to ask one optional question about changes in marital status
or employment. Only that answer sets `RECENT_LIFE_TRANSITION`. A skipped
answer maps to `NaN`, even when both current statuses are known.

### Checklists

Accept each listed value at most once. Set selected flags to `1` and unselected
flags to `0`. An empty list (`[]`) means “None of these”: all flags are `0`.
An omitted or `null` list means unanswered: all flags are `NaN`. The
preprocessor calculates condition and limitation counts after filling missing
values. For each flag below, direct Python callers can also use `No` for `0`
and `Yes` for `1`.

| List | Profile value | Preprocessor column |
| --- | --- | --- |
| `chronic_conditions` | `hypertension` | `HIBPDX` |
| | `high_cholesterol` | `CHOLDX` |
| | `diabetes` | `DIABDX_M18` |
| | `coronary_heart_disease` | `CHDDX` |
| | `stroke` | `STRKDX` |
| | `cancer` | `CANCERDX` |
| | `arthritis` | `ARTHDX` |
| | `asthma` | `ASTHDX` |
| `limitations` | `personal_care_help` | `ADLHLP31` |
| | `daily_tasks_help` | `IADLHP31` |
| | `walking_difficulty` | `WLKLIM31` |
| | `cognitive_difficulty` | `COGLIM31` |
| | `joint_pain` | `JTPAIN31_M18` |

## Direct Python use

Pass one DataFrame row per person with all 27 columns listed in the fitted
preprocessor's `feature_names_in_`. Fill the five required inputs and use
`numpy.nan` for missing optional values. `CostPredictor` orders the columns
for you. Check your values against the tables before predicting.

## Minimal example

A profile with only the required answers:

```json
{
  "age": 42,
  "sex": "female",
  "region": "west",
  "insurance_status": "public_only",
  "physical_health": "good"
}
```

The same inputs in Python, after loading the matching preprocessor and model:

```python
import numpy as np
import pandas as pd
from src.prediction import CostPredictor

features = preprocessor.feature_names_in_
row = dict.fromkeys(features, np.nan)
row.update(AGE23X=42, SEX=0, REGION23=4, INSCOV23=2, RTHLTH31=3)
predictor = CostPredictor(preprocessor, model, features)
quantiles = predictor.predict_quantiles(pd.DataFrame([row]))
```

This returns q25, q50, q75, and q90 in 2023 dollars. The service adds inflation
adjustment and warning flags; see the [API contract](technical_specifications.md#api-contract).

## Still to confirm before implementation

- Decide how to choose the forecast year.
- Define which marital and employment changes count, how far back to ask, and
  how to word the question. Check how someone with a job to return to would
  answer the separate employment question.
- Verify the source and reference year for the form's dollar ranges.
