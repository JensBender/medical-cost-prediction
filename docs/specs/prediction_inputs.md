# Prediction Inputs

This document defines what to send for a prediction and how each answer maps
to the model's inputs. The web form and API rules are planned requirements;
direct prediction through `CostPredictor` already exists.

## How the inputs fit together

- **Web form:** converts answers such as birth year and state into a shared
  prediction profile: a set of named fields such as `age` and `region`.
- **API:** accepts that same profile as JSON. Both interfaces must use the
  same field definitions and validation rules.
- **Prediction service:** maps a validated profile to the model's 27 input
  columns. Both interfaces must use this same mapping.
- **Direct Python use:** supplies those 27 columns in a pandas DataFrame.

The [product requirements](product_requirements.md#user-inputs) describe the
form questions. The [technical specifications](technical_specifications.md#prediction-architecture)
describe how the software is organized.

## Shared rules

- Required fields must have a value. Optional fields may be omitted or `null`.
- Use the exact category values below, integers for age and family size, and
  `true` or `false` for yes/no answers. Reject unknown fields, unsupported
  values, and wrong types; for example, `"42"` is not an integer.
- Missing optional answers become `NaN` in the model inputs. The preprocessor
  fills these gaps with values learned during training. `false` means No,
  not missing. The combined transition flag follows the rule below.
  Optional form controls must start unanswered.
- Use the best available answers for the start of the forecast year, except
  age, which uses the end of that year. Conditions cover diagnoses ever
  received; joint pain covers the prior 12 months.

## Required inputs

The model codes below follow the same order as the listed profile values.

| Profile field | Allowed values | Model input |
| --- | --- | --- |
| `age` | Integer 18–85; `85` means 85 or older | `AGE23X`: same integer |
| `sex` | `female`, `male` | `SEX`: `0`, `1` |
| `region` | `northeast`, `midwest`, `south`, `west` | `REGION23`: `1`, `2`, `3`, `4` |
| `insurance_status` | `any_private`, `public_only`, `uninsured` | `INSCOV23`: `1`, `2`, `3` |
| `physical_health` | [Health rating](#health-ratings) | `RTHLTH31`: `1`–`5` |

The forecast year is the year the estimate covers. The form calculates age as
forecast year minus birth year, rejects ages below
18, and caps age at 85. It converts state (50 states or DC) to a
[Census region](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf).
API callers supply age and region directly. Choose `any_private` if the person
has any private coverage, even alongside public coverage.

## Optional inputs

| Profile field | Allowed values | Model input |
| --- | --- | --- |
| `marital_status` | `married`, `widowed`, `divorced`, `separated`, `never_married` | `MARRY31X_GRP`: `1`, `2`, `3`, `4`, `5` |
| `recent_marital_change` | Boolean: recently married, widowed, divorced, or separated | Contributes to `RECENT_LIFE_TRANSITION`; see [employment](#employment-and-life-transitions) |
| `family_size` | Integer 1–14: related people living together, including the person answering | `FAMSZE23`: same integer; the form's “8+” sends `8` |
| `income_bracket` | `poor`, `near_poor`, `low_income`, `middle_income`, `high_income` | `POVCAT23`: `1`, `2`, `3`, `4`, `5` |
| `education` | [Education choice](#education) | `HIDEG`: code below |
| `employment_status` | [Employment choice](#employment-and-life-transitions) | `EMPST31_GRP` and `RECENT_LIFE_TRANSITION` |
| `mental_health` | [Health rating](#health-ratings) | `MNHLTH31`: `1`–`5` |
| `usual_source_of_care` | Boolean | `HAVEUS42`: `false` → `0`, `true` → `1` |
| `smoker` | Boolean: currently smokes cigarettes | `ADSMOK42`: `false` → `0`, `true` → `1` |
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

| Profile value | Model `HIDEG` code |
| --- | --- |
| `no_degree` | `1` |
| `ged` | `2` |
| `high_school_diploma` | `3` |
| `associate_degree` | `7` (Other) |
| `bachelors_degree` | `4` |
| `masters_degree` | `5` |
| `doctorate_or_professional` | `6` |
| `other` | `7` |

### Employment and life transitions

| Profile `employment_status` | Model `EMPST31_GRP` | Indicates a transition? |
| --- | --- | --- |
| `employed` | `1` | No |
| `job_to_return_to` (on leave or has a job to return to) | `0` | Yes |
| `worked_recently` (currently out of work) | `0` | Yes |
| `not_employed` | `0` | No |

Combine employment status and `recent_marital_change` into
`RECENT_LIFE_TRANSITION`: use `1` if either indicates a transition, `0` if at
least one is answered and neither indicates a transition, and `NaN` if both
are missing. Current marital status alone does not establish a recent change.
This follows the [training rule](../../scripts/preprocess.py).

### Checklists

Accept each listed value at most once. Set selected flags to `1` and unselected
flags to `0`. An empty list (`[]`) means “None of these”: all flags are `0`.
An omitted or `null` list means unanswered: all flags are `NaN`. The form must
keep “None of these” separate from skipping the question. The preprocessor
calculates condition and limitation counts after filling missing values.

| List | Profile value | Model column |
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
for you. Use the cleaned numeric codes above, not raw survey codes: for
example, a nonsmoker is `ADSMOK42=0`, not `2`.

Direct callers must check their values against this contract; the fitted
preprocessor does not enforce every allowed value. It also accepts fitted
labels for nominal and binary fields, ignoring case; see the
[categorical label standardizer](../../src/transformers.py).

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

Decide how the forecast year is chosen, define the time period meant by
“recent” marital and employment changes, and verify the source and reference
year for the form's dollar ranges before adding them here.
