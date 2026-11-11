# Prediction Input Mapping

This document explains how web form answers and API fields map to the 27 input
columns expected by the fitted preprocessor. Direct Python callers provide
those columns directly. API field definitions and validation rules live in
[app/schemas.py](../../app/schemas.py).

The web form and service mappings below are planned.

## How the inputs fit together

- **Web form** converts answers into a profile validated by `PredictionRequest`.
- **API** validates a JSON profile with the same class.
- **Prediction service** maps the profile to the 27 preprocessor columns and
  calls `CostPredictor`.
- **Direct Python use** supplies those columns in a DataFrame and calls
  `CostPredictor` directly.

`CostPredictor` runs the fitted preprocessor and then the fitted model.

## Mapping to preprocessor inputs

Boolean fields map `true` to `1` and `false` to `0`. Unanswered inputs map to
`NaN`. How omitted condition and limitation fields should map is still to decide.

| Profile field | Preprocessor column | Value mapping |
| --- | --- | --- |
| `age` | `AGE23X` | Same integer |
| `sex` | `SEX` | `female` → `0`, `male` → `1` |
| `insurance_status` | `INSCOV23` | `any_private` → `1`, `public_only` → `2`, `uninsured` → `3` |
| `family_size` | `FAMSZE23` | Same integer |
| `education` | `HIDEG` | See [education mapping](#education) |
| `region` | `REGION23` | `northeast` → `1`, `midwest` → `2`, `south` → `3`, `west` → `4` |
| `physical_health` | `RTHLTH31` | See [health rating mapping](#health-ratings) |
| `mental_health` | `MNHLTH31` | See [health rating mapping](#health-ratings) |
| `family_income` | `POVCAT23` | `poor` → `1`, `near_poor` → `2`, `low_income` → `3`, `middle_income` → `4`, `high_income` → `5` |
| `marital_status` | `MARRY31X_GRP` | `married` → `1`, `widowed` → `2`, `divorced` → `3`, `separated` → `4`, `never_married` → `5` |
| `employment_status` | `EMPST31_GRP` | `employed` → `1`, `not_employed` → `0` |
| `recent_life_transition` | `RECENT_LIFE_TRANSITION` | Boolean flag |
| `usual_source_of_care` | `HAVEUS42` | Boolean flag |
| `smoker` | `ADSMOK42` | Boolean flag |
| `hypertension` | `HIBPDX` | Boolean flag |
| `high_cholesterol` | `CHOLDX` | Boolean flag |
| `diabetes` | `DIABDX_M18` | Boolean flag |
| `coronary_heart_disease` | `CHDDX` | Boolean flag |
| `stroke` | `STRKDX` | Boolean flag |
| `cancer` | `CANCERDX` | Boolean flag |
| `arthritis` | `ARTHDX` | Boolean flag |
| `asthma` | `ASTHDX` | Boolean flag |
| `receives_personal_care_help` | `ADLHLP31` | Boolean flag |
| `receives_daily_tasks_help` | `IADLHP31` | Boolean flag |
| `walking_limitation` | `WLKLIM31` | Boolean flag |
| `cognitive_limitation` | `COGLIM31` | Boolean flag |
| `joint_pain` | `JTPAIN31_M18` | Boolean flag |

### Birth year and state

The planned form calculates age as forecast year minus birth year and caps it
at 85. It maps state (50 states or DC) to
[Census region](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf).
API callers provide age and region directly. Other status inputs use the best
available answers for the start of the forecast year.

### Family income ranges

The five income categories mean below 100%, 100% to under 125%, 125% to under
200%, 200% to under 400%, and at least 400% of the applicable poverty line.
The form plans to convert dollar ranges and family size to `family_income`.
Changing family size must clear the income selection. The planned form's “8+”
family-size choice sends `8`. The final income ranges and API mapping are open.

### Health ratings

For both health fields: `excellent` → `1`, `very_good` → `2`, `good` → `3`,
`fair` → `4`, `poor` → `5`.

### Education

| Profile value | Preprocessor code |
| --- | --- |
| `no_degree` | `1` |
| `ged` | `2` |
| `high_school_diploma` | `3` |
| `associate_degree` | `7` |
| `bachelors_degree` | `4` |
| `masters_degree` | `5` |
| `doctorate` | `6` |
| `other` | `7` |

### Recent life transition

The form plans to ask one question about changes in marital status or
employment. Only that answer sets `RECENT_LIFE_TRANSITION`. A skipped answer
maps to `NaN`, even when both current statuses are known.

### Checklists

Gradio checklists return selected items as lists. The planned form handler
converts these selections to individual Boolean fields. API callers supply one
Boolean field per item.

If the user checks at least one item, treat the unchecked items in that
checklist as “No.” How to handle an entirely skipped checklist is still open.
The preprocessor calculates condition and limitation counts after filling
missing values.

## Direct Python use

Pass one DataFrame row per person with all 27 columns listed in the fitted
preprocessor's `feature_names_in_`. Use `numpy.nan` for missing values.
`CostPredictor` orders the columns for you. For nominal and binary fields, the
preprocessor also accepts the labels in
[src/constants.py](../../src/constants.py), regardless of case. Health ratings
and family income use numerical codes.

Direct Python calls follow the fitted preprocessor's missing-value rules. The
saved preprocessor uses `strict=False` and fills missing values. See the
[pipeline setup](../../scripts/preprocess.py).

After loading the matching preprocessor and model:

```python
import numpy as np
import pandas as pd
from src.prediction import CostPredictor

features = preprocessor.feature_names_in_
row = dict.fromkeys(features, np.nan)
row.update(AGE23X=42, SEX=0, INSCOV23=2, FAMSZE23=2, HIDEG=4)
predictor = CostPredictor(preprocessor, model, features)
quantiles = predictor.predict_quantiles(pd.DataFrame([row]))
```

This returns q25, q50, q75, and q90 in 2023 dollars. The service will add
inflation adjustment and warning flags.

## Open mapping decisions

- Decide how to choose the forecast year.
- Define which marital and employment changes count, how far back to ask, and
  how to word the question. Check how someone with a job to return to would
  answer the separate employment question.
- Finalize the form's dollar ranges, their reference year, and how API callers
  provide family income.
- Decide whether omitted API condition and limitation fields mean No or
  unanswered, and whether the form needs an explicit “None of these” choice.
