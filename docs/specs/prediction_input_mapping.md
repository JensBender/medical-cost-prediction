# Prediction Input Mapping

This document maps web form answers and API fields to the 27 preprocessor
inputs. API field definitions and validation rules live in
[app/schemas.py](../../app/schemas.py).

## How the inputs fit together

- **Web form** (planned): converts answers
  into a validated input profile.
- **API**: validates a JSON profile using `PredictionRequest`.
- **Prediction service** (planned): converts the profile to preprocessor inputs
  and calls `CostPredictor`, which runs the fitted preprocessor and model.
- **Direct Python use**: supplies preprocessor inputs directly to `CostPredictor`.

The web form and API both call the prediction service. Direct Python callers bypass it.

## Input mapping

The service receives the same fields and values as the validated API profile.
The third column shows its planned conversions; the last names the resulting
columns, which direct Python callers supply themselves.

| Web form | API field | Prediction service mapping | Direct Python column |
| --- | --- | --- | --- |
| Birth year | `age` | Same integer | `AGE23X` |
| Sex | `sex` | `female` → `0`, `male` → `1` | `SEX` |
| Insurance status | `insurance_status` | `any_private` → `1`, `public_only` → `2`, `uninsured` → `3` | `INSCOV23` |
| Family size | `family_size` | Same integer | `FAMSZE23` |
| Education | `education` | See [education](#education) | `HIDEG` |
| State | `region` | `northeast` → `1`, `midwest` → `2`, `south` → `3`, `west` → `4` | `REGION23` |
| Physical health | `physical_health` | See [health ratings](#health-ratings) | `RTHLTH31` |
| Mental health | `mental_health` | See [health ratings](#health-ratings) | `MNHLTH31` |
| Family income range (mapping to decide) | `family_income` | `poor` → `1`, `near_poor` → `2`, `low_income` → `3`, `middle_income` → `4`, `high_income` → `5` | `POVCAT23` |
| Marital status | `marital_status` | `married` → `1`, `widowed` → `2`, `divorced` → `3`, `separated` → `4`, `never_married` → `5` | `MARRY31X_GRP` |
| Employment status | `employment_status` | `employed` → `1`, `not_employed` → `0` | `EMPST31_GRP` |
| Recent life transition | `recent_life_transition` | Boolean → `0`/`1`; not inferred from current statuses | `RECENT_LIFE_TRANSITION` |
| Usual source of care | `usual_source_of_care` | Boolean → `0`/`1` | `HAVEUS42` |
| Smoker | `smoker` | Boolean → `0`/`1` | `ADSMOK42` |
| Conditions checklist: hypertension | `hypertension` | Boolean → `0`/`1` | `HIBPDX` |
| Conditions checklist: high cholesterol | `high_cholesterol` | Boolean → `0`/`1` | `CHOLDX` |
| Conditions checklist: diabetes | `diabetes` | Boolean → `0`/`1` | `DIABDX_M18` |
| Conditions checklist: coronary heart disease | `coronary_heart_disease` | Boolean → `0`/`1` | `CHDDX` |
| Conditions checklist: stroke | `stroke` | Boolean → `0`/`1` | `STRKDX` |
| Conditions checklist: cancer | `cancer` | Boolean → `0`/`1` | `CANCERDX` |
| Conditions checklist: arthritis | `arthritis` | Boolean → `0`/`1` | `ARTHDX` |
| Conditions checklist: asthma | `asthma` | Boolean → `0`/`1` | `ASTHDX` |
| Limitations checklist: personal care help | `receives_personal_care_help` | Boolean → `0`/`1` | `ADLHLP31` |
| Limitations checklist: daily tasks help | `receives_daily_tasks_help` | Boolean → `0`/`1` | `IADLHP31` |
| Limitations checklist: walking limitation | `walking_limitation` | Boolean → `0`/`1` | `WLKLIM31` |
| Limitations checklist: cognitive limitation | `cognitive_limitation` | Boolean → `0`/`1` | `COGLIM31` |
| Limitations checklist: joint pain | `joint_pain` | Boolean → `0`/`1` | `JTPAIN31_M18` |

### Planned form conversions

- Calculate age as forecast year minus birth year, capped at 85. How to choose
  the forecast year is still to decide.
- Map state to [Census region](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf).
- Send `8` for the family-size choice “8+”. Clear the income selection when
  family size changes. Final income ranges and their reference year are open.

The income categories mean below 100%, 100% to under 125%, 125% to under 200%,
200% to under 400%, and at least 400% of the applicable poverty line.

For recent life transition, decide which marital and employment changes count,
how far back to ask, and the question wording. Also check how someone with a job
to return to would answer the separate employment question.

### Health ratings

For both fields: `excellent` → `1`, `very_good` → `2`, `good` → `3`,
`fair` → `4`, `poor` → `5`.

### Education

`no_degree` → `1`, `ged` → `2`, `high_school_diploma` → `3`,
`associate_degree` → `7`, `bachelors_degree` → `4`, `masters_degree` → `5`,
`doctorate` → `6`, `other` → `7`.

## Checklist and missing-value mappings

The form converts each checklist's selected items into separate Boolean fields.
If any item is selected, the unchecked items in that checklist become `false`.
The service maps `true` to `1`, `false` to `0`, and unanswered inputs to `NaN`.

Still to decide: how an entirely empty checklist maps, whether to offer a
“None of these” choice, and whether omitted API condition and limitation fields
mean No or unanswered. The family income format for API callers is also open.

## Direct Python inputs

Supply a DataFrame with the 27 columns above and `numpy.nan` for missing values.
Nominal and binary inputs also accept the labels in
[src/constants.py](../../src/constants.py), regardless of case. Health ratings
and family income require numerical codes.
