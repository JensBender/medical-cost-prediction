# Prediction Input Mapping

The fitted scikit-learn preprocessing pipeline (the preprocessor) takes 27
inputs. This document shows how each web form answer maps to an API field and
then to a preprocessor column.

## How the inputs fit together

The planned web form converts answers to API values. API callers send those
values as JSON. Both validate requests against the same Pydantic schema,
[`PredictionRequest`](../../app/schemas.py).

The planned prediction service takes a validated request from either source,
converts it to preprocessor inputs, and calls `CostPredictor` to run the
preprocessor and model. [Direct Python callers](#direct-python-inputs) supply
those inputs themselves.

## Input mapping

The API fields and validation rules are in place. The form conversions and
prediction service mappings are planned.

- An asterisk (\*) marks a required input field (age, sex,
  insurance status, family size, education).
- “Label → API value” means the form sends the exact value for the selected
  label. For example, “Very Good” becomes `very_good`.
- Yes/no answers and checklist items follow the [Boolean and missing-value rules](#boolean-and-missing-value-rules).
- “Open” links point to questions still to decide. The [time reference](#7-time-reference)
  also needs review across the form.

| Web form input | Form conversion | API field | Service mapping | Preprocessor column |
| --- | --- | --- | --- | --- |
| Birth year* | Subtract birth year from forecast year. Cap at 85. [Open 1](#1-age-and-forecast-year) | `age` | Same integer (18–85) | `AGE23X` |
| Sex* | Label → API value | `sex` | `female` → `0`, `male` → `1` | `SEX` |
| Insurance status* | For both private and public coverage, use `any_private`. [Open 9](#9-insurance-wording) | `insurance_status` | `any_private` → `1`, `public_only` → `2`, `uninsured` → `3` | `INSCOV23` |
| Family size* | Use the selected count, or `8` for “8+”. [Open 2](#2-family-size) | `family_size` | Same integer (1–14) | `FAMSZE23` |
| Education* | Use the API value for the highest completed qualification. [Open 8](#8-education-choices) | `education` | [education](#education) | `HIDEG` |
| State | Look up the state’s Census region | `region` | `northeast` → `1`, `midwest` → `2`, `south` → `3`, `west` → `4` | `REGION23` |
| Physical health | Label → API value | `physical_health` | [health ratings](#health-ratings) | `RTHLTH31` |
| Mental health | Label → API value | `mental_health` | [health ratings](#health-ratings) | `MNHLTH31` |
| Family income range | Dollar range → category. [Open 3](#3-family-income) | `family_income` | [Income codes](#family-income) | `POVCAT23` |
| Marital status | Label → API value | `marital_status` | `married` → `1`, `widowed` → `2`, `divorced` → `3`, `separated` → `4`, `never_married` → `5` | `MARRY31X_GRP` |
| Employment status | Label → API value. [Open 4](#4-employment-and-recent-life-transition) | `employment_status` | `employed` → `1`, `not_employed` → `0` | `EMPST31_GRP` |
| Recent life transition | Ask a separate yes/no question. [Open 4](#4-employment-and-recent-life-transition) | `recent_life_transition` | Boolean rule | `RECENT_LIFE_TRANSITION` |
| Usual source of care | Yes/no | `usual_source_of_care` | Boolean rule | `HAVEUS42` |
| Current cigarette smoking | Yes/no | `smoker` | Boolean rule | `ADSMOK42` |
| High blood pressure | Conditions checklist | `hypertension` | Boolean rule | `HIBPDX` |
| High cholesterol | Conditions checklist | `high_cholesterol` | Boolean rule | `CHOLDX` |
| Diabetes | Conditions checklist | `diabetes` | Boolean rule | `DIABDX_M18` |
| Coronary heart disease | Conditions checklist | `coronary_heart_disease` | Boolean rule | `CHDDX` |
| Stroke | Conditions checklist | `stroke` | Boolean rule | `STRKDX` |
| Cancer | Conditions checklist | `cancer` | Boolean rule | `CANCERDX` |
| Arthritis | Conditions checklist | `arthritis` | Boolean rule | `ARTHDX` |
| Asthma | Conditions checklist | `asthma` | Boolean rule | `ASTHDX` |
| Help with personal care | Limitations and symptoms checklist | `receives_personal_care_help` | Boolean rule | `ADLHLP31` |
| Help with daily tasks | Limitations and symptoms checklist | `receives_daily_tasks_help` | Boolean rule | `IADLHP31` |
| Walking limitation | Limitations and symptoms checklist. [Open 6](#6-limitations-and-symptoms-wording) | `walking_limitation` | Boolean rule | `WLKLIM31` |
| Cognitive limitation | Limitations and symptoms checklist. [Open 6](#6-limitations-and-symptoms-wording) | `cognitive_limitation` | Boolean rule | `COGLIM31` |
| Joint symptoms in the past 12 months | Limitations and symptoms checklist | `joint_pain` | Boolean rule | `JTPAIN31_M18` |

### Form details

- **State.** Offer all 50 states and DC. Map each to its [Census region](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf).
- **Family size.** Count yourself and anyone in your home related to you by
  blood, marriage, or adoption.
- **Family income.** Clear the selected income range when family size changes.
- **Conditions checklist.** Ask whether a doctor or other health professional
  has ever diagnosed each condition.
- **Limitations and symptoms checklist.** Help with personal care or daily
  tasks includes supervision. Joint symptoms include pain, aching, stiffness,
  or swelling in or around a joint in the past 12 months.

See the [product requirements](product_requirements.md#user-inputs) for the full questions.

### Family income

The API currently accepts these categories. The [dollar ranges and API input
format](#3-family-income) are still under review.

| Family income as a share of the poverty line | API value | Preprocessor code |
| --- | --- | --- |
| Under 100% | `poor` | `1` |
| 100% to under 125% | `near_poor` | `2` |
| 125% to under 200% | `low_income` | `3` |
| 200% to under 400% | `middle_income` | `4` |
| 400% or more | `high_income` | `5` |

### Health ratings

Physical and mental health use the same codes. `excellent` → `1`,
`very_good` → `2`, `good` → `3`, `fair` → `4`, `poor` → `5`.

### Education

`no_degree` → `1`, `ged` → `2`, `high_school_diploma` → `3`,
`associate_degree` → `7`, `bachelors_degree` → `4`, `masters_degree` → `5`,
`doctorate` → `6`, `other` → `7`.

The model treats `associate_degree` and `other` as one category (“Other”).
The `doctorate` choice includes PhD, MD, and JD. Some [education choices](#8-education-choices)
still need clarification.

## Boolean and missing-value rules

For yes/no questions, the form sends `true` for Yes and `false` for No.
The service maps `true` to `1` and `false` to `0` for all Boolean fields.

Each checklist item has its own Boolean field. If someone checks at least one
item, the form sends `true` for each checked item and `false` for the rest of
that checklist. How to handle an empty checklist or a “None of these” choice
is [still open](#5-checklists-and-missing-answers).

For optional fields, both a missing field and JSON `null` give Python
`None`. The planned service converts `None` to `numpy.nan` so the
preprocessor can fill in the value. The schema rejects a request if a
required field is missing or `null`.

Use exact category values, integers for age and family size, and JSON Booleans
for yes/no answers. For example, `"very_good"`, `42`, and `false` are valid.
The schema rejects `"Very Good"`, `"42"`, and `"false"`, as well as unknown fields.

## Open decisions and definitions to review

### 1. Age and forecast year

How should the form choose the forecast year? It calculates age at year-end.
Can someone under 18 use the form if they turn 18 during the forecast year?

The form caps age at 85. The API rejects ages above 85. Should API callers
cap age themselves, or should the API accept older ages and cap them?

### 2. Family size

The form sends `8` for any family of eight or more. The API accepts exact
counts up to 14, the largest family in the training data. Should the form
keep “8+”? Which income ranges should those families see? Should the API
accept counts above 14?

### 3. Family income

Choose the income period, the poverty-line source and year, and any rules
that vary by location. Set exact dollar cutoffs and decide how to round them
for display, including for families of eight or more.

Check that every income falls into exactly one range. For example, one
[draft range](technical_specifications.md#draft-family-income-ranges) ends at
$18,100 and the next starts at $18,200.

The API takes an income category. Should callers keep sending a category or
send a dollar amount? If they send an amount, what else does the service
need to find the category?

### 4. Employment and recent life transition

The [training notebook](../../notebooks/1_eda_and_preprocessing.py) counts
“Job to Return To” and “Job in Ref Period” as `not_employed` and flags both as
recent transitions. Someone temporarily away from work might choose “Employed”
on the form. How should the employment question guide them?

Ask about transitions separately from current marital and employment status.
Which marital or job changes count, and how far back should the question go?
The notebook suggests 12 months. The product requirements leave the period
open. Check that the final wording captures the events used in training.

### 5. Checklists and missing answers

How should the form distinguish a skipped checklist from “None of these”?
If the form offers that choice, can someone select it alongside another item?
When someone checks one item, should every unchecked item count as No, or
should users be able to leave individual items unanswered?

The API currently treats an omitted condition or limitation as unanswered.
Should it default to No? Should an explicit `null` have a different meaning?
Apply the same decision to the form, schema defaults, and service mapping.

### 6. Limitations and symptoms wording

The form draft asks about **serious difficulty** walking or climbing stairs.
The schema also mentions lifting and bending and leaves severity unspecified.

The draft asks about serious difficulty concentrating, remembering, or making
decisions. The schema also mentions confusion and supervision for safety.
Check both questions against the training definitions before finalizing the
form wording, schema descriptions, and table labels.

### 7. Time reference

The specs call for status at the start of the forecast year. Several draft
questions ask about today. For insurance, employment, marital status, and
health, should users report their status today, at the start of the forecast
year, or as they expect it to be during that year?

Keep “ever diagnosed” and “past 12 months” in the relevant questions. Specify
the date from which to count back.

### 8. Education choices

Which choice fits someone who attended college without earning a degree?
What should someone with a vocational qualification choose? Check the training
categories and confirm that an associate degree belongs under “Other” before
writing the help text.

### 9. Insurance wording

Which examples would help users choose between private coverage, public-only
coverage, and no coverage? Anyone with both private and public insurance
selects `any_private`. Decide which coverage to report if it changes during the
year. See [time reference](#7-time-reference).

## Direct Python inputs

Pass `CostPredictor` a DataFrame with all 27 preprocessor columns. Use
`numpy.nan` for missing values. This path skips request validation and the
prediction service mappings.

For categorical columns (nominal and binary), the preprocessor accepts
numerical labels and case-insensitive string labels. For example, `REGION23`
accepts `2`, `"Midwest"`, or `"midwest"`.
See [`CATEGORY_LABELS_PIPELINE`](../../src/constants.py) for the label mappings.

Physical health, mental health, and family income take numerical labels only.
