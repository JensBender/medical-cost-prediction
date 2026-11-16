# Prediction Input Mapping

This document maps web form answers and API fields to the 27 inputs expected by
the fitted preprocessing pipeline (a scikit-learn `Pipeline`), referred to below
as the preprocessor.

## How the inputs fit together

- **Web form** (planned): converts answers into API-compatible inputs and
  validates them.
- **API**: parses JSON inputs into Python values and validates them.
- **Prediction service** (planned): converts validated web form or API inputs
  into the column names and values expected by the preprocessor, then calls
  `CostPredictor`, which runs the fitted preprocessor and model.
- **Direct Python use**: supplies preprocessor inputs directly to `CostPredictor`.

The web form and API both call the prediction service. Direct Python use bypasses it.

**Data validation:** The web form and API use Pydantic for data validation with the
`PredictionRequest` schema in [app/schemas.py](../../app/schemas.py), which
defines the fields, allowed values, and validation rules.

## Input mapping

The service receives validated inputs from the web form or API, using the same
field names and value formats. Form conversions and service mappings below are
planned; API fields and validation rules describe the current request schema.
The last column names the service's output columns, which direct Python callers
supply themselves.

**Required inputs:** An asterisk (*) marks inputs required by the web/API
request schema: age, sex, insurance status, family size, and education. All
other fields currently allow omission or `null`. These labels describe the
request contract, not the pipeline's internal required-feature list; see
[review 10](#10-required-inputs).

**Shared rules:** “Label → API value” means converting a displayed choice to
the exact schema value, such as “Very Good” → `very_good`. “Yes/no” and
“Checklist item” use the [Boolean and missing-value rules](#boolean-and-missing-value-rules)
below. Links marked **Open** identify decisions or definitions still to review.
The [time reference](#7-time-reference) needs review across the form.

| Web form input | Form conversion | API field | Service mapping | Preprocessor input column |
| --- | --- | --- | --- | --- |
| Birth year* | Forecast year − birth year; cap at 85. **Open:** [1](#1-age-and-forecast-year) | `age` | Pass through validated integer (18–85) | `AGE23X` |
| Sex* | Label → API value | `sex` | `female` → `0`, `male` → `1` | `SEX` |
| Insurance status* | Label → API value; private + public → `any_private`. **Open:** [9](#9-insurance-wording) | `insurance_status` | `any_private` → `1`, `public_only` → `2`, `uninsured` → `3` | `INSCOV23` |
| Family size* | “8+” → `8`; otherwise selected count. **Open:** [2](#2-family-size) | `family_size` | Pass through validated integer (1–14) | `FAMSZE23` |
| Education* | Label → API value; highest completed qualification. **Open:** [8](#8-education-choices) | `education` | See [education](#education) | `HIDEG` |
| State | State → Census region | `region` | `northeast` → `1`, `midwest` → `2`, `south` → `3`, `west` → `4` | `REGION23` |
| Physical health | Label → API value | `physical_health` | See [health ratings](#health-ratings) | `RTHLTH31` |
| Mental health | Label → API value | `mental_health` | See [health ratings](#health-ratings) | `MNHLTH31` |
| Family income range | Dollar range → category. **Open:** [3](#3-family-income) | `family_income` | `poor` → `1`, `near_poor` → `2`, `low_income` → `3`, `middle_income` → `4`, `high_income` → `5` | `POVCAT23` |
| Marital status | Label → API value | `marital_status` | `married` → `1`, `widowed` → `2`, `divorced` → `3`, `separated` → `4`, `never_married` → `5` | `MARRY31X_GRP` |
| Employment status | Label → API value. **Open:** [4](#4-employment-and-recent-life-transition) | `employment_status` | `employed` → `1`, `not_employed` → `0` | `EMPST31_GRP` |
| Recent life transition | Yes/no; separate question, not inferred from current statuses. **Open:** [4](#4-employment-and-recent-life-transition) | `recent_life_transition` | Boolean rule | `RECENT_LIFE_TRANSITION` |
| Usual source of care | Yes/no | `usual_source_of_care` | Boolean rule | `HAVEUS42` |
| Current cigarette smoking | Yes/no | `smoker` | Boolean rule | `ADSMOK42` |
| Ever-diagnosed conditions: high blood pressure | Checklist item | `hypertension` | Boolean rule | `HIBPDX` |
| Ever-diagnosed conditions: high cholesterol | Checklist item | `high_cholesterol` | Boolean rule | `CHOLDX` |
| Ever-diagnosed conditions: diabetes | Checklist item | `diabetes` | Boolean rule | `DIABDX_M18` |
| Ever-diagnosed conditions: coronary heart disease | Checklist item | `coronary_heart_disease` | Boolean rule | `CHDDX` |
| Ever-diagnosed conditions: stroke | Checklist item | `stroke` | Boolean rule | `STRKDX` |
| Ever-diagnosed conditions: cancer | Checklist item | `cancer` | Boolean rule | `CANCERDX` |
| Ever-diagnosed conditions: arthritis | Checklist item | `arthritis` | Boolean rule | `ARTHDX` |
| Ever-diagnosed conditions: asthma | Checklist item | `asthma` | Boolean rule | `ASTHDX` |
| Limitations and symptoms: personal care help | Checklist item | `receives_personal_care_help` | Boolean rule | `ADLHLP31` |
| Limitations and symptoms: daily tasks help | Checklist item | `receives_daily_tasks_help` | Boolean rule | `IADLHP31` |
| Limitations and symptoms: walking limitation | Checklist item. **Open:** [6](#6-limitations-and-symptoms-wording) | `walking_limitation` | Boolean rule | `WLKLIM31` |
| Limitations and symptoms: cognitive limitation | Checklist item. **Open:** [6](#6-limitations-and-symptoms-wording) | `cognitive_limitation` | Boolean rule | `COGLIM31` |
| Limitations and symptoms: joint symptoms in the past 12 months | Checklist item | `joint_pain` | Boolean rule | `JTPAIN31_M18` |

### Form details

- **State:** Offer the 50 states and DC, mapped to
  [Census region](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf).
- **Family size:** Count the person and people living in the same home who are
  related by blood, marriage, or adoption, as specified in the PRD.
- **Family income:** Clear the income selection when family size changes.
  The current categories mean below 100%, 100% to under 125%, 125% to under
  200%, 200% to under 400%, and at least 400% of the applicable poverty line.
  The dollar ranges and API income format remain [open](#3-family-income).
- **Conditions:** Ask whether a doctor or other health professional has ever
  diagnosed each condition, rather than whether the person currently has symptoms.
- **Limitations and symptoms:** Personal care and daily tasks refer to receiving
  help or supervision. Joint symptoms include pain, aching, stiffness, or
  swelling in or around a joint during the past 12 months.

The [PRD input questions](product_requirements.md#user-inputs) contain the longer
question text. Wording differences that could change answers are listed below.

### Health ratings

For both fields: `excellent` → `1`, `very_good` → `2`, `good` → `3`,
`fair` → `4`, `poor` → `5`.

### Education

`no_degree` → `1`, `ged` → `2`, `high_school_diploma` → `3`,
`associate_degree` → `7`, `bachelors_degree` → `4`, `masters_degree` → `5`,
`doctorate` → `6`, `other` → `7`.

`associate_degree` and `other` both map to the preprocessor's “Other” category;
the model cannot distinguish them. The `doctorate` choice includes PhD, MD,
and JD. Remaining choice definitions are [open](#8-education-choices).

## Boolean and missing-value rules

The form converts Yes to `true` and No to `false`. The service maps `true` to
`1` and `false` to `0` for every field marked “Boolean rule.”

The form converts each checklist's selected items into separate Boolean fields.
Under the current plan, if any item is selected, it becomes `true` and unchecked
items in that checklist become `false`. Empty checklists and a possible “None
of these” choice remain [open](#5-checklists-and-missing-answers).

In the current schema, an omitted optional field or explicit JSON `null` both
produce Python `None`. The planned service maps these missing values to
`numpy.nan` for the preprocessor to impute. Missing or `null` required inputs
are validation errors. The omission policy for condition and limitation fields
is still [under review](#5-checklists-and-missing-answers).

API callers must use exact categorical values, integers for age and family
size, and JSON Booleans for yes/no answers. The schema rejects unknown fields,
wrong types, and invalid values; for example, `"Very Good"`, `"42"`, and
`"false"` are not substitutes for `"very_good"`, `42`, and `false`.

## Open decisions and definitions to review

These items are unresolved. The table records the current schema and planned
behavior; it does not settle the choices below.

### 1. Age and forecast year

**Decide:** How the forecast year is selected. The planned subtraction gives
age at the end of that year. Confirm whether someone turning 18 during the
forecast year is eligible even if they are under 18 when using the form.

**Review:** The form caps calculated ages at 85, while the API rejects ages
above 85. Decide whether API callers must cap ages themselves or the interface
should accept and cap older ages. The current API does not perform that conversion.

### 2. Family size

**Review:** The planned form collapses all sizes of eight or more to `8`, while
the API accepts exact sizes through 14. Confirm whether this difference is
intentional and how income ranges should work for larger families. Separately,
decide whether the API should allow sizes above the training maximum of 14.

### 3. Family income

**Decide:** The income reference period, poverty-line source and reference year,
and any location-dependent rules. Finalize exact dollar boundaries and display
rounding, including the treatment of families shown as “8+”.

The [draft ranges](technical_specifications.md#draft-family-income-ranges)
require verification and contain gaps: for example, one range ends at $18,100
and the next starts at $18,200. Final ranges must classify every amount without
gaps or overlaps.

**Decide:** Whether API callers continue to provide a category or instead
provide an income amount, and which supporting inputs that conversion needs.
The category-to-code mapping is specified; the dollar-range-to-category
conversion is not final.

### 4. Employment and recent life transition

**Review:** Match the questions to the trained features. In
[training preparation](../../notebooks/1_eda_and_preprocessing.py), employment
codes for “Job to Return To” and “Job in Ref Period” map to not employed and
also set the life-transition flag. Someone temporarily away from work may
instead choose “Employed” in an ordinary two-choice question.

**Decide:** Which marital and employment events count as a recent transition,
the lookback period, and the final question wording. The notebook suggests
12 months, but the current PRD leaves the period open. A general question about
any change may not match the specific training categories. Keep the transition
answer separate from current marital and employment status.

### 5. Checklists and missing answers

**Decide:** How to distinguish an unanswered checklist from an explicit answer
that none apply. If adding “None of these,” specify how it interacts with other
selections. Confirm whether selecting one item means all unchecked items are
known No answers, or whether users need a way to leave individual items unknown.

**Decide:** Whether omitted API condition and limitation fields retain the
current meaning of unanswered or default to No, and whether explicit `null`
should behave differently. Keep the form conversion, schema defaults, and
service mapping consistent with that decision.

### 6. Limitations and symptoms wording

**Review:** The PRD asks about serious difficulty walking or climbing stairs;
the schema also includes lifting and bending without the same severity wording.
The PRD's cognitive question asks about serious difficulty concentrating,
remembering, or deciding; the schema includes confusion and supervision for
safety. Check the training-variable definitions and align the form, schema
descriptions, and table labels before implementing these questions.

### 7. Time reference

**Decide:** Whether status questions refer to today, the beginning of the
forecast year, or expected circumstances during that year. The specs require
beginning-of-year alignment, but several draft questions ask about current
status. Clarify insurance, employment, marital status, and health in particular.
Keep explicit periods such as “ever diagnosed” and “past 12 months” visible,
and define the date from which a lookback period is measured.

### 8. Education choices

**Clarify:** How users with some college but no degree, or vocational
qualifications, should answer. Check these definitions against the trained
categories and confirm the associate-degree-to-“Other” mapping before finalizing
the help text. The existing mapping is documented above, not changed here.

### 9. Insurance wording

**Clarify:** Provide examples that help users distinguish private coverage,
public-only coverage, and no coverage. The combined-coverage rule is already
defined: any private coverage takes precedence. Resolve the coverage period
with [decision 7](#7-time-reference), including how users with changing coverage
should answer.

### 10. Required inputs

**Review:** The request schema requires age, sex, insurance, family size, and
education. The [pipeline constants](../../src/constants.py) instead mark age,
sex, insurance, region, and physical health as required. The notebook creates
the preprocessor with `strict=False`, allowing missing values to be imputed.
Confirm whether this difference is intentional before changing either list;
the table's asterisks follow the current request schema.

## Direct Python inputs

Supply a DataFrame with all 27 columns above, even when some values are missing;
use `numpy.nan` for those values. Direct Python use bypasses `PredictionRequest`
validation and the service mappings.
Nominal and binary inputs also accept the labels in
[src/constants.py](../../src/constants.py), regardless of case. Health ratings
and family income require numerical codes.
