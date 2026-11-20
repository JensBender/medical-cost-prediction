# Prediction Input Mapping

This document traces each of the 27 inputs the model needs, from the question
on the web form, to the API field, to the column the model’s preprocessing
step (the preprocessor) reads. Use it to check that the form, the API, and the
model all mean the same thing by each answer.

> **Status:** The API fields and validation rules are in place. The form
> conversions and prediction service mappings are planned.

## How the inputs fit together

```text
Web form → API (JSON) → Prediction service → Preprocessor + model
```

The planned web form converts answers to API values. API callers send those
values as JSON. Both validate requests against the same Pydantic schema,
[`PredictionRequest`](../../app/schemas.py).

The planned prediction service takes a validated request from either source,
converts it to preprocessor inputs, and calls `CostPredictor` to run the
preprocessor and model. [Direct Python callers](#direct-python-inputs) supply
those inputs themselves.

## Input mapping

How to read the table:

- An asterisk (\*) marks a required input field (age, sex, insurance status,
  family size, education).
- **As selected** means the form sends the exact value for the selected label.
  For example, “Very Good” becomes `very_good`.
- **`true` → `1`, `false` → `0`** is the yes/no rule. It also covers checklist
  items. See the [Boolean and missing-value rules](#boolean-and-missing-value-rules).
- **⚠ Q1, Q2, …** link to open questions. The
  [time reference](#6-time-reference) (Q6) also needs review across the form.

| Web form input | How the form asks | API field | Prediction service mapping | Preprocessor column |
| --- | --- | --- | --- | --- |
| Year of birth* | Subtract the birth year from the current year. | `age` | [Cap at 85](#age-validation). | `AGE23X` |
| Sex* | As selected | `sex` | `female` → `0`, `male` → `1` | `SEX` |
| Insurance status* | People with both private and public coverage choose `any_private`. [⚠ Q7](#7-insurance-wording) | `insurance_status` | `any_private` → `1`, `public_only` → `2`, `uninsured` → `3` | `INSCOV23` |
| Family size* | The selected count, or `8` for “8+”. [⚠ Q1](#1-family-size) | `family_size` | Used as entered (1–14) | `FAMSZE23` |
| Education* | No Degree<br>GED<br>High School Diploma<br>Associate's Degree<br>Bachelor's Degree<br>Master's Degree<br>Doctorate or Professional (MD, JD, etc.)<br>Other | `education` | Associate’s degree and Other both map to `7`. | `HIDEG` |
| State | Look up the state’s Census region | `region` | `northeast` → `1`, `midwest` → `2`, `south` → `3`, `west` → `4` | `REGION23` |
| Physical health | As selected | `physical_health` | [Health rating codes](#health-ratings) | `RTHLTH31` |
| Mental health | As selected | `mental_health` | [Health rating codes](#health-ratings) | `MNHLTH31` |
| Family income range | Dollar range → poverty-line category. [⚠ Q2](#2-family-income) | `family_income` | [Income codes](#family-income) | `POVCAT23` |
| Marital status | As selected | `marital_status` | [Marital status codes](#marital-status) | `MARRY31X_GRP` |
| Employment status | As selected. [⚠ Q3](#3-employment-and-recent-life-transition) | `employment_status` | `employed` → `1`, `not_employed` → `0` | `EMPST31_GRP` |
| Recent life transition | Separate yes/no question. [⚠ Q3](#3-employment-and-recent-life-transition) | `recent_life_transition` | `true` → `1`, `false` → `0` | `RECENT_LIFE_TRANSITION` |
| Usual source of care | Yes/no | `usual_source_of_care` | `true` → `1`, `false` → `0` | `HAVEUS42` |
| Current cigarette smoking | Yes/no | `smoker` | `true` → `1`, `false` → `0` | `ADSMOK42` |
| High blood pressure | Conditions checklist | `hypertension` | `true` → `1`, `false` → `0` | `HIBPDX` |
| High cholesterol | Conditions checklist | `high_cholesterol` | `true` → `1`, `false` → `0` | `CHOLDX` |
| Diabetes | Conditions checklist | `diabetes` | `true` → `1`, `false` → `0` | `DIABDX_M18` |
| Coronary heart disease | Conditions checklist | `coronary_heart_disease` | `true` → `1`, `false` → `0` | `CHDDX` |
| Stroke | Conditions checklist | `stroke` | `true` → `1`, `false` → `0` | `STRKDX` |
| Cancer | Conditions checklist | `cancer` | `true` → `1`, `false` → `0` | `CANCERDX` |
| Arthritis | Conditions checklist | `arthritis` | `true` → `1`, `false` → `0` | `ARTHDX` |
| Asthma | Conditions checklist | `asthma` | `true` → `1`, `false` → `0` | `ASTHDX` |
| Help with personal care | Limitations and symptoms checklist | `receives_personal_care_help` | `true` → `1`, `false` → `0` | `ADLHLP31` |
| Help with daily tasks | Limitations and symptoms checklist | `receives_daily_tasks_help` | `true` → `1`, `false` → `0` | `IADLHP31` |
| Walking limitation | Limitations and symptoms checklist. [⚠ Q5](#5-limitations-and-symptoms-wording) | `walking_limitation` | `true` → `1`, `false` → `0` | `WLKLIM31` |
| Cognitive limitation | Limitations and symptoms checklist. [⚠ Q5](#5-limitations-and-symptoms-wording) | `cognitive_limitation` | `true` → `1`, `false` → `0` | `COGLIM31` |
| Joint symptoms in the past 12 months | Limitations and symptoms checklist | `joint_pain` | `true` → `1`, `false` → `0` | `JTPAIN31_M18` |

### Age validation

The shared request schema accepts integer ages from 18 to 125, inclusive.
This range also applies to ages calculated from birth years in the planned
web form. The prediction service will cap accepted ages at 85 before passing
them to the preprocessor.

### Form details

- **State.** Offer all 50 states and DC. Map each to its [Census region](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf).
- **Family size.** Count yourself and anyone in your home related to you by
  blood, marriage, or adoption. (Open question: [Q1](#1-family-size).)
- **Family income.** Clear the selected income range when family size changes.
- **Conditions checklist.** Ask whether a doctor or other health professional
  has ever diagnosed each condition.
- **Limitations and symptoms checklist.** Help with personal care or daily
  tasks includes supervision. Joint symptoms include pain, aching, stiffness,
  or swelling in or around a joint in the past 12 months. (Open question:
  [Q5](#5-limitations-and-symptoms-wording).)

See the [product requirements](product_requirements.md#user-inputs) for the full questions.

### Family income

The form shows dollar ranges, which the system converts to a share of the
poverty line. The API currently accepts the categories below. The [dollar
ranges and API input format](#2-family-income) are still under review.

| Family income as a share of the poverty line | API value | Preprocessor code |
| --- | --- | --- |
| Under 100% | `poor` | `1` |
| 100% to under 125% | `near_poor` | `2` |
| 125% to under 200% | `low_income` | `3` |
| 200% to under 400% | `middle_income` | `4` |
| 400% or more | `high_income` | `5` |

### Health ratings

Physical and mental health use the same codes.

| Rating | API value | Preprocessor code |
| --- | --- | --- |
| Excellent | `excellent` | `1` |
| Very good | `very_good` | `2` |
| Good | `good` | `3` |
| Fair | `fair` | `4` |
| Poor | `poor` | `5` |

### Marital status

| Status | API value | Preprocessor code |
| --- | --- | --- |
| Married | `married` | `1` |
| Widowed | `widowed` | `2` |
| Divorced | `divorced` | `3` |
| Separated | `separated` | `4` |
| Never married | `never_married` | `5` |

## Boolean and missing-value rules

For yes/no questions, the form sends `true` for Yes and `false` for No.
The service maps `true` to `1` and `false` to `0` for all Boolean fields.

Each checklist item has its own Boolean field. If someone checks at least one
item, the form sends `true` for each checked item and `false` for the rest of
that checklist. How to handle an empty checklist or a “None of these” choice
is [still open](#4-checklists-and-missing-answers).

Required fields must always be answered. The schema rejects a request if a
required field is missing or `null`. Optional fields can be left blank, and the
preprocessor fills in the gap.

Send exact category values, integers for age and family size, and JSON Booleans
for yes/no answers. For example, `"very_good"`, `42`, and `false` are valid.
The schema rejects `"Very Good"`, `"42"`, and `"false"`, as well as unknown fields.

## Open decisions and definitions to review

Each item below states where things stand, what needs deciding, and, where the
specs make it clear, why it matters.

### 1. Family size

**Where things stand:** The form sends `8` for any family of eight or more. The
API accepts exact counts up to 14, the largest family in the training data.

**To decide:**

- Whether the form should keep “8+”.
- Which income ranges families of eight or more should see.
- Whether the API should accept counts above 14.

**Why it matters:** The form clears the selected income range when family size
changes, so family size and income ranges must be decided together (see
[Q2](#2-family-income)).

### 2. Family income

**Where things stand:** The API accepts five poverty-line categories. The
dollar ranges shown on the form are still in draft.

**To decide:**

- The income period (for example, annual or monthly).
- The poverty-line source and year, and any rules that vary by location.
- The exact dollar cutoffs, and how to round them for display, including for
  families of eight or more.
- Whether API callers should keep sending a category or send a dollar amount.
  If they send an amount, what else the service needs to find the category.

**Known problem:** One [draft range](technical_specifications.md#draft-family-income-ranges)
ends at $18,100 and the next starts at $18,200, which leaves a gap. Every
income must fall into exactly one range.

### 3. Employment and recent life transition

**Where things stand:** The [training notebook](../../notebooks/1_eda_and_preprocessing.py)
counts “Job to Return To” and “Job in Ref Period” as `not_employed` and flags
both as recent transitions. Someone temporarily away from work might choose
“Employed” on the form.

**To decide:**

- How the employment question should guide people who are temporarily away
  from work.
- Which marital or job changes count as a recent transition, and how far back
  the question should go. The notebook suggests 12 months. The product
  requirements leave the period open.
- Whether the final wording captures the events used in training.

The transition question is asked separately from current marital and
employment status.

### 4. Checklists and missing answers

**Where things stand:** The API currently treats an omitted condition or
limitation as unanswered, not as No.

**To decide:**

- How the form distinguishes a skipped checklist from “None of these”, and
  whether “None of these” can be selected alongside another item.
- When someone checks one item, whether every unchecked item counts as No, or
  whether users can leave individual items unanswered.
- Whether the API should default an omitted item to No, and whether an
  explicit `null` should mean something different.

Apply the same decision to the form, schema defaults, and service mapping.

### 5. Limitations and symptoms wording

**Where things stand:** The form draft and the schema describe the same two
limitations differently.

- **Walking.** The form draft asks about **serious difficulty** walking or
  climbing stairs. The schema also mentions lifting and bending and leaves
  severity unspecified.
- **Cognitive.** The draft asks about serious difficulty concentrating,
  remembering, or making decisions. The schema also mentions confusion and
  supervision for safety.

**To decide:** Check both questions against the training definitions before
finalizing the form wording, schema descriptions, and table labels.

### 6. Time reference

**Where things stand:** The specs call for status at the start of the forecast
year, but several draft questions ask about today.

**To decide:**

- For insurance, employment, marital status, and health, whether users report
  their status today, at the start of the forecast year, or as they expect it
  to be during that year.
- What date “ever diagnosed” and “past 12 months” should count back from.
  Keep those phrases in the relevant questions.

### 7. Insurance wording

**Where things stand:** Anyone with both private and public insurance selects
`any_private`.

**To decide:**

- Which examples would help users choose between private coverage, public-only
  coverage, and no coverage.
- Which coverage to report if it changes during the year. See [time
  reference](#6-time-reference).

## Direct Python inputs

This section is for developers calling the model directly.

Pass `CostPredictor` a DataFrame with all 27 preprocessor columns. Use
`numpy.nan` for missing values. This path skips request validation and the
prediction service mappings.

Use `85` for `AGE23X` when the person is 85 or older.

For categorical columns (nominal and binary), the preprocessor accepts
numerical labels and case-insensitive string labels. For example, `REGION23`
accepts `2`, `"Midwest"`, or `"midwest"`.
See [`CATEGORY_LABELS_PIPELINE`](../../src/constants.py) for the label mappings.

Physical health, mental health, and family income take numerical labels only.

When requests come through the API, a missing optional field and JSON `null`
both become Python `None`. The planned service converts `None` to `numpy.nan`
so the preprocessor can fill in the value.
