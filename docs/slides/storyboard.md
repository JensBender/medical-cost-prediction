# Medical cost prediction: presentation storyboard

Status: eleven main slides and twenty appendix slides are generated. The remaining
appendix slides are planned below.

## Presentation context

- **Setting:** job interviews, conference talks, meetups, and other settings.
- **Audience:** a technical audience. Assume basic ML
  knowledge, but no familiarity with MEPS or U.S. healthcare costs. Explain
  survey weights, quantiles, and evaluation metrics when introducing them.
- **Emphasis:** data science, with optional emphasis on ML engineering.
- **Language:** English.
- **Length:** 10–12 minutes of speaking, excluding questions. The current
  allocation below totals 11:25, leaving room for transitions.
- **Structure:** a cover, ten content slides, and twenty-three appendix topics.
  Appendix slides follow their first reference in the main presentation;
  unreferenced slides sit with their related topic.
- **Slide numbering:** Display 2–11 on the main content slides, with the cover
  unnumbered, and A1, A2, etc. in the appendix. Use 1–11 in the storyboard.
- **Appendix navigation:** Main slides link to relevant appendix topics;
  appendix slides have a “Back” link to the referring main slide.
- **Appendix speaker notes:** Leave main speaking points unlabelled. Use a standalone
  “For questions:” heading before reference material for follow-up questions.
- **Purpose:** show how the project connects a user need to modeling choices,
  evidence, limitations, and implementation decisions.
- **Project status:** model training, selection, and evaluation for the MVP are
  complete. Web app and API development are next.

## Central story

> The project aims to help U.S. adults plan next year's out-of-pocket healthcare
> costs using information they can provide easily without medical records.
> Out-of-pocket costs vary widely and are difficult to predict, so budgeting
> guidance needs to convey uncertainty alongside a typical estimate. That need
> shaped model selection: XGBoost quantile regression provides a plan-around
> estimate, typical range, and safety cushion.

The key modeling tradeoff was accepting a higher validation median absolute
error than Elastic Net to provide useful uncertainty ranges. Better point-estimate
performance by tree models in several medically complex subgroups also supported
the choice of XGBoost. Model selection considered which estimates and performance
criteria served the intended budgeting use case.

## Source material

| Source | Use in the presentation |
| --- | --- |
| [README](../../README.md) | Project overview, main narrative, headline results, and current status |
| [EDA/preprocessing notebook](../../notebooks/1_eda_and_preprocessing.ipynb) | Findings, diagnostics, and rationale for cleaning, feature, and outlier decisions; detail for speaker notes and appendix slides |
| [Modeling notebook](../../notebooks/2_modeling.ipynb) | Model comparisons, alternatives considered, selection rationale, evaluation, subgroup audits, and explainability; detail for speaker notes, appendix slides, and technical questions |
| [Product requirements](../specs/product_requirements.md) | Intended users, budgeting needs, product scope, and success criteria |
| [Technical specifications](../specs/technical_specifications.md) | Intended architecture, API behavior, and monitoring design |

## Main presentation at a glance

The first three slide timings are rehearsed; the remaining timings are estimates.

| ID | Proposed slide title | Time | Purpose |
| --- | --- | ---: | --- |
| 1 | Medical cost prediction for annual budgeting | 0:10 | Cover and brief orientation |
| 2 | How much should I set aside for healthcare? | 0:45 | Establish the budgeting need and intended value |
| 3 | MEPS links accessible inputs to observed spending | 2:00 | Explain the survey, project data, and population weights |
| 4 | Out-of-pocket costs: 20% of adults account for ~80% | 1:00 | Show the data challenge that shaped modeling |
| 5 | Preprocessing pipeline | 1:00 | Prepare model-ready inputs and preserve consistent transformations |
| 6 | Model selection: median error was not enough | 1:30 | Explain the key model-selection tradeoff |
| 7 | Quantile regression: budgeting for uncertainty | 1:30 | Explain the final model and its estimates |
| 8 | XGBoost quantile audit: all release gates passed | 1:30 | Present held-out evidence and baseline comparisons |
| 9 | Example prediction output (profile with higher cost risk) | 0:35 | Show the main estimates and planning guidance |
| 10 | Example prediction output (profile with higher cost risk) | 0:25 | Show expanded explanations and comparisons |
| 11 | Model evaluation is complete; app development comes next | 1:00 | Show engineering work and close with priorities |
| | **Total** | **11:25** | |

## Main slides

### 1 — Medical cost prediction for annual budgeting

**Purpose:** Introduce the project and presenter before the problem statement.

**On the slide**

- Title: Medical Cost Planner (in the project header)
- Subtitle: Predicting Out-of-Pocket Healthcare Costs with Machine Learning
- Presenter: Jens Bender
- Event or setting: [presentation setting]
- Date: [presentation date]

**Visual:** [Project header](../../assets/header.png) above the subtitle and
presenter details.

**Speaker notes — 0:10 (rehearsed)**

“This project aims to help U.S. adults plan for next year's out-of-pocket
healthcare costs using machine learning.”

**Transition:** Introduce the budgeting problem and why it matters.

**Source:** [README: motivation](../../README.md).

### 2 — How much should I set aside for healthcare?

**Takeaway:** People need an estimate of next year's out-of-pocket
spending to plan their budget and HSA/FSA contributions.

**On the slide**

- **The challenge:** Difficult to plan next year's out-of-pocket costs and HSA/FSA
  contributions.
- **The aim:** A useful ballpark estimate from questions people can answer from
  memory.

**Visual:** Wide [budgeting illustration](assets/budget-planning.png) above
two columns: challenge and aim.

**Appendix link:** [U.S. healthcare cost explainer (A1)](#a1--us-healthcare-costs-out-of-pocket-spending-and-hsafsa-planning).

**Speaker notes — 0:45 (rehearsed)**

“How much should I set aside for healthcare next year? People in the U.S. face
this question even with health insurance, because they still pay some costs out
of their own pocket. Some also need to decide how much to contribute to an HSA
or FSA, accounts that offer tax benefits for setting aside money for healthcare.

But how do you know how much healthcare you're going to need next year? Even
for planned care, working out what you'll actually pay can be cumbersome.

This project aims to provide a quick and useful ballpark estimate for next
year's out-of-pocket costs, using questions people can answer from memory,
without researching procedure costs or digging through medical records.”

**Transition:** Explain the dataset that makes this target measurable.

**Sources:** [README: motivation](../../README.md);
[product requirements: user needs and UX rationale](../specs/product_requirements.md);
[HealthCare.gov: HSA overview](https://www.healthcare.gov/high-deductible-health-plan/);
[HealthCare.gov: FSA overview](https://www.healthcare.gov/have-job-based-coverage/flexible-spending-accounts/).

### 3 — MEPS links accessible inputs to observed spending

**Takeaway:** MEPS links accessible personal information to annual out-of-pocket
spending, with survey weights supporting population-level training and evaluation.

**On the slide**

- **MEPS:** Medical Expenditure Panel Survey
- **Data:** 2023 Household Component (HC-251)
- **Sample:** 14,768 representing ~260 million U.S. adults
- **Target:** Annual out-of-pocket healthcare costs
- **Features:** 26 inputs across demographics, socioeconomics, health profile,
  chronic conditions, and limitations
- **Survey weights:** Used in EDA, model training, and evaluation to reflect
  the population

**Visual:** Bullet list beside a [household survey illustration](assets/household-survey.png).

**Appendix links:** [MEPS overview (A2)](#a2--meps-survey-overview) and
[Features (A3–A4)](#a3--preprocessor-input-features).

**Speaker notes — 2:00 (rehearsed)**

“To do so, I used data from the Medical Expenditure Panel Survey, or MEPS.
MEPS is a nationally representative survey and a leading data source for
U.S. healthcare costs. Households complete five interviews over two years.
To improve cost estimates, MEPS also collects payment records
directly from providers and pharmacies.

I chose out-of-pocket costs as the target because that is what people need to
budget for themselves. This includes copays, deductibles, and uncovered services,
but excludes insurance premiums.

I used the 2023 household data for over 14,000 U.S. adults, representing about 260
million adults in the population. I selected 26 features from over 1,000 variables,
including age, insurance status, family income, joint pain, and high cholesterol.
My selection prioritized information people can provide from memory, prioritizing
variables measured early in the year to reduce data leakage and those with predictive
value supported by the healthcare cost literature.

MEPS oversamples some groups, such as Hispanic households, to get more reliable
estimates. Each person has a survey weight showing how many people they represent
in the population. These weights account for selection probabilities and
nonresponse. I use them in data exploration, model training, and evaluation so the
results reflect the population rather than the sample’s composition. Otherwise,
oversampled groups would have disproportionate influence.”

**Transition:** Show why the spending distribution makes this task difficult.

**Sources:** [README: data and preprocessing](../../README.md);
[EDA notebook script: survey weights and data splitting](../../notebooks/1_eda_and_preprocessing.py);
[preprocessing script](../../scripts/preprocess.py);
[AHRQ: MEPS overview](https://www.ahrq.gov/data/meps.html);
[AHRQ: separate survey components](https://www.ahrq.gov/cpi/about/otherwebsites/meps.ahrq.gov/index.html);
[MEPS: interview design](https://meps.ahrq.gov/survey_comp/hc_data_collection.jsp);
[MEPS 2023 methodology: oversampling](https://meps.ahrq.gov/data_files/publications/annual_contractor_report/MEPS-Methodology-Report-2023.html);
[MEPS: medical provider follow-up and authorization](https://meps.ahrq.gov/communication/participants/faq_mpc.shtml);
[MEPS: pharmacy follow-up and authorization](https://meps.ahrq.gov/communication/participants/faq_pharm.shtml);
[MEPS 2023 documentation: expenditure construction, sampling, and weights](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.shtml).

**Detail for questions:** The target variable, `TOTSLF23`, sums self/family
payments across healthcare services, including prescriptions. It excludes
insurance premiums and over-the-counter medicines. Provider and pharmacy data
supplement household reports for selected services; not every payment is
independently verified. MEPS edits inconsistent reports and imputes missing
expenditures before aggregating the annual total.

See A3–A4 for preprocessor input features, counts, and possible feature reduction,
and A21 for input timing. Preprocessing is fitted on training data. The 80/10/10 split is
random within spending groups, with a separate zero-cost group and finer bins
for high costs to reduce imbalances in rare, expensive cases. Discuss the split
after the spending-distribution slide if asked.

### 4 — Out-of-pocket costs: 20% of adults account for ~80%

**Takeaway:** Typical errors and errors on expensive years describe different
parts of model performance.

**On the slide**

- **22.3%** of the weighted adult population has zero out-of-pocket spending
- The highest-spending **20% accounts for 79.3%** of spending
- The highest-spending **1% accounts for 20.6%** of spending
- **Primary metric:** MdAE
- **Supporting:** MAE

**Visual:** Large [Lorenz curve](../../figures/eda/lorenz_curve.png) with the
primary and supporting metrics in the left margin.

**Appendix links:** [Outlier analysis (A5)](#a5--outliers-greater-health-burden-and-higher-cost-risk),
[Distributions (A6–A8)](#a6--distributions-numerical-features), and
[Feature–target relations (A9–A12)](#a9--featuretarget-relationships-correlations).

**Speaker notes — 1:00**

“Out-of-pocket costs present two challenges: about 22% of adults have no costs
for the entire year, while costs are heavily concentrated among the top spenders.
The highest-spending 20% account for almost 80% of costs, and the top 1% account
for about 21%. High costs could reflect ongoing care or unexpected events such
as hospitalization. Unexpected events make costs especially difficult to predict.

Large errors on high-cost cases can strongly influence mean absolute error.
Therefore, I chose median absolute error as the primary evaluation metric to
describe a typical prediction error, and complemented it with mean absolute
error as a supporting diagnostic.

Outlier profiling revealed greater health burden and a higher likelihood of
extreme costs. Outlier profiles were plausible, so I retained them to preserve
valuable training information.

This distribution also informed a stratified data split. I used 80% training, 10%
validation, and 10% test, stratified by out-of-pocket costs to preserve the
proportions of zero-cost and high-cost cases across splits.”

**Transition:** Before comparing models, explain how the data became model-ready.

**Sources:** [EDA notebook script: zero costs and cost concentration](../../notebooks/1_eda_and_preprocessing.py);
[README: EDA and outlier analysis](../../README.md).

### 5 — Preprocessing pipeline

**Purpose:** Explain data preparation and the reusable preprocessing pipeline.

**On the slide:** [Preprocessing pipeline PNG](assets/preprocessing-pipeline.png),
rendered from the [SVG source](../../assets/preprocessing_pipeline.svg).

**Visual:** Full-slide diagram on a matching dark navy background.

**Appendix link:** [Data preparation (A13)](#a13--data-preparation).

**Speaker notes — 1:00**

“I fitted this scikit-learn pipeline on the training data and applied it to
validation and test data. It standardizes categorical labels, checks and imputes
missing values, derives medical features, scales numerical features, and
one-hot encodes nominal features.

The fitted pipeline is saved with joblib for reuse during prediction. Both the
preprocessor inputs and model-ready datasets are saved as Parquet files.”

**Transition:** With the data prepared, compare the models and their usefulness
for budgeting.

**Sources:** [README: data preprocessing](../../README.md#-data-preprocessing);
[preprocessing script](../../scripts/preprocess.py);
[preprocessing pipeline](../../src/pipeline.py).

### 6 — Model selection: median error was not enough

**Takeaway:** The model with the best typical error offered limited separation
between lower- and higher-cost profiles.

**On the slide**

| Tuned model | MdAE | Overfitting | MAE |
| --- | ---: | ---: | ---: |
| Elastic Net | **$159** | +7.9% | $1,051 |
| Random Forest | $228 | **+3.8%** | $964 |
| XGBoost | $242 | +6.2% | **$954** |

- **Elastic Net:** Lowest MdAE, but never predicts over ~$217 and increasingly underpredicts.
- **Choose XGBoost:** Predicts up to ~$2,100 with less systematic underprediction.
- **Switch to quantile regression:** Errors generally widen at higher predicted costs,
  so add budgeting ranges to communicate that uncertainty.

**Footnote:** “Note: Survey-weighted validation metrics. Overfitting is the percentage
change in MdAE from training to validation.”

**Visual:** Comparison table followed by the three model-selection points.

**Appendix links:** [Baseline models (A14)](#a14--baseline-models) |
[Hyperparameter tuning (A15)](#a15--hyperparameter-tuning) |
[Heteroscedasticity (A16)](#a16--heteroscedasticity) |
[Subgroup reliability (A17)](#a17--subgroup-reliability).

**Speaker notes — 1:30**

I compared six model architectures and selected Elastic Net, Random Forest,
and XGBoost for hyperparameter tuning. This table shows their results after
tuning. Elastic Net achieved the lowest median absolute error. However, its
validation predictions never exceeded about $217, and the higher its predicted
costs, the more it tended to underestimate actual costs.

XGBoost's predictions, on the other hand, reached about $2,100, and its median
residuals generally stayed closer to zero. It also had the lowest mean absolute
error and lower median error than Elastic Net in key subgroups, such as people
with poor physical health. So I chose XGBoost as the more promising model for
planning for higher-cost years.

But even a more useful central estimate doesn’t show how widely someone’s
costs could vary. I initially focused on typical prediction error, but residual
analysis showed that errors were generally more spread out at higher predicted
costs. A single estimate cannot communicate that uncertainty.

For budgeting, this matters because someone could plan around the central
estimate and still be underprepared for a higher-cost year. So I decided to
switch to quantile regression in XGBoost.

**Transition:** Map the quantiles directly to the intended budgeting estimates.

**Sources:** [README: hyperparameter tuning and model decision](../../README.md);
[modeling notebook script](../../notebooks/2_modeling.py).

**Comparison rule:** All values on this slide are validation results for tuned
point-estimate models. Do not compare $159 here with the final model's $240 test
result as if they came from the same evaluation.

### 7 — Quantile regression: budgeting for uncertainty

**Takeaway:** Go beyond a typical estimate: show how costs could vary and
provide a safety cushion for a higher-cost year.

**On the slide**

| Estimate | Model quantity | Intended meaning |
| --- | --- | --- |
| Plan-around estimate | q50 | Predicted median cost |
| Typical range | q25–q75 | Central range targeting 50% coverage |
| Safety cushion | q90 | Upper planning reference targeting 90% coverage |

**Visual:** Quantile markers on a cost axis, with a curved brace linking the typical range
from q25 to q75.

**Speaker notes — 1:30**

“Quantile regression allowed me to give users three estimates based on four
predicted quantiles to help them budget for uncertainty.
The plan-around estimate is the predicted median. It represents the typical costs
for people with a similar profile. The typical range runs from the 25th to the
75th percentile and shows how much costs typically vary. The safety cushion is
the 90th percentile, which provides a conservative estimate to help
users budget for a higher-cost year.

To evaluate these estimates, I checked coverage and mean width. Coverage measures
how often actual costs fall within the typical range or the safety cushion. Width
matters too, because a very wide range could achieve high coverage but be too
broad to be useful for budgeting. The typical range targets 50% coverage, while
the safety cushion targets 90%. The goal is coverage close to these targets
without making the ranges too wide.”

For questions:

Safety cushion: Q90 targets 90% coverage, not a maximum bill. Q95 or q99 would
suggest larger reserves, which may be less practical and discourage users from
following the guidance. Q95/q99 estimates may also be less stable because they
rely on sparser tail data.

Hyperparameters: The model reuses the best tuned XGBoost point-estimate
hyperparameters, switching the objective from absolute error to quantile error
for q25, q50, q75, and q90.

Training: Uses survey weights and log-transformed costs, ln(1 + cost), also called
log1p. Adding one makes the log transformation defined for zero costs.
Predictions are converted back to dollars before postprocessing.

Postprocessing: Clips negative predictions to zero and enforces quantile order,
so higher percentiles never predict lower costs. On the test set, postprocessing
had negligible impact.
Clipping negative predictions affected 5.7% of the survey-weighted test
population, by at most 41 cents. Enforcing quantile order affected 0.24%,
by at most one cent.

**Transition:** Show whether the final model delivers those estimates on held-out data.

**Sources:** [README: final model](../../README.md);
[quantile training script](../../scripts/train_xgboost_quantile.py);
[prediction postprocessing](../../src/prediction.py);
[modeling notebook: postprocessing checks](../../notebooks/2_modeling.ipynb);
[scikit-learn: estimating extreme quantiles](https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html).

### 8 — XGBoost quantile audit: all release gates passed

**Takeaway:** The final model meets the project's overall performance gates,
with its strongest evidence of added value in interval and upper-quantile scores.

**On the slide**

| Test metric | Estimate (95% CI) | Release gate |
| --- | ---: | --- |
| Plan-around estimate (q50): MdAE | **$240** [$215, $279] | < $500 |
| Typical range (q25–q75): coverage | **47.3%** [44.0%, 50.6%] | 45%–55% |
| Safety cushion (q90): coverage | **91.0%** [89.2%, 92.6%] | 85%–95% |
| Typical range (q25–q75): mean width | **$912** [$875, $955] | < $1,500 |
| Safety cushion (q50–q90): mean width | **$2,032** [$1,964, $2,108] | < $3,500 |

**Table note:** “Note: Survey-weighted test metrics. 95% percentile bootstrap confidence
intervals from 1,000 resamples of test-set rows.”

- **Value over population and age-group baselines:** better typical range and safety cushion.
- **Subgroup reliability and fairness:** add planning notes for groups at risk of underbudgeting.

**Visual:** Results table above the baseline comparison and subgroup summary.

**Appendix links:** Value over simple baselines (A18) | Subgroup reliability (A19) | Subgroup fairness (A20).

**Speaker notes — 1:30**

“To make those goals measurable, I defined release gates, which are pass-fail
criteria for the minimum test-set performance required to launch.

The model passed all release gates. For about half of people, plan-around
estimates were within $240 of actual costs. The typical range covered close to
half of actual outcomes, while the safety cushion covered about 90%, close to
their intended targets.

On average, the typical range spanned about $900, and the safety cushion added
about $2,000 above the plan-around estimate. Both stayed below the release limits.

To assess added value, I compared XGBoost's quantile estimates with two simple
baselines. The population baseline gives everyone the same estimates, while the
age-group baseline gives everyone within the same age group the same estimates.
XGBoost's clearest added value was in the typical range and safety cushion. For
the plan-around estimate, median absolute error remained close to the population
baseline.

Subgroup reliability and fairness checks showed coverage close to its targets
for most groups. The main limitation was underprediction of very high actual
costs. Typical-range undercoverage also affected people with poor mental health,
low income, or a doctorate degree. I added planning notes for groups at risk
of underbudgeting, suggesting they plan closer to the safety cushion.”

For questions:

Release gate rationale: Ideally, user feedback would inform thresholds based on
budgeting needs and tolerance for errors. Without that feedback, I chose
thresholds that seemed reasonable before examining the test results. They are
not universal standards, but at least establish a measurable minimum
performance level.

**Transition:** Show how the findings translate into a user-facing prediction output.

**Sources:** [README: release gates](../../README.md);
[modeling notebook: test comparison with simple baselines](../../notebooks/2_modeling.py).
Detailed baseline comparisons and scoring definitions are in A18.

### 9 — Example prediction output (profile with higher cost risk)

**Takeaway:** The estimates and planning note help users prepare for a
higher-cost year, with explanations and comparisons available to expand.

**On the slide**

**Title detail:** (profile with higher cost risk: 69 years, arthritis)

**Your estimated out-of-pocket costs for next year**

| Plan around | Typical range | Safety cushion |
| --- | --- | --- |
| $895 | $400–$2,025 | $3,852 |

Plan around gives you an estimate of your typical costs. The typical range shows
how much those costs usually vary. The safety cushion helps you budget for a
higher-cost year. Costs can still be higher.

**Planning note:** Costs for profiles like yours can vary a lot. To reduce the
risk of underbudgeting, consider planning closer to the safety cushion.

- **Which answers shaped your estimate?** (Click to expand)
- **How you compare to others** (Click to expand)

**About this estimate:** This is a planning estimate, not a bill estimate. It is
based on 2023 national survey data and adjusted to current dollars. It does not
include premiums, over-the-counter costs, family totals, or procedure prices.
New diagnoses, accidents, hospitalizations, and plan-specific billing details
can make actual costs higher.

**Visual:** The title has a smaller, muted gray parenthetical for the higher-cost
risk profile, age, and chronic conditions. A thin border encloses the
user-facing content, separating it from the title and slide footer. Three
prominent estimates sit above the interpretation text and planning note.
The two collapsed section labels link to slide 10. The scope and inflation note
sits at the bottom inside the border, leaving the slide footer clear.

**Speaker notes — 0:35**

“Here is an actual test-set example for a 69-year-old with arthritis. The estimates provide a central amount, a typical range,
and extra room for a higher-cost year. This higher-cost profile receives a planning note
suggesting planning closer to the safety cushion. The scope note explains the
exclusions, and these amounts are adjusted for medical inflation.
Users can expand the explanations and comparisons.”

For questions:

Example: Test case 2817769101 has arthritis and high blood pressure. High
predicted q90 triggers `UNDERBUDGETING_RISK`. The same flag and note apply
when supplied answers indicate poor mental health, low income, or a doctorate
degree. The modeling notebook reproduces the predictions and SHAP values.

Scope: Estimates cover annual out-of-pocket costs for an individual. They
exclude insurance premiums, over-the-counter medicines, family totals, and
procedure prices.

Inflation: Predictions, population and age-group comparisons, and SHAP dollar
contributions are adjusted to May 2026 dollars using the persisted medical
inflation factor of 1.080415.

Next-year use: The test split comes from 2023, and some inputs summarize
information collected during that year. The planned next-year use case needs
feature-timing checks and later-year validation. Scope wording and planning
notices communicate limitations but do not fix calibration.

**Transition:** Open the explanations and comparisons.

**Sources:** [README: example prediction output](../../README.md);
[SHAP explanation module](../../src/explainability.py);
[medical inflation adjustment](../../scripts/update_medical_inflation.py);
[feature timing notes](../research/candidate_features.md).

### 10 — Example prediction output (profile with higher cost risk)

**Takeaway:** SHAP explains which answers moved the plan-around estimate, while
population and age-group medians provide context.

**On the slide**

**Title detail:** (profile with higher cost risk: 69 years, arthritis)

| Plan around | Typical range | Safety cushion |
| --- | --- | --- |
| $895 | $400–$2,025 | $3,852 |

**Which answers shaped your estimate?**

Contributions to the plan-around estimate:

| Your answer | Contribution |
| --- | ---: |
| Arthritis: Yes | +$124 |
| Joint pain: Yes | +$110 |
| High blood pressure: Yes | +$94 |
| Age: 69 | +$92 |
| Insurance: Any private | +$80 |

**How you compare to others**

| Comparison | Annual costs |
| --- | ---: |
| Your plan-around estimate | $895 |
| U.S. adult median | $268 |
| Age 65+ median | $657 |

**Visual:** The title, muted profile text, and border match slide 9. The three
estimates remain at the top as a reference. All five SHAP contributions appear
on the left, with the comparisons on the right. The footer leaves room for the
planned SHAP appendix link.

**Speaker notes — 0:25**

“These five contributions show how the person's answers moved the plan-around
estimate up or down. I use SHAP to calculate them through the full prediction
pipeline. They explain the plan-around estimate, not the typical range or
safety cushion. The population and age-group medians help users put their
estimate in context.”

**Transition:** Close with the implemented components and remaining application
and validation work.

**Sources:** [README: example prediction output](../../README.md);
[SHAP explanation module](../../src/explainability.py).

### 11 — Model evaluation is complete; app development comes next

**Takeaway:** The project has reusable ML components and an explicit plan for
the remaining product and validation work.

**On the slide**

| Implemented | Planned |
| --- | --- |
| DVC stages and MLflow experiment tracking | FastAPI/Gradio application |
| Shared prediction and SHAP modules; unit tests | Complete request-latency and integration checks |
| Model evaluation and application data artifacts | User evaluation and aggregate monitoring |

- Next validation priority: feature timing and performance on a later survey year
- Main lesson: evaluate the estimates needed for the user decision

**Visual:** Workflow from training artifacts through shared inference to the
application, distinguishing implemented and planned components.

**Speaker notes — 1:00**

“The project goes beyond a notebook: scripts handle preprocessing and training,
DVC tracks stages, MLflow records experiments, and shared modules provide
prediction and SHAP explanations. The FastAPI and Gradio application is still
planned. The next delivery work is to connect those components and measure
complete request latency, while the next modeling validation is to review input
timing and test on a later survey year. User testing also needs to establish
whether people understand and can use these ranges. My main lesson is that
model selection depends on the decision the estimate supports: typical error,
uncertainty, and subgroup reliability each reveal something different.”

**Close:** Invite questions with the results and limitations still easy to revisit.

**Sources:** [DVC stages](../../dvc.yaml); [prediction module](../../src/prediction.py);
[explanation module](../../src/explainability.py);
[unit tests](../../tests/unit/); [technical specifications](../specs/technical_specifications.md).

## Appendix slides

Each appendix item is an optional slide with a specific question to answer.
The remaining planned topics will be reviewed after the main slides are finalized.
Timings apply only when promoted into the main presentation.

### A1 — U.S. healthcare costs: out-of-pocket spending and HSA/FSA planning

**Question:** What is out-of-pocket spending, and how does it relate to HSA/FSA planning?

**On the slide:** [Healthcare-cost infographic](../../assets/infographic_healthcare_costs.png).

**Speaker notes**

For questions:

HSA: The account belongs to the individual and stays with them when they
change jobs. Contributions require an HSA-eligible health plan and meeting
other eligibility conditions.

FSA: This is an employer benefit. Although generally “use it or lose it,”
the employer may allow a limited carryover or a grace period. 

Out-of-pocket maximum: This usually limits payments for covered, in-network
care. In-network providers have an agreement with the insurer. Depending on
the plan, care outside that network may cost more or may not be covered.
Premiums, uncovered services, and out-of-network costs generally fall outside
the maximum.

**Transition:** Return to the problem statement slide or continue with questions.

**Detail for questions:** The target excludes premiums and over-the-counter
purchases. It covers annual costs for an individual U.S. civilian
noninstitutionalized adult. Household totals, procedure prices, and insurance
plan comparisons are outside scope.

**Promote:** After 2 when the audience needs U.S. healthcare background; allow 1:00.

**Sources:** [README: target variable](../../README.md);
[product requirements: out of scope](../specs/product_requirements.md);
[HealthCare.gov: HSA overview](https://www.healthcare.gov/high-deductible-health-plan/);
[HealthCare.gov: FSA overview](https://www.healthcare.gov/have-job-based-coverage/flexible-spending-accounts/);
[IRS: HSA ownership and eligibility](https://www.irs.gov/publications/p969);
[HealthCare.gov: out-of-pocket maximum](https://www.healthcare.gov/glossary/out-of-pocket-maximum-limit/);
[HealthCare.gov: provider networks](https://www.healthcare.gov/glossary/network/);
[HealthCare.gov: PPO coverage outside the network](https://www.healthcare.gov/glossary/preferred-provider-organization-ppo/);
[HealthCare.gov: HMO coverage outside the network](https://www.healthcare.gov/glossary/health-maintenance-organization-hmo/).

### A2 — MEPS survey overview

**Question:** How does MEPS collect its data, and which component does this project use?

**On the slide:** [MEPS infographic](../../assets/infographic_meps_data.jpg).

**Speaker notes**

For questions:

Employer survey: Collects information about employers' insurance offerings,
premiums, and contributions. It is separate from the household data used here
and does not supply this project's features or target.

Provider data: Provider and pharmacy information supplements or replaces
household-reported spending where needed. Those improvements are already
incorporated into the released household file; I did not merge provider
records myself.

Overlapping panels: The 2023 file combines the second year of one household
panel with the first year of another. Both contribute information about
costs during 2023.

Population covered: “Civilian noninstitutionalized” excludes active-duty
military personnel and people living in institutions such as nursing homes
or prisons. The project further restricts the data to adults.

**Transition:** Return to the main data slide or continue with questions.

**Detail for questions:** `PERWT23F` supplies person weights. Training, metrics,
and simple benchmarks use these weights. They do not by themselves remove
missing-feature bias, guarantee representativeness of future app users, or
account for survey clustering in uncertainty estimates.

**Promote:** After 3 for survey background; allow 1:00.

**Sources:** [AHRQ: MEPS overview](https://www.ahrq.gov/data/meps.html);
[MEPS medical provider component](https://meps.ahrq.gov/mepsweb/survey_comp/mpc.jsp);
[MEPS employer insurance component](https://meps.ahrq.gov/mepsweb/survey_comp/Insurance.jsp);
[MEPS 2023: consolidated household data](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.shtml);
[MEPS household panel design](https://meps.ahrq.gov/mepsweb/survey_comp/hc_data_collection.jsp);
[EDA notebook](../../notebooks/1_eda_and_preprocessing.py);
[modeling helpers](../../src/modeling.py).

### A3 — Preprocessor input features

**Question:** What information is supplied to the preprocessing pipeline?

**On the slide**

| Feature | Type | Possible values |
| --- | --- | --- |
| Age | Numerical | 18–85 years |
| Sex | Binary | Male, female |
| Region | Categorical | Northeast, Midwest, South, West |
| Marital status | Categorical | Married, widowed, divorced, separated, never married |
| Family income | Numerical | 1 = Poor/Negative, 2 = Near Poor, 3 = Low, 4 = Middle, 5 = High |
| Family size | Numerical | 1–14 |
| Education | Categorical | None, GED, high school, bachelor's, master's, doctorate, other |
| Employment status | Binary | Employed, not employed |
| Insurance | Categorical | Any private, public only, uninsured |
| Physical health | Numerical | 1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor |
| Mental health | Numerical | 1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor |

**Visual:** Feature table; the remaining 16 yes/no inputs continue on A4.

**Speaker notes**

These are the interpretable preprocessor input features, which SHAP also uses.
They have gone through initial data preparation, including grouping marital
and employment categories, rather than being raw MEPS variables.

For questions:

Family income and physical and mental health ratings are conceptually ordinal.
They already use ordered codes from 1 to 5, which the pipeline processes as
numerical features with median imputation and standard scaling. An ordinal
encoder preserving that order would only shift the codes to 0–4, giving the
same standardized values without adding information.

Family income: MEPS collects information about annual income from sources such
as wages, pensions, and benefits. Income is summed across family members, then
divided by the poverty threshold for that family's size and composition to
assign one of five income categories.

Family size generally counts related people living in the same household.
It can also include unmarried partners and college students living away
from home.

**Transition:** Continue to the yes/no inputs or return to the MEPS data slide.

**Detail for questions:** The 26 survey inputs become 27 preprocessor inputs
after adding a life-transition feature and 40 model-ready columns after
transformation. These counts do not equal the number of questions on the planned
form: related conditions can be grouped, and some values are derived. User testing
will assess the burden; feature reduction is an option if fewer inputs would
improve the experience. Refer to A21 for input timing.

**Sources:** [README: feature details](../../README.md#feature-details);
[input definitions](../../src/constants.py);
[data preparation](../../scripts/preprocess.py);
[product requirements: input form](../specs/product_requirements.md);
[MEPS 2023: income construction and family definitions](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.shtml).

### A4 — Preprocessor input features: yes/no

**Question:** Which conditions, limitations, and other yes/no inputs are used?

**On the slide**

| Feature group | Features |
| --- | --- |
| Chronic conditions | High blood pressure, high cholesterol, diabetes, coronary heart disease, stroke, cancer, arthritis, asthma |
| Limitations | ADL help: personal care, such as bathing or dressing<br>IADL help: daily tasks, such as managing bills or medication<br>Walking limitation, cognitive limitation, joint pain |
| Other | Smoker<br>Usual source of care: regular doctor or clinic<br>Recent life transition (derived): marital or employment changes |

**Visual:** Feature table grouped into chronic conditions, limitations, and other inputs.

**Speaker notes**

For questions:

Chronic conditions: MEPS asks about each condition separately during the
household interview. For example, “Have you ever been told by a doctor or
other health professional that you had high cholesterol?” The yes/no answers
record diagnosis history, not severity or current treatment needs.

Smoking: The feature records whether the person currently smokes cigarettes.
It comes from the self-administered questionnaire, rather than the
household interview. It does not describe smoking frequency or amount.

Recent life transition: This flag is derived from marital and employment
status. For example, “divorced” and “divorced during the survey round” are
grouped as “divorced” in the marital-status feature. The latter also
sets the recent life transition flag to yes.

**Transition:** Return to the MEPS data slide.

**Sources:** [README: feature details](../../README.md#feature-details);
[input definitions](../../src/constants.py);
[data preparation](../../scripts/preprocess.py);
[MEPS 2023: condition questions](https://meps.ahrq.gov/survey_comp/hc_survey/2023/PE_2023.htm);
[MEPS 2023: household interviews and smoking questionnaire](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.shtml).

### A5 — Outliers: greater health burden and higher cost risk

**Question:** How were outliers identified, and why were they retained?

**On the slide**

- **Detection:** Isolation Forest (flagging 5%).
- **Profiling:** Comparison table below.
- **Treatment:** Retained all outliers as plausible, valuable training examples.

**Visual:** Profiling table between the detection and treatment summaries.

| Profiling | Inliers | Outliers |
| --- | ---: | ---: |
| In top 1% of spenders | 0.9% | 3.5% |
| Age (median) | 47 | 66 |
| Chronic conditions (median) | 1 | 4 |
| Functional limitations (median) | 0 | 3 |
| Private insurance | 68% | 32% |
| Walking limitations | 9% | 75% |
| Cognitive limitations | 3% | 61% |

**Speaker notes — 1:15 (optional, estimated)**

“To detect multivariate outliers, I used an Isolation Forest configured to flag 5%
of people in the training data. I then compared survey-weighted out-of-pocket
costs and feature distributions between inliers and outliers.

Outliers were 3.9 times as likely as inliers to be among the top 1% of spenders.
They were 19 years older and had three more chronic conditions and three more
functional limitations, based on group medians. They were also much more
likely to report walking and cognitive limitations, and about half as likely
to have private health insurance.

I retained all outliers because their profiles were plausible, with no clear
evidence of data errors. Removing them could discard valuable training
information about people with greater health burden and extreme out-of-pocket
costs, limiting the model's ability to learn patterns relevant to these groups.”

**Transition:** Return to the cost-distribution slide or continue with questions.

**Detail for questions:** Percentages and medians use survey weights. The top-1%
threshold is defined across all people in the training data.

**Sources:** [README: outlier analysis](../../README.md#outlier-analysis-details);
[EDA notebook: detection and profiling](../../notebooks/1_eda_and_preprocessing.py).

### A6 — Distributions: numerical features

**Question:** How are the numerical features distributed?

**On the slide:** [Numerical distributions](../../figures/eda/numerical_distributions.png).

**Speaker notes**

Family size is right-skewed, and health ratings are concentrated around
excellent to good. These distributions supported median imputation because the
median is less affected by extreme values than the mean.

For questions:

“Complete” shows the survey-weighted percentage with nonmissing information.
Missing values can occur when people do not know an answer or decline to
answer. The plots show the data before imputation, which is done later by
the preprocessing pipeline using medians from the training data.

**Transition:** Continue to categorical distributions or return to the cost-distribution slide.

**Sources:** [README: distributions](../../README.md#distributions);
[numerical distributions](../../figures/eda/numerical_distributions.png);
[EDA notebook: distributions and completeness](../../notebooks/1_eda_and_preprocessing.py);
[preprocessing pipeline](../../src/pipeline.py);
[MEPS 2023: missing-value codes](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.shtml).

### A7 — Distributions: categorical features

**Question:** How are the categorical features distributed?

**On the slide:** [Categorical distributions](../../figures/eda/categorical_distributions.png).

**Speaker notes**

About 66% of adults had private insurance. For this majority, out-of-pocket
costs represent their personal share of care, including deductibles, copays,
coinsurance, and uncovered services.

Compared with the weighted population estimates, the sample contains larger
proportions of people with low income and lower education. This reinforces
the importance of survey weights in data exploration, model training, and
evaluation.

**Transition:** Continue to binary distributions or return to the cost-distribution slide.

**Sources:** [README: distributions](../../README.md#distributions);
[categorical distributions](../../figures/eda/categorical_distributions.png);
[EDA notebook: sample and population comparisons](../../notebooks/1_eda_and_preprocessing.py).

### A8 — Distributions: binary features

**Question:** How common are the binary feature categories?

**On the slide:** [Binary distributions](../../figures/eda/binary_distributions.png).

**Speaker notes**

Joint pain, high blood pressure, and high cholesterol are common. Severe conditions such as cancer, coronary heart disease, and stroke are less common.

**Transition:** Return to the cost-distribution slide or continue with questions.

**Sources:** [README: distributions](../../README.md#distributions);
[binary distributions](../../figures/eda/binary_distributions.png).

### A9 — Feature–target relationships: correlations

**Question:** Which features are associated with costs and with one another?

**On the slide:** [Correlation heatmap](../../figures/eda/correlation_heatmap.png).

**Speaker notes**

This heatmap shows survey-weighted Spearman rank correlations. Age has the
strongest positive association with out-of-pocket costs at 0.30, followed by
family income, arthritis, high cholesterol, and joint pain.
Family size has the strongest negative association at −0.22.

**Transition:** Continue to the numerical relationships or return to the cost-distribution slide.

**Sources:** [README: relationships](../../README.md#main-relationships);
[correlation heatmap](../../figures/eda/correlation_heatmap.png).

### A10 — Feature–target relationships: numerical features

**Question:** How do costs vary with the numerical features?

**On the slide:** [Numerical feature–target relationships](../../figures/eda/numerical_feature_target_relationships.png).

**Speaker notes**

The plots show log-transformed costs rather than dollar amounts, so very high
costs do not dominate the plots.

People of similar ages or with the same self-reported health rating can still
have very different out-of-pocket costs, ranging from zero to high costs.

**Transition:** Continue to the categorical relationships or return to the cost-distribution slide.

**Sources:** [README: numerical relationships](../../README.md#numerical-feature-target-relationships);
[numerical relationships](../../figures/eda/numerical_feature_target_relationships.png).

### A11 — Feature–target relationships: categorical features

**Question:** How do cost distributions differ across categorical feature groups?

**On the slide:** [Categorical feature–target relationships](../../figures/eda/categorical_feature_target_relationships.png).

**Speaker notes**

The plots show log-transformed costs rather than dollar amounts, so very high
costs do not dominate the plots. The dollar labels show group medians.

People with higher income, higher education, or private insurance generally
had higher out-of-pocket costs. But the plots also show substantial
variation within groups. Even among people with high income, spending ranges
from zero to very high costs.

The uninsured group has a median of zero, but some people still face very high
out-of-pocket costs. Low observed spending should not be interpreted as low
healthcare need.

**Transition:** Continue to the binary relationships or return to the cost-distribution slide.

**Sources:** [README: categorical relationships](../../README.md#categorical-feature-target-relationships);
[categorical relationships](../../figures/eda/categorical_feature_target_relationships.png).

### A12 — Feature–target relationships: binary features

**Question:** How do costs differ between the two categories of each binary feature?

**On the slide:** [Binary feature–target relationships](../../figures/eda/binary_feature_target_relationships.png).

**Speaker notes**

Arthritis was common and showed a stronger overall correlation with costs,
while cancer was less common but showed a larger difference in median
spending. Women and people with a usual source of care also had higher median
spending.

**Transition:** Return to the cost-distribution slide or continue with questions.

**Sources:** [README: binary relationships](../../README.md#binary-feature-target-relationships);
[binary relationships](../../figures/eda/binary_feature_target_relationships.png).

### A13 — Data preparation

**Question:** How was the raw survey data prepared and split for modeling?

**On the slide**

- **Prepare MEPS data:** Select variables and adults, standardize missing values,
  and group sparse categories.
- **Stratified split by out-of-pocket costs:** Keep zero-cost and rare high-cost cases
  represented across splits. Separate zero costs and use finer percentile groups
  toward the expensive tail. Randomly allocate within each group: 80% training,
  10% validation, and 10% test.

**Visual:** Preparation summary above a diagram showing zero-cost and
positive-cost groups, with finer groups toward the expensive tail.

**Speaker notes — 1:00 (optional)**

“I used a data preparation script to select the variables and adult population.
It recovers implied values from survey skip patterns and standardizes the
remaining MEPS missing-value codes. It also groups sparse marital and
employment categories, such as ‘recently divorced’.
A separate feature indicates whether a recent life transition occurred.

A purely random split could leave rare high-cost cases unevenly distributed.
So I stratified the split by out-of-pocket costs, using a separate group for
zero costs and six percentile groups for positive costs, with finer groups
toward the expensive tail. Within each group, the script randomly allocates
people to 80% training, 10% validation, and 10% test data.”

**Transition:** Return to the pipeline slide or continue with questions.

**Detail for questions:** Positive-cost boundaries are the 50th, 80th, 95th,
99th, and 99.9th percentiles, calculated without survey weights. The script first
holds out 20%, then recalculates the cost groups within that subset before
splitting it equally into validation and test data.

**Promote:** Before the pipeline slide for a longer or more technical talk;
allow 1:00.

**Sources:** [README: data preparation](../../README.md#-data-preprocessing);
[preprocessing script](../../scripts/preprocess.py);
[stratification helper](../../src/stats.py);
[EDA/preprocessing notebook](../../notebooks/1_eda_and_preprocessing.py).

### A14 — Baseline models

**Question:** Which models were compared before tuning, and why were three selected?

**On the slide**

| Model | MdAE | Overfitting | MAE |
| --- | ---: | ---: | ---: |
| **Elastic Net** | **$163** | +6.6% | $1,044 |
| Linear Regression | $219 | +4.8% | $998 |
| **Random Forest** | $232 | +9.6% | **$958** |
| *Median (benchmark)* | *$248* | *0.0%* | *$1,041* |
| Decision Tree | $271 | **+1.5%** | $971 |
| **XGBoost** | $281 | +98.0% | $961 |
| Support Vector Machine | $291 | +190.7% | $1,027 |
| *LLM (benchmark)* | *$518* | *N/A* | *$1,168* |

**Visual:** Model comparison table highlighting the three tuning candidates
and the best values in each metric column.

**Footnote:** “Note: Survey-weighted validation metrics. Overfitting is the percentage
change in MdAE from training to validation.”

**Speaker notes — 1:00 (optional)**

“I evaluated six model architectures and compared their performance with a
median benchmark and a large language model benchmark.

For most specialist models, median absolute error was around $200, but mean
absolute error was around $1,000, meaning that some predictions missed by much
more than the typical one.

Compared with Gemini 3 Flash, every specialist model had a lower median
absolute error, showing the added value of specialized models over a
general-purpose model for this task.

For hyperparameter tuning, I selected Elastic Net for its strong median
accuracy, Random Forest for the lowest mean absolute error, and XGBoost as a
promising candidate for capturing nonlinear interactions.”

**Transition:** Return to the tuned-model comparison.

**Promote:** Before the model-selection slide for a longer presentation; allow 1:00.

**Sources:** [README: baseline models](../../README.md#-baseline-models);
[modeling notebook](../../notebooks/2_modeling.py);
[baseline training script](../../scripts/train_baseline.py);
[LLM benchmark script](../../scripts/benchmark_llm.py).

### A15 — Hyperparameter tuning

**Question:** How were the three models tuned and their configurations selected?

**On the slide**

50 configurations per model; select the lowest weighted validation MdAE.

**Elastic Net**

- Squared terms: on/off
- Regularization: 0.01–1.0
- L1/L2 mix: 0–1

**Random Forest**

- Trees: 200, 300, 400
- Depth: 8–24
- Min. split size: 20–149
- Min. leaf size: 10–79
- Features: sqrt, log2, 30/50/70%
- Row sampling: 60%–100%

**XGBoost**

- Trees: 400, 600, 800
- Depth: 3–9
- Learning rate: 0.01–0.2
- Min. child weight: 1–19
- Row sampling: 60%–100%
- Feature sampling: 50%–100%
- L1 penalty: 0–5
- L2 penalty: 0–5

**Visual:** Search objective above three columns of model-specific search spaces.

**Speaker notes — about 1:15 (estimated)**

I tested 50 configurations per model and selected the one with the lowest
median absolute error on a fixed validation split. Survey weights were used
during training and evaluation. All three models used log-transformed costs
to reduce the impact of the heavy tail. I used a manual loop with
ParameterSampler to pass survey weights correctly through the pipeline and
target-transform wrapper. MLflow recorded each trial's parameters and results.

The search focused on balancing flexibility with regularization. For Elastic
Net, I explored nonlinear effects through polynomial features and varied the
strength and mix of the penalties.

For Random Forest, I varied tree depth and required larger groups at splits
and leaves to limit fitting to small, unusual groups. Row and feature
sampling added diversity between trees.

For XGBoost, I explored more, shallower trees with smaller learning steps,
alongside sampling and penalties to control overfitting.

**Transition:** Return to the tuned model results to discuss the modeling decision.

**Promote:** Before 6 for a longer modeling discussion; allow 1:15.

**Sources:** [README: tuning methodology](../../README.md);
[Elastic Net tuning script](../../scripts/tune_elastic_net.py);
[Random Forest tuning script](../../scripts/tune_random_forest.py);
[XGBoost tuning script](../../scripts/tune_xgboost.py).

### A16 — Heteroscedasticity

**Question:** What did the residual analysis reveal about the tuned models?

**Visual:** Full-slide [tuned-model residual figure](../../figures/evaluation/tuned_models_validation_heteroscedasticity.png).

**Speaker notes**

The plots show prediction errors, or residuals. Positive values mean
underprediction, and negative values mean overprediction. I divided people
into 15 groups with similar predicted costs. The line connects the median
residual for each group, and the shaded band shows the middle 50% of
residuals within each group.

Errors were generally more spread out at higher predicted costs. All three
models also substantially underpredicted high-cost cases.

Elastic Net's median residual generally increased with its predictions,
showing increasing underprediction. Random Forest and XGBoost generally
stayed closer to zero.

The tree models also produced a wider prediction range. Elastic Net never
predicted above about $217, compared with about $1,270 for Random Forest and
$2,100 for XGBoost. Together, these findings made the tree models more
promising for budgeting higher costs.

For questions:

The line and band exclude the highest 1% of
predicted costs, and the x-axis is zoomed accordingly to keep extreme
predictions from compressing the main pattern.

**Transition:** Return to model selection or continue with questions.

**Promote:** After 6 for a longer modeling discussion; allow 1:00.

**Sources:** [README: heteroscedasticity](../../README.md#tuned-models-heteroscedasticity);
[modeling notebook](../../notebooks/2_modeling.py).

### A17 — Subgroup reliability

**Question:** How did the tuned models compare across reliability subgroups?

**Visual:** Full-slide [tuned-model reliability figure](assets/tuned_models_validation_subgroup_reliability.png).

**Speaker notes**

Errors increased across actual and predicted cost tiers for all three
models and were higher for privately insured adults than for publicly
insured or uninsured adults.

Elastic Net achieved the lowest overall median absolute error and had lower
error across many subgroups. However, Random Forest and XGBoost
had lower median absolute error in key health groups, such as people with
poor physical health or four-plus chronic conditions.

For questions:

The metrics are survey-weighted validation median absolute
errors. Sample sizes and actual median costs appear in the group labels.
Predicted cost tiers are defined separately for each model, so they do not
necessarily contain the same people.

**Transition:** Return to model selection or continue with questions.

**Promote:** After 6 for a longer discussion of model selection; allow 1:00.

**Sources:** [README: subgroup reliability](../../README.md#tuned-models-subgroup-reliability);
[modeling notebook](../../notebooks/2_modeling.py).

### A18 — XGBoost quantile: value over simple baselines

**Question:** Does XGBoost improve on population and age-group estimates?

**On the slide**

Compared XGBoost with population and age-group baselines based on training quantiles.
The clearest gains were in the typical range and safety cushion.

| Metric | Versus population baseline (95% CI) | Versus age-group baseline (95% CI) |
| --- | --- | --- |
| Plan-around estimate (q50): MdAE reduction | **3.4%** [-12.7%, 13.1%] | **21.5%** [10.1%, 30.1%] |
| Plan-around estimate (q50): quantile skill score | **9.8%** [7.8%, 11.9%] | **7.6%** [5.9%, 9.6%] |
| Typical range (q25–q75): interval skill score | **11.2%** [8.6%, 13.8%] | **9.0%** [6.9%, 11.3%] |
| Safety cushion (q90): quantile skill score | **15.6%** [10.8%, 21.1%] | **14.3%** [9.7%, 19.0%] |

**Table note:** Survey-weighted test metrics. Skill scores measure the percentage
reduction in quantile or interval loss compared with each baseline. 95% percentile
bootstrap confidence intervals from 1,000 paired resamples of test-set rows.

**Visual:** Comparison table with a Back link to the final model audit.

**Speaker notes — 0:45 (optional, estimated)**

“XGBoost improved the typical range and safety cushion over both baselines, with
the largest improvement for the safety cushion. Plan-around results were mixed:
quantile loss improved against both baselines, while median absolute error
remained close to the population baseline.”

For questions:

Baselines: Both use survey-weighted training quantiles. The population baseline
gives everyone the same estimates. The age-group baseline gives everyone within
the same age band the same estimates: 18–34, 35–49, 50–64, and 65+.

Quantile skill: Percentage reduction in pinball loss. At q90, underprediction
receives nine times the penalty of an equally large overprediction. At q50,
quantile skill equals the percentage reduction in mean absolute error.

Interval skill: Percentage reduction in the Winkler interval score, which
rewards narrower ranges but adds a penalty when actual costs fall outside them.
The farther outside, the larger the penalty.

Uncertainty: The comparison uses 1,000 paired bootstrap resamples of test-set
rows, keeping each person's actual cost, predictions, and survey weight together.
The same sampled rows are used for all three approaches. The confidence interval
for median absolute error reduction versus the population baseline includes
zero, so that improvement remains uncertain.

**Transition:** Return to the final model audit.

**Promote:** After 8 for a longer presentation; allow 0:45.

**Sources:** [README: baseline comparison](../../README.md#final-model-value-over-simple-baselines);
[modeling notebook: baseline comparison and paired bootstrap](../../notebooks/2_modeling.py).

### A19 — XGBoost quantile: subgroup reliability

**Question:** How does performance differ across actual and predicted cost tiers?

**Visual:** Full-slide [test subgroup reliability figure](assets/xgb_quantile_test_subgroup_reliability.png),
showing actual costs and predicted Q90 costs. Each row shows plan-around MdAE,
coverage, and mean width. The figure contains its own title and metric notes.

**Backlink:** XGBoost quantile audit (slide 8).

**Speaker notes — 0:45**

“Very high actual spending remained the largest prediction weakness. Plan-around
error exceeded $7,000, typical-range coverage was zero, and safety-cushion
coverage was only about 7%, with actual costs mostly exceeding those estimates.

Across predicted Q90 cost tiers, higher costs came with larger plan-around
errors and wider budgeting ranges, while coverage stayed broadly close to its
targets. Since actual spending is unknown at prediction time, I used predicted
Q90 as a proxy for triggering a planning note that suggests users plan closer
to the safety cushion.”

For questions:

Cost tiers: Actual zero costs form a separate group. The positive costs are
grouped using sample percentiles: the lower half, 50th–80th, 80th–95th, and top
5%. Predicted Q90 tiers use their own positive-cost percentiles, with zero
predictions merged into the lowest group. These groups are calculated within
the test set; the metrics within them use survey weights.

Interpretation: Actual cost groups are known only after spending occurs.
Selecting groups by actual outcomes changes the question, so their coverage
does not have the same interpretation as coverage across predicted cost groups.

Planning note: The trigger is Q90 at or above $3,525 in 2023 dollars, the
survey-weighted 80th percentile of validation Q90 predictions. This fixed cutoff
is separate from the test-set percentile groups shown in the figure.

**Transition:** Return to the overall audit and the example prediction output.

**Sources:** [README: final subgroup audit](../../README.md#final-model-subgroup-reliability--fairness);
[modeling notebook: test subgroup analysis](../../notebooks/2_modeling.py).

### A20 — XGBoost quantile: subgroup fairness

**Question:** Which fairness subgroups showed typical-range undercoverage?

**Visual:** Full-slide [test subgroup fairness figure](assets/xgb_quantile_test_subgroup_fairness.png),
showing mental health, family income, and education. Each row shows plan-around
MdAE, coverage, and mean width. The figure contains its own title and metric notes.

**Backlink:** XGBoost quantile audit (slide 8).

**Speaker notes — 0:45**

“Coverage was broadly close to its targets for most groups shown here. However,
the typical range undercovered people with poor mental health, low income, or
a doctorate degree.

Across these groups, typical misses were larger above the typical range,
several hundred to over $1,000, than below it, under $35. These findings led to
adding a planning note for these groups that suggests users plan closer to
the safety cushion.”

For questions:

Miss size: These amounts are survey-weighted median distances from the nearest
range boundary among people whose actual costs fell outside that boundary.
They describe typical misses, not the largest errors.

Group size: Poor mental health and doctorate groups contain 34 and 46 test
cases, respectively, so their estimates carry more uncertainty.

Review criteria: Coverage outside 40%–60% for the typical range or 80%–97% for
the safety cushion is flagged for review. Near-poor adults also showed slight
safety-cushion overcoverage. A flag alone does not establish unfairness or mean
the model should not be released.

Planning note: The trigger uses the provided mental health, family income,
or education input. The note uses neutral wording without naming the group.

**Transition:** Return to the overall audit and the example prediction output.

**Sources:** [README: final subgroup audit](../../README.md#final-model-subgroup-reliability--fairness);
[modeling notebook: test subgroup analysis](../../notebooks/2_modeling.py).

### A21 — Feature timing: limits of next-year forecasting

**Question:** Is this truly a next-year forecast?

**On the slide:** Early-round health measures reduce reliance on later-year
information, but interviews occur during the cost year. Variables such as
`INSCOV23` summarize the full year. Current evaluation uses a within-year split.

**Visual:** A timeline separating survey collection, the 2023 spending target,
and the planned prospective application.

**Talking points:** Training-fitted preprocessing addresses one source of
leakage. Feature availability at the intended prediction date is a separate
issue. Audit every input against that date and evaluate on later-year data.
Do not claim the size or direction of the resulting performance change is known.

**Promote:** After 3 or 9 for forecasting and validation questions; allow 1:00.

**Sources:** [input feature timing](../research/candidate_features.md);
[technical specifications: input selection](../specs/technical_specifications.md).

### A22 — SHAP explains how inputs shape the median estimate

**Question:** Which inputs influence predictions, and what can users infer?

**On the slide:** Insurance and family income have the largest average absolute
SHAP contributions. Explanations concern postprocessed q50 in dollars, through
the full inference function. They do not explain q90 or causal effects.

**Visual:** [Test SHAP feature importance](../../figures/evaluation/shap_feature_importance.png).

**Talking points:** Permutation SHAP operates on 27 interpretable preprocessor
inputs, including the effects of transformations and derived medical counts.
Correlated inputs can share attribution. Higher or lower observed spending
does not by itself mean higher or lower healthcare need.

**Promote:** Before 11 for explainability depth; allow 1:00.

**Sources:** [README: feature importance](../../README.md);
[SHAP metadata](../../app/data/shap_metadata.json).

### A23 — SHAP benchmarking: explanation quality and latency

**Question:** How was the SHAP configuration chosen?

**On the slide:** 225 background rows; one permutation round. On 100 held-out
test rows, every row matched at least four of the reference's top five features;
median matched contribution difference **$6.47**; P95 core SHAP latency **0.20 s**.

**Visual:** One compact quality-versus-latency summary, with the measurement
boundary stated explicitly.

**Talking points:** The reference uses 500 background rows and 24 rounds.
The benchmark also checks material sign reversals and prediction reconstruction.
P95 summarizes subsequent explanation calls after a separately measured first
call. It excludes the rest of the request and is not a deployed API latency
claim. Complete request latency on target hardware remains unverified.

**Promote:** Before 11 for ML engineering roles; allow 1:00.

**Sources:** [README: SHAP explanation details](../../README.md);
[SHAP metadata](../../app/data/shap_metadata.json);
[benchmark script](../../scripts/benchmark_shap.py).

## Adaptation and file organization

Adapt the content and slide selection to the audience, time, and setting,
promoting appendix slides as needed. Create variants when requirements are known;
keep local notes in `docs/slides/variants/`, referencing the shared slides.

The storyboard holds content and speaker notes; the generator implements the
slide text and layout. See the [slides README](README.md) for generation,
version retention, and restoring earlier presentations. The root `.gitignore`
defines which sources and assets are tracked.
