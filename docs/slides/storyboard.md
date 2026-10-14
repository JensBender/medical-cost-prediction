# Medical cost prediction: presentation storyboard

Status: the ten-slide main presentation and fifteen appendix slides are ready for
review. The remaining appendix slides are in storyboard form.

## Presentation context

- **Setting:** job interviews, conference talks, meetups, and other settings.
- **Audience:** a technical audience. Assume basic ML
  knowledge, but no familiarity with MEPS or U.S. healthcare costs. Explain
  survey weights, quantiles, and evaluation metrics when introducing them.
- **Emphasis:** data science, with optional emphasis on ML engineering.
- **Language:** English.
- **Length:** 10–12 minutes of speaking, excluding questions. The current
  allocation below totals 11:25, leaving room for transitions.
- **Structure:** a cover, nine content slides, and twenty-six planned appendix slides.
  Order and number appendix slides by their first reference in the main presentation.
  Place unreferenced slides with their related topic. Update numbering and links
  when the presentation order changes.
- **Slide numbering:** Display 2–10 on the main content slides, with the cover
  unnumbered, and A1, A2, etc. in the appendix. Use 1–10 in the storyboard.
- **Title size:** Use 33 pt for all main content-slide titles, including two-line titles.
- **Appendix navigation:** Add links where a specific follow-up is likely, using
  “Appendix: [topic]” at the bottom left and “Back” on the
  appendix slide. Use small, muted gray, underlined text and keep slide IDs at
  the bottom right. Add other footnotes only when needed to interpret the slide.
  On the cost-distribution slide, group EDA links in the left margin under
  “Appendix:” to preserve the plot size.
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
the choice of XGBoost. Model selection considered which outputs and performance
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
| 7 | Quantile regression turns predictions into budgeting ranges | 1:30 | Explain the final model and its outputs |
| 8 | Final model audit: clearest gains in ranges and q90 | 1:30 | Present held-out evidence and baseline comparisons |
| 9 | Final model test audit: overall coverage hides subgroup gaps | 1:00 | Demonstrate critical evaluation and limits |
| 10 | Model evaluation is complete; app development comes next | 1:00 | Show engineering work and close with priorities |
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
[Features (A3–A4)](#a3--preprocessor-input-features). Display as
“Appendix: MEPS overview | Features” with a muted gray vertical bar separating
the two underlined links.

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
and A16 for input timing. Preprocessing is fitted on training data. The 80/10/10 split is
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
- **Supporting:** MAE, R²

**Visual:** Large [Lorenz curve](../../figures/eda/lorenz_curve.png) beneath the
slide title, with the figure's own title cropped. A compact text box in the
upper-left margin lists the primary and supporting metrics, using the same font
size with bold headings. Align it with the top of the plot. Keep the plot large
and shift it right to leave a clear gap beside the text; no extra footnote.

**Appendix links:** A compact block near the bottom left, with a small, muted
“Appendix:” label and
underlined links to [Outlier analysis (A5)](#a5--outliers-greater-health-burden-and-higher-cost-risk),
[Distributions (A6–A8)](#a6--distributions-numerical-features), and
[Feature–target relations (A9–A12)](#a9--featuretarget-relationships-correlations)
closely beneath it. Link to the first slide of each group, then advance through
the group. Keep each link on one line and preserve the plot's current size.

**Speaker notes — 1:00**

“Out-of-pocket costs present two challenges: about 22% of adults have no costs
for the entire year, while costs are heavily concentrated among the top spenders.
The highest-spending 20% account for almost 80% of costs, and the top 1% account
for about 21%. High costs could reflect ongoing care or unexpected events such
as hospitalization. Unexpected events make costs especially difficult to predict.

Large errors on high-cost cases can strongly influence mean absolute error and
R-squared. Therefore, I chose median absolute error as the primary evaluation
metric to describe a typical prediction error, and complemented it with mean
absolute error and R-squared as supporting diagnostics.

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
rendered from the [SVG source](../../assets/preprocessing_pipeline.svg),
using the diagram's existing title and subtitle.

**Visual:** Fit the diagram to the full slide height and match its dark navy
background. Use light gray for the slide number and a small “Appendix:” label
above an underlined “Data preparation” link in the lower-left margin. Link only
“Data preparation” to A13.

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

| Tuned model | MdAE | Overfitting | MAE | R² |
| --- | ---: | ---: | ---: | ---: |
| Elastic Net | **$159** | +7.9% | $1,051 | -0.13 |
| Random Forest | $228 | **+3.8%** | $964 | -0.05 |
| XGBoost | $242 | +6.2% | **$954** | **-0.02** |

- Elastic Net's largest validation prediction was about **$217**
- Subgroup and residual analysis motivated a model that also describes uncertainty

**Footnote:** “Note: Survey-weighted validation metrics. Overfitting is the percentage
change in MdAE from training to validation.”

**Display notes:** Comparison table with its note immediately below, followed by
the $217 finding and its implication. Keep residual diagnostics in A18.

**Appendix links:** [Baseline models (A14)](#a14--baseline-models) |
[Hyperparameter tuning (A15)](#a15--hyperparameter-tuning).

**Speaker notes — 1:30**

“I compared six model architectures and selected Elastic Net, Random Forest,
and XGBoost for tuning. Elastic Net was a strong result, not a failed baseline.
After tuning, its weighted validation median absolute error was $159, lower than either tree
model. But its predictions remained very compressed, with a maximum of about
$217. Residual and subgroup analysis showed why that mattered: good aggregate
median error did not imply useful separation across all health and cost
profiles. XGBoost had a higher median error but lower mean absolute error,
and the tree models performed better in several groups with greater medical
complexity. I therefore moved to predicting several conditional quantiles.
This changed what the model was asked to deliver. It did not establish that
XGBoost is universally better, or that quantiles can predict every expensive
medical event.”

**Transition:** Map the quantiles directly to the intended budgeting outputs.

**Sources:** [README: hyperparameter tuning and model decision](../../README.md);
[modeling notebook script](../../notebooks/2_modeling.py).

**Comparison rule:** All values on this slide are validation results for tuned
point-estimate models. Do not compare $159 here with the final model's $240 test
result as if they came from the same evaluation.

### 7 — Quantile regression turns predictions into budgeting ranges

**Takeaway:** The final model estimates different parts of the spending
distribution for each input profile.

**On the slide**

| Output | Model quantity | Intended meaning |
| --- | --- | --- |
| Plan-around estimate | q50 | Predicted median spending |
| Typical range | q25–q75 | Central range targeting 50% coverage |
| Safety cushion | q90 | Upper planning reference targeting 90% coverage |

- XGBoost with a quantile objective; survey-weighted training on log-transformed costs
- Evaluate coverage **and** width, plus losses that penalize missed outcomes

**Visual:** Cost axis with q25, q50, q75, and q90, labeled “conceptual.”

**Speaker notes — 1:30**

“A conditional median describes the midpoint of spending for an input profile.
The 25th and 75th percentiles describe a central range, and the 90th percentile
provides an upper planning reference. The model learns all four quantiles using
XGBoost's quantile objective. Training uses survey weights and log-transformed
costs; predictions are returned to dollars and postprocessed so the quantiles
are nonnegative and ordered. The final model reuses the tuned point-model
hyperparameters, so a separate search for the quantile objective remains a
possible improvement. Coverage asks how often observed spending falls inside
the range or below q90. Width matters too: a range that covers almost anything
would not help much with budgeting. These are prediction ranges for spending,
not confidence intervals around an average, and q90 is not a maximum possible bill.”

Log-transform reference: Costs are transformed as ln(1 + cost), also called
log1p. Adding one makes the transformation defined for zero costs.

**Transition:** Show whether the final model delivers those outputs on held-out data.

**Sources:** [README: final model](../../README.md);
[quantile training script](../../scripts/train_xgboost_quantile.py);
[prediction postprocessing](../../src/prediction.py).

### 8 — Final model audit: clearest gains in ranges and q90

**Takeaway:** The final model meets the project's overall performance gates,
with its strongest evidence of added value in interval and upper-quantile scores.

**On the slide**

| Held-out test metric | Result | Release gate |
| --- | ---: | --- |
| Plan-around estimate (q50): MdAE | **$240** | < $500 |
| Typical range (q25–q75): coverage | **47.3%** | 45%–55% |
| Safety cushion (q90): coverage | **91.0%** | 85%–95% |

Compared with the population baseline:

| Budgeting output | Comparison result |
| --- | --- |
| Plan-around estimate (q50) | **≈$8** lower MdAE (improvement uncertain) |
| Typical range (q25–q75) | **11.2%** interval skill score |
| Safety cushion (q90) | **15.6%** quantile skill score |

**Visual:** Results table above the baseline comparisons.

**Footnote:** “Survey-weighted test metrics; dollar amounts in 2023 USD.”

**Speaker notes — 1:30**

“On the test set, the median absolute error is $240. This means roughly half of
the weighted test population has an absolute error no larger than that amount;
it is not the mean error. The typical range covers 47.3% of spending outcomes,
close to its 50% target, and q90 covers 91%. Mean range widths are $912 for
q25 to q75 and $2,032 for q50 to q90, and the model meets all five project
performance gates. Release gates are minimum requirements; the product targets
are more ambitious goals. But a simple population estimate is competitive on plan-around
MdAE: $248 versus $240. That is approximately $8 lower MdAE, using the rounded
values shown. The bootstrap comparison does not establish a clear MdAE
improvement. The stronger results are an 11.2% interval skill score and a
15.6% quantile skill score at q90. These express the relative reduction in
interval score and pinball loss compared with the baseline. The underlying
scores assess width and misses, or the size and direction of errors, rather
than coverage alone. The age-group comparison also supports gains on these
measures.”

**Transition:** Overall metrics still need to be checked against the groups
and situations where the model is less reliable.

**Sources:** [README: release gates](../../README.md);
[modeling notebook: test comparison with simple baselines](../../notebooks/2_modeling.py).
Confidence intervals, widths, and scoring definitions are in A19 and A20.

### 9 — Final model test audit: overall coverage hides subgroup gaps

**Takeaway:** The model's limitations affect both interpretation and the next
validation steps.

**On the slide**

- Rare high-cost years remain difficult to anticipate
- Typical ranges undercover some groups, including uninsured users and people
  reporting poor mental health
- One-year holdout performance does not establish prospective performance
- Subgroup audits inform safeguards; they do not prove absence of bias

**Visual:** Bar chart of typical-range coverage: poor mental health 30.1%, low
income 39.2%, overall 47.3%, and target 50%. Label results as point estimates;
A21 adds subgroup counts and uncertainty.

**Speaker notes — 1:00**

“Overall coverage can hide important differences. The typical range misses too
often for several groups, including uninsured users and people reporting poor
mental health. Rare actual high-cost outcomes remain particularly difficult,
and those actual cost tiers are only known after the year ends. They are useful
diagnostics, but cannot be used to route users at prediction time. There is also
a temporal limitation: the holdout comes from 2023, and some inputs summarize
information collected during that year. The planned next-year use case needs
further validation. Clear scope wording and planning notices can communicate
these limitations, but they do not fix calibration.”

**Transition:** Close with what has been implemented and the next checks needed
to turn it into a usable product.

**Sources:** [README: final subgroup audit](../../README.md);
[feature timing notes](../research/candidate_features.md);
[preprocessing split](../../scripts/preprocess.py).

### 10 — Model evaluation is complete; app development comes next

**Takeaway:** The project has reusable ML components and an explicit plan for
the remaining product and validation work.

**On the slide**

| Implemented | Planned |
| --- | --- |
| DVC stages and MLflow experiment tracking | FastAPI/Gradio application |
| Shared prediction and SHAP modules; unit tests | Complete request-latency and integration checks |
| Model evaluation and application data artifacts | User evaluation and aggregate monitoring |

- Next validation priority: feature timing and performance on a later survey year
- Main lesson: evaluate the outputs needed for the user decision

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
model selection depends on the decision the output supports: typical error,
uncertainty, and subgroup reliability each reveal something different.”

**Close:** Invite questions with the results and limitations still easy to revisit.

**Sources:** [DVC stages](../../dvc.yaml); [prediction module](../../src/prediction.py);
[explanation module](../../src/explainability.py);
[unit tests](../../tests/unit/); [technical specifications](../specs/technical_specifications.md).

## Appendix slides

Each appendix item is an optional slide with a specific question to answer.
Timings apply only when promoted into the main presentation.

### A1 — U.S. healthcare costs: out-of-pocket spending and HSA/FSA planning

**Question:** What is out-of-pocket spending, and how does it relate to HSA/FSA planning?

**On the slide:** [Healthcare-cost infographic](../../assets/infographic_healthcare_costs.png)
with a “Back” link.

**Speaker notes — reference for questions**

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

**On the slide:** [MEPS infographic](../../assets/infographic_meps_data.jpg)
with a “Back” link.

**Speaker notes — reference for questions**

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
| Family income | Ordinal | Relative to poverty: poor/negative, near poor, low, middle, high |
| Family size | Numerical | 1–14 |
| Education | Categorical | None, GED, high school, bachelor's, master's, doctorate, other |
| Employment status | Binary | Employed, not employed |
| Insurance | Categorical | Any private, public only, uninsured |
| Physical health | Numerical | 1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor |
| Mental health | Numerical | 1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor |

**Visual:** An editable table of 11 prepared inputs, with a “Back” link to the MEPS
data slide. Continue to A4 for the remaining 16 yes/no features.

**Speaker notes — reference for questions**

These are the interpretable preprocessor input features, which SHAP also uses.
They have gone through initial data preparation, including grouping marital
and employment categories, rather than being raw MEPS variables.

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
improve the experience. Refer to A16 for input timing.

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

**Visual:** An editable table with three groups: chronic conditions, limitations,
and other. Include a “Back” link to the MEPS data slide. Keep explanations for
ADL help, IADL help, usual source of care, and recent life transition.

**Speaker notes — reference for questions**

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

**Visual:** Detection above a wide, editable table; treatment below it. Use
“Profiling” as the first-column header, with no separate heading or footnote.
Leave space between the three sections.

| Profiling | Inliers | Outliers |
| --- | ---: | ---: |
| In top 1% of spenders | 0.9% | 3.5% |
| Age (median) | 47 | 66 |
| Chronic conditions (median) | 1 | 4 |
| Functional limitations (median) | 0 | 3 |
| Private insurance | 68% | 32% |
| Walking limitations | 9% | 75% |
| Cognitive limitations | 3% | 61% |

Include a “Back” link.

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

**On the slide:** [Numerical distributions](../../figures/eda/numerical_distributions.png),
using the figure's existing title.

**Visual:** Fit the unchanged image to the full slide height. Place a small
“Back” link in the unused lower-left side margin, pointing to the
cost-distribution slide. Use the same layout for the next two distribution slides.

**Speaker notes — reference for questions**

Family size is right-skewed, and health ratings are concentrated around
excellent to good. These distributions supported median imputation because the
median is less affected by extreme values than the mean.

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

**On the slide:** [Categorical distributions](../../figures/eda/categorical_distributions.png),
using the figure's existing title and the same layout as A6.

**Speaker notes — reference for questions**

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

**On the slide:** [Binary distributions](../../figures/eda/binary_distributions.png),
using the figure's existing title and the same layout as A6.

**Speaker notes — reference for questions**

Joint pain, high blood pressure, and high cholesterol are common. Severe conditions such as cancer, coronary heart disease, and stroke are less common.

**Transition:** Return to the cost-distribution slide or continue with questions.

**Sources:** [README: distributions](../../README.md#distributions);
[binary distributions](../../figures/eda/binary_distributions.png).

### A9 — Feature–target relationships: correlations

**Question:** Which features are associated with costs and with one another?

**On the slide:** [Correlation heatmap](../../figures/eda/correlation_heatmap.png),
using the figure's existing title.

**Visual:** Fit the unchanged image to the full slide height. Place a small
“Back” link in the unused lower-left side margin, pointing to the
cost-distribution slide. Use the same layout for the next three relationship slides.

**Speaker notes — reference for questions**

This heatmap shows survey-weighted Spearman rank correlations. Age has the
strongest positive association with out-of-pocket costs at 0.30, followed by
family income, arthritis, high cholesterol, and joint pain.
Family size has the strongest negative association at −0.22.

**Transition:** Continue to the numerical relationships or return to the cost-distribution slide.

**Sources:** [README: relationships](../../README.md#main-relationships);
[correlation heatmap](../../figures/eda/correlation_heatmap.png).

### A10 — Feature–target relationships: numerical features

**Question:** How do costs vary with the numerical features?

**On the slide:** [Numerical feature–target relationships](../../figures/eda/numerical_feature_target_relationships.png),
using the figure's existing title and the same layout as A9.

**Speaker notes — reference for questions**

The plots show log-transformed costs rather than dollar amounts, so very high
costs do not dominate the plots.

People of similar ages or with the same self-reported health rating can still
have very different out-of-pocket costs, ranging from zero to high costs.

**Transition:** Continue to the categorical relationships or return to the cost-distribution slide.

**Sources:** [README: numerical relationships](../../README.md#numerical-feature-target-relationships);
[numerical relationships](../../figures/eda/numerical_feature_target_relationships.png).

### A11 — Feature–target relationships: categorical features

**Question:** How do cost distributions differ across categorical feature groups?

**On the slide:** [Categorical feature–target relationships](../../figures/eda/categorical_feature_target_relationships.png),
using the figure's existing title and the same layout as A9.

**Speaker notes — 0:20 (optional)**

“These plots compare cost distributions across categories. The dollar labels
show group medians, while the vertical axes use a log scale. Higher income,
higher education, and private insurance generally coincide with higher
out-of-pocket costs. The plots also show substantial variation within groups.”

**Transition:** Continue to the binary relationships or return to the cost-distribution slide.

**Sources:** [README: categorical relationships](../../README.md#categorical-feature-target-relationships);
[categorical relationships](../../figures/eda/categorical_feature_target_relationships.png).

### A12 — Feature–target relationships: binary features

**Question:** How do costs differ between the two categories of each binary feature?

**On the slide:** [Binary feature–target relationships](../../figures/eda/binary_feature_target_relationships.png),
using the figure's existing title and the same layout as A9.

**Speaker notes — 0:20 (optional)**

“These plots compare costs between the two groups for each binary feature.
Arthritis shows a stronger rank association with costs, while cancer shows a
larger difference in median spending. Women and people with a usual source of
care also have higher median costs. These comparisons describe associations.”

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

**Visual:** A short preparation sentence above the stratified-split explanation.
Place zero, lower, and extreme cost labels above the full-width group diagram,
with uniform gaps and the finer-percentile-group explanation below it. Keep the
preparation sentence and split allocation on one line each. Add a small “Back”
link to the pipeline slide. Set the split allocation in regular dark blue text
at 18.75 pt, matching the explanation beneath the diagram.

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

| Model | MdAE | Overfitting | MAE | R² |
| --- | ---: | ---: | ---: | ---: |
| **Elastic Net** | **$163** | +6.6% | $1,044 | -0.12 |
| Linear Regression | $219 | +4.8% | $998 | -0.06 |
| **Random Forest** | $232 | +9.6% | **$958** | -0.04 |
| *Median (benchmark)* | *$248* | *0.0%* | *$1,041* | *-0.10* |
| Decision Tree | $271 | **+1.5%** | $971 | -0.03 |
| **XGBoost** | $281 | +98.0% | $961 | 0.00 |
| Support Vector Machine | $291 | +190.7% | $1,027 | -0.03 |
| *LLM (benchmark)* | *$518* | *N/A* | *$1,168* | ***0.04*** |

**Visual:** Editable table matching the README's baseline comparison. Bold the
three selected model names and the best values in each metric column. Place the
note beneath the table and add a small “Back” link to the model-selection slide.

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

**Display notes:** One opening sentence above three columns, with balanced gaps
between the visible text. Use 24 pt bold model names and 18 pt regular text,
with one parameter per line and no bullet markers.
Separate parameter names and values with colons.
Leave the search rationale and implementation details to the speaker notes.
Return link to model selection.

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

### A16 — Feature timing: limits of next-year forecasting

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

### A17 — LLM benchmark: General versus Specific Intelligence

**Question:** Why not ask a general-purpose language model for an estimate?

**On the slide:** The documented Gemini 3 Flash benchmark has validation MdAE
of **$518**, versus **$163** for baseline Elastic Net.

**Visual:** Two bars labeled with model, metric, and validation split. Do not
mix these with test-set results from the final quantile model.

**Talking points:** This result describes one model and prompting setup, not all
LLMs.

**Promote:** Optional, after A15; allow 1:00.

**Sources:** [README: LLM benchmark](../../README.md);
[LLM benchmark script](../../scripts/benchmark_llm.py).

### A18 — Point-model diagnostics: residuals and prediction ranges

**Question:** Why did the lowest-MdAE model not meet the full product need?

**On the slide:** Elastic Net's narrow prediction range; larger errors on
expensive outcomes; differences across medical-complexity groups.

**Visual:** [Tuned-model residual analysis](../../figures/evaluation/tuned_models_validation_heteroscedasticity.png).
Use a readable crop or split the figure during design rather than shrinking it.

**Talking points:** MdAE and MAE summarize different aspects of the error
distribution. A lower MdAE alone does not establish useful risk separation.
Wider prediction spread alone also does not establish accuracy; inspect
residuals and subgroup errors alongside it.

**Promote:** After 6; allow 1:00.

**Sources:** [README: heteroscedasticity and subgroup analysis](../../README.md);
[modeling notebook](../../notebooks/2_modeling.py).

### A19 — Final model test results: coverage, width, and release gates

**Question:** What does passing the project's gates mean?

**On the slide:** All results are survey-weighted test estimates. Dollar amounts
are in 2023 USD; intervals below are approximate 95% bootstrap confidence intervals.

| Metric | Estimate [95% CI] | Project gate |
| --- | --- | --- |
| q50 MdAE | $240 [$215, $279] | < $500 |
| q25–q75 coverage | 47.3% [44.0%, 50.6%] | 45%–55% |
| q90 coverage | 91.0% [89.2%, 92.6%] | 85%–95% |
| Mean q25–q75 width | $912 [$875, $955] | < $1,500 |
| Mean q50–q90 width | $2,032 [$1,964, $2,108] | < $3,500 |

**Talking points:** Point estimates pass the gates. The entire confidence
interval does not need to fall inside the gate under the reported decision
rule. These are project criteria, not external certification. The notebook
resamples rows and retains their weights.

**Promote:** After 8 for evaluation depth; allow 1:00.

**Sources:** [README: release gates](../../README.md);
[modeling notebook: bootstrap implementation](../../notebooks/2_modeling.py).

### A20 — Final model test benchmarks: gains over simple estimates

**Question:** Why not give everyone a population or age-group estimate?

**On the slide**

| Test comparison | Versus population baseline | Versus age-group baseline |
| --- | ---: | ---: |
| q50 MAE reduction | 9.8% | 7.6% |
| Typical-range interval-score reduction | 11.2% | 9.0% |
| q90 pinball-loss reduction | 15.6% | 14.3% |

MdAE: XGBoost **$240**; population baseline **$248**; age-group baseline **$305**.

**Visual:** Grouped comparison of the loss reductions, with paired-bootstrap
intervals from the saved modeling notebook.

**Talking points:** Benchmarks use training data. The interval score penalizes
width and observations outside the range. At q90, pinball loss penalizes
underprediction more heavily than overprediction. The notebook reports
confidence intervals above zero for the three reductions shown, but the small
MdAE improvement versus the population baseline remains uncertain. Positive
skill is not a percentage-point improvement in coverage.

**Promote:** After 8; allow 1:00.

**Source:** [Modeling notebook: test baseline comparisons](../../notebooks/2_modeling.py).

### A21 — Final model test audit: subgroup coverage gaps

**Question:** Who receives less reliable ranges?

**On the slide:** Typical-range test coverage is **30.1%** for poor mental health,
**39.2%** for low income, and **34.7%** for doctorate degree holders. The target
is 50%; uninsured users are also on the undercoverage watchlist.

**Visual:** Selected rows from [test subgroup fairness](../../figures/evaluation/xgb_quantile_test_subgroup_fairness.png)
and [test subgroup reliability](../../figures/evaluation/xgb_quantile_test_subgroup_reliability.png).
Retain sample sizes and uncertainty intervals where shown.

**Talking points:** Smaller subgroups carry greater uncertainty. The audit uses
wider diagnostic review bands than the overall gates and a minimum raw count
of 30. Similar error patterns across models do not establish absence of bias.
Planning notices communicate a limitation; further evaluation or recalibration
would be needed to address it.

**Promote:** After 9; allow 1:00.

**Sources:** [README: final subgroup audit](../../README.md);
[modeling notebook](../../notebooks/2_modeling.py).

### A22 — Final model test audit: actual versus predicted cost tiers

**Question:** Does the safety cushion protect the largest spenders?

**On the slide:** q90 covers **91.0% overall**, but only **6.7%** in the audit's
actual “Very High” spending group. Actual cost groups are retrospective;
predicted cost groups are available at inference.

**Visual:** The actual-versus-predicted cost-tier section of
[test reliability](../../figures/evaluation/xgb_quantile_test_subgroup_reliability.png).
Preserve the notebook's group definitions in the figure or footnote.

**Talking points:** Selecting a subgroup using the outcome changes the question.
Low coverage in the observed tail is a useful description of missed expensive
years, not the same as conditional calibration by an available input. An upper
quantile is not catastrophe insurance or a hard spending cap.

**Promote:** After 9 when discussing tail risk; allow 1:00.

**Sources:** [README: final reliability audit](../../README.md);
[modeling notebook](../../notebooks/2_modeling.py).

### A23 — SHAP explains how inputs shape the median estimate

**Question:** Which inputs influence predictions, and what can users infer?

**On the slide:** Insurance and family income have the largest average absolute
SHAP contributions. Explanations concern postprocessed q50 in dollars, through
the full inference function. They do not explain q90 or causal effects.

**Visual:** [Test SHAP feature importance](../../figures/evaluation/shap_feature_importance.png).

**Talking points:** Permutation SHAP operates on 27 interpretable preprocessor
inputs, including the effects of transformations and derived medical counts.
Correlated inputs can share attribution. Higher or lower observed spending
does not by itself mean higher or lower healthcare need.

**Promote:** Before 10 for explainability depth; allow 1:00.

**Sources:** [README: feature importance](../../README.md);
[SHAP metadata](../../app/data/shap_metadata.json).

### A24 — SHAP benchmarking: explanation quality and latency

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

**Promote:** Before 10 for ML engineering roles; allow 1:00.

**Sources:** [README: SHAP explanation details](../../README.md);
[SHAP metadata](../../app/data/shap_metadata.json);
[benchmark script](../../scripts/benchmark_shap.py).

### A25 — Inference architecture: shared prediction and explanation code

**Question:** How will training and serving stay consistent?

**On the slide:** DVC stages produce fitted preprocessing and model artifacts.
Shared prediction code applies the fitted preprocessor, predicts quantiles, and
enforces ordered nonnegative outputs. The planned service handles user-input
mapping, inflation adjustment, notices, and response formatting.

**Visual:** An architecture diagram separating implemented modules from planned
FastAPI/Gradio integration. Show explanations calling the same q50 inference path.

**Talking points:** Keep UI concerns at the interface. DVC records stage
dependencies; MLflow records experiments. Existing unit tests cover core
prediction and explanation behavior. Integration, end-to-end behavior, and
deployment performance still need verification. Do not imply that every
experiment is a DVC stage or that the full service is implemented.

**Promote:** Before 10 for ML engineering roles; allow 1:30.

**Sources:** [DVC pipeline](../../dvc.yaml); [prediction code](../../src/prediction.py);
[technical specifications](../specs/technical_specifications.md); [unit tests](../../tests/unit/).

### A26 — Production monitoring: drift signals without observed outcomes

**Question:** How would the product be monitored without retaining user records?

**On the slide:** Planned aggregate counters track app health, input and output
distributions, and warning rates. Individual inputs and predictions are not
retained. Actual annual spending is unavailable by default.

**Visual:** A table separating observable signals from unavailable outcomes.

**Talking points:** Drift can trigger investigation, but it does not measure
MdAE or interval coverage. Those require observed outcomes. Periodic evaluation
on later MEPS data is a separate route for assessing the model. Medical
inflation adjusts the dollar scale; it cannot correct all changes in insurance,
care use, or the population. These are planned operating choices.

**Promote:** Before 10 for deployment and product discussion; allow 1:00.

**Sources:** [Technical specifications: privacy-preserving monitoring](../specs/technical_specifications.md);
[product requirements](../specs/product_requirements.md).

## Adaptation and file organization

Adapt the content and slide selection to the audience, time, and setting,
promoting appendix slides as needed. Create variants when requirements are known;
keep local notes in `docs/slides/variants/`, referencing the shared slides.

The storyboard holds content and spoken notes; the generator holds slide text
and layout. Update both when content changes. See the [slides README](README.md)
for generation, version retention, and restoring earlier presentations. Track
the storyboard, generator, README, and required images. Keep exports, build files,
previews, variant notes, and image prompts local, as defined in the root `.gitignore`.
