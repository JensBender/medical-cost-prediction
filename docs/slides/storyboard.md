# Medical cost prediction: presentation storyboard

Status: the nine-slide main presentation and the U.S. healthcare costs, MEPS,
and outlier analysis appendix slides are ready for review. The remaining appendix
slides are in storyboard form.

## Presentation context

- **Setting:** job interviews, conference talks, meetups, and other settings.
- **Audience:** a technical audience. Assume basic ML
  knowledge, but no familiarity with MEPS or U.S. healthcare costs. Explain
  survey weights, quantiles, and evaluation metrics when introducing them.
- **Emphasis:** data science, with optional emphasis on ML engineering.
- **Language:** English.
- **Length:** 10–12 minutes of speaking, excluding questions. The current
  allocation below totals 10:25, leaving room for transitions.
- **Structure:** a cover, eight content slides, and sixteen planned appendix slides.
  Order and number appendix slides by their first reference in the main presentation.
  Place unreferenced slides with their related topic. Update numbering and links
  when the presentation order changes.
- **Slide numbering:** Display 2–9 on the main content slides, with the cover
  unnumbered, and A1, A2, etc. in the appendix. Keep M1–M9 as internal IDs.
- **Appendix navigation:** Add links where a specific follow-up is likely, using
  “Appendix: [topic]” at the bottom left and “Back to [main slide topic]” on the
  appendix slide. Use small, muted gray, underlined text and keep slide IDs at
  the bottom right. Add other footnotes only when needed to interpret the slide.
  On the cost-distribution slide, group EDA links in the left margin under
  “Appendix” to preserve the plot size.
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
| M1 | Medical cost prediction for annual budgeting | 0:10 | Cover and brief orientation |
| M2 | How much should I set aside for healthcare? | 0:45 | Establish the budgeting need and intended value |
| M3 | MEPS links accessible inputs to observed spending | 2:00 | Explain the survey, project data, and population weights |
| M4 | Out-of-pocket costs: 20% of adults account for ~80% | 1:00 | Show the data challenge that shaped modeling |
| M5 | Model selection: median error was not enough | 1:30 | Explain the key model-selection tradeoff |
| M6 | Quantile regression turns predictions into budgeting ranges | 1:30 | Explain the final model and its outputs |
| M7 | Final model audit: clearest gains in ranges and q90 | 1:30 | Present held-out evidence and baseline comparisons |
| M8 | Final model test audit: overall coverage hides subgroup gaps | 1:00 | Demonstrate critical evaluation and limits |
| M9 | Model evaluation is complete; app development comes next | 1:00 | Show engineering work and close with priorities |
| | **Total** | **10:25** | |

## Main slides

### M1 — Medical cost prediction for annual budgeting

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

### M2 — How much should I set aside for healthcare?

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

### M3 — MEPS links accessible inputs to observed spending

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

**Appendix link:** [MEPS overview (A2)](#a2--meps-survey-overview).

**Speaker notes — 2:00 (rehearsed)**

“To do so, I used data from the Medical Expenditure Panel Survey, or MEPS. 
MEPS is a nationally representative survey and a leading data source for 
U.S. healthcare costs. Households complete five interviews over two years.
To improve healthcare cost estimates, MEPS obtains participants’ written
permission to collect medical records directly from providers and pharmacies,
including payment details.

MEPS records healthcare spending and who paid for it. I chose out-of-pocket costs 
as the target because that is what people need to budget for themselves. This 
includes copays, deductibles, and uncovered services, but excludes insurance premiums.

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

See A4 for feature examples, counts, and possible feature reduction, and A5 for
input timing. Preprocessing is fitted on training data. The 80/10/10 split is
random within spending groups, with a separate zero-cost group and finer bins
for high costs to reduce imbalances in rare, expensive cases. Discuss the split
after the spending-distribution slide if asked.

### M4 — Out-of-pocket costs: 20% of adults account for ~80%

**Takeaway:** Typical errors and errors on expensive years describe different
parts of model performance.

**On the slide**

- **22.3%** of the weighted adult population has zero out-of-pocket spending
- The highest-spending **20% accounts for 79.3%** of spending
- The highest-spending **1% accounts for 20.6%** of spending

**Visual:** Large [Lorenz curve](../../figures/eda/lorenz_curve.png) beneath the
slide title, with the figure's own title cropped. No extra body text or footnote.

**Appendix links:** A small, muted “Appendix” label in the lower left margin with
an underlined [Outlier analysis (A3)](#a3--outlier-analysis-retaining-plausible-cases)
link beneath it. Keep the plot's current size and position. Add feature-distribution
and correlation links here when those appendix slides are created.

**Speaker notes — 1:00**

“Out-of-pocket costs present two challenges: many people have no costs, while a
small share have very high costs. About 22% of adults have zero costs for the year.
The highest-spending 20% account for almost 80% of costs, and the top 1% account
for about 21%.

I chose median absolute error, or MdAE, as the primary evaluation metric to
describe a typical miss without letting rare, extreme errors dominate. I also
tracked mean absolute error and R-squared to assess performance beyond typical
errors.

Outlier profiling revealed unusual health profiles consistent with greater
medical needs. I retained these cases because unusual values alone were not
evidence of data errors, and removing them would exclude people relevant to the
intended use.”

**Transition:** This distribution explains why the validation leaderboard was
only the start of model selection.

**Sources:** [EDA notebook script: zero costs and cost concentration](../../notebooks/1_eda_and_preprocessing.py);
[README: EDA and outlier analysis](../../README.md).

### M5 — Model selection: median error was not enough

**Takeaway:** The model with the best typical error offered limited separation
between lower- and higher-cost profiles.

**On the slide**

| Tuned point-estimate model | Validation MdAE | Validation MAE |
| --- | ---: | ---: |
| Elastic Net | **$159** | $1,051 |
| Random Forest | $228 | $964 |
| XGBoost | $242 | **$954** |

- Elastic Net's largest validation prediction was about **$217**
- Subgroup and residual analysis motivated a model that also describes uncertainty

**Display notes:** Comparison table with the $217 finding below it. Keep residual
diagnostics in A8. Define the metrics here: “MdAE: median absolute error;
MAE: mean absolute error.”

**Speaker notes — 1:30**

“Elastic Net was a strong result, not a failed baseline. After tuning, its
weighted validation median absolute error was $159, lower than either tree
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

### M6 — Quantile regression turns predictions into budgeting ranges

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

**Transition:** Show whether the final model delivers those outputs on held-out data.

**Sources:** [README: final model](../../README.md);
[quantile training script](../../scripts/train_xgboost_quantile.py);
[prediction postprocessing](../../src/prediction.py).

### M7 — Final model audit: clearest gains in ranges and q90

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
Confidence intervals, widths, and scoring definitions are in A9 and A10.

### M8 — Final model test audit: overall coverage hides subgroup gaps

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
A11 adds subgroup counts and uncertainty.

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

### M9 — Model evaluation is complete; app development comes next

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
with a “Back to problem statement” link.

**Speaker notes — 1:00 (optional)**

“This is background for anyone unfamiliar with the U.S. healthcare system.
Insurance may cover part of the bill, but people can still face deductibles,
copays, and coinsurance. Those payments, plus uncovered services, are the costs
the project aims to predict. Insurance premiums are excluded. HSA and FSA accounts
offer tax benefits for eligible healthcare expenses, so choosing contributions
is another reason to estimate costs ahead of time. The infographic is a simplified
overview; actual payments depend on insurance coverage and the care received.”

**Transition:** Return to the problem statement slide or continue with questions.

**Detail for questions:** The target excludes premiums and over-the-counter
purchases. It covers annual costs for an individual U.S. civilian
noninstitutionalized adult. Household totals, procedure prices, and insurance
plan comparisons are outside scope.

**Promote:** After M2 when the audience needs U.S. healthcare background; allow 1:00.

**Sources:** [README: target variable](../../README.md);
[product requirements: out of scope](../specs/product_requirements.md);
[HealthCare.gov: HSA overview](https://www.healthcare.gov/high-deductible-health-plan/);
[HealthCare.gov: FSA overview](https://www.healthcare.gov/have-job-based-coverage/flexible-spending-accounts/).

### A2 — MEPS survey overview

**Question:** How does MEPS collect its data, and which component does this project use?

**On the slide:** [MEPS infographic](../../assets/infographic_meps_data.jpg)
with a “Back to MEPS data” link.

**Speaker notes — 1:00 (optional)**

“MEPS includes household, medical provider, and employer insurance surveys.
The project uses the 2023 full-year Household Component file, HC-251. Households
are interviewed repeatedly, and provider information supplements or replaces
reported spending where needed. The employer insurance survey is a separate
component, not an extra set of linked inputs in this project. The overlapping
panels shown here contribute observations to the same calendar-year file.”

**Transition:** Return to the main data slide or continue with questions.

**Detail for questions:** `PERWT23F` supplies person weights. Training, metrics,
and simple benchmarks use these weights. They do not by themselves remove
missing-feature bias, guarantee representativeness of future app users, or
account for survey clustering in uncertainty estimates.

**Promote:** After M3 for survey background; allow 1:00.

**Sources:** [AHRQ: MEPS overview](https://www.ahrq.gov/data/meps.html);
[MEPS medical provider component](https://meps.ahrq.gov/mepsweb/survey_comp/mpc.jsp);
[EDA notebook](../../notebooks/1_eda_and_preprocessing.py);
[modeling helpers](../../src/modeling.py).

### A3 — Outlier analysis: retaining plausible cases

**Question:** How were outliers identified, and why were they retained?

**On the slide**

- **Detection:** Isolation Forest (flagging 5%).
- **Profiling:** Outliers compared with inliers.
  - **Extreme costs:** 3.9× as likely to be among the top 1% of spenders.
  - **Age and health burden:** Older, with more chronic conditions and limitations.
  - **Insurance:** About half as likely to have private insurance.
- **Treatment:** Retained all outliers as plausible, valuable training examples.

**Visual:** Bullet list uses most of the slide width, with the profiling findings
as nested bullets. A narrow, editable horizontal bar chart on the right compares:

| Feature | Inliers | Outliers |
| --- | ---: | ---: |
| Walking limitations | 9% | 75% |
| Cognitive limitations | 3% | 61% |
| Private insurance | 68% | 32% |

Keep the legend, percentages beside the bars, and two-line feature labels. Use
thin bars and smaller labels so the chart supports the bullets. Omit the chart
heading, axis percentages, and footnote. Include a “Back to cost distribution”
link.

**Speaker notes — 1:15 (optional, estimated)**

“To detect multivariate outliers, I used Isolation Forest configured to flag 5%
of respondents in the training data. I then compared out-of-pocket costs and
feature distributions between inliers and outliers, using survey weights.

Outliers were 3.9 times as likely as inliers to be among the top 1% of spenders.
Comparing group medians, they were 19 years older and had three more chronic
conditions and three more functional limitations. They were also much more
likely to report walking and cognitive limitations, and about half as likely
to have private health insurance.

I retained all outliers because their profiles were plausible, with no clear
evidence of data errors. Removing them could discard valuable training
information about people with greater health burden and extreme out-of-pocket
costs, limiting the model's ability to learn patterns relevant to these groups.”

**Transition:** Return to the cost-distribution slide or continue with questions.

**Sources:** [README: outlier analysis](../../README.md#outlier-analysis-details);
[EDA notebook: detection and profiling](../../notebooks/1_eda_and_preprocessing.py).

### A4 — Input features: what users provide

**Question:** What information does the model use, and how much does it ask of users?

**On the slide**

| Input group | Examples |
| --- | --- |
| Demographics | Age, region, family size |
| Socioeconomic information | Family income, education, employment status |
| Insurance and access | Insurance status, usual source of care |
| Health and lifestyle | Self-rated physical and mental health, smoking |
| Conditions and limitations | Diabetes, arthritis, difficulty walking |

- Examples from the 26 survey inputs; the full list is in the README
- Inputs chosen for expected predictive value and answers users can provide
  without medical records

**Talking points:** The current model uses all 26 survey inputs, which become
27 preprocessor inputs after adding a life-transition feature and 40 model-ready
columns after transformation. These counts do not equal the number of questions
on the planned form: related conditions can be grouped, and some values are
derived. User testing will assess the burden; feature reduction is an option if
fewer inputs would improve the experience. Refer to A5 for input timing.

**Promote:** After M3 when input design is relevant to the audience; allow 0:45.
Otherwise, use only for questions.

**Sources:** [README: feature details](../../README.md#feature-details);
[input definitions](../../src/constants.py);
[product requirements: input form](../specs/product_requirements.md).

### A5 — Feature timing: limits of next-year forecasting

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

**Promote:** After M3 or M8 for forecasting and validation questions; allow 1:00.

**Sources:** [input feature timing](../research/candidate_features.md);
[technical specifications: input selection](../specs/technical_specifications.md).

### A6 — Model tuning: comparisons on a fixed validation split

**Question:** What alternatives and tuning procedure were used?

**On the slide:** Compare linear regression, Elastic Net, decision tree, Random
Forest, SVM, and XGBoost with a population-median benchmark. Tune Elastic Net,
Random Forest, and XGBoost with 50 sampled configurations each. Select by
survey-weighted validation MdAE.

**Visual:** The compact baseline and tuned comparison from the README, clearly
labeled “validation.”

**Talking points:** A manual `ParameterSampler` loop routes weights through
nested pipeline and target-transform wrappers. This is not cross-validation.
Repeated selection on one validation split has uncertainty. The quantile model
reuses the tuned point-model parameters rather than receiving its own search.

**Promote:** Before M5 for a longer modeling discussion; allow 1:30.

**Sources:** [README: modeling](../../README.md);
[XGBoost tuning script](../../scripts/tune_xgboost.py).

### A7 — LLM benchmark: General versus Specific Intelligence

**Question:** Why not ask a general-purpose language model for an estimate?

**On the slide:** The documented Gemini 3 Flash benchmark has validation MdAE
of **$518**, versus **$163** for baseline Elastic Net.

**Visual:** Two bars labeled with model, metric, and validation split. Do not
mix these with test-set results from the final quantile model.

**Talking points:** This result describes one model and prompting setup, not all
LLMs.

**Promote:** Optional, after A6; allow 1:00.

**Sources:** [README: LLM benchmark](../../README.md);
[LLM benchmark script](../../scripts/benchmark_llm.py).

### A8 — Point-model diagnostics: residuals and prediction ranges

**Question:** Why did the lowest-MdAE model not meet the full product need?

**On the slide:** Elastic Net's narrow prediction range; larger errors on
expensive outcomes; differences across medical-complexity groups.

**Visual:** [Tuned-model residual analysis](../../figures/evaluation/tuned_models_validation_heteroscedasticity.png).
Use a readable crop or split the figure during design rather than shrinking it.

**Talking points:** MdAE and MAE summarize different aspects of the error
distribution. A lower MdAE alone does not establish useful risk separation.
Wider prediction spread alone also does not establish accuracy; inspect
residuals and subgroup errors alongside it.

**Promote:** After M5; allow 1:00.

**Sources:** [README: heteroscedasticity and subgroup analysis](../../README.md);
[modeling notebook](../../notebooks/2_modeling.py).

### A9 — Final model test results: coverage, width, and release gates

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

**Promote:** After M7 for evaluation depth; allow 1:00.

**Sources:** [README: release gates](../../README.md);
[modeling notebook: bootstrap implementation](../../notebooks/2_modeling.py).

### A10 — Final model test benchmarks: gains over simple estimates

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

**Promote:** After M7; allow 1:00.

**Source:** [Modeling notebook: test baseline comparisons](../../notebooks/2_modeling.py).

### A11 — Final model test audit: subgroup coverage gaps

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

**Promote:** After M8; allow 1:00.

**Sources:** [README: final subgroup audit](../../README.md);
[modeling notebook](../../notebooks/2_modeling.py).

### A12 — Final model test audit: actual versus predicted cost tiers

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

**Promote:** After M8 when discussing tail risk; allow 1:00.

**Sources:** [README: final reliability audit](../../README.md);
[modeling notebook](../../notebooks/2_modeling.py).

### A13 — SHAP explains how inputs shape the median estimate

**Question:** Which inputs influence predictions, and what can users infer?

**On the slide:** Insurance and family income have the largest average absolute
SHAP contributions. Explanations concern postprocessed q50 in dollars, through
the full inference function. They do not explain q90 or causal effects.

**Visual:** [Test SHAP feature importance](../../figures/evaluation/shap_feature_importance.png).

**Talking points:** Permutation SHAP operates on 27 interpretable preprocessor
inputs, including the effects of transformations and derived medical counts.
Correlated inputs can share attribution. Higher or lower observed spending
does not by itself mean higher or lower healthcare need.

**Promote:** Before M9 for explainability depth; allow 1:00.

**Sources:** [README: feature importance](../../README.md);
[SHAP metadata](../../app/data/shap_metadata.json).

### A14 — SHAP benchmarking: explanation quality and latency

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

**Promote:** Before M9 for ML engineering roles; allow 1:00.

**Sources:** [README: SHAP explanation details](../../README.md);
[SHAP metadata](../../app/data/shap_metadata.json);
[benchmark script](../../scripts/benchmark_shap.py).

### A15 — Inference architecture: shared prediction and explanation code

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

**Promote:** Before M9 for ML engineering roles; allow 1:30.

**Sources:** [DVC pipeline](../../dvc.yaml); [prediction code](../../src/prediction.py);
[technical specifications](../specs/technical_specifications.md); [unit tests](../../tests/unit/).

### A16 — Production monitoring: drift signals without observed outcomes

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

**Promote:** Before M9 for deployment and product discussion; allow 1:00.

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
