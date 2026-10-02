# Medical cost prediction: presentation storyboard

Status: first draft for content review. No slide design yet.

## Presentation context

- **Setting:** job interviews, conference talks, meetups, and other settings.
- **Audience:** a technical audience. Assume basic ML
  knowledge, but no familiarity with MEPS or U.S. healthcare costs. Explain
  survey weights, quantiles, and evaluation metrics when introducing them.
- **Emphasis:** data science, with optional emphasis on ML engineering.
- **Language:** English.
- **Length:** 10–12 minutes of speaking, excluding questions. The initial
  allocation below totals 10 minutes, leaving room for transitions.
- **Structure:** a cover, eight content slides, and fourteen appendix slides. Appendix IDs
  remain stable so we can promote slides without rewriting their references.
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

| ID | Proposed slide title | Time | Purpose |
| --- | --- | ---: | --- |
| M1 | Medical cost prediction for annual budgeting | 0:15 | Cover and brief orientation |
| M2 | How much should I set aside for healthcare next year? | 1:00 | Establish the budgeting need and intended value |
| M3 | MEPS data connect accessible inputs to observed spending | 1:15 | Explain data scope, weights, and evaluation setup |
| M4 | Most out-of-pocket spending comes from a small share of adults | 1:00 | Show the data challenge that shaped modeling |
| M5 | Model selection: the lowest median error was not enough | 1:30 | Explain the key model-selection tradeoff |
| M6 | Quantile regression turns predictions into budgeting ranges | 1:30 | Explain the final model and its outputs |
| M7 | Final model test results: ranges and q90 show the clearest gains | 1:30 | Present held-out evidence and baseline comparisons |
| M8 | Final model test audit: overall coverage hides subgroup gaps | 1:00 | Demonstrate critical evaluation and limits |
| M9 | Model evaluation is complete; app development comes next | 1:00 | Show engineering work and close with priorities |
| | **Total** | **10:00** | |

## Main slides

### M1 — Medical cost prediction for annual budgeting

**Purpose:** Introduce the project and presenter before the problem statement.

**On the slide**

- Title: Medical Cost Planner (in the project header)
- Subtitle: Predicting annual out-of-pocket healthcare costs
- Presenter: Jens [surname]
- Event or setting: [presentation setting]
- Date: [presentation date]

**Visual:** Use the [project header](../../assets/header.png) as a wide banner,
preserving its proportions. Place the subtitle and presenter details below it.

**Speaker notes — 0:15**

“This project aims to help U.S. adults plan for next year's out-of-pocket
healthcare costs using machine learning.”

**Transition:** Introduce the budgeting problem and why it matters.

**Source:** [README: motivation](../../README.md).

### M2 — How much should I set aside for healthcare next year?

**Takeaway:** People need an estimate of next year's out-of-pocket
spending to plan their budget and HSA/FSA contributions.

**On the slide**

- **Why it matters:** annual out-of-pocket budgeting and HSA/FSA contribution planning
- **The challenge:** next year's care needs and out-of-pocket spending are uncertain
- **Our aim:** a quick, personalized out-of-pocket estimate using the person's
  insurance status and questions they can answer from memory

**Speaker notes — 1:00**

“How much should I set aside for healthcare next year? Here, we mean
out-of-pocket spending: the healthcare costs a person pays themselves. People
need to plan their budget and HSA or FSA contributions before they know what
care they will need. This project aims to give them a useful starting estimate
quickly, using demographic and health questions they can answer from memory,
without medical records or a list of anticipated treatments.”

**Transition:** Explain the dataset that makes this target measurable.

**Sources:** [README: motivation](../../README.md);
[product requirements: user needs and UX rationale](../specs/product_requirements.md).

### M3 — MEPS data connect accessible inputs to observed spending

**Takeaway:** MEPS links accessible personal information to annual out-of-pocket
spending; weighting and stratification shape training and evaluation.

**On the slide**

- **MEPS 2023:** 14,768 adult respondents represent approximately **260 million
  U.S. adults**
- **Inputs → target:** 26 candidate features covering demographics, insurance,
  and health → observed annual out-of-pocket spending
- **Survey weights:** how many people each respondent represents; used in
  training and evaluation
- **80% training / 10% validation / 10% test:** stratified by spending, with a
  separate zero-cost group and finer bins for high costs

**Visual:** A compact diagram connecting the input groups to annual out-of-pocket
spending, with a sample-to-population callout and an 80/10/10 split bar.
Footnote: “U.S. civilian noninstitutionalized adults; one survey year.”

**Speaker notes — 1:15**

“MEPS—the Medical Expenditure Panel Survey—records healthcare spending alongside
demographics, insurance, and health information. I selected 26 candidate features
that users can report without medical records, with annual out-of-pocket spending
as the target. The 14,768 adult respondents represent about 260 million adults.
Each survey weight tells us how many people that respondent represents. These
weights account for unequal selection probabilities, including oversampling of
certain groups, and survey nonresponse. I use them in both training and evaluation.
The 80/10/10 split is random within spending groups: zero costs have their own
group, and high costs use finer bins. This helps preserve the spending distribution
across splits and reduces chance imbalances in rare, expensive cases.”

**Transition:** Show why the spending distribution makes this task difficult.

**Sources:** [README: data and preprocessing](../../README.md);
[EDA notebook script: survey weights and data splitting](../../notebooks/1_eda_and_preprocessing.py);
[preprocessing script](../../scripts/preprocess.py);
[MEPS 2023 documentation: sampling and weights](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.shtml).

**Detail for questions:** The 26 candidate variables become 27 preprocessor
inputs after adding a life-transition feature. Medical feature derivation and
encoding produce 40 model-ready columns. These counts describe different stages.
Preprocessing is fitted on training data. The exact spending-bin boundaries and
weighting methodology belong in technical questions or the appendix.

### M4 — Most out-of-pocket spending comes from a small share of adults

**Takeaway:** Typical errors and errors on expensive years describe different
parts of model performance.

**On the slide**

- **22.3%** of the weighted adult population has zero out-of-pocket spending
- The highest-spending **20% accounts for 79.3%** of spending
- Evaluate typical error alongside large errors and uncertainty

**Visual:** Use the [Lorenz curve](../../figures/eda/lorenz_curve.png), with its two
headline annotations. Simplify labels for slide readability during design.

**Speaker notes — 1:00**

“The distribution has a large mass at zero and a long right tail. About 22% of
adults have no out-of-pocket spending, while the top fifth accounts for almost
80% of spending. This makes a single error metric an incomplete summary.
Median absolute error describes a typical miss, but it tells us little about
the worst misses. Mean absolute error and R² provide additional diagnostics.
I also retained unusual but valid health profiles instead of deleting them just
because they were outliers. Removing them would make the task look easier while
discarding cases relevant to the intended use.”

**Transition:** This distribution explains why the validation leaderboard was
only the start of model selection.

**Sources:** [EDA notebook script: zero costs and cost concentration](../../notebooks/1_eda_and_preprocessing.py);
[README: EDA and outlier analysis](../../README.md).

### M5 — Model selection: the lowest median error was not enough

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

**Visual:** A small comparison table and a focused excerpt from
[validation residual diagnostics](../../figures/evaluation/tuned_models_validation_heteroscedasticity.png).
Keep the complete figure in A4 if it cannot be read comfortably on this slide.

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

**Visual:** A schematic cost axis with q25, q50, q75, and q90. Label it
“conceptual”; avoid implying it is a prediction for an actual person.

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

### M7 — Final model test results: ranges and q90 show the clearest gains

**Takeaway:** The final model meets the project's overall performance gates,
with its strongest evidence of added value in interval and upper-quantile scores.

**On the slide**

| Held-out test metric | Result | Reference |
| --- | ---: | --- |
| Median absolute error for q50 | **$240** | Population baseline: $248 |
| Typical-range coverage | **47.3%** | Target: 50% |
| Safety-cushion coverage | **91.0%** | Target: 90% |

- Versus population baseline: **11.2% lower interval score**, **15.6% lower q90 loss**
- Median-error improvement over the population baseline remains uncertain

**Visual:** One results table with the two baseline improvements beneath it.
Footnote: “Survey-weighted test metrics; dollar amounts in 2023 USD.”

**Speaker notes — 1:30**

“On the test set, the median absolute error is $240. This means roughly half of
the weighted test population has an absolute error no larger than that amount;
it is not the mean error. The typical range covers 47.3% of spending outcomes,
close to its 50% target, and q90 covers 91%. Mean range widths are $912 for
q25 to q75 and $2,032 for q50 to q90, and the model meets all five project
performance gates. But a simple population estimate is competitive on median
error: $248 versus $240. The bootstrap comparison does not establish a clear
MdAE improvement. The stronger result is an 11.2% reduction in interval score
and a 15.6% reduction in q90 pinball loss. Those assess width and misses, or the
size and direction of errors, rather than coverage alone. The age-group
comparison also supports gains on these measures.”

**Transition:** Overall metrics still need to be checked against the groups
and situations where the model is less reliable.

**Sources:** [README: release gates](../../README.md);
[modeling notebook: test comparison with simple baselines](../../notebooks/2_modeling.py).
Confidence intervals, widths, and scoring definitions are in A5 and A6.

### M8 — Final model test audit: overall coverage hides subgroup gaps

**Takeaway:** The model's limitations affect both interpretation and the next
validation steps.

**On the slide**

- Rare high-cost years remain difficult to anticipate
- Typical ranges undercover some groups, including uninsured users and people
  reporting poor mental health
- One-year holdout performance does not establish prospective performance
- Subgroup audits inform safeguards; they do not prove absence of bias

**Visual:** Two selected subgroup coverage markers against the 50% target,
or a focused excerpt from the final audit. Use A7 for the complete audit.

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

**Visual:** A simple workflow from training artifacts to shared inference to
the planned application. Visually distinguish implemented and planned elements.

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

### A1 — Survey weights in training and population evaluation

**Question:** Why use weights instead of evaluating the raw sample?

**On the slide:** `PERWT23F` supplies person weights. Training passes weights
through to the estimator; metrics and simple benchmarks use weights too.
Population summaries concern civilian noninstitutionalized adults.

**Visual:** A small illustrative weighted-versus-unweighted example, clearly
labeled as conceptual rather than MEPS observations.

**Talking points:** Weights change whose errors contribute most to the objective
and evaluation. They do not by themselves remove missing-feature bias, guarantee
representativeness of future app users, or account for survey clustering in
uncertainty estimates.

**Promote:** After M3 for statistical depth; allow 1:00.

**Sources:** [EDA notebook](../../notebooks/1_eda_and_preprocessing.py);
[modeling helpers](../../src/modeling.py).

### A2 — Feature timing: limits of next-year forecasting

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

**Sources:** [candidate feature timing](../research/candidate_features.md);
[technical specifications: candidate features](../specs/technical_specifications.md).

### A3 — Model tuning: comparisons on a fixed validation split

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

### A4 — Point-model diagnostics: residuals and prediction ranges

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

### A5 — Final model test results: coverage, width, and release gates

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

**Visual:** This table, with gate results readable beside each estimate.

**Talking points:** Point estimates pass the gates. The entire confidence
interval does not need to fall inside the gate under the reported decision
rule. These are project criteria, not external certification. The notebook
resamples rows and retains their weights.

**Promote:** After M7 for evaluation depth; allow 1:00.

**Sources:** [README: release gates](../../README.md);
[modeling notebook: bootstrap implementation](../../notebooks/2_modeling.py).

### A6 — Final model test benchmarks: gains over simple estimates

**Question:** Why not give everyone a population or age-group estimate?

**On the slide**

| Test comparison | Versus population baseline | Versus age-group baseline |
| --- | ---: | ---: |
| q50 MAE reduction | 9.8% | 7.6% |
| Typical-range interval-score reduction | 11.2% | 9.0% |
| q90 pinball-loss reduction | 15.6% | 14.3% |

MdAE: XGBoost **$240**; population baseline **$248**; age-group baseline **$305**.

**Visual:** A grouped comparison of these three loss reductions. Add the saved
notebook's paired-bootstrap intervals when preparing the chart; do not invent
interval endpoints.

**Talking points:** Benchmarks use training data. The interval score penalizes
width and observations outside the range. At q90, pinball loss penalizes
underprediction more heavily than overprediction. The notebook reports
confidence intervals above zero for the three reductions shown, but the small
MdAE improvement versus the population baseline remains uncertain. Positive
skill is not a percentage-point improvement in coverage.

**Promote:** After M7; allow 1:00.

**Source:** [Modeling notebook: test baseline comparisons](../../notebooks/2_modeling.py).

### A7 — Final model test audit: subgroup coverage gaps

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

### A8 — Final model test audit: actual versus predicted cost tiers

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

### A9 — SHAP explains how inputs shape the median estimate

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

### A10 — SHAP benchmarking: explanation quality and latency

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

### A11 — Inference architecture: shared prediction and explanation code

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

### A12 — Production monitoring: drift signals without observed outcomes

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

### A13 — LLM benchmark: General versus Specific Intelligence

**Question:** Why not ask a general-purpose language model for an estimate?

**On the slide:** The documented Gemini 3 Flash benchmark has validation MdAE
of **$518**, versus **$163** for baseline Elastic Net. 

**Visual:** Two bars labeled with model, metric, and validation split. Do not
mix these with test-set results from the final quantile model.

**Talking points:** This result describes one model and prompting setup, not all
LLMs. 

**Promote:** Optional, after A3; allow 1:00.

**Sources:** [README: LLM benchmark](../../README.md);
[LLM benchmark script](../../scripts/benchmark_llm.py).

### A14 — U.S. healthcare costs: out-of-pocket spending and HSA/FSA planning

**Question:** What is out-of-pocket spending, and how does it relate to HSA/FSA planning?

**On the slide**

- The infographic explains out-of-pocket costs, HSA/FSA planning, and the costs
  included in the prediction target
- Scope note: annual costs for an individual U.S. civilian noninstitutionalized adult

**Visual:** Use the existing [healthcare-cost infographic](../../assets/infographic_healthcare_costs.png)
as the main content.

**Talking points:** Out-of-pocket spending includes copays, deductibles, and uncovered services; premiums and over-the-counter purchases are excluded from the target. The estimate supports  individual annual budgeting, with household totals, procedure prices, and insurance plan comparisons outside scope.

**Sources:** [README: target variable](../../README.md);
[product requirements: out of scope](../specs/product_requirements.md).

## Adaptation and file organization

This storyboard defines the default 10–12-minute presentation. Adapt the
emphasis, level of detail, and slide selection when the audience, available time,
and setting are known. Move appendix slides into the main presentation as needed.

Create variants as concrete requirements arise, such as a data science or ML
engineering interview, tech meetup, or conference talk. Keep variant notes in
`docs/slides/variants/`, referencing shared slide IDs and documenting the changes.

Keep the storyboard, variant notes, and slide-generation code in Git under
`docs/slides/`. Reuse existing project figures; add presentation-specific visuals
in `assets/` and generated PPTX/PDF files in `exports/` when needed. Keep
exports out of Git.
