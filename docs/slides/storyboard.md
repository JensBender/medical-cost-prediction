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
  allocation below totals 10 minutes 30 seconds, leaving room for transitions.
- **Structure:** nine main slides and thirteen appendix slides. Appendix IDs
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

