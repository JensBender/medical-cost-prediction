# Responsible AI

*Last reviewed: September 2026*

This document explains how the Medical Cost Planner manages AI risk and which
legal frameworks may become relevant.

## Intended Use

The Medical Cost Planner is a US-focused consumer budgeting tool. It uses
self-reported demographic, insurance, and health information to estimate an
individual's annual out-of-pocket healthcare costs. It presents a plan-around
estimate, a typical range, and a safety cushion rather than a single precise
forecast.

The tool does not make decisions about:

- Medical diagnosis, treatment, or patient care
- Insurance eligibility, underwriting, pricing, or plan selection
- Specific bills, procedures, or provider prices

These boundaries are part of the product design. Expanding beyond them requires
a new legal and risk review.

## Responsible AI Approach

| Principle | Project practice |
| :--- | :--- |
| **Population-aware evaluation** | Use the MEPS person-level survey weight (`PERWT23F`) for training and population-level evaluation. |
| **Subgroup reliability** | Compare error and prediction-interval coverage across demographic, health, insurance, income, education, and regional groups. |
| **Uncertainty communication** | Show XGBoost quantile predictions as a plan-around estimate (`q50`), typical range (`q25`-`q75`), and safety cushion (`q90`). |
| **Transparency** | Explain the intended use, data year, target definition, exclusions, known limitations, and factors that most influenced each estimate. |
| **Privacy** | Require no account or direct identifiers, process prediction inputs in memory, and retain only coarse aggregate monitoring counters. |
| **Ongoing review** | Monitor aggregate input and prediction drift and repeat weighted performance and subgroup audits on newer MEPS data. |

The project uses the voluntary
[NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
as a reference for identifying, measuring, and managing AI risk.

## Model Evidence and Decision

The tuned point-estimate models show broadly similar subgroup error patterns,
which makes an architecture-specific fairness issue less likely. Tree-based
models are more reliable than Elastic Net for several groups with greater
medical complexity, including people in poor health, people with multiple
chronic conditions, and people with walking limitations. Elastic Net performs
best mainly for lower-complexity groups.

This subgroup analysis, together with the heteroscedasticity analysis, informed
the decision to use XGBoost quantile regression. The final model communicates
the large person-level uncertainty in healthcare costs instead of hiding it
behind one point estimate.

The final-model audit found no broad pattern of undercoverage across sex, age,
race or ethnicity, region, or walking-limitation groups. It also identified
specific limitations: typical-range coverage is low for uninsured users, people
reporting poor mental health, people with low family income, and people with a
doctorate degree. Rare high-cost years remain the largest source of error.

These findings shape the product safeguards:

- Display prediction ranges and a safety cushion
- Show a planning note when uncertainty is elevated
- State clearly that the result is a budgeting estimate, not a bill estimate
- Recheck performance and subgroup results on future MEPS survey years

## US Legal Context

The NIST AI RMF provides a voluntary structure for the project's responsible AI
work. Project alignment includes documenting the intended use, measuring
overall and subgroup performance, communicating uncertainty, protecting user
data, and planning post-launch monitoring.

Section 5 of the Federal Trade Commission Act prohibits unfair or deceptive
practices. For this product, the main implications are to support accuracy and
fairness claims with reliable evidence, describe the model's capabilities
honestly, disclose important limitations, and follow the stated privacy
practices.

## Privacy and Monitoring

The MVP is designed for anonymous, stateless predictions:

- Do not request names, email addresses, account credentials, or medical
  records
- Do not persist prediction inputs, exact outputs, SHAP explanations, IP
  addresses, user agents, request IDs, or session identifiers
- Disable Gradio analytics and flagging
- Store only coarse aggregate counters with small-cell suppression
- Keep hosting-provider infrastructure logs separate from model monitoring

The hosting provider may process its own infrastructure logs under its privacy
policy. That behavior must be reviewed before launch and described accurately
in user-facing privacy information.

## Future Scope Changes

This assessment is based on the current use as a US consumer budgeting tool
with anonymous, stateless predictions. Review the project's responsible AI and
legal requirements whenever its intended use, data handling, deployment
region, or relevant policies change.

A separate assessment is especially important before:

- Using predictions in healthcare-provider, payer, or insurance decisions,
  which may introduce health nondiscrimination, insurance, or health-data
  requirements
- Collecting accounts, medical records, prediction histories, or follow-up
  spending outcomes, which would change the project's current privacy model
- Launching in the European Union, where the
  [GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj) may apply to the
  processing of demographic and health information even without long-term
  storage

Under the [EU AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), the
current consumer budgeting purpose is different from listed high-risk uses
such as health-insurance risk assessment and pricing. Repurposing the model for
one of those uses would change the regulatory assessment.
