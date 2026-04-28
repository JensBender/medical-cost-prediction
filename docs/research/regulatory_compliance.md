# Regulatory Compliance and Ethical AI

This document outlines the regulatory and ethical frameworks governing the Medical Cost Prediction project. It focuses on the primary **US Market** context while providing a roadmap for future expansion into the **European Market** under the EU AI Act.

## Overview

Predicting medical expenditures is a high-stakes application that falls under specific regulatory scrutiny in both the US and the EU. This project adheres to "best-in-class" standards for transparency, fairness, and robustness to ensure it is both compliant and trustworthy.

## Primary Market: United States

The model is trained on the **Medical Expenditure Panel Survey (MEPS) 2023** data and is intended for use in the US market.

### Regulatory Frameworks
*   **FTC Act Section 5**: Prohibits unfair or deceptive acts, including algorithmic discrimination that leads to disparate impacts on protected groups.
*   **NIST AI Risk Management Framework (AI RMF)**: Provides a voluntary but standard-setting framework for managing risks to individuals and organizations.
*   **Executive Order 14110 (2023)**: Sets new standards for AI safety and security, emphasizing the need for representative data and bias mitigation.
*   **Sector-Specific Laws**:
    *   **Healthcare**: ACA Section 1557 (Nondiscrimination).
    *   **Finance**: ECOA (Equal Credit Opportunity Act) if used for financial eligibility.

## Expansion Roadmap: European Union

While not the primary target market for the current iteration, the project's technical foundation is designed for future compliance with EU standards.

### Regulatory Frameworks
*   **EU AI Act**: In healthcare and insurance pricing, AI systems are likely classified as **"High-Risk"** (Annex III).
*   **GDPR (DSGVO)**: Article 22 regulates automated individual decision-making, requiring transparency and the right to human intervention.

## Detailed Regulatory Mapping

| Dimension | Primary: US Market Perspective | Expansion: EU Market Roadmap | Implementation in this Project |
| :--- | :--- | :--- | :--- |
| **Risk Classification** | Categorized under **NIST** based on impact on human rights and civil liberties. | Classified as **"High-Risk"** (Annex III) for insurance/medical prioritization. | We categorize the system as "Advisory" for planning purposes to manage risk. |
| **Bias & Fairness** | **FTC** enforces against "unfair" disparate impacts on protected groups. | Mandatory bias monitoring and **conformity assessments**. | **Stratified Error Analysis** monitors MdAE across Age, Income, and Health segments. |
| **Transparency** | Per **AI Bill of Rights**, providing clear notices on AI use is a priority. | Formal **Model Cards** and technical documentation are strict requirements. | All performance metrics are disclosed per group in a structured format. |
| **Data Governance** | Ensuring representativeness per the **2023 AI Executive Order**. | Strict **GDPR** compliance and data quality standards (Art. 10 AI Act). | Use of **Survey Weights** ensures the training set represents the true US population. |

## Implementation of Fairness

To satisfy these requirements, the project implements the following technical safeguards:
1.  **Population Weighting**: Using MEPS survey weights to ensure the model doesn't "over-optimize" for majority groups at the expense of minorities.
2.  **Stratified Evaluation**: Disaggregation of error metrics (MdAE) across protected and vulnerable populations.
3.  **Proxy Detection**: Analyzing performance across socioeconomic status (Poverty Category) as a proxy for other protected attributes.
4.  **Advisory Focus**: Positioning the tool as a financial planning aid rather than a binding decision-making engine to lower the regulatory risk profile.
