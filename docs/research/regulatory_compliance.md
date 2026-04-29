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

## Risk Management Philosophy

This project adopts a **Governance-First** approach to AI safety, based on three core principles:

1.  **Risk Governance over Risk Elimination**: We acknowledge that "zero bias" is mathematically impossible. Our goal is **Defensible Transparency**—using rigorous audit trails (Stratified Error Analysis) to identify, document, and justify any performance disparities.
2.  **Decision Support, Not Automation**: The tool is designed as an **advisory aid** for consumers. By keeping a "Human-in-the-loop," we significantly lower the regulatory risk profile compared to automated decision-making systems (e.g., automated insurance denials).
3.  **The "Audited Alternative"**: We argue that an audited ML model is a safer and more equitable alternative to the status quo: **unaudited human bias** or **crude rules-of-thumb** (e.g., flat age-based pricing) which are often more discriminatory but harder to detect.

## Use Case & Risk Escalation

The regulatory requirements for this project depend heavily on its **Business Context**.

| Use Case | Risk Category (NIST/EU) | Regulatory Impact |
| :--- | :--- | :--- |
| **Consumer Planning Helper** (Current) | **Limited/Low Risk** | Focus on transparency and user notice. |
| **Provider Financial Counseling** | **Medium Risk** | Requires clear documentation of "Advisory" status. |
| **Insurance Underwriting/Pricing** | **High-Risk (Annex III)** | Requires formal **Conformity Assessments**, strict data quality audits, and human oversight logs. |

## Detailed Regulatory Mapping

| Dimension | Primary: US Market Perspective | Expansion: EU Market Roadmap | Implementation in this Project |
| :--- | :--- | :--- | :--- |
| **Risk Classification** | Categorized under **NIST** based on impact on human rights and civil liberties. | Classified as **"High-Risk"** (Annex III) for insurance/medical prioritization. | We categorize the system as "Advisory" for planning purposes to manage risk. |
| **Bias & Fairness** | **FTC** enforces against "unfair" disparate impacts on protected groups. | Mandatory bias monitoring and **conformity assessments**. | **Stratified Error Analysis** monitors MdAE across Legally Protected and Vulnerable & Proxy Groups. |
| **Transparency** | Per **AI Bill of Rights**, providing clear notices on AI use is a priority. | Formal **Model Cards** and technical documentation are strict requirements. | All performance metrics are disclosed per group in a structured format. |
| **Data Governance** | Ensuring representativeness per the **2023 AI Executive Order**. | Strict **GDPR** compliance and data quality standards (Art. 10 AI Act). | Use of **Survey Weights** ensures the training set represents the true US population. |

## Implementation of Fairness

To satisfy these requirements, the project implements the following technical safeguards:
1.  **Population Weighting**: Using MEPS survey weights to ensure the model doesn't "over-optimize" for majority groups at the expense of minorities.
2.  **Tiered Evaluation**: Stratified error metrics (MdAE) across:
    *   **Legally Protected Groups**: (Sex, Age, Race) for direct statutory compliance.
    *   **Vulnerable & Proxy Groups**: (Income, Education, Region, Disability Proxies) to detect indirect Disparate Impact.
3.  **Legitimate Business Necessity Defense**: Any detected disparities are investigated to determine if they reflect legitimate medical complexity (e.g., chronic condition counts) rather than algorithmic bias.
4.  **Advisory Focus**: The tool as a financial planning aid for consumer (low risk), not a binding automated decision-maker (high risk).
