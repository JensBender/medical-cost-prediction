# Candidate Features for Medical Cost Prediction
Deep Research report on the 30 best candidate features for predicting out-of-pocket medical costs using MEPS-HC 2023 data.  
> **Tool:** Gemini Deep Research  
> **Date:** 2025-12-19  

# **Analytical Framework for Personalized Healthcare Out-of-Pocket Expenditure Forecasting: A Multi-Dimensional Approach Utilizing the 2023 Medical Expenditure Panel Survey**

## **The Structural Imperative for Consumer-Centric Health Expenditure Prediction**

The current landscape of the United States healthcare economy is defined by a paradoxical state of "opaque transparency." While legislative mandates have accelerated the release of raw pricing data from both providers and payers, the resulting datasets remain fundamentally decoupled from the consumer's decision-making process. For the individual, the "black box" of healthcare pricing is not merely a failure of data availability, but a failure of data translation. Consumers are rarely concerned with the unit price of a single medical event, such as a diagnostic image or a laboratory test, in isolation. Instead, the primary financial anxiety revolves around the aggregate, longitudinal exposure of the household budget over an annual cycle.1

The rise of high-deductible health plans (HDHPs) and the proliferation of tax-advantaged accounts, such as Health Savings Accounts (HSAs) and Flexible Spending Accounts (FSAs), have transferred significant actuarial risk from the employer and insurer to the individual. During open enrollment periods, employees are forced to act as their own risk managers, often with little more than historical anecdotes or generic averages to guide their financial commitments. The Medical Cost Planner addresses this systemic gap by employing machine learning to predict annual out-of-pocket (OOP) healthcare costs. By utilizing the Medical Expenditure Panel Survey (MEPS), the tool provides a personalized forecasting mechanism that transforms high-fidelity national data into actionable financial intelligence.4

The fundamental challenge in this domain is the reconciliation of predictive accuracy with user accessibility. Traditional medical risk adjustment models rely on clinical data, such as ICD-10 diagnosis codes or pharmacy claims, which are inaccessible to the average layperson. The Medical Cost Planner pivots toward a "UX-First" methodology, identifying a subset of demographic and self-reported health variables that serve as high-fidelity proxies for clinical burden. This strategy ensures that a user can complete a cost forecast in under 90 seconds, leveraging information they already know, while still maintaining a degree of accuracy—specifically a Median Absolute Error (MdAE) within $500—that is meaningful for personal financial planning.1

## **The Medical Expenditure Panel Survey as the Actuarial Foundation**

The efficacy of a predictive model is intrinsically linked to the representativeness of its training data. The Medical Cost Planner is anchored in the MEPS-HC 2023 Full Year Consolidated Data File (H251), which represents the gold standard for individual-level healthcare expenditure research in the United States. Managed by the Agency for Healthcare Research and Quality (AHRQ), the MEPS dataset utilizes an overlapping panel design that captures a nationally representative sample of the civilian noninstitutionalized population.5

| MEPS Component | Data Focus | Utility in Prediction |
| :---- | :---- | :---- |
| Household Component (HC) | Individual demographics, health status, and expenditures. | Primary source for user-reported features and target variables. |
| Medical Provider Component (MPC) | Claims-based verification from doctors and hospitals. | Enhances the accuracy of the self-reported expenditure data. |
| Insurance Component (IC) | Employer-sponsored insurance plan details. | Provides context for the distribution of total vs. out-of-pocket costs. |

The H251 file is particularly valuable because it integrates a wide array of variables across survey administration, demographics, income, person-level conditions, health status, and utilization.9 For the purposes of the Medical Cost Planner, the MEPS longitudinal structure allows for the observation of how individual health characteristics in one period translate into financial outcomes over the course of a full calendar year. This historical perspective is essential for training models to recognize the latent correlations between lifestyle factors and expenditure.11

## **Detailed Analysis of the Target Variable: TOTSLF23**

The central target of the predictive model is TOTSLF23, defined as the total amount paid out-of-pocket by an individual or their family for all medical events during the year 2023\. This variable is a composite of multiple sub-components that reflect the real-world financial friction experienced by consumers in the healthcare market.2

### **Composition of Out-of-Pocket Expenditure**

Out-of-pocket costs are the net result of insurance plan design, provider pricing, and individual utilization. The TOTSLF23 variable captures the sum of self-payments across diverse service categories:

* **Office-Based Visits (OBSLF23):** This includes co-pays and deductibles for routine primary care, specialist consultations, and preventive screenings. These often represent the most frequent point-of-service financial interactions for the average user.2  
* **Prescribed Medicines (RXSLF23):** This component tracks the cost of prescription drugs. Research indicates that pharmacy spending is a highly significant driver of overall health costs, particularly for individuals with chronic conditions managed through long-term medication regimens.1  
* **Emergency and Acute Care (ERSLF23, IPSLF23):** While less frequent, these costs are often the most devastating to a household budget. The model must account for the statistical probability of these high-magnitude events based on the user's health profile.9  
* **Specialty Services (DVSLF23, VISLF23):** Dental and vision services are frequently siloed in insurance discussions but are integral to the consumer’s actual cash outflow. Their inclusion in TOTSLF23 ensures a comprehensive view of healthcare's financial impact.9

### **Exclusions and Actuarial Rationale**

It is critical to note that TOTSLF23 excludes insurance premiums. From a personal finance perspective, premiums are fixed, known costs determined at the time of enrollment. The Medical Cost Planner focuses on the variable, uncertain expenditure that occurs during the delivery of care. Furthermore, over-the-counter (OTC) medications are excluded because they are not systematically tracked in the MEPS provider-verified components and do not typically interact with the deductible structures that FSAs/HSAs are designed to mitigate.5

For the uninsured segment of the population, TOTSLF23 often converges with total expenditure (TOTTCH23), reflecting the full market price of care. However, for most users, the delta between total cost and out-of-pocket cost is dictated by their insurance status (INSCOV23), which serves as a primary feature in the predictive logic.2

## **Persona-Driven Financial Risk Assessment**

The utility of the Medical Cost Planner is manifested differently across various consumer segments. Each persona has a unique set of financial constraints and decision-making deadlines.

### **The Open Enrollment Planner**

The "Open Enrollment Planner" represents the most common use case: employees navigating the annual benefit selection process. For these users, the primary need is to optimize the allocation of pre-tax dollars into FSAs or HSAs. Under-contributing leads to missing out on tax savings, while over-contributing (particularly to FSAs) risks the loss of funds under "use-it-or-lose-it" rules. The tool provides a quantitative basis for choosing between $1,000 and $3,000 contributions based on the user's anticipated utilization.2

### **The Budgeter and the Gig Worker**

For the "Budgeter," the focus shifts to liquidity management. These individuals require an estimate of the "worst-case scenario" to ensure their emergency funds are sufficiently sized to cover deductibles and out-of-pocket maximums. Similarly, the "Gig Worker"—who may be uninsured or underinsured—uses the tool to weigh the financial risk of skipping coverage against the cost of a private plan. In this context, the model functions as a comparative risk-assessment tool, highlighting the potential financial liability of a sudden medical event.2

### **The Newly Diagnosed and the Caregiver**

The "Newly Diagnosed" persona faces a sudden shift in their financial trajectory. A diagnosis of diabetes or hypertension introduces recurring medication and monitoring costs that were previously absent from their budget. The Medical Cost Planner allows these users to update their health profile and immediately see the projected impact on their annual bottom line. For the "Caregiver," the tool facilitates the management of healthcare costs for dependents, such as an elderly parent. This is particularly relevant for the "sandwich generation," who must balance their own children's costs with those of an aging relative.2

## **Feature Engineering and the "UX-First" Constraint**

The primary technical innovation of the Medical Cost Planner lies in its feature selection process. To satisfy the 90-second completion target, the candidate features must be "accessible"—defined as information a user knows offhand without consulting external records.2

### **Candidate Screening and Feature Importance**

The feature selection process begins with the identification of all MEPS variables that meet the accessibility criterion. These candidates are then ranked using machine learning models to determine their predictive importance. Research suggests that approximately 30 features represent the optimal pool for screening, which are eventually narrowed down to the 10–12 interactions required for the final UI.1

| Feature Category | Candidate MEPS Variables | Predictive Significance |
| :---- | :---- | :---- |
| Demographics | AGE31X, SEX, MARRY23X, FAMSZE23 | Fundamental indicators of medical need and social risk. |
| Socio-Economics | POVCAT23, HIDEG, REGION23 | Correlates with access to care and insurance quality. |
| Health Status | RTHLTH31, MNHLTH31, ADSMOK42, BMI | High-fidelity proxies for clinical and lifestyle risk. |
| Priority Conditions | HIBPDX, DIABDX, ASTHDX, CANCERDX | Primary drivers of recurring and high-magnitude costs. |
| Insurance Status | INSCOV23, INSURC23 | The primary filter for out-of-pocket exposure. |

### **The Power of Self-Rated Health Proxies**

One of the most robust findings in health expenditure research is the predictive power of subjective health ratings. RTHLTH31 (Perceived Physical Health) and MNHLTH31 (Perceived Mental Health) consistently rank among the top predictors of cost, often outperforming specific diagnosis codes. This is because these variables capture a holistic sense of well-being and functional limitation that precedes or encompasses formal medical diagnoses. A user reporting "Fair" or "Poor" health is statistically likely to incur higher costs across all categories, making these variables essential "quick-answer" inputs for a 90-second form.1

### **The Multiplier Effect of Chronic Conditions**

The presence of "priority conditions" serves as a major driver of expense. Conditions such as diabetes (DIABDX) and hypertension (HIBPDX) are associated with continuous pharmacy costs and frequent office visits. However, the model must also account for the multiplier effect of comorbidities. The financial burden of having both diabetes and a history of heart disease is greater than the sum of their individual expected costs. Machine learning models, particularly tree-based ensembles, are superior to linear models in detecting these non-linear interactions.2

## **Analysis of the 30 Best Candidate Features for Expenditure Prediction**

To develop a high-performance model that adheres to the user-centric constraints, 30 specific MEPS variables have been identified as the most suitable candidates. These variables are selected because they are known offhand by users, have established statistical significance in the literature, and are well-documented in the H251 codebook.1

### **Demographic Variables (Primary Drivers)**

1. **AGE31X (Age):** Perhaps the single most important demographic factor. Healthcare consumption follows a clear U-shaped curve, with higher costs for the very young (neonatal) and the elderly (chronic disease management).2  
2. **SEX (Gender):** Historically associated with differing utilization patterns, particularly in reproductive health and the prevalence of certain chronic conditions.4  
3. **FAMSZE23 (Family Size):** Total household expenditure is a function of family size, as more individuals increase the statistical probability of a high-cost event.2  
4. **MARRY23X (Marital Status):** Often used as a proxy for social support and household income stability, both of which correlate with healthcare access and preventive care utilization.6

### **Socio-Economic and Location Variables (Access Proxies)**

5. **POVCAT23 (Poverty Category):** A constructed variable indicating the ratio of household income to the federal poverty line. This determines eligibility for subsidies and the likelihood of having comprehensive insurance.8  
6. **HIDEG (Highest Degree):** Education level is a well-established social determinant of health, often correlating with health literacy and the use of preventive services.8  
7. **REGION23 (Census Region):** Geographic variation in healthcare pricing and provider density significantly impacts total expenditures.4

### **Subjective Health and Lifestyle (High-Fidelity Proxies)**

8. **RTHLTH31 (Perceived Physical Health):** As discussed, this is a powerful, low-friction input for predicting overall utilization intensity.1  
9. **MNHLTH31 (Perceived Mental Health):** Poor mental health is increasingly recognized as a multiplier for physical healthcare costs due to treatment adherence and systemic physiological effects.1  
10. **ADSMOK42 (Smoking Status):** A direct indicator of future respiratory and cardiovascular risk.4  
11. **BMI (Body Mass Index):** While not explicitly in every PUF, it is often derived from weight/height variables in MEPS. High BMI is a primary driver of metabolic and orthopedic costs.4

### **Priority Conditions (The "Cost Engine")**

12. **HIBPDX (Hypertension):** Requires lifelong medication and monitoring, making it a predictable "base" cost.9  
13. **CHOLDX (High Cholesterol):** Similar to hypertension, it drives pharmacy spending and periodic lab work.9  
14. **CHDDX (Coronary Heart Disease):** Represents significant risk for high-cost acute events.9  
15. **ANGIDX (Angina):** Indicator of advanced cardiovascular disease.9  
16. **MIDX (Heart Attack History):** A history of myocardial infarction is a strong predictor of future inpatient and specialist costs.5  
17. **STRKDX (Stroke):** Associated with high rehabilitation and long-term care costs.5  
18. **EMPHDX (Emphysema):** Drives costs for oxygen, medications, and frequent respiratory events.9  
19. **CHBRON31 (Chronic Bronchitis):** Adds to the respiratory disease burden.9  
20. **ASTHDX (Asthma):** Especially significant for younger personas, driving emergency room visits and inhaler costs.2  
21. **CANCERDX (Cancer):** A primary driver of extreme "tail" costs in the expenditure distribution.5  
22. **DIABDX (Diabetes):** One of the most significant predictors of sustained, high-magnitude outpatient and pharmacy spending.2  
23. **ARTHDX (Arthritis):** Drives long-term pharmaceutical and physical therapy costs.5  
24. **DEPRDX (Depression):** Associated with higher utilization across all medical service categories.5

### **Insurance and Activity Limitations (Functional Burden)**

25. **INSCOV23 (Insurance Coverage Indicator):** The most critical variable for converting total cost to out-of-pocket cost.2  
26. **ADLHLP31 (ADL Help Needed):** Indicates a need for assistance with activities of daily living, a high-cost functional indicator.5  
27. **IADLHP31 (IADL Help Needed):** Similar to ADL, this tracks higher-level functional limitations.5  
28. **COGLIM31 (Cognitive Limitations):** Tracks mental functional capacity, which correlates with specialized care needs.6  
29. **WRKLIM31 (Work Limitation):** A proxy for the severity of underlying health conditions and their impact on daily life.6  
30. **ANYLIM23 (Any Limitation):** A composite variable of all physical or cognitive limitations, serving as a broad indicator of high healthcare needs.6

## **Machine Learning Methodologies for Healthcare Expenditure**

Modeling healthcare costs presents a significant statistical challenge due to the distribution of the data. Healthcare spending is notoriously "long-tailed"—the vast majority of individuals have low expenditures, while a small fraction (often referred to as "super-utilizers") accounts for the bulk of total spending. Furthermore, there is a large mass of "zeroes"—respondents who incurred no medical expenses during the year.1

### **The Two-Part Modeling Strategy**

To address the zero-inflation and skewness, researchers often employ a Two-Part Model (TPM). The first part is a binary classification model (e.g., Logistic Regression or a Random Forest classifier) that predicts the probability of having *any* expenditure. The second part is a regression model that predicts the *amount* of expenditure, conditional on it being greater than zero. This approach ensures that the model is not biased toward under-prediction by the large number of zero-cost cases.1

### **Regression Algorithms and Optimization**

The choice of regression algorithm is critical for balancing interpretability and accuracy.

* **Linear Regression with Log Transformation:** Standard linear models often struggle with skewed cost data. Applying a natural log transformation to the target variable (ln(TOTSLF23 \+ 1)) can normalize the distribution, though it complicates the interpretation of the coefficients.2  
* **Ridge and LASSO Regression:** These techniques introduce regularization to prevent overfitting, which is essential when dealing with the high dimensionality of the MEPS dataset. LASSO is particularly useful for feature selection, as it can shrink the coefficients of non-essential variables to zero.1  
* **Tree-Based Ensemble Methods (Random Forest, XGBoost):** These algorithms are the current state-of-the-art for expenditure prediction. They naturally handle non-linear relationships and interaction effects between variables (e.g., age and diabetes). XGBoost, in particular, has shown superior performance in medical cost prediction tasks, often achieving R-squared values near 0.90.3  
* **Artificial Neural Networks (ANN):** ANNs have shown high performance for identifying complex patterns in high-cost individuals. While less interpretable than decision trees, they excel at capturing the nuanced interplay of multiple chronic conditions and socio-economic factors.1

### **Evaluation Metrics in a Personal Finance Context**

The primary evaluation metric for the Medical Cost Planner is the Median Absolute Error (MdAE). While Mean Absolute Error (MAE) is a common metric, it can be heavily distorted by extreme outliers—one "miss" on a cancer patient can significantly skew the average error. In contrast, the MdAE provides a realistic estimate of the "typical" error experienced by the majority of users. For an application focused on FSA/HSA budgeting, an MdAE of $500 or less is considered the "sweet spot" for decision support.1

The Pearson correlation coefficient ($R^2$) is also used to measure the overall variance explained by the model:

$$R^2 \= 1 \- \\frac{\\sum (y\_i \- \\hat{y}\_i)^2}{\\sum (y\_i \- \\bar{y})^2}$$  
In research using MEPS data, $R^2$ values typically range from 0.40 to 0.85 depending on the feature set and model complexity. The Medical Cost Planner aims for the higher end of this range by prioritizing high-importance health and insurance features.1

## **UX Design and Cognitive Load Management**

The design of the Medical Cost Planner UI is as important as the backend model. The "90-second rule" is a response to the "paradox of choice" and the cognitive fatigue associated with health insurance decisions.2

### **Designing for Frictionless Interaction**

To achieve the speed and usability goals, the UI must avoid any input that requires the user to "leave their chair." This means no asking for:

* **Specific Deductible Amounts:** Most users do not know their exact deductible without looking up their summary of benefits.  
* **ICD-10 or CPT Codes:** These are technical terms unknown to laypeople.  
* **Historical Expenditure Data:** While prior costs are excellent predictors of future costs, users rarely remember their exact out-of-pocket total from the previous year.1

Instead, the form uses high-level groupings. A "Chronic Condition Checklist" is presented as a single multi-select interaction, which the backend then decomposes into the relevant MEPS variables (DIABDX, HIBPDX, etc.). This approach minimizes the perceived length of the form while maximizing the data density for the model.2

### **Strategic Transparency: Explaining the "Why"**

While simplicity is the priority, the tool must also build trust. This is achieved through "Strategic Transparency"—providing the user with a high-level explanation of the factors driving their estimate. For instance, after receiving a cost forecast, a user might see a brief note stating: "Your estimate is based on typical costs for individuals in your age group with managed hypertension." This explainability is crucial for financial risk management, as it allows users to understand the model's logic without needing to understand the underlying machine learning.6

## **Socio-Economic Implications of Expenditure Prediction**

The deployment of a personalized cost planner has broader implications for healthcare equity and literacy.

### **Mitigating the "Sandwich Generation" Burden**

The "Caregiver" persona highlights a growing demographic: individuals managing the health and finances of both children and aging parents. By providing a tool that can quickly estimate costs for an elderly parent, the Medical Cost Planner reduces the administrative burden on caregivers, allowing for more stable multi-generational financial planning.2

### **Improving Insurance Literacy for Gig Workers**

The rise of the gig economy has led to an increase in individuals purchasing insurance through individual marketplaces. These users often suffer from low "health insurance literacy," struggling to understand the trade-offs between low premiums and high deductibles. The Medical Cost Planner acts as an educational bridge, demonstrating the real-world financial consequences of different coverage choices.2

### **Financial Resilience and Emergency Planning**

For the "Budgeter," the tool provides a quantitative basis for emergency fund sizing. In the United States, medical debt is a leading cause of bankruptcy. By allowing individuals to anticipate their potential OOP exposure, the planner contributes to household financial resilience, helping users avoid high-interest debt when a medical event occurs.2

## **Future Outlook: From Static Estimates to Dynamic Financial Health**

The Medical Cost Planner represents the first step toward a more integrated model of personal health finance. As the accuracy of these models improves, we can anticipate several evolutionary stages:

### **Real-Time Adjustment and "What-If" Modeling**

Future iterations may allow for real-time adjustments as a user moves through the year. For example, if a user experiences an unexpected ER visit in March, the model could update its annual forecast to account for the impact on their deductible and the increased likelihood of follow-up care. "What-if" modeling could also allow users to see the financial impact of lifestyle changes, such as quitting smoking or managing BMI.4

### **Integration with Value-Based Care**

As the healthcare system moves toward value-based care, cost prediction tools will become essential for helping users navigate "bundled payments" and "narrow networks." A future version of the planner could integrate provider-specific quality and price data, helping users choose not only *how much* to save but *where* to receive care to minimize their OOP exposure.3

## **Conclusions and Practical Implementation Strategy**

The development of the Medical Cost Planner underscores a critical shift in the healthcare market: the move from institutional risk management to individual empowerment. By harnessing the predictive power of the MEPS 2023 dataset and aligning it with a user-centric design philosophy, we can provide consumers with the tools they need to navigate an increasingly complex financial landscape.1

The analysis of the 30 best candidate features reveals that a small number of high-impact variables—age, insurance status, self-rated health, and a handful of priority conditions—can provide the foundation for a robust, accessible forecasting tool. When these features are fed into sophisticated machine learning architectures, such as XGBoost or Two-Part Models, the result is a predictive capability that is both accurate enough for personal finance and simple enough for everyday use.2

Ultimately, the goal of the Medical Cost Planner is to eliminate the "financial surprise" of healthcare. By transforming a "black box" of pricing into a clear, personalized forecast, we empower users to make data-driven decisions that protect their health and their financial future.1 The integration of advanced machine learning with the gold standard of U.S. healthcare data provides the essential roadmap for this transformation, ensuring that every consumer has the insight they need to plan with confidence.

#### **Works cited**

1. Supervised Learning Methods for Predicting Healthcare Costs \- NIH, accessed December 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5977561/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5977561/)  
2. Who determines United States Healthcare out-of-pocket costs ..., accessed December 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8184979/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8184979/)  
3. Refining Medical Insurance Cost Predictions with Advanced ..., accessed December 19, 2025, [https://www.scitepress.org/Papers/2025/137022/137022.pdf](https://www.scitepress.org/Papers/2025/137022/137022.pdf)  
4. Predictive Modelling of Healthcare Insurance Costs Using Machine ..., accessed December 19, 2025, [https://www.preprints.org/manuscript/202502.1873](https://www.preprints.org/manuscript/202502.1873)  
5. Dataset: Medical Expenditure Panel Survey (MEPS), accessed December 19, 2025, [https://www.disabilitystatistics.org/dataset-directory/dataset/70](https://www.disabilitystatistics.org/dataset-directory/dataset/70)  
6. AIX360/examples/tutorials/MEPS.ipynb at master \- GitHub, accessed December 19, 2025, [https://github.com/Trusted-AI/AIX360/blob/master/examples/tutorials/MEPS.ipynb](https://github.com/Trusted-AI/AIX360/blob/master/examples/tutorials/MEPS.ipynb)  
7. Chapter 7 Story MEPS: Healthcare expenditures of individuals, accessed December 19, 2025, [https://pbiecek.github.io/xai\_stories/story-meps-healthcare-expenditures-of-individuals.html](https://pbiecek.github.io/xai_stories/story-meps-healthcare-expenditures-of-individuals.html)  
8. MEPS HC 251 2023 Full Year Consolidated Data File August 2025, accessed December 19, 2025, [https://meps.ahrq.gov/data\_stats/download\_data/pufs/h251/h251doc.pdf](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.pdf)  
9. MEPS HC 251 2023 Full Year Consolidated Data File, accessed December 19, 2025, [https://meps.ahrq.gov/data\_stats/download\_data/pufs/h251/h251doc.shtml](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.shtml)  
10. MEPS HC 247: 2023 Full-Year Population Characteristics, accessed December 19, 2025, [https://meps.ahrq.gov/data\_stats/download\_data/pufs/h247/h247doc.shtml](https://meps.ahrq.gov/data_stats/download_data/pufs/h247/h247doc.shtml)  
11. Medical Expenditure Panel Survey (MEPS) \- GitHub, accessed December 19, 2025, [https://github.com/HHS-AHRQ/MEPS](https://github.com/HHS-AHRQ/MEPS)  
12. MEPS HC-233 2021 Full Year Consolidated Data File August 2023, accessed December 19, 2025, [https://meps.ipums.org/meps/resources/h233doc.pdf](https://meps.ipums.org/meps/resources/h233doc.pdf)  
13. Comparison of Machine Learning Algorithms for Predicting the Out, accessed December 19, 2025, [https://www.iomcworld.org/articles/comparison-of-machine-learning-algorithms-for-predicting-the-out-of-pocket-medical-expenditures-in-rwanda-44322.html](https://www.iomcworld.org/articles/comparison-of-machine-learning-algorithms-for-predicting-the-out-of-pocket-medical-expenditures-in-rwanda-44322.html)  
14. MEPS Topics: Priority Conditions \-- General, accessed December 19, 2025, [https://meps.ahrq.gov/data\_stats/MEPS\_topics.jsp?topicid=41Z-1](https://meps.ahrq.gov/data_stats/MEPS_topics.jsp?topicid=41Z-1)  
15. Medical Expenditure Panel Survey (MEPS) Household Component ..., accessed December 19, 2025, [https://datatools.ahrq.gov/meps-hc/](https://datatools.ahrq.gov/meps-hc/)  
16. BalaElangovan/Forecasting-Medical-Insurance-Costs-Using ..., accessed December 19, 2025, [https://github.com/BalaElangovan/Forecasting-Medical-Insurance-Costs-Using-Machine-Learning](https://github.com/BalaElangovan/Forecasting-Medical-Insurance-Costs-Using-Machine-Learning)  
17. Predictive Modeling for Healthcare Insurance Costs Using Machine ..., accessed December 19, 2025, [https://www.ijraset.com/best-journal/predictive-modeling-for-healthcare-insurance-costs-using-machine-learning](https://www.ijraset.com/best-journal/predictive-modeling-for-healthcare-insurance-costs-using-machine-learning)  
18. Condition, Event, and Prescribed Medicine Records \- IPUMS MEPS, accessed December 19, 2025, [https://meps.ipums.org/meps/userNotes\_conditioneventpmed.shtml](https://meps.ipums.org/meps/userNotes_conditioneventpmed.shtml)  
19. Chapter 6 Story MEPS: Explainable predictions for healthcare ..., accessed December 19, 2025, [https://pbiecek.github.io/xai\_stories/story-meps-explainable-predictions-for-healthcare-expenditures.html](https://pbiecek.github.io/xai_stories/story-meps-explainable-predictions-for-healthcare-expenditures.html)  
20. Multivariable prediction models for health care spending using ... \- NIH, accessed December 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8943988/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8943988/)