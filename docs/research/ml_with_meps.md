# Machine Learning with MEPS Data 
Deep Research report on existing machine learning projects using MEPS data.
> **Tool:** ChatGPT Deep Research  
> **Date:** 2025-12-05  
> **Prompt:** *"give overview of all machine learning projects with Medical expenditure panel survey MEPS data including what ml model was trained, what meps dataset was used, if the ml model was deployed e.g. via an app and if there are any consumer-facing apps built on meps data"*

A number of recent studies (mostly academic and government-sponsored) have applied machine learning (ML) to the Medical Expenditure Panel Survey (MEPS) to predict healthcare costs. Most use the Household Component (HC) data (often the Full-Year Consolidated file) over one or more years, and apply models like random forests, gradient boosting, neural nets, and ensemble (“super learner”) methods. 

For example, **McClellan et al. (2023)** used MEPS HC data from 2016–2017 and replaced the usual linear regression in predictive mean-matching with several ML algorithms (gradient boosting, random forests, extreme random forests, deep neural networks and a stacked ensemble) to impute missing medical-event expenditures  
<https://pubmed.ncbi.nlm.nih.gov/36495183/>

Similarly, **Li et al. (2022)** developed a **two-stage super-learner** (an ensemble of models) that separately predicts (1) the probability of any expenditure and (2) the conditional mean cost when positive, using 2016 MEPS as training and 2017 MEPS as testing.  
<https://pmc.ncbi.nlm.nih.gov/articles/PMC9683480/>

In their MEPS analysis (10,925 training and 10,815 test persons), this two-part ensemble (mixing GLMs and machine learners like random forests) improved mean-squared-error over standard one-step models.  
<https://pmc.ncbi.nlm.nih.gov/articles/PMC9683480/>

---

## AHRQ / Survey-imputation projects
AHRQ (often via RTI) has applied ML to accelerate MEPS data processing. For instance, **Cohen et al. (JSM 2018)** describe an AI/ML-enhanced imputation pipeline for MEPS (household component), using methods like random forests to fast-track expenditure estimates.  
<https://www.rti.org/sites/default/files/publications/3b7cefaa-37d5-4e81-9f9e-aa32a594daa5/versions/867091_cohen_proceedings_2018.pdf>

In a recent paper, **McClellan et al. (2023)** (AHRQ authors) showed that ML algorithms (gradient boosting, deep nets, etc.) can substantially improve predictive mean matching for imputing individual event costs in the 2016–2017 MEPS.  
<https://pubmed.ncbi.nlm.nih.gov/36495183/>

(These efforts are internal research to improve MEPS data quality; no public app was released.)

---

## Health economics research
**Wang & Shi (2020)** used MEPS HC data from 2000–2015 (household interviews) to predict total annual medical expenditures for diagnosed diabetics. They trained a random forest model (in R) on full-year consolidated MEPS files and identified top predictors (e.g. insulin use, insurance type).  
<https://pubmed.ncbi.nlm.nih.gov/32159759/>

The RF model outperformed a traditional regression baseline, with a Spearman ρ ≈ 0.64 between predicted and actual costs.  
<https://pubmed.ncbi.nlm.nih.gov/32159759/>

**Maidman & Wang (2018)** developed a semiparametric model to classify “high-cost” patients using a Bayes-rule-like approach; they applied this to a subset of MEPS data to predict which individuals fall in the upper tail of expenditures.  
<https://pubmed.ncbi.nlm.nih.gov/29228454/>

(This work led to an R package *plaqr* for upper-tail prediction.)

---

## Explainable AI tutorials
In industry and open-source, MEPS has been used for demonstration. For example, IBM’s AIX360 toolkit published a tutorial (“XAI Stories”) in 2023 where analysts built three models (ridge regression, neural network, and gradient boosting) to predict individual annual healthcare expenditures from a MEPS-derived dataset of ~18,350 individuals.  
<https://pbiecek.github.io/xai_stories/story-meps-healthcare-expenditures-of-individuals.html>

This exercise (mentored by McKinsey) focused on interpretation (SHAP, LIME) and did not result in a consumer app. The data include demographics, health status, diagnoses, etc., matching MEPS content.  
<https://pbiecek.github.io/xai_stories/story-meps-healthcare-expenditures-of-individuals.html>

---

## Table 1. Representative ML projects using MEPS for expenditure prediction
| Project (source) | ML model(s) | MEPS data (years, component) | Deployed? (app/tool) | Reference |
|------------------|-------------|-------------------------------|-----------------------|-----------|
| **McClellan et al., 2023** | Gradient boosting, RF, XRF, DNN, stacked ensemble (for imputation) | MEPS-HC, 2016–2017 (event-level spending) | No – research on imputation quality | <https://pubmed.ncbi.nlm.nih.gov/36495183/> |
| **Li et al., 2022** | Two-stage super learner (two-part ensemble: logistic + GLM, RF, etc.) | MEPS-HC, 2016–2017 (total annual costs) | No – method evaluation | <https://pmc.ncbi.nlm.nih.gov/articles/PMC9683480/> |
| **Wang & Shi, 2020** | Random Forest | MEPS-HC, 2000–2015 (diabetic subsample) | No – academic study | <https://pubmed.ncbi.nlm.nih.gov/32159759/> |
| **Maidman & Wang, 2018** | Semiparametric upper-tail classifier | MEPS (subset) | No – methodology paper | <https://pubmed.ncbi.nlm.nih.gov/29228454/> |
| **Kim et al., 2024 (ISPOR abstract)** | ML (unspecified) | MEPS-HC 2021 | No – conference abstract | — |
| **Bankiewicz et al., 2023 (XAI tutorial)** | Ridge, ANN, XGBoost | MEPS (AIX360 sample ~18K) | No – tutorial/code example | <https://pbiecek.github.io/xai_stories/story-meps-healthcare-expenditures-of-individuals.html> |
| **Cohen et al., 2018 (RTI/AHRQ)** | Random forests and others for imputation | MEPS-HC (annual) | No – internal AHRQ project | <https://www.rti.org/sites/default/files/publications/3b7cefaa-37d5-4e81-9f9e-aa32a594daa5/versions/867091_cohen_proceedings_2018.pdf> |

---

## Government / Agency initiatives
MEPS sponsors have funded ML work. AHRQ/RTI projects have applied AI/ML to speed up MEPS processing—e.g., Cohen et al. (2018) describe an ML-enhanced pipeline generating interim expenditure estimates aligned with final MEPS results.  
<https://www.rti.org/sites/default/files/publications/3b7cefaa-37d5-4e81-9f9e-aa32a594daa5/versions/867091_cohen_proceedings_2018.pdf>

AHRQ authors also tested ML-based imputation (as noted above). None of these produced a public product; they were focused on improving data processing.

AHRQ does provide **MEPS Data Tools**, which allow users to explore aggregate spending data (but not predictive modeling).  
<https://meps.ahrq.gov/mepsweb/data_stats/data_tools.jsp>

---

## Industry and tools
We found **no major commercial products** that embed MEPS-based cost prediction.

MEPS data do appear in:
- Open-source ML tutorials  
- Kaggle datasets  
- Fairness/interpretability demo datasets (e.g., AIF360 uses MEPS)
