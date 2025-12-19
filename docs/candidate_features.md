# Candidate Features for Medical Cost Prediction
Deep Research report on the 30 best candidate features for predicting out-of-pocket medical costs using MEPS-HC 2023 data. These 30 MEPS variables are selected because they are known offhand by users, have established statistical significance in the literature, and are well-documented in the H251 codebook. 
> **Tool:** Gemini Deep Research  
> **Date:** 2025-12-19  

### **Demographic Variables (Primary Drivers)**

1. **AGE31X (Age):** Perhaps the single most important demographic factor. Healthcare consumption follows a clear U-shaped curve, with higher costs for the very young (neonatal) and the elderly (chronic disease management). [[1]](#ref1)  
2. **SEX (Gender):** Historically associated with differing utilization patterns, particularly in reproductive health and the prevalence of certain chronic conditions. [[2]](#ref2)  
3. **FAMSZE23 (Family Size):** Total household expenditure is a function of family size, as more individuals increase the statistical probability of a high-cost event. [[1]](#ref1)  
4. **MARRY23X (Marital Status):** Often used as a proxy for social support and household income stability, both of which correlate with healthcare access and preventive care utilization. [[3]](#ref3)

### **Socio-Economic and Location Variables (Access Proxies)**

5. **POVCAT23 (Poverty Category):** A constructed variable indicating the ratio of household income to the federal poverty line. This determines eligibility for subsidies and the likelihood of having comprehensive insurance. [[4]](#ref4)  
6. **HIDEG (Highest Degree):** Education level is a well-established social determinant of health, often correlating with health literacy and the use of preventive services. [[4]](#ref4)  
7. **REGION23 (Census Region):** Geographic variation in healthcare pricing and provider density significantly impacts total expenditures. [[2]](#ref2)

### **Subjective Health and Lifestyle (High-Fidelity Proxies)**

8. **RTHLTH31 (Perceived Physical Health):** As discussed, this is a powerful, low-friction input for predicting overall utilization intensity. [[5]](#ref5)  
9. **MNHLTH31 (Perceived Mental Health):** Poor mental health is increasingly recognized as a multiplier for physical healthcare costs due to treatment adherence and systemic physiological effects. [[5]](#ref5)  
10. **ADSMOK42 (Smoking Status):** A direct indicator of future respiratory and cardiovascular risk. [[2]](#ref2)  
11. **BMI (Body Mass Index):** While not explicitly in every PUF, it is often derived from weight/height variables in MEPS. High BMI is a primary driver of metabolic and orthopedic costs. [[2]](#ref2)

### **Priority Conditions (The "Cost Engine")**

12. **HIBPDX (Hypertension):** Requires lifelong medication and monitoring, making it a predictable "base" cost. [[6]](#ref6)  
13. **CHOLDX (High Cholesterol):** Similar to hypertension, it drives pharmacy spending and periodic lab work. [[6]](#ref6)  
14. **CHDDX (Coronary Heart Disease):** Represents significant risk for high-cost acute events. [[6]](#ref6)  
15. **ANGIDX (Angina):** Indicator of advanced cardiovascular disease. [[6]](#ref6)  
16. **MIDX (Heart Attack History):** A history of myocardial infarction is a strong predictor of future inpatient and specialist costs. [[7]](#ref7)  
17. **STRKDX (Stroke):** Associated with high rehabilitation and long-term care costs. [[7]](#ref7)  
18. **EMPHDX (Emphysema):** Drives costs for oxygen, medications, and frequent respiratory events. [[6]](#ref6)  
19. **CHBRON31 (Chronic Bronchitis):** Adds to the respiratory disease burden. [[6]](#ref6)  
20. **ASTHDX (Asthma):** Especially significant for younger personas, driving emergency room visits and inhaler costs. [[1]](#ref1)  
21. **CANCERDX (Cancer):** A primary driver of extreme "tail" costs in the expenditure distribution. [[7]](#ref7)  
22. **DIABDX (Diabetes):** One of the most significant predictors of sustained, high-magnitude outpatient and pharmacy spending. [[1]](#ref1)  
23. **ARTHDX (Arthritis):** Drives long-term pharmaceutical and physical therapy costs. [[7]](#ref7)  
24. **DEPRDX (Depression):** Associated with higher utilization across all medical service categories. [[7]](#ref7)

### **Insurance and Activity Limitations (Functional Burden)**

25. **INSCOV23 (Insurance Coverage Indicator):** The most critical variable for converting total cost to out-of-pocket cost. [[1]](#ref1)  
26. **ADLHLP31 (ADL Help Needed):** Indicates a need for assistance with activities of daily living, a high-cost functional indicator. [[7]](#ref7)  
27. **IADLHP31 (IADL Help Needed):** Similar to ADL, this tracks higher-level functional limitations. [[7]](#ref7)  
28. **COGLIM31 (Cognitive Limitations):** Tracks mental functional capacity, which correlates with specialized care needs. [[3]](#ref3)  
29. **WRKLIM31 (Work Limitation):** A proxy for the severity of underlying health conditions and their impact on daily life. [[3]](#ref3)  
30. **ANYLIM23 (Any Limitation):** A composite variable of all physical or cognitive limitations, serving as a broad indicator of high healthcare needs. [[3]](#ref3)


#### **Works cited**

1. <a id="ref1"></a>Who determines United States Healthcare out-of-pocket costs ..., accessed December 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8184979/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8184979/)  
2. <a id="ref2"></a>Predictive Modelling of Healthcare Insurance Costs Using Machine ..., accessed December 19, 2025, [https://www.preprints.org/manuscript/202502.1873](https://www.preprints.org/manuscript/202502.1873)  
3. <a id="ref3"></a>AIX360/examples/tutorials/MEPS.ipynb at master \- GitHub, accessed December 19, 2025, [https://github.com/Trusted-AI/AIX360/blob/master/examples/tutorials/MEPS.ipynb](https://github.com/Trusted-AI/AIX360/blob/master/examples/tutorials/MEPS.ipynb)  
4. <a id="ref4"></a>MEPS HC 251 2023 Full Year Consolidated Data File August 2025, accessed December 19, 2025, [https://meps.ahrq.gov/data\_stats/download\_data/pufs/h251/h251doc.pdf](https://meps.ahrq.gov/data\_stats/download\_data/pufs/h251/h251doc.pdf)  
5. <a id="ref5"></a>Supervised Learning Methods for Predicting Healthcare Costs \- NIH, accessed December 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5977561/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5977561/)  
6. <a id="ref6"></a>MEPS HC 251 2023 Full Year Consolidated Data File, accessed December 19, 2025, [https://meps.ahrq.gov/data\_stats/download_data/pufs/h251/h251doc.shtml](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.shtml)  
7. <a id="ref7"></a>Dataset: Medical Expenditure Panel Survey (MEPS), accessed December 19, 2025, [https://www.disabilitystatistics.org/dataset-directory/dataset/70](https://www.disabilitystatistics.org/dataset-directory/dataset/70)