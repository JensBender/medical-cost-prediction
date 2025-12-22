# Candidate Features for Medical Cost Prediction
Deep Research report on the 30 best candidate features for predicting out-of-pocket medical costs using MEPS-HC 2023 data. These 30 MEPS variables are selected because they are known offhand by users, have established significance in the literature, and are well-documented in the data codebook. 
> **Tool:** Gemini Deep Research  
> **Date:** 2025-12-19  

### **Demographic Variables (Primary Drivers)**

1. **AGE31X (Age):** Primary driver of healthcare utilization; costs typically follow a U-curve, peaking for infants and the elderly. [[1]](#ref1)  
2. **SEX (Gender):** Influences utilization patterns based on reproductive health and gender-specific conditions. [[2]](#ref2)  
3. **FAMSZE23 (Family Size):** A larger household increases the statistical probability of high-cost medical events. [[1]](#ref1)  
4. **MARRY23X (Marital Status):** Used as a proxy for social support and income stability, both correlating with healthcare access. [[3]](#ref3)

### **Socio-Economic and Location Variables (Access Proxies)**

5. **POVCAT23 (Poverty Category):** Income relative to the federal poverty line; determines subsidy eligibility and insurance quality. [[4]](#ref4)  
6. **HIDEG (Highest Degree):** Education level correlates with health literacy and the use of preventive services. [[4]](#ref4)  
7. **REGION23 (Census Region):** Captures geographic variations in healthcare pricing and provider density. [[2]](#ref2)

### **Subjective Health and Lifestyle (High-Fidelity Proxies)**

8. **RTHLTH31 (Perceived Physical Health):** A powerful subjective indicator of overall utilization intensity. [[5]](#ref5)  
9. **MNHLTH31 (Perceived Mental Health):** A significant cost multiplier via its impact on treatment adherence and physical health. [[5]](#ref5)  
10. **ADSMOK42 (Smoking Status):** A risk factor for respiratory and cardiovascular diseases. [[2]](#ref2)  
11. **BMI (Body Mass Index):** A primary driver of high-cost metabolic and orthopedic conditions. [[2]](#ref2)

### **Priority Conditions (The "Cost Engine")**

12. **HIBPDX (Hypertension):** Drives predictable base costs for lifelong medication and routine monitoring. [[6]](#ref6)  
13. **CHOLDX (High Cholesterol):** Increases pharmacy spending and periodic laboratory requirements. [[6]](#ref6)  
14. **CHDDX (Coronary Heart Disease):** Indicates high risk for expensive acute cardiac events. [[6]](#ref6)  
15. **ANGIDX (Angina):** Indicator of advanced cardiovascular disease and localized care needs. [[6]](#ref6)  
16. **MIDX (Heart Attack History):** Predictor of future intensive inpatient and specialist care. [[7]](#ref7)  
17. **STRKDX (Stroke):** Associated with substantial long-term rehabilitation and care costs. [[7]](#ref7)  
18. **EMPHDX (Emphysema):** Drives costs for specialized medication, oxygen, and emergency care. [[6]](#ref6)  
19. **CHBRON31 (Chronic Bronchitis):** Contributes significantly to the overall respiratory healthcare burden. [[6]](#ref6)  
20. **ASTHDX (Asthma):** Drives ER visits and specialized medication costs, especially for younger patients. [[1]](#ref1)  
21. **CANCERDX (Cancer):** A primary driver of extreme "tail" costs and intensive treatment regimens. [[7]](#ref7)  
22. **DIABDX (Diabetes):** Strong predictor of sustained, high-magnitude outpatient and pharmacy expenditures. [[1]](#ref1)  
23. **ARTHDX (Arthritis):** Drives long-term spending on pharmaceuticals and physical therapy. [[7]](#ref7)  
24. **DEPRDX (Depression):** Correlates with increased utilization across all medical service categories. [[7]](#ref7)

### **Insurance and Activity Limitations (Functional Burden)**

25. **INSCOV23 (Insurance Coverage Indicator):** The critical factor for converting total medical expenditures into out-of-pocket costs. [[1]](#ref1)  
26. **ADLHLP31 (ADL Help Needed):** Indicates a need for help with basic daily activities; a high-cost functional indicator. [[7]](#ref7)  
27. **IADLHP31 (IADL Help Needed):** Tracks limitations in complex tasks, signaling high-cost care requirements. [[7]](#ref7)  
28. **COGLIM31 (Cognitive Limitations):** Signifies mental functional capacity, correlating with specialized care needs. [[3]](#ref3)  
29. **WRKLIM31 (Work Limitation):** A proxy for condition severity and its impact on daily functioning. [[3]](#ref3)  
30. **ANYLIM23 (Any Limitation):** A composite indicator of physical or cognitive limitations, signaling high healthcare needs. [[3]](#ref3)


#### **Works cited**

1. <a id="ref1"></a>Who determines United States Healthcare out-of-pocket costs ..., accessed December 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8184979/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8184979/)  
2. <a id="ref2"></a>Predictive Modelling of Healthcare Insurance Costs Using Machine ..., accessed December 19, 2025, [https://www.preprints.org/manuscript/202502.1873](https://www.preprints.org/manuscript/202502.1873)  
3. <a id="ref3"></a>AIX360/examples/tutorials/MEPS.ipynb at master \- GitHub, accessed December 19, 2025, [https://github.com/Trusted-AI/AIX360/blob/master/examples/tutorials/MEPS.ipynb](https://github.com/Trusted-AI/AIX360/blob/master/examples/tutorials/MEPS.ipynb)  
4. <a id="ref4"></a>MEPS HC 251 2023 Full Year Consolidated Data File August 2025, accessed December 19, 2025, [https://meps.ahrq.gov/data\_stats/download\_data/pufs/h251/h251doc.pdf](https://meps.ahrq.gov/data\_stats/download\_data/pufs/h251/h251doc.pdf)  
5. <a id="ref5"></a>Supervised Learning Methods for Predicting Healthcare Costs \- NIH, accessed December 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5977561/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5977561/)  
6. <a id="ref6"></a>MEPS HC 251 2023 Full Year Consolidated Data File, accessed December 19, 2025, [https://meps.ahrq.gov/data\_stats/download_data/pufs/h251/h251doc.shtml](https://meps.ahrq.gov/data_stats/download_data/pufs/h251/h251doc.shtml)  
7. <a id="ref7"></a>Dataset: Medical Expenditure Panel Survey (MEPS), accessed December 19, 2025, [https://www.disabilitystatistics.org/dataset-directory/dataset/70](https://www.disabilitystatistics.org/dataset-directory/dataset/70)