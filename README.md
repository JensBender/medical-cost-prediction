<!-- anchor tag for back-to-top links -->
<a id="readme-top"></a>

<!-- HEADER IMAGE  -->
<img src="assets/header.png" alt="Header Image">

<!-- SHORT SUMMARY  -->


## 📋 Table of Contents
<ol>
  <li>
    <a href="#-summary">Summary</a>
  </li>
  <li>
    <a href="#-motivation">Motivation</a>
  </li>
  <li>
    <a href="#️-data">Data</a>
  </li>
  <li>
    <a href="#-exploratory-data-analysis-eda">Exploratory Data Analysis (EDA)</a>
  </li>
  <li>
    <a href="#-data-preprocessing">Data Preprocessing</a>
  </li>
  <li>
    <a href="#-modeling">Modeling</a>
    <ul>
      <li><a href="#-baseline-models">Baseline Models</a></li>      
      <li><a href="#️-hyperparameter-tuning">Hyperparameter Tuning</a></li>
      <li><a href="#-final-model">Final Model</a></li>
    </ul>
  </li>
  <li>
    <a href="#-project-structure">Project Structure</a>
  </li>
  <li>
    <a href="#️-getting-started">Getting Started</a>
    <ul>
      <li><a href="#installation-and-setup">Installation and Setup</a></li>
      <li><a href="#production-deployment">Production Deployment</a></li>
    </ul>
  </li>
  <li>
    <a href="#️-license">License</a>
  </li>
  <li>
    <a href="#-credits">Credits</a>
  </li>
  <li>
    <a href="#-appendix">Appendix</a>
    <ul>
      <li><a href="#distributions">Distributions</a></li>        
      <li><a href="#feature-target-relationships">Feature-Target Relationships</a></li>      
      <li><a href="#outlier-analysis">Outlier Analysis</a></li>      
    </ul>
  </li>
</ol>


## 🎯 Summary
Currently developing an end-to-end machine learning application to predict annual out-of-pocket healthcare costs. Initial phase: authoring [Product Requirements](./docs/specs/product_requirements.md) and [Technical Specifications](./docs/specs/technical_specifications.md) for a user-centric system design (task completion time <90s). Engineering the ML pipeline to translate complex survey data (MEPS) into personalized budgeting insights.

🛠️ **Built With**
- [![Python][Python-badge]][Python-url]
- [![Pandas][Pandas-badge]][Pandas-url]
- [![Matplotlib][Matplotlib-badge]][Matplotlib-url] 
- [![Seaborn][Seaborn-badge]][Seaborn-url]
- [![scikit-learn][scikit-learn-badge]][scikit-learn-url]
- [![DVC][DVC-badge]][DVC-url]
- [![Jupyter Notebook][JupyterNotebook-badge]][JupyterNotebook-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 💡 Motivation
**The Problem:** Healthcare pricing is a "black box." While insurance portals show prices for individual treatments (e.g., an MRI), consumers lack tools to predict their total expected costs for the year. Existing calculators are often too generic (ignoring health conditions) or too complex (requiring specific procedure codes).

**Our Solution:** A personalized forecasting tool based on accessible inputs. Users simply enter demographic and health details such as age, insurance status, and chronic conditions to receive a cost estimate for the upcoming year. This empowers users to make data-driven decisions for FSA/HSA contributions and emergency planning.

**How It Works:** The web app is powered by a machine learning model trained on the Medical Expenditure Panel Survey (MEPS), the gold standard for U.S. healthcare data. By analyzing what people with similar demographic and health profiles actually spent, our model learns real-world cost patterns and translates them into actionable financial insights without requiring complex medical records.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 🗂️ Data
The **Medical Expenditure Panel Survey (MEPS)** is the most complete source of data on the cost and use of health care and health insurance coverage in the United States. Administered by the **Agency for Healthcare Research and Quality (AHRQ)**, it is a set of large-scale surveys of families and individuals, their medical providers, and employers. MEPS provides nationally representative estimates for the **U.S. civilian noninstitutionalized population**:
- **Included:** People living in households (houses, apartments) and non-institutional group quarters (e.g., college dorms).
- **Excluded:** Active-duty military, and people in institutions like nursing homes or prisons. Also excludes individual periods where a respondent moved outside the U.S. during the survey year.

**MEPS Household Component**  
The Household Component (MEPS-HC) collects comprehensive data directly from families and individuals. To ensure accuracy, household-reported expenditures are validated and imputed using the Medical Provider Component (MEPS-MPC), which draws data directly from doctors, hospitals, and pharmacies.

**MEPS-HC 2023**  
This project utilizes the 2023 Full-Year Consolidated Data File (HC-251). The data was released in August 2025.
- **Sample Size:** 18,919 individuals.
- **Variables:** 1,374.

![MEPS Data Infographic](assets/infographic_meps_data.jpg)

**Target Variable**  
The target variable is **total out-of-pocket health care costs in 2023** (`TOTSLF23`). This variable represents what the person or their family actually paid directly for all medical events throughout 2023. It includes copays and coinsurance, deductibles, and services not covered by insurance. `TOTSLF23` directly answers practical user questions like "How much should I contribute to my FSA/HSA?" and "What's my financial exposure?" For uninsured users, out-of-pocket costs approximate total costs, making this target appropriate across all insurance statuses.  

<details>
<summary>ℹ️<strong>US Healthcare Costs Explained</strong> (click to expand)</summary>

![US Healthcare Costs Infographic](./assets/infographic_healthcare_costs.png)
</details>
<br>

**Feature Selection**  
Candidate features were selected from MEPS-HC 2023 based on the following criteria:
- **Consumer Accessibility:** Users can answer from memory without looking up records, ensuring the model is usable in a consumer-facing app.
- **Beginning-of-Year Data:** To enable the app to be used during Open Enrollment for predicting *upcoming* costs, only variables measured at the beginning of the year (`31` suffix) or stable traits are used to prevent data leakage.
- **Predictive Power:** Features have established significance in healthcare cost literature.

These 26 candidate features will be further reduced based on importance scores to meet the UX goal of a form completion time of less than 90 seconds.

**Candidate Features**
| Label | Variable | Description | Data Type | Value Range |
| :--- | :--- | :--- | :--- | :--- |
| Age | `AGE23X` | Age as of Dec 31, 2023. | Numerical (Int) | 0–85 |
| Sex | `SEX` | Biological sex. | Binary (Int) | 1=Male, 2=Female |
| Region | `REGION23` | Census region. | Nominal (Int) | 1=Northeast, 2=Midwest, 3=South, 4=West |
| Marital Status | `MARRY31X` | Status at beginning of year. | Nominal (Int) | 1=Married, 2=Widowed, 3=Divorced, 4=Separated, 5=Never Married |
| Poverty Category | `POVCAT23` | Family income relative to poverty line. | Ordinal (Int) | 1=Poor, 2=Near Poor, 3=Low Income, 4=Middle Income, 5=High Income |
| Family Size | `FAMSZE23` | Number of related persons residing together. | Numerical (Int) | 1–14 |
| Education | `HIDEG` | Highest degree attained. | Ordinal (Int) | 1=No Degree, 2=GED, 3=HS Diploma, 4=Bachelor's, 5=Master's, 6=Doctorate, 7=Other |
| Employment Status | `EMPST31` | Status at beginning of year. | Nominal (Int) | 1=Employed, 2=Job to return to, 3=Job during reference period, 4=Not employed |
| Insurance | `INSCOV23` | Coverage status. | Nominal (Int) | 1=Any Private, 2=Public Only, 3=Uninsured |
| Usual Source of Care | `HAVEUS42` | Regular doctor or clinic. | Binary (Int) | 1=Yes, 2=No |
| Physical Health | `RTHLTH31` | Self-rated physical health. | Numerical (Int) | 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor |
| Mental Health | `MNHLTH31` | Self-rated mental health. | Numerical (Int) | 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor |
| Smoker | `ADSMOK42` | Currently smokes cigarettes. | Binary (Int) | 1=Yes, 2=No |
| ADL Help | `ADLHLP31` | Needs help with activities of daily living (personal care, bathing, dressing). | Binary (Int) | 1=Yes, 2=No |
| IADL Help | `IADLHP31` | Needs help with instrumental activities of daily living (paying bills, taking medications, doing laundry). | Binary (Int) | 1=Yes, 2=No |
| Walking Limitation | `WLKLIM31` | Difficulty walking or climbing stairs. | Binary (Int) | 1=Yes, 2=No |
| Cognitive Limitation | `COGLIM31` | Confusion or memory loss. | Binary (Int) | 1=Yes, 2=No |
| Joint Pain | `JTPAIN31_M18` | Pain/stiffness in past year. | Binary (Int) | 1=Yes, 2=No |
| Hypertension | `HIBPDX` | Diagnosed with high blood pressure. | Binary (Int) | 1=Yes, 2=No |
| High Cholesterol | `CHOLDX` | Diagnosed with high cholesterol. | Binary (Int) | 1=Yes, 2=No |
| Diabetes | `DIABDX_M18` | Diagnosed with diabetes. | Binary (Int) | 1=Yes, 2=No |
| Heart Disease | `CHDDX` | Diagnosed with coronary heart disease. | Binary (Int) | 1=Yes, 2=No |
| Stroke | `STRKDX` | Diagnosed with stroke. | Binary (Int) | 1=Yes, 2=No |
| Cancer | `CANCERDX` | Diagnosed with cancer or malignancy. | Binary (Int) | 1=Yes, 2=No |
| Arthritis | `ARTHDX` | Diagnosed with arthritis. | Binary (Int) | 1=Yes, 2=No |
| Asthma | `ASTHDX` | Diagnosed with asthma. | Binary (Int) | 1=Yes, 2=No |

**Sample Weights**  
MEPS-HC 2023 includes survey sample weights (`PERWT23F`) to account for the complex survey design and non-response. This project incorporates these weights during model training to correct for the intentional oversampling of specific subgroups (e.g., the elderly and low-income), ensuring the model remains representative of the general population and prevents bias toward overrepresented groups.

**MEPS Resources**
| Resource | Description | Link |
| :--- | :--- | :--- |
| Data | MEPS-HC 2023 Full Year Consolidated Data File (HC-251). | [Visit Page](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-251) |
| Full Documentation | Technical details on data collection, variable editing, and survey sampling. | [View PDF](docs/h251doc.pdf) |
| Codebook | Variables, labels, coding schemes, and frequencies. | [View PDF](docs/h251cb.pdf) |
| MEPS Overview | Background on MEPS components and larger survey history. | [Visit Page](https://meps.ahrq.gov/mepsweb/about_meps/survey_back.jsp) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 🔍 Exploratory Data Analysis (EDA)
Analyzed distributions and relationships to inform data preprocessing, feature engineering, and modeling decisions.  

**Distributions (Univariate EDA)** 
- **Sample Weights:** Verified survey weights represent ~260M adults and confirmed weighting is essential for population-level representativeness.
- **Target Variable:** Identified a zero-inflated (22.3%) and extremely right-skewed distribution where the top 20% of spenders drive 79.3% of costs (see Lorenz curve below).
- **Numerical Features:** Visualized distribution of age, family size, and self-reported health, informing robust median-based imputation for right-skewed and discrete features [(see histograms)](#numerical-distributions).
- **Categorical Features:** Revealed 66% hold private insurance, suggesting costs will be driven by plan-specific cost-sharing. Identified oversampling of healthy and low socio-economic status individuals, confirming the importance of sample weights [(see bar plots)](#categorical-distributions).
- **Binary Features:** Identified high prevalence of joint pain (45%), high bood pressure (32%), and high cholesterol (31%), while severe conditions such as cancer (11%), coronary heart disease (5%), and stroke (4%) are more sparse [(see bar plots)](#binary-distributions).

![Lorenz Curve](figures/eda/lorenz_curve.png)

**Relationships (Bivariate EDA)** 
- **Correlations:** Spearman rank correlations (see heatmap below) revealed age (0.30) and poverty category (0.26) as primary cost correlates, alongside arthritis, high cholesterol, and joint pain (~0.22).
- **Numerical Features vs. Target:** Visualized feature-target relationships, revealing age as the primary cost driver and a negative relationship with family size likely due to shared family insurance limits [(see scatter plots)](#numerical-feature-target-relationships).
- **Categorical Features vs. Target:** Grouped box plots revealed higher out-of-pocket spending for individuals with high income, high education, and private insurance, suggesting financial access drives healthcare utilization [(see grouped box plots)](#categorical-feature-target-relationships).
- **Binary Features vs. Target:** Identified high-prevalence "global drivers" (arthritis) vs. high-severity "local triggers" (cancer), and confirmed a massive "utilization hurdle" where women and people with a usual source of care spend more [(see grouped box plots)](#binary-feature-target-relationships).

![Correlation Heatmap](figures/eda/correlation_heatmap.png)

**Modeling Strategy**  
Based on EDA-driven insights, decided to implement sample weights for population representativeness and align models with the Median Absolute Error (MdAE) success metric through tailored loss functions, target log transformation, and polynomial features to effectively handle the zero-inflated, heavy-tailed cost distribution.


## 🧹 Data Preprocessing
This project implements a hybrid workflow that bridges interactive exploration and production-grade reproducibility. While [Jupyter notebooks](notebooks/) were used for initial EDA and logic prototyping, the final pipeline is executed via a standalone script and orchestrated by [DVC](https://dvc.org/).

**Workflow Overview:**
- **Interactive Exploration:** [1_eda_and_preprocessing.ipynb](notebooks/1_eda_and_preprocessing.ipynb) was used to develop the transformation logic and perform outlier profiling.
- **Reproducible Pipeline:** All preprocessing steps are encapsulated in [scripts/preprocess.py](scripts/preprocess.py), which uses modular components from `src/` for consistent execution across environments.
- **Data Versioning:** The pipeline is tracked in [dvc.yaml](dvc.yaml), ensuring that data lineage is preserved and results can be reproduced with a single command:
  ```bash
  dvc repro preprocess
  ```

**Preprocessing Architecture**  
To ensure a seamless transition from raw survey data to live application predictions, the preprocessing workflow follows a structured three-step process:

**Step 1: Raw Data Preparation** (via `scripts/preprocess.py`)  
This stage converts the raw MEPS data to the clean format expected by the production-ready pipeline. These steps are primarily for data cleaning and population filtering:
- **Data Loading:** Imports the MEPS-HC 2023 SAS data as a pandas DataFrame.
- **Variable Selection:** Filters 29 essential columns (target variable, candidate features, ID, sample weights) from the original 1,374 columns.
- **Target Population Filtering:** Filters rows for adults with positive person weights (14,768 out of 18,919 respondents).
- **Data Type Handling:** Converts ID to string and sets as index.
- **Missing Value Standardization:** Recovers missing values from survey skip patterns and converts MEPS-specific missing codes to `np.nan`.
- **Binary Feature Standardization:** Standardizes binary features to 0/1 encoding.
- **Stateless Feature Engineering:** Creates a recent life transition feature and collapses sparse categories (e.g., recent divorce, job loss) into stable parent categories.
- **Train-Validation-Test Split:** Splits data into training (80%), validation (10%), and test (10%) sets using a distribution-informed stratified split to balance zero-inflation and the extreme tail of the target variable.

**Step 2: Production-Ready Pipeline** (via `src/pipeline.py`)  
Once the raw data is cleaned and prepared, the `preprocess.py` script *calls* a production-ready Scikit-learn pipeline. This pipeline is used for both training and inference (Web UI and API), ensuring absolute consistency across all environments:
- **Standardization:** Normalizes categorical inputs. Accepts both numeric codes (e.g. 0/1) and string labels (e.g. no/yes). 
- **Validation & Imputation:** Implements a `MissingValueChecker` to catch required fields and a `RobustSimpleImputer` for median/mode-based imputation.
- **Medical Feature Derivation:** Calculates aggregate chronic condition and functional limitation counts to capture health burden.
- **Scaling & Encoding:** Implements a `ColumnTransformer` with `RobustStandardScaler` and `RobustOneHotEncoder`.

**Step 3: Data Persistence** (via `scripts/preprocess.py`)  
Finally, the script merges the processed features with the target variable and survey weights to generate the final model-ready artifacts:
- **Parquet Export:** Saves the training, validation, and test sets as `.parquet` files to preserve data types and ensure high-performance loading during model training.

![Preprocessing Pipeline](assets/pipeline.svg)

**Exploratory Phase** (via `notebooks/1_eda_and_preprocessing.ipynb`):  
- **Handling Duplicates**: Verified the absence of duplicates based on the ID column, complete rows, and all columns except ID.
- **Handling Outliers**: Detected univariate outliers with 3SD and 1.5 IQR methods and multivariate outliers with an isolation forest (5% contamination). Profiled outliers by comparing out-of-pocket costs and feature distributions between inliers and outliers. Confirmed that outliers represent legitimate high risk profiles rather than data errors, and retained all outliers to preserve the model's ability to predict extreme out-of-pocket costs [(see detailed outlier analysis)](#outlier-analysis).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 🧠 Modeling

### 📏 Baseline Models  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### 🎛️ Hyperparameter Tuning  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### 🏆 Final Model
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 📂 Project Structure
```text
├── notebooks/               # Jupyter Notebooks 
│   ├── 1_eda_and_preprocessing.ipynb  # EDA, preprocessing, and pipeline development
│   ├── 1_eda_and_preprocessing.py     # Script version (generated via Jupytext)
│   ├── 2_modeling.ipynb               # Model training, evaluation and hyperparameter tuning 
│   └── 2_modeling.py                  # Script version (generated via Jupytext)
│
├── scripts/                 # Reproducible pipeline scripts 
│   ├── preprocess.py        # Production-ready data preprocessing
│   └── train_baseline.py    # Baseline model training 
│
├── src/                     # Core package source code 
│   ├── constants.py         # Feature lists and display labels
│   ├── transformers.py      # Custom Scikit-learn transformers
│   └── pipeline.py          # Preprocessing and prediction pipelines
│
├── app/                     # (Planned) Web application source code
│
├── models/                  # (Planned) Trained model artifacts (ignored by Git)
│
├── data/                    # Raw and processed datasets (ignored by Git)
│   ├── h251.sas7bdat        # MEPS-HC 2023 dataset (SAS V9 format)
│   └── *_preprocessed.*     # Training, validation, and test sets (CSV/Parquet)
│
├── figures/                 # Generated figures
│   ├── eda/                 # Distribution and relationship plots 
│   ├── outliers/            # Outlier analysis plots 
│   ├── training/            # (Planned) Training curves and learning rates (ignored by Git)
│   ├── evaluation/          # (Planned) Model performance plots (ignored by Git)
│   └── tuning/              # (Planned) Hyperparameter tuning results (ignored by Git)
│
├── assets/                  # Images and other assets for README
│   ├── header.png           # Header image
│   ├── data_infographic.jpg # MEPS data overview infographic
│   └── healthcare_costs_infographic.png  # U.S. healthcare cost explainer
│
├── tests/                   # (Planned) Software testing for web application
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── e2e/                 # End-to-end tests
│
├── docs/                    # Project documentation and resources
│   ├── specs/               # PRD and tech specs
│   │   ├── product_requirements.md
│   │   └── technical_specifications.md
│   ├── references/          # MEPS documentation, codebook, and data dictionary 
│   ├── research/            # Background research 
│   └── workflow/            # Git conventions
│
├── pyproject.toml           # Project configuration and dependencies
├── requirements.txt         # Proxy for production dependencies 
├── requirements-train.txt   # Training dependencies 
├── requirements-test.txt    # Test dependencies 
│
├── dvc.yaml                 # Preprocessing and modeling pipeline definitions (stages, deps, outs)
├── dvc.lock                 # Hash-based data lineage lockfile
├── .dvc/                    # DVC configuration 
│
├── README.md                # Project overview 
├── AGENTS.md                # Context and instructions for AI agents
├── LICENSE                  # MIT License
└── .gitignore               # Files and directories excluded from version control
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## ⚙️ Getting Started

### Installation and Setup
This project uses three isolated virtual environments to keep application dependencies lightweight. In all three setups, the project is installed as a local package, ensuring that the `src/` module can be reliably imported from any folder.

**1. Training Environment (`.venv-train`)**
- **Purpose:** Model development (preprocessing, EDA, training, evaluation, tuning).
- **Setup:**
  ```bash
  python -m venv .venv-train
  source .venv-train/bin/activate  # or .venv-train\Scripts\activate on Windows
  pip install -r requirements-train.txt
  ```
- **Import Logic:** This environment uses an **editable install** (`-e .[train]`). Changes you make to `src/` are instantly available in your notebooks without re-installation.

**2. Application Environment (`.venv-app`)**
- **Purpose:** Run and test the web application.
- **Setup:**
  ```bash
  python -m venv .venv-app
  source .venv-app/bin/activate  # or .venv-app\Scripts\activate on Windows
  pip install -r requirements.txt
  ```
- **Import Logic:** This environment installs the project as a **regular package** (`.[app]`). This mirrors the production environment, allowing the app to reliably import from `src/` regardless of where it is launched.

**3. Testing Environment (`.venv-test`)**
- **Purpose:** Web App/API testing using unit, integration, and end-to-end tests with `pytest`.
- **Setup:**
  ```bash
  python -m venv .venv-test
  source .venv-test/bin/activate  # or .venv-test\Scripts\activate on Windows
  pip install -r requirements-test.txt
  ```
- **Import Logic:** This environment uses an **editable install** (`-e .[app,test]`). It combines both the application dependencies and the testing tools, allowing you to run tests against your latest code.

**4. Data Management (DVC)**
- **Purpose:** Version control for local data and reproducibility of preprocessing and modeling.
- **Workflow:**
  - **Run Full Pipeline:** To execute all stages (preprocessing through baseline modeling):
    ```bash
    dvc repro
    ```
  - **Run Specific Stages:**
    - `dvc repro preprocess`: Reproduce only the data preparation, feature engineering, and preprocessing.
    - `dvc repro baseline`: Reproduce baseline model training (will re-run `preprocess` if data or script changed).

#### Production Deployment 
The project is optimized for deployment on Hugging Face. When you connect your repository to Hugging Face Spaces (or any platform using `requirements.txt`), it automatically runs:
```bash
pip install -r requirements.txt
```
Because `requirements.txt` contains `. [app]`, the platform installs the project itself as a package. This ensures your application can always find the `src` module regardless of the working directory.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## ©️ License
This project is licensed under the [MIT License](LICENSE).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 👏 Credits
This project was made possible with the help of the following resources:
- **Dataset**: [2023 Full Year Consolidated Data File (HC-251)](https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-251) from the [Medical Expenditure Panel Survey (MEPS)](https://meps.ahrq.gov/mepsweb/), provided by the [Agency for Healthcare Research and Quality (AHRQ)](https://www.ahrq.gov/).
- **Images**: 
  - Header: The [header image](./assets/header.png) was generated using [GPT Image 1.5](https://openai.com/index/new-chatgpt-images-is-here/) via the [ChatGPT app](https://chatgpt.com/) by OpenAI. 
  - Infographics: The [MEPS data infographic](./assets/infographic_meps_data.jpg) and the [US healthcare costs infographic](./assets/infographic_healthcare_costs.png) were generated using [Gemini 3 Pro Image](https://deepmind.google/models/gemini-image/pro/) via the [Gemini app](https://gemini.google.com/app) by Google.
- **AI Coding Assistant**: [Antigravity](https://antigravity.google/) by Google.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- APPENDIX -->
## 📎 Appendix
### Distributions
<a id="numerical-distributions">![Numerical Distributions](figures/eda/numerical_distributions.png)</a>

Table of population statistics for all numerical features:
| Feature         | Count       | Mean  | Std   | Min  | 25%  | 50%  | 75%  | Max  |
|-----------------|-------------|-------|-------|------|------|------|------|------|
| Age             | 259,681,066 | 48.32 | 18.54 | 18.0 | 32.0 | 47.0 | 63.0 | 85.0 |
| Family Size     | 259,568,347 | 2.88  | 1.59  | 1.0  | 2.0  | 2.0  | 4.0  | 14.0 |
| Physical Health | 258,917,544 | 2.37  | 1.04  | 1.0  | 2.0  | 2.0  | 3.0  | 5.0  |
| Mental Health   | 258,635,089 | 2.26  | 1.03  | 1.0  | 1.0  | 2.0  | 3.0  | 5.0  |

<p align="right">(<a href="#-exploratory-data-analysis-eda">back to EDA</a> | <a href="#readme-top">back to top</a>)</p>

<a id="categorical-distributions">![Categorical Distributions](figures/eda/categorical_distributions.png)</a>
<p align="right">(<a href="#-exploratory-data-analysis-eda">back to EDA</a> | <a href="#readme-top">back to top</a>)</p>

<a id="binary-distributions">![Binary Distributions](figures/eda/binary_distributions.png)</a>
<p align="right">(<a href="#-exploratory-data-analysis-eda">back to EDA</a> | <a href="#readme-top">back to top</a>)</p>

### Feature-Target Relationships
<a id="numerical-feature-target-relationships">![Numerical Feature-Target Relationships](figures/eda/numerical_feature_target_relationships.png)</a>
<p align="right">(<a href="#-exploratory-data-analysis-eda">back to EDA</a> | <a href="#readme-top">back to top</a>)</p>

<a id="categorical-feature-target-relationships">![Categorical Feature-Target Relationships](figures/eda/categorical_feature_target_relationships.png)</a>
<p align="right">(<a href="#-exploratory-data-analysis-eda">back to EDA</a> | <a href="#readme-top">back to top</a>)</p>

<a id="binary-feature-target-relationships">![Binary Feature-Target Relationships](figures/eda/binary_feature_target_relationships.png)</a>
<p align="right">(<a href="#-exploratory-data-analysis-eda">back to EDA</a> | <a href="#readme-top">back to top</a>)</p>

### Outlier Analysis
**1. Outlier Detection:** Utilized an Isolation Forest (5% contamination) to identify multivariate outliers in training data.  
**2. Outlier Profiling:** Compared out-of-pocket costs and feature distributions between inliers and outliers.  
**3. Outlier Treatment:** Retained all outliers as legitimate "High Comorbidity" health profiles essential for robust tail-risk prediction.

**Cost Concentration**  
While outliers are only 1.1x more likely to cross the median cost threshold, they are **3.4x more likely** to be in the Top 1% of spenders.

![Outlier Lorenz Curves](figures/outliers/outlier_lorenz_curve.png)
![Outlier Profile for Numerical Features and Target](figures/outliers/outlier_numeric_profile.png)
![Outlier Profile for Binary Features](figures/outliers/outlier_binary_profile.png)
![Outlier Profile for Categorical Features](figures/outliers/outlier_categorical_profile.png)

<p align="right">(<a href="#-data-preprocessing">back to Preprocessing</a> | <a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS -->
[Python-badge]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[Pandas-badge]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-%23DDDDDD?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxODAiIGhlaWdodD0iMTgwIiBzdHJva2U9ImdyYXkiPgo8ZyBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkYiPgo8Y2lyY2xlIGN4PSI5MCIgY3k9IjkwIiByPSI4OCIvPgo8Y2lyY2xlIGN4PSI5MCIgY3k9IjkwIiByPSI2NiIvPgo8Y2lyY2xlIGN4PSI5MCIgY3k9IjkwIiByPSI0NCIvPgo8Y2lyY2xlIGN4PSI5MCIgY3k9IjkwIiByPSIyMiIvPgo8cGF0aCBkPSJtOTAsMnYxNzZtNjItMjYtMTI0LTEyNG0xMjQsMC0xMjQsMTI0bTE1MC02MkgyIi8+CjwvZz48ZyBvcGFjaXR5PSIuOCI+CjxwYXRoIGZpbGw9IiM0NEMiIGQ9Im05MCw5MGgxOGExOCwxOCAwIDAsMCAwLTV6Ii8+CjxwYXRoIGZpbGw9IiNCQzMiIGQ9Im05MCw5MCAzNC00M2E1NSw1NSAwIDAsMC0xNS04eiIvPgo8cGF0aCBmaWxsPSIjRDkzIiBkPSJtOTAsOTAtMTYtNzJhNzQsNzQgMCAwLDAtMzEsMTV6Ii8+CjxwYXRoIGZpbGw9IiNEQjMiIGQ9Im05MCw5MC01OC0yOGE2NSw2NSAwIDAsMC01LDM5eiIvPgo8cGF0aCBmaWxsPSIjM0JCIiBkPSJtOTAsOTAtMzMsMTZhMzcsMzcgMCAwLDAgMiw1eiIvPgo8cGF0aCBmaWxsPSIjM0M5IiBkPSJtOTAsOTAtMTAsNDVhNDYsNDYgMCAwLDAgMTgsMHoiLz4KPHBhdGggZmlsbD0iI0Q3MyIgZD0ibTkwLDkwIDQ2LDU4YTc0LDc0IDAgMCwwIDEyLTEyeiIvPgo8L2c+PC9zdmc+
[Matplotlib-url]: https://matplotlib.org/
[Seaborn-badge]: https://img.shields.io/badge/seaborn-%230C4A89.svg?style=for-the-badge&logo=seaborn&logoColor=white
[Seaborn-url]: https://seaborn.pydata.org/
[scikit-learn-badge]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/
[DVC-badge]: https://img.shields.io/badge/DVC-9cf?style=for-the-badge&logo=data-version-control&logoColor=white
[DVC-url]: https://dvc.org/
[JupyterNotebook-badge]: https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white
[JupyterNotebook-url]: https://jupyter.org/
