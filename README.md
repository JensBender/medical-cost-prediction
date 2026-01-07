<!-- anchor tag for back-to-top links -->
<a name="readme-top"></a>

<!-- HEADER IMAGE  -->

<!-- SHORT SUMMARY  -->


## 📋 Table of Contents
<ol>
  <li>
    <a href="#-summary">Summary</a>
    <ul>
      <li><a href="#️-built-with">Built With</a></li>
    </ul>
  </li>
  <li>
    <a href="#-motivation">Motivation</a>
  </li>
  <li>
    <a href="#️-data">Data</a>
  </li>
  <li>
    <a href="#-data-preprocessing">Data Preprocessing</a>
  </li>
  <li>
    <a href="#-exploratory-data-analysis-eda">Exploratory Data Analysis (EDA)</a>
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
      <li><a href="#-virtual-environments">Virtual Environments</a></li>
    </ul>
  </li>
  <li>
    <a href="#️-license">License</a>
  </li>
  <li>
    <a href="#-credits">Credits</a>
  </li>
</ol>


## 🎯 Summary
### 🛠️ Built With
- [![Python][Python-badge]][Python-url]
- [![Pandas][Pandas-badge]][Pandas-url]
- [![Matplotlib][Matplotlib-badge]][Matplotlib-url] 
- [![Seaborn][Seaborn-badge]][Seaborn-url]
- [![scikit-learn][scikit-learn-badge]][scikit-learn-url]
- [![Jupyter Notebook][JupyterNotebook-badge]][JupyterNotebook-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 💡 Motivation
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 🗂️ Data
The **Medical Expenditure Panel Survey (MEPS)** is the most complete source of data on the cost and use of health care and health insurance coverage in the United States. Administered by the **Agency for Healthcare Research and Quality (AHRQ)**, it is a set of large-scale surveys of families and individuals, their medical providers, and employers. MEPS provides nationally representative estimates for the **U.S. civilian noninstitutionalized population**:
- **Included:** People living in households (houses, apartments) and non-institutional group quarters (e.g., college dorms).
- **Excluded:** Active-duty military, and people in institutions like nursing homes or prisons. Also excludes individual periods where a respondent moved outside the U.S. during the survey year.

**MEPS Household Component**  
The Household Component (MEPS-HC) collects comprehensive data directly from families and individuals. To ensure accuracy, household-reported expenditures are validated and imputed using the Medical Provider Component (MEPS-MPC), which draws data directly from doctors, hospitals, and pharmacies.

**MEPS-HC 2023**  
This project utilizes the 2023 Full-Year Consolidated Data File (HC-251).
- **Sample Size:** 18,919 individuals.
- **Variables:** 1,374.
- **Data Domains:** Medical expenditures, conditions, and events, demographics (e.g., age, ethnicity, and income), health insurance coverage, access to care, health status, and jobs held.
- **Format:** Available in ASCII, SAS transport, SAS V9, XLSX, and Stata file formats.
- **Collection Period:** Rounds 3, 4, and 5 of Panel 27, and Rounds 1, 2, and the 2023 portion of Round 3 of Panel 28.
- **Released:** August 2025.

![MEPS Data Infographic](assets/data_infographic.jpg)

**Target Variable**  
The target variable is **total out-of-pocket health care costs in 2023** (`TOTSLF23`). This variable represents what the person or their family actually paid directly for all medical events throughout 2023, including:
- Copays and coinsurance
- Deductibles
- Services not covered by insurance

`TOTSLF23` directly answers practical user questions like "How much should I contribute to my FSA/HSA?" and "What's my financial exposure?" For uninsured users, out-of-pocket costs approximate total costs, making this target appropriate across all insurance statuses.  

<details>
<summary><strong>US Healthcare Costs Explained</strong> (click to expand)</summary>

![US Healthcare Costs Infographic](./assets/healthcare_costs_infographic.png)
</details>

**Feature Selection**  
A subset of 26 candidate features was selected from MEPS-HC 2023 based on the following criteria:
- **Consumer Accessibility:** Users can answer from memory without looking up records, ensuring the model is usable in a consumer-facing app.
- **Beginning-of-Year Data:** To enable the app to be used during Open Enrollment for predicting *upcoming* costs, only variables measured at the beginning of the year (`31` suffix) or stable traits are used to prevent data leakage.
- **Predictive Power:** Features have established significance in healthcare cost literature.

**Features**
| Category | Variable | Label | Description |
| :--- | :--- | :--- | :--- |
| **Demographics** | `AGE23X` | Age | Age as of Dec 31, 2023. |
| | `SEX` | Sex | Biological sex. |
| | `REGION23` | Region | Census region (Northeast, Midwest, South, West). |
| | `MARRY31X` | Marital Status | Status at beginning of year. |
| **Socioeconomic** | `POVCAT23` | Poverty Category | Family income relative to poverty line. |
| | `FAMSZE23` | Family Size | Number of related persons residing together. |
| | `HIDEG` | Education | Highest degree attained. |
| | `EMPST31` | Employment Status | Status at beginning of year. |
| **Insurance & Access** | `INSCOV23` | Insurance | Coverage status (Private, Public, Uninsured). |
| | `HAVEUS42` | Usual Source of Care | Regular doctor or clinic. |
| **Perceived Health & Lifestyle** | `RTHLTH31` | Physical Health | Self-rated physical health (1–5). |
| | `MNHLTH31` | Mental Health | Self-rated mental health (1–5). |
| | `ADSMOK42` | Smoker | Currently smokes cigarettes. |
| **Limitations & Symptoms** | `ADLHLP31` | ADL Help | Needs help with activities of daily living (personal care, bathing, dressing). |
| | `IADLHP31` | IADL Help | Needs help with instrumental activities of daily living (paying bills, taking medications, doing laundry). |
| | `WLKLIM31` | Walking Limitation | Difficulty walking or climbing stairs. |
| | `COGLIM31` | Cognitive Limitation | Confusion or memory loss. |
| | `JTPAIN31_M18` | Joint Pain | Pain/stiffness in past year. |
| **Chronic Conditions** | `HIBPDX` | Hypertension | Diagnosed with high blood pressure. |
| | `CHOLDX` | High Cholesterol | Diagnosed with high cholesterol. |
| | `DIABDX_M18` | Diabetes | Diagnosed with diabetes. |
| | `CHDDX` | Heart Disease | Diagnosed with coronary heart disease. |
| | `STRKDX` | Stroke | Diagnosed with stroke. |
| | `CANCERDX` | Cancer | Diagnosed with cancer or malignancy. |
| | `ARTHDX` | Arthritis | Diagnosed with arthritis. |
| | `ASTHDX` | Asthma | Diagnosed with asthma. |

**Sample Weights**  
MEPS-HC 2023 includes survey sample weights (`PERWT23F`) to adjust for the complex survey design (stratification, clustering, oversampling) and non-response. This machine learning project utilizes these weights in model training to correct for the survey's intentional oversampling of specific subgroups (e.g., the elderly, low-income), preventing the model from being biased toward these groups and ensuring it learns relationships that are true for the general population.

**MEPS Resources**
| Resource | Description | Link |
| :--- | :--- | :--- |
| Data | MEPS-HC 2023 Full Year Consolidated Data File (HC-251). | [Visit Page](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-251) |
| Full Documentation | Technical details on data collection, variable editing, and survey sampling. | [View PDF](docs/h251doc.pdf) |
| Codebook | Variables, labels, coding schemes, and frequencies. | [View PDF](docs/h251cb.pdf) |
| MEPS Overview | Background on MEPS components and larger survey history. | [Visit Page](https://meps.ahrq.gov/mepsweb/about_meps/survey_back.jsp) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 🧹 Data Preprocessing
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 🔍 Exploratory Data Analysis (EDA)
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
│   └── medical_cost_prediction.ipynb  # Preprocessing, EDA, model training, evaluation, tuning and selection
│
├── app/                     # Web application source code
│   └── app.py               # Main application file
│
├── models/                  # Models and pipelines (ignored by Git)
│   ├── model.joblib         # Trained final model  
│   └── pipeline.joblib      # Pipeline with model and preprocessing
│
├── data/                    # Raw and processed datasets (ignored by Git)
│   └── h251.sas7bdat        # MEPS-HC 2023 dataset (SAS V9 format)
│
├── figures/                 # Generated figures (ignored by Git)
│   ├── eda/                 # Exploratory data analysis visualizations
│   ├── training/            # Training curves and learning rates
│   ├── evaluation/          # Model performance plots
│   └── tuning/              # Hyperparameter tuning results
│
├── assets/                  # Images and other assets for README
│   ├── data_infographic.jpg # MEPS data overview infographic
│   └── header.png           # Project header image
│
├── tests/                   # Software testing for web application
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── e2e/                 # End-to-end tests
│
├── docs/                    # Project documentation and resources
│   ├── specs/               # PRD and tech specs
│   │   ├── product_requirements.md
│   │   └── technical_specifications.md
│   ├── references/          # Official MEPS documentation and codebook 
│   ├── research/            # Background research 
│   └── workflow/            # Git conventions
│
├── requirements.txt         # Production dependencies 
├── requirements-train.txt   # Training dependencies 
├── requirements-test.txt    # Test dependencies 
│
├── README.md                # Project overview 
├── AGENTS.md                # Context and instructions for AI agents
├── LICENSE                  # MIT License
└── .gitignore               # Files and directories excluded from version control
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## ⚙️ Getting Started

### Virtual Environments
This project uses two isolated environments to keep application dependencies lightweight for production deployment.

**Training Environment** (`.venv-train`)
- **Requirements File:** `requirements-train.txt`
- **Purpose:** Model development (preprocessing, EDA, training, evaluation, tuning, selection)
- **Key Libraries:** `jupyterlab`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

**Application Environment** (`.venv-app`)
- **Requirements Files:** 
    - `requirements.txt`: Web application dependencies for production deployment (used by deployment platforms)
    - `requirements-test.txt`: Inherits from `requirements.txt` and adds `pytest` for local testing
- **Purpose:** Run and test the web application 
- **Key Libraries:** `fastapi`, `gradio`, `scikit-learn`

**Note:** Both environments use the same version of `scikit-learn` to ensure model consistency across training and deployment.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## ©️ License
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 👏 Credits
This project was made possible with the help of the following resources:
- **Dataset**: [2023 Full Year Consolidated Data File (HC-251)](https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-251) from the [Medical Expenditure Panel Survey (MEPS)](https://meps.ahrq.gov/mepsweb/), provided by the [Agency for Healthcare Research and Quality (AHRQ)](https://www.ahrq.gov/).
- **Infographics**: The [MEPS data infographic](./assets/data_infographic.jpg) and [US healthcare costs infographic](./assets/healthcare_costs_infographic.png) were generated using [Gemini 3 Pro Image](https://deepmind.google/models/gemini-image/pro/) via the [Gemini app](https://gemini.google.com/app) by Google.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


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
[JupyterNotebook-badge]: https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white
[JupyterNotebook-url]: https://jupyter.org/
