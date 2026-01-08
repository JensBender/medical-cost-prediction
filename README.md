<!-- anchor tag for back-to-top links -->
<a name="readme-top"></a>

<!-- HEADER IMAGE  -->
<img src="assets/header.png" alt="Header Image">

<!-- SHORT SUMMARY  -->
Designing an end-to-end ML application to predict out-of-pocket healthcare costs for personalized financial planning. Adopting a product-driven approach, authoring comprehensive Product Requirements and Technical Specifications that prioritize UX-centric design and model transparency. Engineering the ML pipeline to translate complex survey data (MEPS) into actionable budgeting insights.

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

![MEPS Data Infographic](assets/data_infographic.jpg)

**Target Variable**  
The target variable is **total out-of-pocket health care costs in 2023** (`TOTSLF23`). This variable represents what the person or their family actually paid directly for all medical events throughout 2023. It includes copays and coinsurance, deductibles, and services not covered by insurance. `TOTSLF23` directly answers practical user questions like "How much should I contribute to my FSA/HSA?" and "What's my financial exposure?" For uninsured users, out-of-pocket costs approximate total costs, making this target appropriate across all insurance statuses.  

<details>
<summary>ℹ️<strong>US Healthcare Costs Explained</strong> (click to expand)</summary>

![US Healthcare Costs Infographic](./assets/healthcare_costs_infographic.png)
</details>

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
This project is licensed under the [MIT License](LICENSE).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 👏 Credits
This project was made possible with the help of the following resources:
- **Dataset**: [2023 Full Year Consolidated Data File (HC-251)](https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-251) from the [Medical Expenditure Panel Survey (MEPS)](https://meps.ahrq.gov/mepsweb/), provided by the [Agency for Healthcare Research and Quality (AHRQ)](https://www.ahrq.gov/).
- **Images**: The [header image](./assets/header.png), the MEPS [data infographic](./assets/data_infographic.jpg), and the [US healthcare costs infographic](./assets/healthcare_costs_infographic.png) were generated using [Gemini 3 Pro Image](https://deepmind.google/models/gemini-image/pro/) via the [Gemini app](https://gemini.google.com/app) by Google.

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
