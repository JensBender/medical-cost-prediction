## Project Structure

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
│   └── h251.sas7bdat        # Medical cost dataset (SAS V9 format)
│
├── figures/                 # Generated figures (ignored by Git)
│   ├── eda/                 # Exploratory data analysis visualizations
│   ├── training/            # Training curves and learning rates
│   ├── evaluation/          # Model performance plots
│   └── tuning/              # Hyperparameter tuning results
│
├── assets/                  # Images and other assets for README
│   └── header.png           # Project header image
│
├── tests/                   # Software testing for web application
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── e2e/                 # End-to-end tests
│
├── docs/                    # Project documentation and references
│   ├── project_structure.md # Overview of project structure
│   └── git_conventions.md   # Conventions for Git commit messages 
│
├── requirements.txt         # Production dependencies 
├── requirements-train.txt   # Training dependencies 
├── requirements-test.txt    # Test dependencies 
│
├── README.md                # Project overview 
├── LICENSE                  # MIT License
└── .gitignore               # Files and directories excluded from version control
```
