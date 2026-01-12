# Agent Context and Guidelines

This document provides essential context for AI agents working on this project. It serves as the primary source of truth for project structure, design philosophy, and development workflows.

## Project Overview
A machine learning project aimed at predicting US medical out-of-pocket costs using the MEPS (Medical Expenditure Panel Survey) 2023 dataset. The project includes data preprocessing, exploratory data analysis, model training, and deployment as a web application.

## Project Structure Map
Agents should follow this directory structure strictly when adding new files:

- `app/`: Web application source code (FastAPI + Gradio).
- `assets/`: Images and static assets for the README and documentation.
- `data/`: Local copies of subsets of the MEPS dataset (Gitignored).
- `docs/`: Project documentation organized by purpose:
    - `references/`: Official manuals and quantitative data mapping.
        - `data_dictionary.md`: Variable names, labels, and types.
        - `h251cb.pdf`: Official MEPS Codebook.
        - `h251doc.pdf`: Official MEPS dataset documentation.
    - `research/`: Domain knowledge and exploratory analysis.
        - `candidate_features.md`: Initial list of selected variables and rationale.
        - `ml_with_meps.md`: Notes on machine learning with survey weights.
        - `us_healthcare_costs_guide.md`: Background on US insurance and billing.
    - `specs/`: Project definition and technical architecture.
        - `product_requirements.md`: The "What" and "Why" (PRD).
        - `technical_specifications.md`: The "How" (Architecture, Cleaning, Modeling).
    - `workflow/`: Internal processes and developer guides.
        - `git_conventions.md`: Commit message and branching rules.
- `models/`: Serialized model artifacts (.joblib).
- `notebooks/`: Exploratory Data Analysis and model training experiments.
- `tests/`: Software testing suites (Unit, Integration, E2E).

## Development Context

### Dual Environment Strategy
This project uses two isolated virtual environments to minimize production dependencies:
1. **Training (`.venv-train`)**: Used for notebooks, data science work, and model training.
2. **App (`.venv-app`)**: Used for the FastAPI/Gradio web application.

**Crucial for AI Agents:** When using the `run_command` tool to execute Python code or scripts, you **must** use the python executable located within the relevant virtual environment to ensure all dependencies (like pandas, sklearn) are available.

- For **Training tasks**, use: `./.venv-train/Scripts/python` (Windows) or `./.venv-train/bin/python` (Unix).
- For **App tasks**, use: `./.venv-app/Scripts/python` (Windows) or `./.venv-app/bin/python` (Unix).

When suggesting or running installation commands, always clarify which environment is being targeted.


### Documentation and Commits
- PRD vs Tech Spec: The `specs/product_requirements.md` defines the problem and requirements. The `specs/technical_specifications.md` defines the implementation details.
- Atomic Commits: All commits must follow the conventions defined in `docs/workflow/git_conventions.md`.

## Task-Specific Instructions
- Model Files: Do not commit large binary files in `models/`. Instead, document the scripts or parameters used to generate them.
- Reproducibility: Ensure all model training steps in notebooks include random seeds for reproducibility.
