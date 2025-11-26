## Virtual Environments
This project uses two separate environments to keep dependencies isolated and production deployments lightweight.

**Training Environment** (`.venv-train`)
- **Requirements File:** `requirements-train.txt`
- **Purpose:** Model development (preprocessing, EDA, training, evaluation, tuning, selection)
- **Key Libraries:** `notebook`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

**Application Environment** (`.venv-app`)
- **Requirements Files:** 
    - `requirements.txt`: Web application dependencies for production deployment (used by deployment platforms)
    - `requirements-test.txt`: Inherits from `requirements.txt` and adds `pytest` for local testing
- **Purpose:** Run and test the web application 
- **Key Libraries:** `fastapi`, `gradio`, `scikit-learn`

**Note:** Both environments use the same version of `scikit-learn` to ensure model consistency across training and deployment.
