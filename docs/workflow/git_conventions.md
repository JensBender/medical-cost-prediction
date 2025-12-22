# Git Conventions
Conventions for writing Git commit messages in this machine learning project.

## 1. The Format
Conventional structure of a commit message with a header, body, and footer:

```text
<type>(<scope>): <subject>

<body>

<footer>
```

**The Header**
- **Type**: Describes **what** changed.
    - `feat`: A new feature (e.g., adding a new API endpoint).
    - `fix`: A bug fix.
    - `docs`: Documentation only changes.
    - `style`: Formatting, missing semi-colons, etc. (no code change).
    - `refactor`: Refactoring production code (no new features or bug fixes).
    - `perf`: Performance improvements.
    - `test`: Adding or refactoring tests.
    - `chore`: Build tasks, package manager configs, etc.
- **Scope**: Describes **where** it changed, i.e. the specific section of the codebase. For this project, use:
    - `data`: Data loading, preprocessing, cleaning, or dataset versioning.
    - `eda`: Exploratory data analysis.
    - `model`: Model architecture, training logic, or hyperparameter tuning.
    - `eval`: Model evaluation, metric calculations, feature importance or error analysis.
    - `notebook`: Jupyter Notebook.
    - `ui`: Gradio user interface frontend.
    - `api`: FastAPI backend, prediction endpoint.
    - `deploy`: Deployment scripts, Dockerfiles, or serving code.
    - `docs`: Files in the `docs/` directory.
    - `readme`: README.md file.
    - `prd`: Product Requirements Document.
    - `tech-spec`: Technical Specifications Document.
    - `tests`: Unit tests.
- **Subject**: A concise description of the change (maximum 50 characters).

**The Body** (optional)
- More detailed explanation of **what** and **why**.
- Wrap lines at 72 characters.
- Separate from the header with a blank line.

**The Footer** (optional)
- References to issues (e.g., `Closes #123`) or notes on breaking changes.


## 2. Best Practices
1.  **Imperative Mood**: Write the subject line as a command.
    - ✅ `feat(model): train random forest classifier`
    - ❌ `feat(model): trained random forest classifier`
    - ❌ `feat(model): training random forest classifier`
2.  **Atomic Commits**: Keep commits focused on a single task. Don't mix a bug fix with a new feature.
3.  **Jupyter Notebooks**: 
    - **Clear Outputs**: Clear cell outputs of notebooks before committing to keep diffs small and readable.
    - **Description**: Use the **Body** of the commit message to list key findings or changes, since the raw diff will just be JSON metadata.
4.  **No Large Files**: Never commit large datasets or model artifacts (`.pkl`, `.joblib`, `.h5`) directly. Commit the scripts that generate them or the pointer files.
5.  **Header Formatting**: Use all lowercase for the type, scope, and subject, except for proper nouns or acronyms (e.g., `FastAPI`, `JSON`). Do not end the subject with a period.
6.  **Body Formatting**: Use standard English grammar, capitalization, and punctuation for prose. Omit trailing periods for bullet points.


## 3. Examples
**Data Preprocessing**
```text
feat(data): implement standard scaling for numerical features

Added StandardScaler to the preprocessing pipeline to normalize 
age and bmi columns before training.
```

**Exploratory Data Analysis**
```text
feat(eda): analyze correlation between smoking and charges

- Created scatter plots for bmi vs charges
- Calculated Pearson correlation coefficients
- Found strong positive correlation for smokers
```

**Model Training**
```text
feat(model): switch from LinearRegression to XGBoost

Linear models were underfitting the data. XGBoost provides 
better handling of non-linear relationships.
```

**Hyperparameter Tuning**
```text
chore(model): update learning rate and max_depth

- Increased learning rate to 0.01
- Reduced max_depth to 5 to prevent overfitting
```

**Model Evaluation**
```text
feat(eval): add RMSE and MAE metrics in model evaluation section
```

**API**
```text
feat(api): add /predict endpoint using FastAPI

Created the main inference route that accepts JSON input 
and returns the predicted medical cost.
```

**Breaking Change**
```text
feat(api): change input format for prediction endpoint

The API now expects a list of dictionaries instead of a single dictionary 
to support batch predictions.

BREAKING CHANGE: The /predict endpoint payload structure has changed.
Closes #42
```
