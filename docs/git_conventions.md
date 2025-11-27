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
    - **Standard Types**:
        - `feat`: A new feature (e.g., adding a new API endpoint).
        - `fix`: A bug fix.
        - `docs`: Documentation only changes.
        - `style`: Formatting, missing semi-colons, etc. (no code change).
        - `refactor`: Refactoring production code (no new features or bug fixes).
        - `test`: Adding or refactoring tests.
        - `chore`: Build tasks, package manager configs, etc.
    - **ML-Specific Types**:
        - `data`: Changes to data loading, preprocessing, cleaning, or dataset versioning.
        - `eda`: Exploratory data analysis.
        - `model`: Model architecture, training logic, or hyperparameter tuning.
        - `eval`: Model evaluation, metric calculations, feature importance or error analysis.
        - `exp`: Configuration files for specific experiments.
        - `deploy`: Deployment scripts, Dockerfiles, or serving code.
- **Scope** (optional): Describes **where** it changed, i.e. the specific section of the codebase. For this project, use:
    - `app`: Web application code.
    - `notebook`: Analysis and experiments.
    - `data`: Data handling scripts.
    - `docs`: Files in the `docs/` directory.
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
    - ✅ `model: train random forest classifier`
    - ❌ `model: trained random forest classifier`
    - ❌ `model: training random forest classifier`
2.  **Atomic Commits**: Keep commits focused on a single task. Don't mix a bug fix with a new feature.
3.  **Jupyter Notebooks**: 
    - **Clear Outputs**: Clear cell outputs of notebooks before committing to keep diffs small and readable.
    - **Description**: Use the **Body** of the commit message to list key findings or changes, since the raw diff will just be JSON metadata.
4.  **No Large Files**: Never commit large datasets or model artifacts (`.pkl`, `.joblib`, `.h5`) directly. Commit the scripts that generate them or the pointer files.
5.  **Header Formatting**: Use all lowercase for the type, scope, and subject, except for proper nouns or acronyms (e.g., `FastAPI`, `JSON`). Do not end the subject with a period.
6.  **Body Formatting**: Use standard English grammar, capitalization, and punctuation for the body.


## 3. Examples

**Data Preprocessing**
```text
data(notebook): implement standard scaling for numerical features

Added StandardScaler to the preprocessing pipeline to normalize 
age and bmi columns before training.
```

**Exploratory Data Analysis**
```text
eda(notebook): analyze correlation between smoking and charges

- Created scatter plots for bmi vs charges
- Calculated Pearson correlation coefficients
- Found strong positive correlation for smokers
```

**Model Training**
```text
model(notebook): switch from LinearRegression to XGBoost

Linear models were underfitting the data. XGBoost provides 
better handling of non-linear relationships.
```

**Hyperparameter Tuning**
```text
model(notebook): update learning rate and max_depth

- Increased learning rate to 0.01
- Reduced max_depth to 5 to prevent overfitting
```

**Evaluation**
```text
eval(notebook): add RMSE and MAE metrics in model evaluation section
```

**Deployment**
```text
feat(app): add /predict endpoint using FastAPI

Created the main inference route that accepts JSON input 
and returns the predicted medical cost.
```
