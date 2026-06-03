## Project

Medical cost prediction project using MEPS 2023 data to estimate US
out-of-pocket healthcare costs. The project includes preprocessing, EDA,
model evaluation, and a planned FastAPI/Gradio app.

## Key Workflow Rules

- Use the correct virtual environment:
  - Training and notebooks: `.\.venv-train\Scripts\python`
  - App: `.\.venv-app\Scripts\python`
  - Tests: `.\.venv-test\Scripts\python`
- Prefer editing Jupytext notebook scripts (`notebooks/*.py`) instead of
  `.ipynb` files unless the user asks otherwise.

## Architecture Rules

- Keep the sklearn pipeline lean: core validation and ML logic only.
- Handle UI/API formatting and user-input cleanup at the interface layer.
- Use sample weights for MEPS metrics when applicable.
- Treat the final test set as locked: do not tune model choices, thresholds, or
  explanations based on final test results.
  
## Important Files

- `src/constants.py`: pipeline-critical constants; DVC-tracked, changes may
  trigger reruns.
- `src/display.py`: display labels and presentation constants; safer to edit
  for notebook/UI wording.
- `docs/specs/product_requirements.md`: product goals and requirements.
- `docs/specs/technical_specifications.md`: technical design.
- `docs/workflow/git_conventions.md`: commit message rules.

## DVC And Artifacts

- Do not commit large model artifacts from `models/`.
- If DVC-tracked files or pipeline inputs change, mention whether `dvc repro`
  is likely needed.

## Commits

- Do not commit directly unless explicitly asked.
- For code or docs changes, suggest an atomic commit message following
  `docs/workflow/git_conventions.md`.
