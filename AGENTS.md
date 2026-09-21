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
- After editing a notebook script, synchronize its tracked `.ipynb` pair with
  Jupytext. Do not execute the notebook unless the task requires it.

## Working Style

- Make the smallest change needed for the current request.
- Stay close to the existing code. Do not add cleanup, refactors, abstractions,
  or related improvements unless asked.
- For multi-step work, complete one meaningful step and wait for confirmation.
- Mention additional opportunities instead of implementing them.

## Architecture Rules

- Keep the sklearn pipeline lean: core preprocessing and ML logic only.
- Handle UI/API formatting and user-input cleanup at the interface layer.
- Use `PERWT23F` as `sample_weight` during model training. Use survey-weighted
  metrics and benchmarks when estimating population-level performance.
  
## Important Files

- `src/constants.py`: dependency of all DVC stages; changes can invalidate
  cached stages and trigger reruns.
- `src/display.py`: display labels and presentation constants; safer to edit
  for notebook/UI wording.
- `docs/specs/product_requirements.md`: product goals and requirements.
- `docs/specs/technical_specifications.md`: technical design.
- `docs/workflow/git_conventions.md`: commit message rules.

## Writing

- Apply `$plain-writing` when writing or revising project documentation, README,
  product requirements, technical specifications, code comments, docstrings, 
  notebook analysis, or commit messages.

## Unit Tests

- When writing or revising unit tests, use `tests/unit/test_prediction.py` as
  the style reference. Prefer behavior-focused names, concrete inputs and
  expected outputs, and a clear setup/action/assertion structure. Split
  unrelated behaviors and parameterize closely related cases with descriptive
  IDs.

## Commits

- Do not commit directly unless explicitly asked.
- After changing project files, suggest one atomic commit message. Immediately
  before suggesting it, inspect the current Git state and relevant diff. 
- Follow `docs/workflow/git_conventions.md` and apply `$plain-writing`. Write the
  message for someone reading the Git history without this conversation.
