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

- Keep changes focused on the requested outcome and preserve existing behavior
  outside its scope.
- Implement and verify the requested change, but pause when a decision needs
  my input, the scope would materially expand, or the next action is
  destructive or irreversible.

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
- `docs/git_conventions.md`: commit message rules.

## Writing

- Apply `$plain-writing` to project documentation, specs, code comments,
  docstrings, notebook analysis, and commit messages.

## Unit Tests

- When writing or revising unit tests, use `tests/unit/test_prediction.py` as
  the style reference. Prefer behavior-focused names, concrete inputs and
  expected outputs, and a clear setup/action/assertion structure. Split
  unrelated behaviors and parameterize closely related cases with descriptive
  IDs.

## Commits

- Do not commit directly unless explicitly asked.
- After changing project files, check the current Git status and suggest one
  atomic commit message that follows `docs/git_conventions.md`.
