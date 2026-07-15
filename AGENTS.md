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

## Writing style

Use plain, natural English.

- Write for a smart reader who is new to the project.
- Use short sentences and common words.
- Prefer concrete examples over abstract explanations.
- Use active voice.
- Put the main point first.
- Explain necessary technical terms the first time you use them.
- Avoid academic language, corporate jargon, marketing language, and buzzwords.
- Do not use a complicated word when a simple word means the same thing.
- Remove filler, repetition, and unnecessary background.
- Keep headings and lists short.
- Do not make the writing sound formal just because it is documentation.

Before making changes to any document, rewrite sentences that a new team member might
need to read twice.

## Commits

- Do not commit directly unless explicitly asked.
- After making code or documentation changes, suggest one atomic commit message
  that follows `docs/workflow/git_conventions.md`.
