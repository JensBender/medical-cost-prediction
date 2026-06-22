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

- Keep the sklearn pipeline lean: core preprocessing and ML logic only.
- Handle UI/API formatting and user-input cleanup at the interface layer.
- Use sample weights for MEPS metrics when applicable.
  
## Important Files

- `src/constants.py`: pipeline-critical constants; DVC-tracked, changes may
  trigger reruns.
- `src/display.py`: display labels and presentation constants; safer to edit
  for notebook/UI wording.
- `docs/specs/product_requirements.md`: product goals and requirements.
- `docs/specs/technical_specifications.md`: technical design.
- `docs/workflow/git_conventions.md`: commit message rules.

## Writing Style

- Write in clear, precise, plain language. Be accurate and specific. Avoid vague terms, marketing language, and decorative phrasing. Prefer short sentences. If something is uncertain, say so plainly.
- For code or docs changes, always include a unified diff of all changed files at the bottom of your response.

## Commits

- Do not commit directly unless explicitly asked.
- For code or docs changes, always suggest an atomic commit message following
  `docs/workflow/git_conventions.md`.
