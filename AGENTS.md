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

Apply these rules to docstrings, comments, explanatory strings in code,
notebook Markdown, analysis summaries, the README, the product requirements
and technical specifications documents, and suggested commit messages.

- Use the same plain, direct tone as a helpful work summary. Documentation must
  make sense without the conversation that led to it.
- Write for a smart reader who is new to the project. Put the main point first,
  then explain what happens, why it matters, or what the reader needs to do.
- Use common words, active verbs, and concrete examples. Replace abstract labels
  with explanations of the actual behavior.
- Keep necessary technical terms and explain unfamiliar ones where they first
  matter. Use real code names and call each thing by the same name throughout.
- Prefer readable sentences over the fewest words. Vary sentence length and keep
  related conditions and consequences together when that makes them clearer.
- Remove filler, repetition, jargon, and unnecessary formality. Use headings and
  lists only when they help the reader follow or find information.
- In docstrings, explain behavior and any inputs, outputs, or limits the caller
  needs to know. In comments, explain reasons that are not obvious from the code.
- When writing or revising unit tests, use `tests/unit/test_prediction.py` as the
  reference for behavior-focused names, concrete inputs and expected outputs,
  descriptive variables, and clear setup/action/assertion structure. Split
  unrelated behaviors and parameterize variations with descriptive IDs.
- In analysis summaries, state the finding and its evidence, then explain what
  it means. Preserve uncertainty, units, population scope, and whether results
  use survey weights. Do not turn associations into causal claims.
- Distinguish planned behavior, requirements, and implemented behavior. Preserve
  technical meaning when simplifying wording.

Before finishing, reread the text you added or changed from a new team member's
perspective. Rewrite passages that need a second read, without expanding the edit
to unrelated text. Use `$plain-writing` for a focused revision when requested.

## Commits

- Do not commit directly unless explicitly asked.
- After changing project files, suggest one atomic commit message. Immediately
  before suggesting it, inspect the current Git state and relevant diff. 
- Follow `docs/workflow/git_conventions.md` and apply `$plain-writing`. Write the
  message for someone reading the Git history without this conversation.
