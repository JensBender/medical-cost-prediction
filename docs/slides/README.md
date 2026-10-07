# Presentation files

`storyboard.md` holds the content and speaker notes. The main presentation
contains M1–M9, using the approved M2 and M7 design. It uses a white background,
dark blue text, and teal for model results in comparisons. Gray is reserved for
supporting information and slide IDs. Titles can use one or two lines; the
content starts below the title with a consistent gap. Appendix slides remain
in storyboard form.

`generate_slides.py` creates an editable PowerPoint, with optional PNG previews.
Speaker notes come from the storyboard; slide copy and layout are in the Python script.
After agreeing on content changes, update both where needed.
The previous JavaScript generator, `generate_slides.mjs`, is retained for comparison.

## Generate a revision

Install the training dependencies, which include `python-pptx`, and run the
Python generator from the repository root:

```powershell
.\.venv-train\Scripts\python -m pip install -r requirements-train.txt
.\.venv-train\Scripts\python docs/slides/generate_slides.py python-v1
```

The script writes `medical-cost-planner-main-python-v1.pptx` under
`exports/main-python-v1/`. PowerPoint generation needs no ChatGPT or Codex runtime,
model artifacts, or survey data. It uses `storyboard.md`, the images in `assets/`,
and the project images `../../assets/header.png` and
`../../assets/infographic_meps_data.jpg`. All these inputs are tracked by Git.

Choose a new revision name for each run. The Python generator refuses to overwrite
an existing revision and keeps previous exports. Generated files remain ignored
by Git. To recreate an older version, restore the generator, storyboard, and images
from the same Git commit.

## Optional PNG previews

In ChatGPT or Codex, use the paths returned by `load_workspace_dependencies`
to set these environment variables:

- `RUNTIME_NODE`: bundled Node.js executable
- `RUNTIME_NODE_MODULES`: bundled Node.js package directory
- `SKILL_DIR`: installed presentation skill directory

Then add `--render`:

```powershell
.\.venv-train\Scripts\python docs/slides/generate_slides.py python-v2 --render
```

The renderer reads the generated PowerPoint and writes `M1.png` through `M9.png`
and `A1.png` beside it. Rendering failures leave the PowerPoint file available.
Preview generation requires the bundled renderer; it is optional for someone
cloning the repository. The bundled presentation validators can also inspect
Python-generated decks separately. Previews do not verify behavior in PowerPoint.

## Previous JavaScript generator

The JavaScript generator uses `RUNTIME_NODE_MODULES`, `SKILL_DIR`, and
`RUNTIME_PYTHON` (the bundled Python executable). Run it with the bundled Node.js
executable and a new revision name:

```text
node docs/slides/generate_slides.mjs v12
```

That script writes the deck and previews under `exports/main-v12/`.
It keeps intermediate files and validation reports in `.build/v12/`.
Both directories are ignored by Git. Existing exported decks are not overwritten.
After a numbered draft passes validation and its previews render, the script
keeps the new export, the most recent complete previous export, and only the new
build directory. It preserves named versions such as `conference-final`.
