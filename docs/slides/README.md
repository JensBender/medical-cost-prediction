# Presentation files

`storyboard.md` holds the content and speaker notes. The main presentation
contains M1–M9, using the approved M2 and M7 design. It uses a white background,
dark blue text, and teal for model results in comparisons. Gray is reserved for
supporting information and slide IDs. Main slides display plain numbers, the
cover is unnumbered, and appendix slides display A1, A2, etc. M1–M9 remain
internal IDs. Titles can use one or two lines; the content starts below the title
with a consistent gap. The U.S. healthcare costs, MEPS overview, and outlier
analysis appendix slides are included; the remaining appendix slides are
in storyboard form. Order and number appendix slides by their first reference
in the main presentation. Place unreferenced slides with their related topic.

`generate_slides.py` creates an editable PowerPoint, with optional PNG previews.
Speaker notes come from the storyboard; slide copy and layout are in the Python
script. After agreeing on content changes, update both where needed.

## Generate a revision

Install the training dependencies, which include `python-pptx`, and run the
Python generator from the repository root:

```powershell
.\.venv-train\Scripts\python -m pip install -r requirements-train.txt
.\.venv-train\Scripts\python docs/slides/generate_slides.py v4
```

The script writes `medical-cost-planner-v4.pptx` under
`exports/v4/`. PowerPoint generation needs no ChatGPT or Codex runtime,
model artifacts, or survey data. It uses `storyboard.md`, the images in `assets/`,
and the project images `../../assets/header.png`,
`../../assets/infographic_meps_data.jpg`,
`../../assets/infographic_healthcare_costs.png`, and
`../../figures/eda/lorenz_curve.png`. All these inputs are tracked by Git.

Choose a new revision name for each run. The Python generator refuses to overwrite
an existing revision and keeps previous exports. Generated files remain ignored
by Git. When revising slides in the ChatGPT desktop app or Codex, keep the newest
export and the previous complete export after validation and preview review.
Remove older draft exports and obsolete build files; preserve named presentation
milestones. The Python generator does not perform this cleanup automatically.
To recreate an older version, restore the generator, storyboard, and images
from the same Git commit. The earlier JavaScript generator is available in Git
history for versions created with it.

## Optional PNG previews

In ChatGPT or Codex, use the paths returned by `load_workspace_dependencies`
to set these environment variables:

- `RUNTIME_NODE`: bundled Node.js executable
- `RUNTIME_NODE_MODULES`: bundled Node.js package directory
- `SKILL_DIR`: installed presentation skill directory

Then add `--render`:

```powershell
.\.venv-train\Scripts\python docs/slides/generate_slides.py v5 --render
```

The renderer reads the generated PowerPoint and writes `M1.png` through `M9.png`,
`A1.png` through `A3.png` beside it. Rendering failures leave the PowerPoint file
available.
Preview generation requires the bundled renderer; it is optional for someone
cloning the repository. The bundled presentation validators can also inspect
Python-generated decks separately. Previews do not verify behavior in PowerPoint.
