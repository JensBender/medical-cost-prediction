# Presentation files

`storyboard.md` holds the slide content, speaker notes, timings, and sources.
`generate_slides.py` implements the slide text and layout and creates an editable
PowerPoint with optional PNG previews. Update both files when changing slide
content; speaker-note changes only need the storyboard.

The deck contains 12 main slides and 27 appendix slides. The cover is unnumbered;
the remaining main slides display 2–12, and appendix slides display A1–A27.
Main slides link to related appendix topics, which link back to the referring
main slide. The storyboard lists the topics and presentation order.

Most slides use a white background, dark blue text, teal highlights, and muted
gray supporting information. The preprocessing slide uses the diagram's dark
navy background.

## Generate a revision

Install the training dependencies, which include `python-pptx`, and run the
Python generator from the repository root:

```powershell
.\.venv-train\Scripts\python -m pip install -r requirements-train.txt
.\.venv-train\Scripts\python docs/slides/generate_slides.py v4
```

The script writes `medical-cost-planner-v4.pptx` under `exports/v4/`. It uses
the storyboard and tracked images in `assets/`, `../../assets/`, and
`../../figures/`. Generating the PowerPoint does not load model artifacts or
survey data and does not require the ChatGPT or Codex runtime.

The preprocessing slide uses the tracked `assets/preprocessing-pipeline.png`.
Its editable source is `../../assets/preprocessing_pipeline.svg`. When changing
the SVG, regenerate the PNG and include both files in the same commit. The slide
generator embeds the PNG directly.

Choose a new revision name for each run. The Python generator refuses to overwrite
an existing revision and keeps previous exports. Generated files remain ignored
by Git. When revising slides in the ChatGPT desktop app or Codex, keep the newest
export and the previous complete export after validation and preview review.
Remove older draft exports and obsolete build files; preserve named presentation
milestones. The Python generator does not perform this cleanup automatically.
To recreate an older version, restore the generator, storyboard, and images
from the same Git commit.

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

The renderer reads the generated PowerPoint and writes `1.png` through `12.png`,
`A1.png` through `A27.png` beside it. Rendering failures leave the PowerPoint file
available.
Preview generation requires the bundled renderer; it is optional for someone
cloning the repository. The bundled presentation validators can also inspect
Python-generated decks separately. Previews do not verify behavior in PowerPoint.
