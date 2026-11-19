# Run from the project root: .\.venv-app\Scripts\python.exe -m app.ui
# For reload mode, run .\.venv-app\Scripts\gradio.exe app/ui.py --demo-name app

import gradio as gr


# Opt out of sending Gradio usage telemetry.
with gr.Blocks(title="Medical Cost Planner", analytics_enabled=False) as app:
    gr.Markdown("# Medical Cost Planner")

    with gr.Row():
        gr.Number(
            label="Year of birth",
            placeholder="e.g. 1985",
            precision=0,
            min_width=280,
            interactive=True,
        )
        gr.Radio(
            ["Male", "Female"],
            label="Sex",
            value=None,
            min_width=280,
            interactive=True,
        )

    with gr.Row():
        gr.Dropdown(
            [
                "Private insurance (with or without public coverage)",
                "Public insurance only",
                "No health insurance",
            ],
            label="Health insurance",
            value=None,
            min_width=280,
            interactive=True,
        )
        gr.Dropdown(
            [1, 2, 3, 4, 5, 6, 7, "8+"],
            label="How many people (including yourself) live in your home who are related to you by blood, marriage, or adoption?",
            value=None,
            min_width=280,
            interactive=True,
        )

    gr.Dropdown(
        [
            "No Degree", "GED", "High School Diploma", "Associate's Degree",
            "Bachelor's Degree", "Master's Degree", "Doctorate or Professional (MD, JD, etc.)", "Other",
        ],
        label="What is the highest degree you have obtained?",
        value=None,
        interactive=True,
    )


if __name__ == "__main__":
    # Keep Gradio's traffic dashboard off under our aggregate-only monitoring policy.
    app.launch(server_name="127.0.0.1", share=False, enable_monitoring=False)
