"""Generate the presentation from the storyboard and image assets.

PowerPoint generation needs only python-pptx. Optional PNG previews use the
bundled presentation renderer supplied by ChatGPT or Codex.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


SLIDES_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SLIDES_DIR.parent.parent
SLIDE_IDS = [f"M{number}" for number in range(1, 10)] + ["A1"]
FONT = "Arial"
BACKGROUND = "FFFFFF"
INK = "163846"
TEAL = "167F83"
SECONDARY = "4B626B"
PALE = "EAF4F3"
RULE = "D8E5E5"


def pixels(value):
    """Convert the original generator's 96-DPI coordinates to slide units."""
    return Inches(value / 96)


def font_size(value):
    return Pt(value * 72 / 96)


def style_font(font, size, *, color=INK, bold=False, underline=False):
    font.name = FONT
    font.size = font_size(size)
    font.color.rgb = RGBColor.from_string(color)
    font.bold = bold
    font.underline = underline


def text(slide, name, value, x, y, width, height, size, *, color=INK,
         bold=False, alignment=PP_ALIGN.LEFT, underline=False):
    shape = slide.shapes.add_textbox(
        pixels(x), pixels(y), pixels(width), pixels(height)
    )
    shape.name = name
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.vertical_anchor = MSO_ANCHOR.TOP
    paragraphs = value.split("\n") if isinstance(value, str) else [value]
    for index, content in enumerate(paragraphs):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = alignment
        paragraph.space_before = paragraph.space_after = Pt(0)
        style_font(paragraph.font, size, color=color, bold=bold, underline=underline)
        runs = [content] if isinstance(content, str) else content
        for item in runs:
            run = paragraph.add_run()
            run.text = item if isinstance(item, str) else item[0]
            style_font(run.font, size, color=color,
                       bold=bold if isinstance(item, str) else item[1],
                       underline=underline)
    return shape


def notes_for(storyboard, slide_id):
    start = re.search(rf"^### {re.escape(slide_id)} ", storyboard, re.MULTILINE)
    if start is None:
        raise ValueError(f"Storyboard section missing for {slide_id}.")
    section = storyboard[start.start():]
    section = re.split(r"\n### ", section, maxsplit=1)[0]
    notes = re.search(
        r"\*\*Speaker notes[^\n]*\*\*\n\n([\s\S]*?)\n\n\*\*(?:Transition|Close)",
        section,
    )
    if notes is None:
        raise ValueError(f"Speaker notes missing for {slide_id}.")
    return notes[1].replace("“", "").replace("”", "").replace("\n", " ")


def new_slide(presentation, storyboard, slide_id, title=None, *, two_lines=False):
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string(BACKGROUND)
    if title is not None:
        text(slide, f"{slide_id} title", title, 72, 48, 1136,
             112 if two_lines else 66, 46, bold=True)
    if slide_id != "M1":
        text(slide, "Slide ID", slide_id, 1166, 675, 42, 22, 16,
             color=SECONDARY, alignment=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = notes_for(storyboard, slide_id)
    return slide


def footnote(slide, value):
    text(slide, "Scope", value, 72, 674, 1070, 24, 18, color=SECONDARY)


def rect(slide, name, x, y, width, height, fill):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, pixels(x), pixels(y), pixels(width), pixels(height)
    )
    shape.name = name
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.fill.background()
    return shape


def picture(slide, path, alt, x, y, width, height):
    with Image.open(path) as image:
        image_width, image_height = image.size
    scale = min(width / image_width, height / image_height)
    fitted_width, fitted_height = image_width * scale, image_height * scale
    shape = slide.shapes.add_picture(
        str(path), pixels(x + (width - fitted_width) / 2),
        pixels(y + (height - fitted_height) / 2),
        width=pixels(fitted_width), height=pixels(fitted_height),
    )
    shape.name = path.stem
    shape._element.nvPicPr.cNvPr.set("descr", alt)
    return shape


def cell_borders(cell):
    # python-pptx does not expose table-cell borders through its public API.
    properties = cell._tc.get_or_add_tcPr()
    for edge in ("lnL", "lnR", "lnT", "lnB"):
        line = OxmlElement(f"a:{edge}")
        line.set("w", str(font_size(1)))
        fill = OxmlElement("a:solidFill")
        color = OxmlElement("a:srgbClr")
        color.set("val", RULE)
        fill.append(color)
        line.append(fill)
        dash = OxmlElement("a:prstDash")
        dash.set("val", "solid")
        line.append(dash)
        properties.append(line)


def native_table(slide, values, y, widths, row_height=66, *, result_column=False):
    table = slide.shapes.add_table(
        len(values), len(widths), pixels(72), pixels(y), pixels(sum(widths)),
        pixels(48 + row_height * (len(values) - 1)),
    ).table
    table.first_row = False
    table.horz_banding = False
    for column, width in zip(table.columns, widths):
        column.width = pixels(width)
    for row_index, row_values in enumerate(values):
        table.rows[row_index].height = pixels(48 if row_index == 0 else row_height)
        for column_index, value in enumerate(row_values):
            cell = table.cell(row_index, column_index)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(
                PALE if row_index == 0 else BACKGROUND
            )
            cell.margin_left = cell.margin_right = pixels(18)
            cell.margin_top = cell.margin_bottom = pixels(10)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.text_frame.auto_size = MSO_AUTO_SIZE.NONE
            cell.text_frame.word_wrap = True
            is_result = result_column and row_index > 0 and column_index == 1
            size = 23 if row_index == 0 else 32 if is_result else 26
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.space_before = paragraph.space_after = Pt(0)
            style_font(paragraph.font, size, bold=row_index == 0 or is_result,
                       color=TEAL if is_result else INK)
            for run in paragraph.runs:
                style_font(run.font, size, bold=row_index == 0 or is_result,
                           color=TEAL if is_result else INK)
            cell_borders(cell)
    return table


def survey_bullets(slide):
    shape = text(slide, "Survey and project data", "", 72, 148, 744, 500, 30)
    items = [
        ("MEPS:", " Medical Expenditure Panel Survey"),
        ("Data:", " 2023 Household Component (HC-251)"),
        ("Sample:", " 14,768 adults"),
        ("Target:", " Annual out-of-pocket healthcare costs"),
        ("Features:", " 26 inputs covering demographics, insurance, and health"),
        ("Survey weights:", " How many people each sampled person represents. "
         "This sample represents approximately 260 million U.S. adults."),
    ]
    for index, (label, value) in enumerate(items):
        paragraph = (shape.text_frame.paragraphs[0] if index == 0
                     else shape.text_frame.add_paragraph())
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(18)
        style_font(paragraph.font, 30)
        # Native bullets keep wrapped lines aligned with the text.
        properties = paragraph._p.get_or_add_pPr()
        properties.set("marL", str(Pt(18)))
        properties.set("indent", str(-Pt(12)))
        bullet_font = OxmlElement("a:buFont")
        bullet_font.set("typeface", FONT)
        bullet = OxmlElement("a:buChar")
        bullet.set("char", "•")
        properties.append(bullet_font)
        properties.append(bullet)
        for content, bold in ((label, True), (value, False)):
            run = paragraph.add_run()
            run.text = content
            style_font(run.font, 30, bold=bold)


def native_chart(slide, categories, values, name, x, y, width, height, *,
                 horizontal=False, color=INK, gap_width=140):
    chart_data = CategoryChartData()
    chart_data.categories = categories
    chart_data.add_series(name, values, number_format="0.0%")
    chart_type = (XL_CHART_TYPE.BAR_CLUSTERED if horizontal
                  else XL_CHART_TYPE.COLUMN_CLUSTERED)
    shape = slide.shapes.add_chart(
        chart_type, pixels(x), pixels(y), pixels(width), pixels(height), chart_data
    )
    shape.name = name
    chart = shape.chart
    # Normalize python-pptx's signed axis IDs for renderers that require unsigned IDs.
    for axis_id in chart._chartSpace.xpath(".//c:axId | .//c:crossAx"):
        axis_id.set("val", str(int(axis_id.get("val")) % (2 ** 32)))
    style_font(chart.font, 24)
    chart.has_legend = False
    chart.has_title = False
    plot = chart.plots[0]
    plot.gap_width = gap_width
    plot.has_data_labels = True
    plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    plot.data_labels.show_value = True
    plot.data_labels.show_category_name = False
    plot.data_labels.show_series_name = False
    plot.data_labels.show_legend_key = False
    plot.data_labels.number_format = "0.0%"
    plot.data_labels.number_format_is_linked = False
    style_font(plot.data_labels.font, 27 if horizontal else 28, bold=True)
    series = chart.series[0]
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = RGBColor.from_string(color)
    series.format.line.fill.background()
    category_axis, value_axis = chart.category_axis, chart.value_axis
    style_font(category_axis.tick_labels.font, 25 if horizontal else 24)
    style_font(value_axis.tick_labels.font, 22 if horizontal else 20)
    category_axis.has_major_gridlines = False
    value_axis.minimum_scale = 0
    value_axis.maximum_scale = 0.6 if horizontal else 1
    value_axis.major_unit = 0.2 if horizontal else 0.25
    value_axis.tick_labels.number_format = "0%"
    value_axis.tick_labels.number_format_is_linked = False
    value_axis.has_major_gridlines = True
    value_axis.major_gridlines.format.line.color.rgb = RGBColor.from_string(RULE)
    value_axis.major_gridlines.format.line.width = font_size(1)
    for axis in (category_axis, value_axis):
        axis.major_tick_mark = XL_TICK_MARK.NONE
        axis.minor_tick_mark = XL_TICK_MARK.NONE
        axis.format.line.color.rgb = RGBColor.from_string(RULE)
        axis.format.line.width = font_size(1)
    if horizontal:
        # Keep the first category at the top, matching the original slide.
        category_axis.reverse_order = True
        series.points[0].format.fill.solid()
        series.points[0].format.fill.fore_color.rgb = RGBColor.from_string("A7B6BC")
    return chart


def build_presentation():
    storyboard = (SLIDES_DIR / "storyboard.md").read_text(encoding="utf-8")
    presentation = Presentation()
    presentation.slide_width = pixels(1280)
    presentation.slide_height = pixels(720)

    cover = new_slide(presentation, storyboard, "M1")
    picture(cover, WORKSPACE_DIR / "assets/header.png",
            "Medical Cost Planner project header.", 72, 72, 1136, 422)
    text(cover, "Subtitle", "Predicting Out-of-Pocket Healthcare Costs with Machine Learning",
         72, 522, 1136, 56, 34, bold=True)
    text(cover, "Presenter", "Jens Bender", 72, 594, 400, 36, 28)
    text(cover, "Event details", "[Event / setting] · [Date]", 72, 632, 1000, 32, 24,
         color=SECONDARY)

    problem = new_slide(presentation, storyboard, "M2",
                        "How much should I set aside for healthcare?")
    text(problem, "Challenge label", "The challenge", 72, 520, 532, 36, 30, bold=True)
    text(problem, "Budgeting challenge", "Planning next year's out-of-pocket\n"
         "costs and HSA/FSA contributions\nis difficult.", 72, 560, 532, 108, 30)
    text(problem, "Aim label", "Our aim", 676, 520, 532, 36, 30, bold=True)
    text(problem, "Intended estimate", "A useful ballpark estimate from\n"
         "questions people can answer\nfrom memory.", 676, 560, 532, 108, 30)
    picture(problem, SLIDES_DIR / "assets/budget-planning.png",
            "Illustration of an adult considering a budget with a planner and calculator.",
            72, 120, 1136, 379)

    data = new_slide(presentation, storyboard, "M3",
                     "MEPS links accessible inputs to observed spending")
    survey_bullets(data)
    picture(data, SLIDES_DIR / "assets/household-survey.png",
            "Illustration of a household survey interview: a respondent shows an empty "
            "wallet beside bills, while an interviewer listens with a laptop.",
            868, 148, 340, 453.333)
    appendix_link = text(data, "MEPS appendix link", "Appendix: MEPS overview",
                         72, 674, 800, 24, 18, color=SECONDARY, underline=True)

    distribution = new_slide(
        presentation, storyboard, "M4", "Most out-of-pocket spending comes\n"
        "from a small share of adults", two_lines=True,
    )
    text(distribution, "Chart explanation", "Share of total out-of-pocket spending",
         72, 190, 760, 38, 28, bold=True)
    native_chart(distribution, ["Lower-spending 80%", "Highest-spending 20%"],
                 [0.207, 0.793], "Share of spending", 72, 240, 740, 342)
    text(distribution, "Zero spending", "22.3%", 867, 246, 330, 76, 56, bold=True)
    text(distribution, "Zero spending explanation", "of adults have zero\n"
         "out-of-pocket spending", 867, 328, 341, 90, 28)
    text(distribution, "Metric implication", "Evaluate typical error alongside large "
         "errors and uncertainty", 72, 610, 1136, 40, 28, bold=True)
    footnote(distribution, "Survey-weighted MEPS 2023 estimates. Source: EDA notebook.")

    selection = new_slide(presentation, storyboard, "M5",
                          "Model selection: median error was not enough")
    model_table = native_table(selection, [
        ["Tuned point-estimate model", "Validation MdAE", "Validation MAE"],
        ["Elastic Net", "$159", "$1,051"],
        ["Random Forest", "$228", "$964"],
        ["XGBoost", "$242", "$954"],
    ], 142, [520, 308, 308])
    for row, column in ((1, 1), (3, 2)):
        for run in model_table.cell(row, column).text_frame.paragraphs[0].runs:
            run.font.bold = True
    text(selection, "Prediction compression", "Elastic Net's largest validation "
         "prediction was only about $217", 72, 424, 1136, 42, 29, bold=True)
    text(selection, "Tradeoff", "Good typical error, but limited separation across cost profiles.\n"
         "Residual and subgroup checks motivated richer budgeting outputs.",
         72, 480, 1136, 86, 28)
    text(selection, "Quantile decision", "Next step: XGBoost quantile regression",
         72, 602, 1136, 40, 30, bold=True)
    footnote(selection, "MdAE: median absolute error; MAE: mean absolute error. "
             "Validation; 2023 USD.")

    quantiles = new_slide(presentation, storyboard, "M6", "Quantile regression turns "
                          "predictions\ninto budgeting ranges", two_lines=True)
    text(quantiles, "Objective", "XGBoost quantile objective · survey weights · "
         "log-transformed costs", 72, 190, 1136, 38, 27)
    text(quantiles, "Plan-around label", "Plan-around estimate", 238, 260, 320, 38, 27,
         bold=True, alignment=PP_ALIGN.CENTER)
    text(quantiles, "Safety label", "Safety cushion", 834, 260, 320, 38, 27,
         bold=True, alignment=PP_ALIGN.CENTER)
    # This axis is conceptual; positions do not encode a person's dollar values.
    rect(quantiles, "Cost axis", 122, 334, 1040, 2, INK)
    rect(quantiles, "Typical range q25 to q75", 248, 318, 398, 34, PALE)
    for x, label in ((248, "q25"), (398, "q50"), (646, "q75"), (994, "q90")):
        rect(quantiles, f"Quantile marker {label}", x, 311, 3, 49, INK)
        text(quantiles, label, label, x - 40, 373, 83, 38, 27, alignment=PP_ALIGN.CENTER)
    text(quantiles, "Range label", "Typical range", 275, 427, 340, 38, 28,
         bold=True, alignment=PP_ALIGN.CENTER)
    text(quantiles, "Coverage target", "Targets 50% coverage", 258, 466, 380, 38, 26,
         alignment=PP_ALIGN.CENTER)
    text(quantiles, "Upper coverage target", "Targets 90% below q90", 821, 427, 387, 38, 26,
         alignment=PP_ALIGN.CENTER)
    text(quantiles, "Direction", "Higher annual spending →", 856, 496, 352, 38, 24,
         alignment=PP_ALIGN.RIGHT)
    text(quantiles, "Evaluation principle", "Useful ranges need both coverage and "
         "reasonable width", 72, 566, 1136, 42, 30, bold=True)
    text(quantiles, "Not a cap", "The safety cushion is an upper planning reference, "
         "not a spending cap.", 72, 611, 1136, 36, 27)
    footnote(quantiles, "Conceptual diagram, not a prediction for an individual. "
             "q50 is the predicted median.")

    results = new_slide(presentation, storyboard, "M7")
    text(results, "M7 title", [("Final model audit:", True),
         " clearest gains in ranges and q90"], 72, 48, 1136, 66, 46)
    native_table(results, [
        ["Held-out test metric", "Result", "Release gate"],
        ["Plan-around estimate (q50): MdAE", "$240", "< $500"],
        ["Typical range (q25–q75): coverage", "47.3%", "45%–55%"],
        ["Safety cushion (q90): coverage", "91.0%", "85%–95%"],
    ], 142, [520, 210, 406], row_height=68, result_column=True)
    text(results, "Baseline comparison", "Compared with the population baseline",
         72, 419, 1000, 34, 26, bold=True)
    text(results, "Plan-around comparison", "Plan-around estimate (q50)",
         72, 464, 370, 38, 26)
    text(results, "Median error improvement", [("≈$8", True),
         " lower MdAE (improvement uncertain)"], 450, 464, 758, 38, 26)
    text(results, "Typical range comparison", "Typical range (q25–q75)",
         72, 513, 370, 38, 26)
    text(results, "Interval score gain", [("11.2%", True), " interval skill score"],
         450, 513, 758, 38, 26)
    text(results, "Safety cushion comparison", "Safety cushion (q90)",
         72, 562, 370, 38, 26)
    text(results, "q90 loss gain", [("15.6%", True), " quantile skill score"],
         450, 562, 758, 38, 26)
    footnote(results, "Survey-weighted test metrics. Dollar amounts in 2023 USD.")

    audit = new_slide(presentation, storyboard, "M8", "Final model test audit: overall "
                      "coverage\nhides subgroup gaps", two_lines=True)
    text(audit, "Chart metric", "Typical-range coverage (q25–q75)",
         72, 190, 780, 38, 28, bold=True)
    native_chart(audit, ["Target", "Overall", "Low income", "Poor mental health"],
                 [0.5, 0.473, 0.392, 0.301], "Coverage", 72, 242, 740, 326,
                 horizontal=True, color=TEAL, gap_width=85)
    text(audit, "Rare events label", "Rare expensive years", 860, 246, 348, 38, 28, bold=True)
    text(audit, "Rare events detail", "Remain difficult\nto anticipate", 860, 287, 348, 80, 28)
    text(audit, "Timing label", "Future-year use", 860, 392, 348, 38, 28, bold=True)
    text(audit, "Timing detail", "Still needs feature-timing\nand later-year validation",
         860, 433, 348, 116, 28)
    text(audit, "Safeguard limitation", "Scope wording and planning notices communicate limits;\n"
         "they do not fix calibration.", 72, 588, 1136, 74, 28, bold=True)
    footnote(audit, "Survey-weighted test point estimates; subgroup uncertainty matters. "
             "Source: modeling notebook.")

    next_steps = new_slide(presentation, storyboard, "M9", "Model evaluation is complete;\n"
                           "app development comes next", two_lines=True)
    text(next_steps, "Implemented", "Implemented", 72, 190, 700, 38, 30, bold=True)
    text(next_steps, "Planned", "Planned", 895, 190, 313, 38, 30, bold=True)
    text(next_steps, "Training heading", "Training artifacts", 72, 265, 330, 38, 29, bold=True)
    text(next_steps, "Training tools", "DVC stages\nMLflow experiments\nEvaluation artifacts",
         72, 310, 330, 127, 27)
    text(next_steps, "Training to inference", "→", 418, 295, 64, 58, 42)
    text(next_steps, "Shared heading", "Shared inference", 510, 265, 320, 38, 29, bold=True)
    text(next_steps, "Inference modules", "Prediction and SHAP\nReusable modules\nUnit tests",
         510, 310, 320, 127, 27)
    text(next_steps, "Inference to app", "→", 822, 295, 64, 58, 42)
    text(next_steps, "App heading", "FastAPI / Gradio", 895, 265, 313, 38, 29, bold=True)
    text(next_steps, "Remaining app work", "Integration and latency\nUser evaluation\n"
         "Aggregate monitoring", 895, 310, 313, 127, 27)
    text(next_steps, "Validation heading", "Next validation priority", 72, 478, 1136, 38, 28,
         bold=True)
    text(next_steps, "Validation work", "Check feature timing and performance on a later "
         "survey year", 72, 520, 1136, 38, 28)
    text(next_steps, "Closing lesson", "Evaluate the outputs needed for the user decision",
         72, 605, 1136, 44, 32, bold=True)

    meps_overview = new_slide(presentation, storyboard, "A1")
    picture(meps_overview, WORKSPACE_DIR / "assets/infographic_meps_data.jpg",
            "MEPS household, provider, and employer survey components and the 2023 "
            "data used in this project.", 20, 0, 1240, 660)
    return_link = text(meps_overview, "Return to data slide", "Back to MEPS data",
                       72, 674, 800, 24, 18, color=SECONDARY, underline=True)
    appendix_link.click_action.target_slide = meps_overview
    return_link.click_action.target_slide = data
    return presentation


def render_previews(pptx_path):
    """Use the bundled renderer when the caller requests PNG previews."""
    required = ("SKILL_DIR", "RUNTIME_NODE", "RUNTIME_NODE_MODULES")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"PNG previews require: {', '.join(missing)}.")
    renderer = Path(os.environ["SKILL_DIR"]) / "container_tools/render_presentation.mjs"
    result = subprocess.run(
        [os.environ["RUNTIME_NODE"], str(renderer), "--input", str(pptx_path),
         "--output_dir", str(pptx_path.parent), "--scale", "1.5"],
        check=False, capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(f"PNG rendering failed:\n{result.stderr or result.stdout}")
    paths = json.loads(result.stdout)["paths"]
    if len(paths) != len(SLIDE_IDS):
        raise ValueError(f"Expected {len(SLIDE_IDS)} previews, got {len(paths)}.")
    for slide_id, path in zip(SLIDE_IDS, paths):
        Path(path).rename(pptx_path.parent / f"{slide_id}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", help="New revision name, for example v4")
    parser.add_argument("--render", action="store_true",
                        help="Also create PNG previews using the bundled renderer")
    args = parser.parse_args()
    if re.fullmatch(r"[a-zA-Z0-9_-]+", args.revision) is None:
        parser.error("Revision names can contain only letters, numbers, underscores and hyphens.")
    output_dir = SLIDES_DIR / "exports" / args.revision
    if output_dir.exists():
        parser.error(f"Revision already exists: {output_dir}. Choose a new name.")
    if args.render:
        missing = [name for name in ("SKILL_DIR", "RUNTIME_NODE", "RUNTIME_NODE_MODULES")
                   if not os.environ.get(name)]
        if missing:
            parser.error(f"PNG previews require: {', '.join(missing)}.")
    presentation = build_presentation()
    output_dir.mkdir(parents=True)
    pptx_path = output_dir / f"medical-cost-planner-{args.revision}.pptx"
    presentation.save(pptx_path)
    print(f"PowerPoint: {pptx_path}")
    if args.render:
        render_previews(pptx_path)
        print(f"PNG previews: {output_dir}")


if __name__ == "__main__":
    main()
