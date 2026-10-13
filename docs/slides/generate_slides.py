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
from pptx.enum.chart import (
    XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_MARK,
)
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


SLIDES_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SLIDES_DIR.parent.parent
SLIDE_IDS = [str(number) for number in range(1, 11)] + [f"A{number}" for number in range(1, 13)]
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
    paragraphs = re.split(r"\n\s*\n", notes[1].replace("“", "").replace("”", ""))
    return "\n\n".join(" ".join(paragraph.split()) for paragraph in paragraphs)


def new_slide(presentation, storyboard, slide_id, title=None, *, two_lines=False):
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string(BACKGROUND)
    if title is not None:
        text(slide, f"{slide_id} title", title, 72, 48, 1136,
             112 if two_lines else 66, 44 if slide_id.isdigit() else 46, bold=True)
    if slide_id != "1":
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
        ("Sample:", " 14,768 representing ~260 million U.S. adults"),
        ("Target:", " Annual out-of-pocket healthcare costs"),
        ("Features:", " 26 inputs across demographics, socioeconomics, "
         "health profile, chronic conditions, and limitations"),
        ("Survey weights:", " Used in EDA, model training, and evaluation "
         "to reflect the population"),
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

    cover = new_slide(presentation, storyboard, "1")
    picture(cover, WORKSPACE_DIR / "assets/header.png",
            "Medical Cost Planner project header.", 72, 72, 1136, 422)
    text(cover, "Subtitle", "Predicting Out-of-Pocket Healthcare Costs with Machine Learning",
         72, 522, 1136, 56, 34, bold=True)
    text(cover, "Presenter", "Jens Bender", 72, 594, 400, 36, 28)
    text(cover, "Event details", "[Event / setting] · [Date]", 72, 632, 1000, 32, 24,
         color=SECONDARY)

    problem = new_slide(presentation, storyboard, "2",
                        "How much should I set aside for healthcare?")
    text(problem, "Challenge label", "The challenge", 72, 520, 556, 36, 30, bold=True)
    text(problem, "Budgeting challenge", "Difficult to plan next year's out-of-pocket\n"
         "costs and HSA/FSA contributions.", 72, 560, 556, 72, 30)
    text(problem, "Aim label", "The aim", 652, 520, 556, 36, 30, bold=True)
    text(problem, "Intended estimate", "A useful ballpark estimate from questions "
         "people can answer from memory.", 652, 560, 556, 72, 30)
    picture(problem, SLIDES_DIR / "assets/budget-planning.png",
            "Illustration of an adult considering a budget with a planner and calculator.",
            72, 120, 1136, 379)
    healthcare_appendix_link = text(
        problem, "Healthcare costs appendix link", "Appendix: U.S. healthcare cost explainer",
        72, 674, 800, 24, 18, color=SECONDARY, underline=True,
    )

    data = new_slide(presentation, storyboard, "3",
                     "MEPS links accessible inputs to observed spending")
    survey_bullets(data)
    picture(data, SLIDES_DIR / "assets/household-survey.png",
            "Illustration of a household survey interview: a respondent shows an empty "
            "wallet beside bills, while an interviewer listens with a laptop.",
            868, 148, 340, 453.333)
    appendix_link = text(data, "MEPS appendix link", "Appendix: MEPS overview",
                         72, 674, 800, 24, 18, color=SECONDARY, underline=True)

    distribution = new_slide(presentation, storyboard, "4")
    text(distribution, "4 title", "Out-of-pocket costs: 20% of adults account for ~80%",
         72, 48, 1136, 66, 44, bold=True)
    lorenz_path = WORKSPACE_DIR / "figures/eda/lorenz_curve.png"
    with Image.open(lorenz_path) as lorenz_image:
        image_width, image_height = lorenz_image.size
    # Crop the embedded title while preserving the axes and annotations.
    title_crop = 70
    chart_height = 580
    chart_width = image_width / (image_height - title_crop) * chart_height
    lorenz = distribution.shapes.add_picture(
        str(lorenz_path), pixels((1280 - chart_width) / 2 + 64), pixels(120),
        pixels(chart_width), pixels(chart_height),
    )
    lorenz.crop_top = title_crop / image_height
    lorenz.name = "Survey-weighted Lorenz curve"
    lorenz._element._nvXxPr.cNvPr.set(
        "descr", "Lorenz curve of annual out-of-pocket costs: 22.3% of adults "
        "have zero costs; the highest-spending 20% account for 79.3% of costs "
        "and the highest-spending 1% for 20.6%.",
    )
    metrics = text(distribution, "Evaluation metrics",
                   "Primary metric\nMdAE\nSupporting\nMAE, R²",
                   75.8, 126.9, 184, 170, 24)
    for paragraph, heading, spacing in zip(
        metrics.text_frame.paragraphs, (True, False, True, False), (4, 18, 4, 0),
    ):
        paragraph.space_after = font_size(spacing)
        for run in paragraph.runs:
            style_font(run.font, 24, bold=heading)
    text(distribution, "EDA appendix label", "Appendix:", 72, 588, 164, 22, 18,
         color=SECONDARY)
    outlier_appendix_link = text(
        distribution, "Outlier appendix link", "Outlier analysis",
        72, 614, 164, 24, 18, color=SECONDARY, underline=True,
    )
    distributions_appendix_link = text(
        distribution, "Distributions appendix link", "Distributions",
        72, 640, 164, 24, 18, color=SECONDARY, underline=True,
    )
    relationships_appendix_link = text(
        distribution, "Feature-target relationships appendix link",
        "Feature–target relations",
        72, 666, 240, 24, 18, color=SECONDARY, underline=True,
    )

    preprocessing = new_slide(presentation, storyboard, "5")
    preprocessing.background.fill.fore_color.rgb = RGBColor.from_string("020617")
    for shape in preprocessing.shapes:
        if shape.name == "Slide ID":
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.color.rgb = RGBColor.from_string("94A3B8")
    picture(preprocessing, SLIDES_DIR / "assets/preprocessing-pipeline.png",
            "Preprocessing pipeline fitted on training data: categorical input "
            "standardization, missing-value validation and imputation, health-burden "
            "feature derivation, scaling, and encoding.", 0, 0, 1280, 720)
    text(preprocessing, "Preprocessing appendix label", "Appendix:",
         20, 648, 200, 22, 18, color="94A3B8")
    preparation_appendix_link = text(
        preprocessing, "Data preparation appendix link", "Data preparation",
        20, 674, 200, 24, 18, color="94A3B8", underline=True,
    )

    selection = new_slide(presentation, storyboard, "6",
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
    text(selection, "Metric definitions", "MdAE: median absolute error; "
         "MAE: mean absolute error. Validation; 2023 USD.",
         72, 648, 1136, 24, 18, color=SECONDARY)
    baseline_appendix_link = text(
        selection, "Baseline models appendix link", "Appendix: Baseline models",
        72, 674, 800, 24, 18, color=SECONDARY, underline=True,
    )

    quantiles = new_slide(presentation, storyboard, "7", "Quantile regression turns "
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

    results = new_slide(presentation, storyboard, "8")
    text(results, "8 title", [("Final model audit:", True),
         " clearest gains in ranges and q90"], 72, 48, 1136, 66, 44)
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

    audit = new_slide(presentation, storyboard, "9", "Final model test audit: overall "
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

    next_steps = new_slide(presentation, storyboard, "10", "Model evaluation is complete;\n"
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

    healthcare_overview = new_slide(presentation, storyboard, "A1")
    picture(healthcare_overview, WORKSPACE_DIR / "assets/infographic_healthcare_costs.png",
            "Overview of U.S. healthcare payments, out-of-pocket costs, insurance "
            "terms, HSA/FSA planning, and the project target.", 20, 0, 1240, 660)
    healthcare_return_link = text(
        healthcare_overview, "Return to problem statement", "Back",
        72, 674, 800, 24, 18, color=SECONDARY, underline=True,
    )
    healthcare_appendix_link.click_action.target_slide = healthcare_overview
    healthcare_return_link.click_action.target_slide = problem

    meps_overview = new_slide(presentation, storyboard, "A2")
    picture(meps_overview, WORKSPACE_DIR / "assets/infographic_meps_data.jpg",
            "MEPS household, provider, and employer survey components and the 2023 "
            "data used in this project.", 20, 0, 1240, 660)
    return_link = text(meps_overview, "Return to data slide", "Back",
                       72, 674, 800, 24, 18, color=SECONDARY, underline=True)
    appendix_link.click_action.target_slide = meps_overview
    return_link.click_action.target_slide = data

    outliers = new_slide(
        presentation, storyboard, "A3",
        "Outliers: greater health burden and higher cost risk",
    )
    text(outliers, "Outlier detection",
         [("Detection: ", True), ("Isolation Forest (flagging 5%)", False)],
         72, 142, 1136, 42, 30)
    profile_table = native_table(outliers, [
        ["Profiling", "Inliers", "Outliers"],
        ["In top 1% of spenders", "0.9%", "3.5%"],
        ["Age (median)", "47", "66"],
        ["Chronic conditions (median)", "1", "4"],
        ["Functional limitations (median)", "0", "3"],
        ["Private insurance", "68%", "32%"],
        ["Walking limitations", "9%", "75%"],
        ["Cognitive limitations", "3%", "61%"],
    ], 206, [680, 228, 228], row_height=42)
    for row_index in range(len(profile_table.rows)):
        for column_index in range(len(profile_table.columns)):
            cell = profile_table.cell(row_index, column_index)
            cell.margin_top = cell.margin_bottom = pixels(6)
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if column_index == 0 else PP_ALIGN.CENTER
            style_font(paragraph.font, 26, bold=row_index == 0)
            for run in paragraph.runs:
                style_font(run.font, 26, bold=row_index == 0)
    text(outliers, "Outlier treatment",
         [("Treatment: ", True),
          ("Retained all outliers as plausible, valuable training examples.", False)],
         72, 583.56, 1136, 42, 30)
    outlier_return_link = text(
        outliers, "Return to cost distribution", "Back",
        72, 674, 800, 24, 18, color=SECONDARY, underline=True,
    )
    outlier_appendix_link.click_action.target_slide = outliers
    outlier_return_link.click_action.target_slide = distribution
    for slide_id, filename, description in [
        ("A4", "numerical_distributions.png", "Population distributions of age, "
         "family size, physical health, and mental health."),
        ("A5", "categorical_distributions.png", "Population distributions of "
         "region, marital status, employment, insurance, education, and family income."),
        ("A6", "binary_distributions.png", "Population distributions of binary "
         "features, including chronic conditions, limitations, and access to care."),
    ]:
        feature_distributions = new_slide(presentation, storyboard, slide_id)
        picture(feature_distributions, WORKSPACE_DIR / "figures/eda" / filename,
                description, 0, 0, 1280, 720)
        distribution_return_link = text(
            feature_distributions, "Return to cost distribution", "Back",
            20, 674, 100, 24, 18, color=SECONDARY, underline=True,
        )
        distribution_return_link.click_action.target_slide = distribution
        if slide_id == "A4":
            distributions_appendix_link.click_action.target_slide = feature_distributions
    for slide_id, filename, description in [
        ("A7", "correlation_heatmap.png", "Population Spearman rank correlations "
         "between out-of-pocket costs and numerical, ordinal, and binary features."),
        ("A8", "numerical_feature_target_relationships.png", "Relationships between "
         "out-of-pocket costs and age, family size, physical health, and mental health."),
        ("A9", "categorical_feature_target_relationships.png", "Out-of-pocket cost "
         "distributions by region, marital status, employment, insurance, education, "
         "and family income."),
        ("A10", "binary_feature_target_relationships.png", "Out-of-pocket cost "
         "distributions by binary features, including chronic conditions, limitations, "
         "and access to care."),
    ]:
        feature_relationships = new_slide(presentation, storyboard, slide_id)
        picture(feature_relationships, WORKSPACE_DIR / "figures/eda" / filename,
                description, 0, 0, 1280, 720)
        relationships_return_link = text(
            feature_relationships, "Return to cost distribution", "Back",
            20, 674, 100, 24, 18, color=SECONDARY, underline=True,
        )
        relationships_return_link.click_action.target_slide = distribution
        if slide_id == "A7":
            relationships_appendix_link.click_action.target_slide = feature_relationships
    preparation = new_slide(presentation, storyboard, "A11")
    text(preparation, "A11 title", "Data preparation",
         72, 48, 1136, 66, 44, bold=True)
    text(preparation, "Preparation heading", "Prepare MEPS data",
         72, 148, 1136, 42, 30, bold=True)
    text(preparation, "Preparation steps", "Select variables and adults, "
         "standardize missing values, and group sparse categories.",
         72, 195, 1136, 42, 28)
    text(preparation, "Split heading", "Stratified split by out-of-pocket costs",
         72, 283.31, 1136, 42, 30, bold=True)
    text(preparation, "Split rationale", "Keep zero-cost and rare high-cost cases "
         "represented across splits.", 72, 329.31, 1136, 42, 28)
    text(preparation, "Zero-cost group label", "Zero costs",
         72, 391.33, 140, 36, 25, alignment=PP_ALIGN.CENTER)
    text(preparation, "Lower-cost groups", "Lower costs",
         218, 391.33, 210, 36, 25)
    text(preparation, "Extreme-cost groups", "Extreme costs",
         995, 391.33, 213, 36, 25, alignment=PP_ALIGN.RIGHT)
    rect(preparation, "Separate zero-cost group", 72, 435.33, 140, 54, INK)
    group_x = 218
    for index, width in enumerate((290, 240, 180, 126, 80, 44)):
        rect(preparation, f"Positive-cost percentile group {index + 1}",
             group_x, 435.33, width, 54, INK)
        group_x += width + 6
    text(preparation, "Tail grouping", "Finer percentile groups toward the expensive tail",
         218, 503.32, 990, 36, 25)
    text(preparation, "Split allocation",
         "Random split within each group: 80% train, 10% validation, 10% test",
         72, 554.31, 1136, 38, 25)
    preparation_return_link = text(
        preparation, "Return to preprocessing pipeline", "Back",
        72, 674, 800, 24, 18, color=SECONDARY, underline=True,
    )
    preparation_appendix_link.click_action.target_slide = preparation
    preparation_return_link.click_action.target_slide = preprocessing

    baselines = new_slide(presentation, storyboard, "A12")
    text(baselines, "A12 title", "Baseline models",
         72, 48, 1136, 66, 44, bold=True)
    baseline_table = native_table(baselines, [
        ["Model", "MdAE", "Overfitting", "MAE", "R²"],
        ["Elastic Net", "$163", "+6.6%", "$1,044", "-0.12"],
        ["Linear Regression", "$219", "+4.8%", "$998", "-0.06"],
        ["Random Forest", "$232", "+9.6%", "$958", "-0.04"],
        ["Median (benchmark)", "$248", "0.0%", "$1,041", "-0.10"],
        ["Decision Tree", "$271", "+1.5%", "$971", "-0.03"],
        ["XGBoost", "$281", "+98.0%", "$961", "0.00"],
        ["Support Vector Machine", "$291", "+190.7%", "$1,027", "-0.03"],
        ["LLM (benchmark)", "$518", "N/A", "$1,168", "0.04"],
    ], 142, [420, 160, 200, 180, 176], row_height=48)
    for row in baseline_table.rows:
        for cell in row.cells:
            cell.margin_top = cell.margin_bottom = pixels(6)
    for row in (1, 3, 6):
        for run in baseline_table.cell(row, 0).text_frame.paragraphs[0].runs:
            run.font.bold = True
    for row, column in ((1, 1), (3, 3), (5, 2), (8, 4)):
        for run in baseline_table.cell(row, column).text_frame.paragraphs[0].runs:
            run.font.bold = True
    for row in (4, 8):
        for cell in baseline_table.rows[row].cells:
            for run in cell.text_frame.paragraphs[0].runs:
                run.font.italic = True
    text(baselines, "Validation note", "Note: Survey-weighted validation metrics. "
         "Overfitting is the percentage change in MdAE from training to validation.",
         72, 602, 1136, 24, 18, color=SECONDARY)
    baseline_return_link = text(
        baselines, "Return to model selection", "Back",
        72, 674, 800, 24, 18, color=SECONDARY, underline=True,
    )
    baseline_appendix_link.click_action.target_slide = baselines
    baseline_return_link.click_action.target_slide = selection
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
