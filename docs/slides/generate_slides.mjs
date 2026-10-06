// Generate and revise slides through the ChatGPT desktop app or Codex using
// the bundled presentation runtime and validation helpers. Runtime paths come
// from load_workspace_dependencies; this project does not install those
// dependencies or provide a standalone presentation build. To recreate an older
// version, restore the generator, storyboard, and assets from the same Git commit.

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';

// Use the bundled presentation runtime returned by load_workspace_dependencies.
const { RUNTIME_NODE_MODULES, RUNTIME_PYTHON, SKILL_DIR } = process.env;
if (!RUNTIME_NODE_MODULES || !RUNTIME_PYTHON || !SKILL_DIR) {
  throw new Error('Set RUNTIME_NODE_MODULES, RUNTIME_PYTHON, and SKILL_DIR.');
}
const runtimeRequire = createRequire(path.join(RUNTIME_NODE_MODULES, 'package.json'));
const { Presentation, PresentationFile, FileBlob } = await import(
  pathToFileURL(runtimeRequire.resolve('@oai/artifact-tool')).href
);
const { finalizePresentation, resolvePresentationFont, applyPresentationChartFont, makeNativeBulletParagraphs } = await import(
  pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href
);

const slidesDir = path.dirname(fileURLToPath(import.meta.url));
const workspaceDir = path.resolve(slidesDir, '../..');
const revision = process.argv[2];
if (!revision) throw new Error('Pass a new revision name, for example v11.');
if (!/^[a-zA-Z0-9_-]+$/.test(revision)) throw new Error('Invalid revision name.');
const buildDir = path.join(slidesDir, '.build', revision);
const outputDir = path.join(slidesDir, 'exports', `main-${revision}`);
await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const theme = {
  font: resolvePresentationFont(),
  background: '#FFFFFF',
  ink: '#163846',
  teal: '#167F83',
  secondary: '#4B626B',
  pale: '#EAF4F3',
  rule: '#D8E5E5',
};
const fontPolicy = { basis: 'design', families: [theme.font] };
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function text(slide, name, value, x, y, width, height, size, options = {}) {
  const shape = slide.shapes.add({
    geometry: 'textbox', name,
    position: { left: x, top: y, width, height },
    fill: 'none', line: { fill: 'none', width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    typeface: theme.font, fontSize: size, color: theme.ink,
    autoFit: 'none', wrap: 'square', verticalAlignment: 'top',
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
    ...options,
  };
  return shape;
}

const storyboard = await fs.readFile(path.join(slidesDir, 'storyboard.md'), 'utf8');
function notesFor(id) {
  const start = storyboard.indexOf(`### ${id} `);
  const next = storyboard.indexOf('\n### ', start + 1);
  const section = storyboard.slice(start, next < 0 ? undefined : next);
  const notes = section.match(/\*\*Speaker notes[^\n]*\*\*\r?\n\r?\n([\s\S]*?)\r?\n\r?\n\*\*(?:Transition|Close)/);
  if (!notes) throw new Error(`Speaker notes missing for ${id}.`);
  return notes[1].replace(/[“”]/g, '').replace(/\r?\n/g, ' ');
}

function newSlide(id, title, twoLines = false) {
  const slide = presentation.slides.add();
  slide.background.fill = theme.background;
  text(slide, `${id} title`, title, 72, 48, 1136, twoLines ? 112 : 66, 46,
    { bold: true });
  text(slide, 'Slide ID', id, 1166, 675, 42, 22, 16,
    { color: theme.secondary, alignment: 'right' });
  slide.speakerNotes.textFrame.setText(notesFor(id));
  return slide;
}

function footnote(slide, value) {
  text(slide, 'Scope', value, 72, 674, 1070, 24, 18, { color: theme.secondary });
}

function rect(slide, name, x, y, width, height, fill) {
  return slide.shapes.add({ geometry: 'rect', name,
    position: { left: x, top: y, width, height }, fill,
    line: { fill: 'none', width: 0 } });
}

function nativeTable(slide, values, y, widths, rowHeight = 66) {
  const table = slide.tables.add({ rows: values.length, columns: widths.length,
    left: 72, top: y, width: 1136, height: 48 + rowHeight * (values.length - 1),
    columnWidths: widths, values });
  table.styleOptions = { headerRow: false, bandedRows: false, firstColumn: false };
  table.borders.assign({ fill: theme.rule, width: 1, style: 'solid' });
  table.cells.block({ row: 0, column: 0, rowCount: values.length, columnCount: widths.length }).assign({
    fill: theme.background, margins: { left: 18, right: 18, top: 10, bottom: 10 },
    textStyle: { typeface: theme.font, fontSize: 26, color: theme.ink },
  });
  for (let row = 0; row < values.length; row++) {
    table.rows[row].height = row === 0 ? 48 : rowHeight;
    for (let column = 0; column < widths.length; column++) {
      const cell = table.getCell(row, column);
      cell.text.style = { typeface: theme.font, fontSize: row === 0 ? 23 : 26,
        bold: row === 0, color: theme.ink, verticalAlignment: 'middle', autoFit: 'none' };
      if (row === 0) cell.fill = theme.pale;
    }
  }
  return table;
}

// M1: retain the project artwork and leave event details for a specific occasion.
const cover = presentation.slides.add();
cover.background.fill = theme.background;
cover.images.add({
  blob: new Uint8Array(await fs.readFile(path.join(workspaceDir, 'assets/header.png'))),
  contentType: 'image/png', alt: 'Medical Cost Planner project header.', fit: 'contain',
  position: { left: 72, top: 72, width: 1136, height: 422 },
});
text(cover, 'Subtitle', 'Predicting Out-of-Pocket Healthcare Costs with Machine Learning',
  72, 522, 1136, 56, 34, { bold: true });
text(cover, 'Presenter', 'Jens Bender', 72, 594, 400, 36, 28);
text(cover, 'Event details', '[Event / setting] · [Date]', 72, 632, 1000, 32, 24,
  { color: theme.secondary });
cover.speakerNotes.textFrame.setText(notesFor('M1'));

// M2: question first, with an illustration of the planning situation.
const problem = presentation.slides.add();
problem.background.fill = theme.background;
text(problem, 'M2 title', 'How much should I set aside for healthcare?',
  72, 48, 1136, 66, 46, { bold: true });
text(problem, 'Challenge label', 'The challenge',
  72, 520, 532, 36, 30, { bold: true });
text(problem, 'Budgeting challenge', "Planning next year's out-of-pocket\ncosts and HSA/FSA contributions\nis difficult.",
  72, 560, 532, 108, 30);
text(problem, 'Aim label', 'Our aim',
  676, 520, 532, 36, 30, { bold: true });
text(problem, 'Intended estimate', 'A useful ballpark estimate from\nquestions people can answer\nfrom memory.',
  676, 560, 532, 108, 30);
problem.images.add({
  blob: new Uint8Array(await fs.readFile(path.join(slidesDir, 'assets/budget-planning.png'))),
  contentType: 'image/png',
  alt: 'Illustration of an adult considering a budget with a planner and calculator.',
  fit: 'contain', position: { left: 72, top: 120, width: 1136, height: 379 },
});
text(problem, 'Slide ID', 'M2', 1166, 675, 42, 22, 16,
  { color: theme.secondary, alignment: 'right' });
problem.speakerNotes.textFrame.setText(notesFor('M2'));

// M3: introduce the survey before discussing the spending distribution.
const data = newSlide('M3', 'MEPS links accessible inputs to observed spending');
const dataBullets = makeNativeBulletParagraphs([
  'MEPS: Medical Expenditure Panel Survey',
  'Data: 2023 Household Component (HC-251)',
  'Sample: 14,768 adult respondents',
  'Features: 26 inputs covering demographics, insurance, and health',
  'Target: Annual out-of-pocket spending',
  'Survey weights: How many people each respondent represents. This sample represents approximately 260 million U.S. adults.',
], { marginLeftPoints: 18, hangingPoints: 12, spaceAfterPoints: 18 });
for (const paragraph of dataBullets) {
  const value = paragraph.runs[0];
  const colon = value.indexOf(':') + 1;
  paragraph.runs = [
    { run: value.slice(0, colon), textStyle: { bold: true } },
    { run: value.slice(colon) },
  ];
}
text(data, 'Survey and project data', dataBullets, 72, 148, 1136, 448, 30);
const appendixLink = text(data, 'MEPS appendix link', 'Appendix: MEPS overview',
  72, 674, 800, 24, 18, { color: theme.secondary, underline: 'sng' });
appendixLink.text.get('Appendix: MEPS overview').link = {
  uri: 'slide10.xml', isExternal: false, action: 'ppaction://hlinksldjump',
};

// M4: display the concentration figures directly instead of a dense Lorenz plot.
const distribution = newSlide('M4', 'Most out-of-pocket spending comes\nfrom a small share of adults', true);
text(distribution, 'Chart explanation', 'Share of total out-of-pocket spending', 72, 190, 760, 38, 28,
  { bold: true });
const concentration = distribution.charts.add('bar', {
  position: { left: 72, top: 240, width: 740, height: 342 },
  categories: ['Lower-spending 80%', 'Highest-spending 20%'],
  series: [{ name: 'Share of spending', values: [0.207, 0.793], fill: theme.ink,
    valuesFormatCode: '0.0%' }],
  barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 140 },
  hasLegend: false,
  xAxis: { textStyle: { typeface: theme.font, fontSize: 24, fill: theme.ink }, majorGridlines: null },
  yAxis: { min: 0, max: 1, majorUnit: 0.25, numberFormatCode: '0%',
    textStyle: { typeface: theme.font, fontSize: 20, fill: theme.ink },
    majorGridlines: { fill: theme.rule, width: 1 } },
  dataLabels: { showValue: true, position: 'outEnd',
    textStyle: { typeface: theme.font, fontSize: 28, fill: theme.ink, bold: true } },
});
applyPresentationChartFont(concentration, { fontFamily: theme.font });
text(distribution, 'Zero spending', '22.3%', 867, 246, 330, 76, 56, { bold: true });
text(distribution, 'Zero spending explanation', 'of adults have zero\nout-of-pocket spending',
  867, 328, 341, 90, 28);
text(distribution, 'Metric implication', 'Evaluate typical error alongside large errors and uncertainty',
  72, 610, 1136, 40, 28, { bold: true });
footnote(distribution, 'Survey-weighted MEPS 2023 estimates. Source: EDA notebook.');

const selection = newSlide('M5', 'Model selection: median error was not enough');
const modelTable = nativeTable(selection, [
  ['Tuned point-estimate model', 'Validation MdAE', 'Validation MAE'],
  ['Elastic Net', '$159', '$1,051'],
  ['Random Forest', '$228', '$964'],
  ['XGBoost', '$242', '$954'],
], 142, [520, 308, 308]);
for (const [row, column] of [[1, 1], [3, 2]]) {
  modelTable.getCell(row, column).text.style = {
    typeface: theme.font, fontSize: 26, color: theme.ink, bold: true,
    verticalAlignment: 'middle', autoFit: 'none',
  };
}
text(selection, 'Prediction compression', "Elastic Net's largest validation prediction was only about $217",
  72, 424, 1136, 42, 29, { bold: true });
text(selection, 'Tradeoff', 'Good typical error, but limited separation across cost profiles.\nResidual and subgroup checks motivated richer budgeting outputs.',
  72, 480, 1136, 86, 28);
text(selection, 'Quantile decision', 'Next step: XGBoost quantile regression',
  72, 602, 1136, 40, 30, { bold: true });
footnote(selection, 'MdAE: median absolute error; MAE: mean absolute error. Validation; 2023 USD.');

const quantiles = newSlide('M6', 'Quantile regression turns predictions\ninto budgeting ranges', true);
text(quantiles, 'Objective', 'XGBoost quantile objective · survey weights · log-transformed costs',
  72, 190, 1136, 38, 27);
// This axis is conceptual: the positions do not encode a person's dollar values.
text(quantiles, 'Plan-around label', 'Plan-around estimate', 238, 260, 320, 38, 27,
  { bold: true, alignment: 'center' });
text(quantiles, 'Safety label', 'Safety cushion', 834, 260, 320, 38, 27,
  { bold: true, alignment: 'center' });
rect(quantiles, 'Cost axis', 122, 334, 1040, 2, theme.ink);
rect(quantiles, 'Typical range q25 to q75', 248, 318, 398, 34, theme.pale);
for (const [x, label] of [[248, 'q25'], [398, 'q50'], [646, 'q75'], [994, 'q90']]) {
  rect(quantiles, `Quantile marker ${label}`, x, 311, 3, 49, theme.ink);
  text(quantiles, label, label, x - 40, 373, 83, 38, 27, { alignment: 'center' });
}
text(quantiles, 'Range label', 'Typical range', 275, 427, 340, 38, 28,
  { bold: true, alignment: 'center' });
text(quantiles, 'Coverage target', 'Targets 50% coverage', 258, 466, 380, 38, 26,
  { alignment: 'center' });
text(quantiles, 'Upper coverage target', 'Targets 90% below q90', 821, 427, 387, 38, 26,
  { alignment: 'center' });
text(quantiles, 'Direction', 'Higher annual spending →', 856, 496, 352, 38, 24,
  { alignment: 'right' });
text(quantiles, 'Evaluation principle', 'Useful ranges need both coverage and reasonable width',
  72, 566, 1136, 42, 30, { bold: true });
text(quantiles, 'Not a cap', 'The safety cushion is an upper planning reference, not a spending cap.',
  72, 611, 1136, 36, 27);
footnote(quantiles, 'Conceptual diagram, not a prediction for an individual. q50 is the predicted median.');

// M7: a native table preserves editable values and their comparison references.
const results = presentation.slides.add();
results.background.fill = theme.background;
text(results, 'M7 title', [[
  { run: 'Final model audit:', textStyle: { bold: true } },
  ' clearest gains in ranges and q90',
]], 72, 48, 1136, 66, 46);
const table = results.tables.add({
  rows: 4, columns: 3, left: 72, top: 142, width: 1136, height: 252,
  columnWidths: [520, 210, 406],
  values: [
    ['Held-out test metric', 'Result', 'Release gate'],
    ['Plan-around estimate (q50): MdAE', '$240', '< $500'],
    ['Typical range (q25–q75): coverage', '47.3%', '45%–55%'],
    ['Safety cushion (q90): coverage', '91.0%', '85%–95%'],
  ],
});
table.styleOptions = { headerRow: false, bandedRows: false, firstColumn: false };
table.borders.assign({ fill: theme.rule, width: 1, style: 'solid' });
table.cells.block({ row: 0, column: 0, rowCount: 4, columnCount: 3 }).assign({
  fill: theme.background,
  margins: { left: 18, right: 18, top: 10, bottom: 10 },
  textStyle: { typeface: theme.font, fontSize: 26, color: theme.ink },
});
for (let row = 0; row < 4; row++) {
  table.rows[row].height = row === 0 ? 48 : 68;
  for (let column = 0; column < 3; column++) {
    const cell = table.getCell(row, column);
    cell.text.style = {
      typeface: theme.font, fontSize: row === 0 ? 23 : column === 1 ? 32 : 26,
      bold: row === 0 || column === 1,
      color: row > 0 && column === 1 ? theme.teal : theme.ink,
      verticalAlignment: 'middle', autoFit: 'none',
    };
    if (row === 0) cell.fill = theme.pale;
  }
}
text(results, 'Baseline comparison', 'Compared with the population baseline',
  72, 419, 1000, 34, 26, { bold: true });
text(results, 'Plan-around comparison', 'Plan-around estimate (q50)',
  72, 464, 370, 38, 26);
text(results, 'Median error improvement', [[
  { run: '≈$8', textStyle: { bold: true } }, ' lower MdAE (improvement uncertain)',
]], 450, 464, 758, 38, 26);
text(results, 'Typical range comparison', 'Typical range (q25–q75)',
  72, 513, 370, 38, 26);
text(results, 'Interval score gain', [[
  { run: '11.2%', textStyle: { bold: true } }, ' interval skill score',
]], 450, 513, 758, 38, 26);
text(results, 'Safety cushion comparison', 'Safety cushion (q90)',
  72, 562, 370, 38, 26);
text(results, 'q90 loss gain', [[
  { run: '15.6%', textStyle: { bold: true } }, ' quantile skill score',
]], 450, 562, 758, 38, 26);
text(results, 'Metric scope', 'Survey-weighted test metrics. Dollar amounts in 2023 USD.',
  72, 674, 1040, 24, 18, { color: theme.secondary });
text(results, 'Slide ID', 'M7', 1166, 675, 42, 22, 16,
  { color: theme.secondary, alignment: 'right' });
results.speakerNotes.textFrame.setText(notesFor('M7'));

const audit = newSlide('M8', 'Final model test audit: overall coverage\nhides subgroup gaps', true);
text(audit, 'Chart metric', 'Typical-range coverage (q25–q75)', 72, 190, 780, 38, 28,
  { bold: true });
const coverage = audit.charts.add('bar', {
  position: { left: 72, top: 242, width: 740, height: 326 },
  categories: ['Target', 'Overall', 'Low income', 'Poor mental health'],
  series: [{ name: 'Coverage', values: [0.5, 0.473, 0.392, 0.301], fill: theme.teal,
    valuesFormatCode: '0.0%', points: [{ idx: 0, fill: '#A7B6BC' }] }],
  barOptions: { direction: 'bar', grouping: 'clustered', gapWidth: 85 },
  hasLegend: false,
  xAxis: { min: 0, max: 0.6, majorUnit: 0.2, numberFormatCode: '0%',
    textStyle: { typeface: theme.font, fontSize: 22, fill: theme.ink },
    majorGridlines: { fill: theme.rule, width: 1 } },
  yAxis: { textStyle: { typeface: theme.font, fontSize: 25, fill: theme.ink }, majorGridlines: null },
  dataLabels: { showValue: true, position: 'outEnd',
    textStyle: { typeface: theme.font, fontSize: 27, fill: theme.ink, bold: true } },
});
applyPresentationChartFont(coverage, { fontFamily: theme.font });
text(audit, 'Rare events label', 'Rare expensive years', 860, 246, 348, 38, 28, { bold: true });
text(audit, 'Rare events detail', 'Remain difficult\nto anticipate', 860, 287, 348, 80, 28);
text(audit, 'Timing label', 'Future-year use', 860, 392, 348, 38, 28, { bold: true });
text(audit, 'Timing detail', 'Still needs feature-timing\nand later-year validation', 860, 433, 348, 116, 28);
text(audit, 'Safeguard limitation', 'Scope wording and planning notices communicate limits;\nthey do not fix calibration.',
  72, 588, 1136, 74, 28, { bold: true });
footnote(audit, 'Survey-weighted test point estimates; subgroup uncertainty matters. Source: modeling notebook.');

const next = newSlide('M9', 'Model evaluation is complete;\napp development comes next', true);
text(next, 'Implemented', 'Implemented', 72, 190, 700, 38, 30, { bold: true });
text(next, 'Planned', 'Planned', 895, 190, 313, 38, 30, { bold: true });
text(next, 'Training heading', 'Training artifacts', 72, 265, 330, 38, 29, { bold: true });
text(next, 'Training tools', 'DVC stages\nMLflow experiments\nEvaluation artifacts', 72, 310, 330, 127, 27);
text(next, 'Training to inference', '→', 418, 295, 64, 58, 42);
text(next, 'Shared heading', 'Shared inference', 510, 265, 320, 38, 29, { bold: true });
text(next, 'Inference modules', 'Prediction and SHAP\nReusable modules\nUnit tests', 510, 310, 320, 127, 27);
text(next, 'Inference to app', '→', 822, 295, 64, 58, 42);
text(next, 'App heading', 'FastAPI / Gradio', 895, 265, 313, 38, 29, { bold: true });
text(next, 'Remaining app work', 'Integration and latency\nUser evaluation\nAggregate monitoring', 895, 310, 313, 127, 27);
text(next, 'Validation heading', 'Next validation priority', 72, 478, 1136, 38, 28, { bold: true });
text(next, 'Validation work', 'Check feature timing and performance on a later survey year',
  72, 520, 1136, 38, 28);
text(next, 'Closing lesson', 'Evaluate the outputs needed for the user decision',
  72, 605, 1136, 44, 32, { bold: true });

const mepsOverview = presentation.slides.add();
mepsOverview.background.fill = theme.background;
mepsOverview.images.add({
  blob: new Uint8Array(await fs.readFile(path.join(workspaceDir, 'assets/infographic_meps_data.jpg'))),
  contentType: 'image/jpeg',
  alt: 'MEPS household, provider, and employer survey components and the 2023 data used in this project.',
  fit: 'contain', position: { left: 20, top: 0, width: 1240, height: 660 },
});
const returnLink = text(mepsOverview, 'Return to data slide', 'Back to MEPS data',
  72, 674, 800, 24, 18, { color: theme.secondary, underline: 'sng' });
returnLink.text.get('Back to MEPS data').link = {
  uri: 'slide3.xml', isExternal: false, action: 'ppaction://hlinksldjump',
};
text(mepsOverview, 'Slide ID', 'A1', 1166, 675, 42, 22, 16,
  { color: theme.secondary, alignment: 'right' });
mepsOverview.speakerNotes.textFrame.setText(notesFor('A1'));

const candidatePath = path.join(buildDir, 'candidate.pptx');
const finalPath = path.join(outputDir, `medical-cost-planner-main-${revision}.pptx`);
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
// The runtime exports internal slide jumps as generic hyperlinks. Give those
// relationships the slide type required by PowerPoint before validation.
execFileSync(RUNTIME_PYTHON, ['-c', `
import os, sys, zipfile
import xml.etree.ElementTree as ET
source = sys.argv[1]
namespace = 'http://schemas.openxmlformats.org/package/2006/relationships'
prefix = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
ET.register_namespace('', namespace)
with zipfile.ZipFile(source) as package:
    entries = [(info, package.read(info.filename)) for info in package.infolist()]
replacements = {}
for slide, target in [(3, 10), (10, 3)]:
    part = f'ppt/slides/_rels/slide{slide}.xml.rels'
    root = ET.fromstring(dict((info.filename, data) for info, data in entries)[part])
    links = [rel for rel in root if rel.get('Target') == f'slide{target}.xml']
    assert len(links) == 1, f'Expected one slide jump in {part}'
    links[0].set('Type', prefix + 'slide')
    links[0].attrib.pop('TargetMode', None)
    replacements[part] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
temporary = source + '.links.tmp'
with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as package:
    for info, data in entries:
        package.writestr(info, replacements.get(info.filename, data))
os.replace(temporary, source)
`, candidatePath]);
const existingChartStaging = new Set(await fs.readdir(slidesDir));
await finalizePresentation({
  workspaceDir: slidesDir, candidatePath, finalPath,
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [5, 7], requiredNativeChartOwnerSlides: [4, 8],
  materializeLiteralChartWorkbooks: true,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: [
    '--expected-slide-size-emu', '12192000,6858000',
    '--validate-bullet-geometry', '--validate-heading-fit',
    '--require-native-table-slide', '5', '--require-native-table-slide', '7',
  ],
  fontPolicy, verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, 'validation.json'),
});

// Keep the runtime's chart workbook staging files with this revision's build.
for (const entry of await fs.readdir(slidesDir, { withFileTypes: true })) {
  if (!entry.isDirectory() || !/^\.chart-data-/.test(entry.name) || existingChartStaging.has(entry.name)) continue;
  const stagingPath = await fs.realpath(path.join(slidesDir, entry.name));
  const destination = path.join(await fs.realpath(buildDir), entry.name);
  if (path.dirname(stagingPath) !== await fs.realpath(slidesDir)) {
    throw new Error(`Unexpected chart staging path: ${stagingPath}`);
  }
  await fs.rename(stagingPath, destination);
}

// Render the exported file so the previews represent the delivered deck.
const finalDeck = await PresentationFile.importPptx(await FileBlob.load(finalPath));
for (const [index, id] of [...Array.from({ length: 9 }, (_, i) => `M${i + 1}`), 'A1'].entries()) {
  const slide = finalDeck.slides.getItem(index);
  const preview = await finalDeck.export({ slide, format: 'png', scale: 1.5 });
  await fs.writeFile(path.join(outputDir, `${id}.png`), new Uint8Array(await preview.arrayBuffer()));
  const layout = await slide.export({ format: 'layout' });
  await fs.writeFile(path.join(buildDir, `${id}.layout.json`), await layout.text());
}

// Remove numbered drafts only after validation and rendering succeed.
async function pruneDrafts(parentDir, namePattern, currentName, previousCount) {
  const root = await fs.realpath(parentDir);
  const expectedRoot = path.join(await fs.realpath(workspaceDir), 'docs', 'slides', path.basename(parentDir));
  if (root !== expectedRoot) throw new Error(`Unexpected cleanup directory: ${root}`);
  const drafts = [];
  for (const entry of await fs.readdir(root, { withFileTypes: true })) {
    if (!entry.isDirectory() || !namePattern.test(entry.name) || entry.name === currentName) continue;
    const draftPath = await fs.realpath(path.join(root, entry.name));
    if (path.dirname(draftPath) !== root) throw new Error(`Unsafe cleanup path: ${draftPath}`);
    drafts.push({ name: entry.name, path: draftPath, modified: (await fs.stat(draftPath)).mtimeMs });
  }
  drafts.sort((a, b) => b.modified - a.modified);
  let retained = 0;
  for (const draft of drafts) {
    if (retained < previousCount) {
      const isMain = draft.name.startsWith('main-');
      const draftRevision = draft.name.replace(/^(?:design-sample|main)-/, '');
      const ids = isMain ? Array.from({ length: 9 }, (_, i) => `M${i + 1}`) : ['M2', 'M7'];
      const requiredFiles = [`medical-cost-planner-${isMain ? 'main' : 'design'}-${draftRevision}.pptx`,
        ...ids.map(id => `${id}.png`)];
      const complete = await Promise.all(requiredFiles.map(file =>
        fs.stat(path.join(draft.path, file)).then(stat => stat.isFile()).catch(() => false)
      ));
      if (complete.every(Boolean)) {
        retained++;
        continue;
      }
    }
    await fs.rm(draft.path, { recursive: true });
  }
}

if (/^v\d+$/.test(revision)) {
  await pruneDrafts(path.dirname(outputDir), /^(?:design-sample|main)-v\d+$/, path.basename(outputDir), 1);
  await pruneDrafts(path.dirname(buildDir), /^v\d+$/, path.basename(buildDir), 0);
}
console.log(JSON.stringify({ finalPath, outputDir, font: theme.font }, null, 2));
