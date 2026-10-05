// Generate and revise slides through the ChatGPT desktop app or Codex using
// the bundled presentation runtime and validation helpers. Runtime paths come
// from load_workspace_dependencies; this project does not install those
// dependencies or provide a standalone presentation build. To recreate an older
// version, restore the generator, storyboard, and assets from the same Git commit.

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

// Use the bundled presentation runtime returned by load_workspace_dependencies.
const { RUNTIME_NODE_MODULES, RUNTIME_PYTHON, SKILL_DIR } = process.env;
if (!RUNTIME_NODE_MODULES || !RUNTIME_PYTHON || !SKILL_DIR) {
  throw new Error('Set RUNTIME_NODE_MODULES, RUNTIME_PYTHON, and SKILL_DIR.');
}
const runtimeRequire = createRequire(path.join(RUNTIME_NODE_MODULES, 'package.json'));
const { Presentation, PresentationFile, FileBlob } = await import(
  pathToFileURL(runtimeRequire.resolve('@oai/artifact-tool')).href
);
const { finalizePresentation, resolvePresentationFont } = await import(
  pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href
);

const slidesDir = path.dirname(fileURLToPath(import.meta.url));
const workspaceDir = path.resolve(slidesDir, '../..');
const revision = process.argv[2] || 'v1';
if (!/^[a-zA-Z0-9_-]+$/.test(revision)) throw new Error('Invalid revision name.');
const buildDir = path.join(slidesDir, '.build', revision);
const outputDir = path.join(slidesDir, 'exports', `design-sample-${revision}`);
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
  const notes = section.match(/\*\*Speaker notes[^\n]*\*\*\r?\n\r?\n([\s\S]*?)\r?\n\r?\n\*\*Transition/);
  if (!notes) throw new Error(`Speaker notes missing for ${id}.`);
  return notes[1].replace(/[“”]/g, '').replace(/\r?\n/g, ' ');
}

// M2: question first, with an illustration of the planning situation.
const problem = presentation.slides.add();
problem.background.fill = theme.background;
text(problem, 'M2 title', 'How much to set aside for healthcare next year?',
  72, 48, 1136, 66, 46, { bold: true });
text(problem, 'Budgeting need', 'Annual out-of-pocket budgeting',
  72, 142, 644, 38, 28, { bold: true });
text(problem, 'Contribution planning', 'Including HSA/FSA contributions',
  72, 180, 644, 40, 27);
text(problem, 'Challenge label', 'The challenge',
  72, 273, 620, 34, 28, { bold: true });
text(problem, 'Uncertainty', "Next year's care needs and\nout-of-pocket spending are uncertain.",
  72, 309, 644, 78, 28);
text(problem, 'Aim label', 'Our aim',
  72, 429, 620, 34, 28, { bold: true });
text(problem, 'Intended estimate', 'A quick estimate using your insurance\nstatus and answers you can give from memory.',
  72, 465, 650, 78, 28);
problem.images.add({
  blob: new Uint8Array(await fs.readFile(path.join(slidesDir, 'assets/budget-planning.png'))),
  contentType: 'image/png',
  alt: 'Illustration of an adult considering a budget with a planner and calculator.',
  fit: 'contain', position: { left: 748, top: 142, width: 460, height: 460 },
});
text(problem, 'Slide ID', 'M2', 1166, 675, 42, 22, 16,
  { color: theme.secondary, alignment: 'right' });
problem.speakerNotes.textFrame.setText(notesFor('M2'));

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

const candidatePath = path.join(buildDir, 'candidate.pptx');
const finalPath = path.join(outputDir, `medical-cost-planner-design-${revision}.pptx`);
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
await finalizePresentation({
  workspaceDir, candidatePath, finalPath,
  explicitTotalSlideCount: 2,
  requiredNativeTableOwnerSlides: [2], requiredNativeChartOwnerSlides: [],
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: [
    '--expected-slide-size-emu', '12192000,6858000',
    '--validate-bullet-geometry', '--validate-heading-fit', '--require-native-table-slide', '2',
  ],
  fontPolicy, verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, 'validation.json'),
});

// Render the exported file so the previews represent the delivered deck.
const finalDeck = await PresentationFile.importPptx(await FileBlob.load(finalPath));
for (const [index, id] of ['M2', 'M7'].entries()) {
  const slide = finalDeck.slides.getItem(index);
  const preview = await finalDeck.export({ slide, format: 'png', scale: 1.5 });
  await fs.writeFile(path.join(outputDir, `${id}.png`), new Uint8Array(await preview.arrayBuffer()));
  const layout = await slide.export({ format: 'layout' });
  await fs.writeFile(path.join(buildDir, `${id}.layout.json`), await layout.text());
}
console.log(JSON.stringify({ finalPath, outputDir, font: theme.font }, null, 2));
