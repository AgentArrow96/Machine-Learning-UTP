'use strict';

const { esc, pct } = FN_CHARTS;

const METHOD = {
    rf: { name: 'SLIC + Random Forest', short: 'SLIC + RF', colour: '#059669', kind: 'Traditional · CPU' },
    unet: { name: 'ResNet-18 U-Net', short: 'U-Net', colour: '#4f46e5', kind: 'Deep learning · CUDA' }
};
const METHODS = Object.keys(METHOD);
const SEED_COLOURS = ['#0891b2', '#d97706', '#7c3aed', '#e11d48'];
const CLASS_SHORT = ['Other', 'Building', 'Road'];
const PAGE_SIZE = 24;

const NAV = [
    { group: 'START', items: [['overview', 'Overview']] },
    { group: 'THE DATA', items: [['dataset', 'Dataset'], ['preprocessing', 'Data preprocessing']] },
    { group: 'THE MODELS', items: [['methodology', 'Methods & equations'], ['walkthrough', 'How the models work']] },
    { group: 'THE RESULTS', items: [['results', 'Results'], ['explorer', 'Prediction explorer']] },
    { group: 'NOTES', items: [['training', 'Training & compute'], ['interpretation', 'Interpretation'], ['reproducibility', 'Reproducibility']] }
];

let data = null;
let uniform = null;
let drawToken = 0;
const imageCache = new Map();
const chartRenderers = new Map();
const state = {
    mode: 'details', budget: 100, presence: 'all', overlap: 'all', sort: 'id', search: '',
    selected: null, page: 0, opacity: 0.55, overlay: true, visible: [true, true, true],
    metricRun: null, lossSeed: null
};

const $ = (query, root = document) => root.querySelector(query);
const $$ = (query, root = document) => [...root.querySelectorAll(query)];


function num(value) {
    return value == null ? 'N/A' : Number(value).toLocaleString('en-US');
}

function seconds(value) {
    if (value == null) return 'N/A';
    return value < 60 ? `${value.toFixed(1)} s` : `${(value / 60).toFixed(1)} min`;
}

function points(value) {
    if (value == null) return 'N/A';
    const rounded = Math.abs(value * 100).toFixed(1);
    const sign = rounded === '0.0' ? '±' : value > 0 ? '+' : '−';
    return `${sign}${rounded} pp`;
}

function meanSd(stat, format = pct) {
    if (!stat || stat.mean == null) return 'N/A';
    return stat.sd == null ? format(stat.mean) : `${format(stat.mean)} ± ${(stat.sd * 100).toFixed(1)}`;
}

function seedValues(stat, format = pct) {
    return stat?.values ? stat.values.map(v => format(v)).join(' · ') : '';
}

function mean(values) {
    const defined = values.filter(v => v != null);
    return defined.length ? defined.reduce((a, b) => a + b, 0) / defined.length : null;
}

function plain(text) {
    return `<span data-plain>${esc(text)}</span>`;
}


function bundle(source = 'before') {
    return { original: data, before: uniform }[source] || null;
}


function navGroups() {
    return state.mode === 'presentation' ? FN_PRES.NAV : NAV;
}

function navItems() {
    return navGroups().flatMap(g => g.items.map(([route, label, number]) => ({ route, label, group: g.group, number }))).map((item, i) => ({ ...item, number: item.number || String(i + 1).padStart(2, '0') }));
}

function homeRoute(mode = state.mode) {
    return mode === 'presentation' ? FN_PRES.routes()[0] : 'overview';
}

function modeOf(route) {
    if (FN_PRES.routes().includes(route)) return 'presentation';
    return NAV.some(g => g.items.some(([r]) => r === route)) ? 'details' : null;
}

function summaryFor(method, budget, source = 'before') {
    return bundle(source)?.summary.find(s => s.method === method && s.budget === budget);
}

function runsFor(method, budget, source = 'before') {
    return (bundle(source)?.runs || []).filter(r => r.method === method && r.budget === budget).sort((a, b) => a.seed - b.seed);
}

function explorerRun(method, budget, source = 'before') {
    const b = bundle(source);
    return b?.runs.find(r => b.explorer_runs.includes(r.id) && r.method === method && r.budget === budget);
}

function effectFor(method, source = 'before') {
    return bundle(source)?.budget_effects.find(e => e.method === method);
}

function comparisonFor(method, budget, source = 'before') {
    return bundle(source)?.comparisons?.find(c => c.method === method && c.budget === budget);
}

function sampleData(sample, source) {
    const b = bundle(source);
    return b === data ? sample : b?.samples[sample.id];
}

function sampleMetrics(sample, runId, source = 'before') {
    return sampleData(sample, source)?.metrics[runId];
}

function overlapInfo() {
    return data.dataset?.overlap || null;
}

function testLabel() {
    return `Official test split (${num(data.dataset?.counts.test)} images)`;
}


function currentRoute() {
    return location.hash.slice(1) || homeRoute();
}

function kicker(route) {
    const item = navItems().find(i => i.route === route);
    return item ? `${item.number} / ${item.label}` : route;
}


function header(route, title, intro = '') {
    return `<header class="fn-page-header"><div class="fn-eyebrow-row"><div class="fn-eyebrow" data-plain>${esc(kicker(route))}</div></div><h1>${esc(title)}</h1>${intro ? `<p>${esc(intro)}</p>` : ''}</header>`
        + `<nav class="fn-toc" data-toc aria-label="On this page" hidden></nav>`;
}

function card(title, body, extra = '') {
    return `<section class="fn-card ${extra}">${title ? `<h2>${title}</h2>` : ''}${body}</section>`;
}

function takeaways(items, title = 'Key takeaways') {
    const list = items.filter(Boolean);
    return list.length ? `<section class="fn-takeaways" aria-label="${esc(title)}"><p class="fn-takeaways-title">${esc(title)}</p><ul>${list.map(i => `<li>${i}</li>`).join('')}</ul></section>` : '';
}

function chartBlock(title, render, { heading = 'h2', svg = true, note = '' } = {}) {
    const id = `chart-${chartRenderers.size + 1}`;
    chartRenderers.set(id, { title, render });
    return `<div class="fn-chart-block" data-chart-block><div class="fn-card-top"><${heading} class="fn-vis-title">${esc(title)}</${heading}><div class="fn-chart-tools">`
        + `<button type="button" class="fn-btn" data-action="toggle-notes" aria-pressed="true">Notes on</button>`
        + `<button type="button" class="fn-btn" data-expand-chart="${id}">Expand</button>`
        + (svg ? `<button type="button" class="fn-btn" data-action="download-chart">Save SVG</button>` : '')
        + `</div></div><div data-chart-body="${id}">${render({})}</div>${note ? `<p class="fn-figure-note">${esc(note)}</p>` : ''}</div>`;
}

function vis(title, render, note = '') {
    return `<section class="fn-card">${chartBlock(title, render, { note })}</section>`;
}

function zoomable(inner, label, attrs = {}) {
    const extra = Object.entries(attrs).filter(([, v]) => v != null).map(([k, v]) => ` data-zoom-${k}="${esc(v)}"`).join('');
    return `<button type="button" class="fn-zoom" data-zoom${extra} aria-label="Expand: ${esc(label)}">${inner}<span class="fn-zoom-icon" aria-hidden="true">⤢</span></button>`;
}

function targetShare(counts, ids) {
    const total = counts.reduce((a, b) => a + b, 0);
    return ids.reduce((sum, i) => sum + counts[i], 0) / total;
}

function metricCards(items) {
    return `<div class="fn-grid fn-four-col">${items.map(item => `<div class="fn-metric-card"><strong>${item.value}</strong><span>${esc(item.label)}</span>${item.note ? `<small>${esc(item.note)}</small>` : ''}</div>`).join('')}</div>`;
}

function empty(title = 'Results are being prepared', text = 'This view fills from completed evaluations. No example scores have been substituted.') {
    return `<div class="fn-empty"><h3>${esc(title)}</h3><p>${esc(text)}</p></div>`;
}

function notice() {
    if (data.status === 'complete') return '';
    return `<div class="fn-notice" role="status"><strong>Experiment in progress · ${data.runs.length} of ${data.protocol.expected_runs} final evaluations</strong>${esc(data.message || 'The remaining final evaluations have not been exported yet.')}</div>`;
}

function legend() {
    return `<div class="fn-legend">${data.classes.map(c => `<span><i style="background:${c.colour}"></i>${esc(c.name)}</span>`).join('')}</div>`;
}

function download(key, label) {
    return data.downloads[key] ? `<a href="${esc(data.downloads[key])}" download>${esc(label)} ↓</a>` : '';
}

function methodPill(method) {
    return `<span class="fn-pill ${method}">${esc(METHOD[method].kind)}</span>`;
}

function budgetToggle() {
    const button = budget => `<button type="button" class="${state.budget === budget ? 'on' : ''}" data-budget="${budget}" aria-pressed="${state.budget === budget}">${budget}% labels</button>`;
    return `<div class="fn-seg" role="group" aria-label="Training budget">${button(25)}${button(100)}</div>`;
}

function decisionPill(decision) {
    const kind = decision === 'helps' ? 'good' : decision === 'hurts' ? 'warn' : '';
    return `<span class="fn-pill ${kind}">${decision === 'helps' ? '▲ helps' : decision === 'hurts' ? '▼ hurts' : '● no clear effect'}</span>`;
}

function interval(ci) {
    const bare = v => points(v).replace(' pp', '');
    return `[${bare(ci[0])}, ${bare(ci[1])}] pp`;
}


function beforeNote() {
    return 'Every model is treated the same way (capped inverse-frequency class weights, all training data).';
}


function summaryTable() {
    const b = bundle();
    if (!b.summary.length) return empty();
    const head = '<tr><th>Method</th><th>Labels</th><th>Flooded mIoU</th><th>3-class mIoU</th><th>Building IoU</th><th>Road IoU</th></tr>';
    const rows = b.summary.map(item => {
        const s = item.full;
        const cell = key => `<td>${meanSd(s[key])}<span class="fn-seed-values"><br>${seedValues(s[key])}</span></td>`;
        return `<tr><td>${methodPill(item.method)}<br>${esc(METHOD[item.method].name)}</td><td>${item.budget}%</td>`
            + `<td class="fn-score">${meanSd(s.flooded_miou)}<small>seeds: ${seedValues(s.flooded_miou)}</small></td>`
            + `${cell('miou')}${cell('building_iou')}${cell('road_iou')}</tr>`;
    }).join('');
    const which = beforeNote();
    return `<div class="fn-table-wrap"><table class="fn-table"><thead>${head}</thead><tbody>${rows}</tbody></table></div>`
        + `<p class="fn-figure-note">${which} Mean ± standard deviation across ${data.protocol.train_seeds.length} training seeds (${data.protocol.train_seeds.join(', ')}), pooled pixel counts per run. ${esc(testLabel())}. <a href="#results">Every metric (majority IoU, weighted mIoU, accuracy, precision, recall, Dice) →</a></p>`;
}


function overviewTakeaways() {
    const best = [...bundle().summary].filter(s => s.budget === 100).sort((a, b) => b.full.flooded_miou.mean - a.full.flooded_miou.mean)[0];
    const unet = effectFor('unet')?.full.mean;
    return takeaways([
        best ? `Best model at 100% labels: <strong>${esc(METHOD[best.method].name)}</strong>, ${meanSd(best.full.flooded_miou)} flooded mIoU.` : '',
        unet != null ? `Four times the labels change the U-Net by ${points(unet)} on average.` : '',
        `SLIC + RF stays far behind (about ${pct(summaryFor('rf', 100)?.full.flooded_miou.mean, 0)}): colour and texture alone confuse floodwater with flooded roads.`,
        'Both models handle the class imbalance the same way: the same capped class weights and all of the training data.'
    ]);
}

function overview() {
    const sample = data.samples.find(s => s.id === data.representatives[0]) || data.samples[0];
    const total = data.dataset ? Object.values(data.dataset.counts).reduce((a, b) => a + b, 0) : null;
    const best = [...bundle().summary].filter(s => s.full.flooded_miou.mean != null).sort((a, b) => b.full.flooded_miou.mean - a.full.flooded_miou.mean)[0];
    const overlap = overlapInfo();
    const testOverlap = overlap ? `${num(overlap.summary.test.overlapping_train)} / ${num(overlap.summary.test.images)}` : '—';
    const hero = `<div class="fn-hero"><div class="fn-hero-copy"><div class="fn-tag">FLOODNET · A COMPARATIVE STUDY</div>`
        + `<h1>Reading the flood.<br><em>One pixel at a time.</em></h1>`
        + `<p>How well can traditional machine learning and a pretrained U-Net find flooded buildings and roads in post-hurricane aerial images, with all the labels or only a quarter of them?</p>`
        + `<div class="fn-actions"><a class="fn-primary-btn" href="#results">Explore the findings →</a><a class="fn-secondary-btn" href="#explorer">Open the prediction explorer</a></div></div>`
        + `<div class="fn-hero-visual">${sample ? zoomable(`<img src="${esc(sample.rgb)}" alt="Held-out FloodNet aerial image ${esc(sample.image_id)}">`, `held-out image ${sample.image_id}`, { title: `Held-out test image ${sample.image_id}`, hires: sample.id, caption: 'Official test split' }) + `<span class="fn-hero-stamp" data-plain>HELD-OUT IMAGE ${esc(sample.image_id)}</span>` : ''}</div></div>`;
    let tiles;
    tiles = metricCards([
            { value: num(total), label: 'Audited image–mask pairs', note: 'Official train / validation / test' },
            { value: num(data.protocol.trained_runs), label: 'Trained models', note: `2 methods × 2 budgets × ${data.protocol.train_seeds.length} seeds` },
            { value: best ? meanSd(best.full.flooded_miou) : '—', label: 'Best flooded-class mIoU', note: best ? `${METHOD[best.method].short} · ${best.budget}% · mean ± SD` : 'Awaiting evaluation' },
            { value: testOverlap, label: 'Test images overlapping a train frame', note: 'Verified by keypoint matching' }
        ]);
    const intro = `<div class="fn-grid fn-two-col">`
        + card('A small experiment, a specific question', `<p>The primary score is the mean IoU of the two target classes, flooded buildings and flooded roads (${FN_MATH.inline('fmiou')}). Image IDs, input resolution and scoring rules stay fixed while the method and the labelled-data budget change.</p><div class="fn-flow"><div>RGB image</div><span>→</span><div>Learn / classify</div><span>→</span><div>Pixel labels</div></div><a href="#methodology">See how the methods work →</a>`)
        + card('What this study can tell us', `<p>Measured segmentation quality, missed target regions, data-efficiency trade-offs, and the compute cost on one laptop.</p><p>It does not establish road safety, flood depth, emergency-response benefit or performance on a different flood event.</p><a href="#interpretation">Read the practical limits →</a>`)
        + `</div>`;
    const comparison = card('The held-out comparison', summaryTable());
    return hero + notice() + tiles + overviewTakeaways() + comparison + intro;
}

function datasetPage() {
    if (!data.dataset) {
        return header('dataset', 'Understand the data first', 'FloodNet provides high-resolution aerial RGB images with pixelwise annotations.')
            + notice() + empty('Dataset audit pending', 'Counts and distributions come from the downloaded release, not from the proposal figures.');
    }
    const d = data.dataset;
    const train = d.processed_distributions.train;
    const processedTotal = train.pixel_counts.reduce((a, b) => a + b, 0);
    const original = d.distributions.train;
    const originalTotal = original.pixel_counts.reduce((a, b) => a + b, 0);
    const originalTargets = targetShare(original.pixel_counts, [1, 3]);
    const processedTargets = targetShare(train.pixel_counts, [1, 2]);
    const overlap = overlapInfo();
    let html = header('dataset', 'One event. A focused target.', 'Hurricane Harvey UAV imagery, the published FloodNet partitions, and a three-class remapping.');
    html += metricCards([
        { value: num(d.counts.train), label: 'Training images' },
        { value: num(d.counts.val), label: 'Validation images' },
        { value: num(d.counts.test), label: 'Held-out test images' },
        { value: num(d.budget_counts['25']), label: 'Images in the 25% budget', note: `Stratified, seed ${data.protocol.subset_seed}` }
    ]);
    html += takeaways([
        `Flooded buildings and roads are only ${pct(processedTargets)} of the processed training pixels; "Other" (including floodwater) dominates.`,
        overlap ? `${num(overlap.summary.test.overlapping_train)} / ${num(overlap.summary.test.images)} test images share ground with a training frame, so the test split mostly measures performance on areas already seen.` : '',
        `No flooded target is lost by the 256 px resize in more than ${num(Math.max(d.targets_lost_images['1'].length, d.targets_lost_images['2'].length))} images.`
    ]);
    html += card('From ten labels to three', `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Original ID</th><th>Original label</th><th>Experiment class</th></tr></thead><tbody>`
        + `<tr><td>1</td><td>Building flooded</td><td>1 · Flooded building</td></tr><tr><td>3</td><td>Road flooded</td><td>2 · Flooded road</td></tr>`
        + `<tr><td>0, 2, 4–9</td><td>All remaining valid classes</td><td>0 · Other</td></tr><tr><td>Generated padding</td><td>No annotation</td><td>255 · Ignored</td></tr></tbody></table></div>`
        + `${legend()}<div class="fn-callout">“Other” includes floodwater. This study detects flooded infrastructure; it does not outline all water.</div>`);
    html += `<div class="fn-grid fn-two-col">`
        + vis('Original training-pixel distribution', options => FN_CHARTS.bars(d.original_classes.map((name, i) => ({
            label: name, value: original.pixel_counts[i] / originalTotal, colour: i === 1 ? '#f59e0b' : i === 3 ? '#0891b2' : '#94a3b8', emphasis: i === 1 || i === 3,
            tip: [pct(original.pixel_counts[i] / originalTotal, 2), `${name} · original ID ${i}`, `${num(original.pixel_counts[i])} training pixels`]
        })), { ...options, left: 140, label: 'Original training class pixel shares', note: `The two targets (IDs 1 and 3) are only ${pct(originalTargets)} of all training pixels` }), 'Measured on the original indexed training annotations.')
        + vis('Processed training-pixel distribution', options => FN_CHARTS.bars(data.classes.map((c, i) => ({
            label: c.name, value: train.pixel_counts[i] / processedTotal, colour: c.colour, emphasis: i > 0,
            tip: [pct(train.pixel_counts[i] / processedTotal, 2), c.name, `${num(train.pixel_counts[i])} pixels at 256 px`]
        })), { ...options, label: 'Processed training class pixel shares', note: `Flooded targets = ${pct(processedTargets)} of pixels; “Other” dominates` }), 'After the 256×256 resize and remapping; padding excluded.')
        + `</div>`;
    html += overlapSection();
    html += card('Audit checks', `<dl class="fn-definition-list"><dt>Exact duplicate groups</dt><dd>${num(d.exact_duplicate_groups.length)}</dd>`
        + `<dt>EXIF-rotated photos</dt><dd>${num(d.exif_rotated_images)} (RGB rotated to match the annotation coordinates)</dd>`
        + `<dt>Building lost at 256 px</dt><dd>${num(d.targets_lost_images['1'].length)} images</dd><dt>Road lost at 256 px</dt><dd>${num(d.targets_lost_images['2'].length)} images</dd></dl>`
        + `<p class="fn-small fn-muted">Original and processed pixel counts use different resolutions; compare their shares, not raw totals.</p>`);
    return html;
}

function overlapSection() {
    const overlap = overlapInfo();
    if (!overlap) return card('Do test scenes also appear in training?', empty('Overlap analysis pending', 'Run the overlap stage to verify shared scenes between splits.'));
    const t = overlap.summary.test;
    const v = overlap.summary.val;
    const histogram = options => FN_CHARTS.bars(overlap.test_max_overlap_histogram.map(h => ({
        label: h.label, value: h.count / t.images, text: `${h.count} images`, colour: h.label === 'None' ? '#059669' : '#d97706',
        tip: [`${h.count} test images`, `Largest overlap: ${h.label}`, `${pct(h.count / t.images)} of the test split`]
    })), { ...options, left: 100, label: 'Largest verified overlap with any training frame, per test image', note: `${num(t.overlapping_train)} of ${num(t.images)} test images (${pct(t.overlapping_train / t.images, 0)}) share ground with a training frame` });
    const examples = overlap.examples.map((e, i) => {
        const group = `overlap-${i}`;
        const caption = `${pct(e.overlap, 0)} of the test frame is also in the training frame`;
        return `<div class="fn-pair"><div class="fn-pair-images">`
        + `<figure>${zoomable(`<img src="${esc(e.eval_rgb)}" loading="lazy" alt="Test image ${esc(e.eval_id)}">`, `test image ${e.eval_id}`, { group, title: `Test ${e.eval_id.replace('test_', '')}`, hires: e.eval_id, caption })}<figcaption data-plain>Test ${esc(e.eval_id.replace('test_', ''))}</figcaption></figure>`
        + `<figure>${zoomable(`<img src="${esc(e.train_rgb)}" loading="lazy" alt="Training image ${esc(e.train_id)}">`, `training image ${e.train_id}`, { group, title: `Train ${e.train_id.replace('train_', '')}`, hires: e.train_id, caption })}<figcaption data-plain>Train ${esc(e.train_id.replace('train_', ''))}</figcaption></figure></div>`
        + `<p>${pct(e.overlap, 0)} of the test frame is also in the training frame · ${num(e.inliers)} matched keypoints</p></div>`;
    }).join('');
    return `<section class="fn-card"><h2>Do test scenes also appear in training?</h2>`
        + `<p>The drone photographed overlapping consecutive frames, and the official split assigns frames independently, so the same ground can appear in both training and test images. Candidate pairs (frames within ±${overlap.settings.frame_window} IDs plus the most similar images by ImageNet features) were checked by matching ORB keypoints and fitting a homography. A pair counts as overlapping only when at least ${overlap.settings.min_inliers} keypoints agree and the training frame covers at least ${pct(overlap.settings.min_overlap, 0)} of the test frame.</p>`
        + metricCards([
            { value: `${num(t.overlapping_train)} / ${num(t.images)}`, label: 'Test images overlapping training', note: pct(t.overlapping_train / t.images, 0) },
            { value: `${num(v.overlapping_train)} / ${num(v.images)}`, label: 'Validation images overlapping training', note: pct(v.overlapping_train / v.images, 0) },
            { value: `${num(overlap.target_images.overlapping)} / ${num(overlap.target_images.images)}`, label: 'Test images with flooded targets that overlap training', note: 'Almost every flooded test scene' },
            { value: num(t.clean), label: 'Test images with no overlap', note: `Only ${overlap.target_images.clean_with_building} contain flooded buildings, ${overlap.target_images.clean_with_road} flooded roads` }
        ])
        + chartBlock('Largest overlap with any training frame', histogram, { heading: 'h3' })
        + (examples ? `<h3 class="fn-vis-title" style="margin:18px 0 10px">Strongest examples</h3><p class="fn-small fn-muted">Select a photo to expand it; ← / → flips between the test and training frame.</p><div class="fn-pair-grid">${examples}</div>` : '')
        + `<div class="fn-callout">The official split is kept for the headline results, as the methodology requires. Because the flooded scenes were photographed densely, an overlap-free test subset holds almost no flooded targets and cannot be scored fairly. The Results page instead compares target images with lower and higher overlap.</div>`
        + `<div class="fn-downloads">${download('overlap', 'Verified overlap pairs CSV')}</div></section>`;
}

function regionCounts(budget) {
    const run = explorerRun('rf', budget) || runsFor('rf', budget)[0];
    return run?.region_class_counts ? run.region_class_counts.map(num).join(' / ') : 'N/A';
}


function rfWeights() {
    const w = runsFor('rf', 100)[0]?.class_weights;
    return w ? `${[0, 1, 2].map(c => Number(w[c]).toFixed(2)).join(' / ')} at 100%` : '';
}

function methodology() {
    let html = header('methodology', 'Two approaches. Shared rules.', `The study compares practical segmentation pipelines under matched image budgets and identical evaluation.`);
    html += takeaways([
        `Every method sees the same images, resolution (256×256), budgets (25% / 100%) and seeds (${data.protocol.train_seeds.join(', ')}); only the method changes.`,
        `The primary score is ${FN_MATH.inline('fmiou')}, from pooled pixel counts.`,
        'The class imbalance is handled the same way for both models: the same capped inverse-frequency class weights, and all of the training data.',
    ]);
    html += card('The shared preparation path', `<div class="fn-flow"><div>Audit originals</div><span>→</span><div>Remap labels</div><span>→</span><div>Resize + pad</div><span>→</span><div>Cache once</div><span>→</span><div>25% / 100%</div></div>`
        + `<p>Aspect ratio is preserved inside a 256×256 frame. Masks use nearest-neighbour resizing, and padded pixels are ignored in training and scoring. One stratified 25% subset (seed ${data.protocol.subset_seed}) is shared by every method and seed. Validation and test sets never change.</p>`);
    const rf = `<section class="fn-card fn-method-card"><span class="fn-pill rf">Traditional · CPU</span><h2 style="margin-top:12px">SLIC + Random Forest</h2>`
        + `<p>Neighbouring pixels are grouped into about 250 superpixels. Each region is described by colour and texture, then classified by a class-weighted forest.</p>`
        + `<div class="fn-method-path"><span>RGB</span>→<span>SLIC regions</span>→<span>22 features</span>→<span>Forest</span></div>`
        + `<dl class="fn-definition-list"><dt>Colour</dt><dd>RGB/HSV mean and standard deviation</dd><dt>Texture</dt><dd>10-bin uniform LBP histogram</dd>`
        + (`<dt>Region label</dt><dd>Majority annotation, padding excluded</dd><dt>Forest</dt><dd>150 trees · maximum depth 20</dd>`
                + `<dt>Class weights</dt><dd>The U-Net's capped inverse-frequency weights: ${esc(rfWeights())}<br><small class="fn-muted">Other / building / road</small></dd>`
                + `<dt>Training data</dt><dd>Every training region, none discarded</dd>`)
        + `<dt>Regions used</dt><dd>25%: ${regionCounts(25)}<br>100%: ${regionCounts(100)}<br><small class="fn-muted">Other / building / road</small></dd></dl>`
        + (`<p class="fn-small fn-muted">First version (kept for reference): sklearn “balanced” weights and a ${num(data.protocol.rf_max_regions)}-region cap that dropped two thirds of the “Other” regions at 100%. It was changed so that both models handle the imbalance the same way. <a href="#results">Both versions compared →</a></p>`)
        + `</section>`;
    const unet = `<section class="fn-card fn-method-card unet"><span class="fn-pill unet">Deep learning · CUDA</span><h2 style="margin-top:12px">Lightweight U-Net</h2>`
        + `<p>A pretrained ResNet-18 encoder extracts features. A decoder with skip connections restores spatial detail and outputs three scores per pixel.</p>`
        + `<div class="fn-method-path"><span>RGB</span>→<span>Encoder</span>→<span>Decoder + skips</span>→<span>3 logits</span></div>`
        + `<dl class="fn-definition-list"><dt>Pretraining</dt><dd>ImageNet ResNet-18</dd><dt>Decoder</dt><dd>128 → 64 → 32 → 16 → 8</dd>`
        + `<dt>Loss</dt><dd>Class-weighted cross-entropy</dd><dt>Optimiser</dt><dd>AdamW · LR 1e-4 · WD 1e-4 · batch 4</dd>`
        + `<dt>Selection</dt><dd>Best validation flooded mIoU, patience 6, up to 30 epochs</dd></dl>`
        + `<p class="fn-small">BF16 mixed precision on the GPU. Training stops with an error instead of falling back to the CPU.</p>` + `</section>`;
    html += `<div class="fn-grid fn-two-col">${rf}${unet}</div>`;
    html += `<div class="fn-callout">Want to see these steps on a real test image? <a href="#walkthrough">Open the step-by-step walkthrough →</a> It replays each model on six representative scenes, and every final mask it shows is checked against the frozen test prediction.</div>`;
    html += card('Notation and equations', `<p>Every formula used on this site, written once. ${FN_MATH.tex('c')} is a class, ${FN_MATH.tex(String.raw`TP_c,\ FP_c,\ FN_c`)} are pooled pixel counts over the test images and ${FN_MATH.tex('s')} is a training seed.</p>`
        + `<h3 class="fn-eq-group">Metrics</h3>${FN_MATH.equations(['fmiou', 'iou', 'majority', 'miou3', 'fwiou', 'acc', 'precision', 'recall', 'dice'])}`
        + `<h3 class="fn-eq-group">Data and training</h3>${FN_MATH.equations(['norm', 'weights', 'ce'])}`
        + ('')
        + ('')
        + `<h3 class="fn-eq-group">Statistics</h3>${FN_MATH.equations(['budget', 'bootstrap', 'decision'])}`);
    html += `<div class="fn-dark-card"><h3>Three seeds, one subset</h3><p>Every configuration is trained with seeds <strong>${data.protocol.train_seeds.join(', ')}</strong>. The seed changes forest sampling, network initialisation, data order and augmentation; the 25% image subset stays the same. Results report the mean, standard deviation and range, so a lucky or unlucky run cannot decide the comparison.</p></div>`;
    html += card('Fair comparison and model selection', `<details class="fn-details" open><summary>What is held constant?</summary><p>Image IDs within each budget, input resolution, target labels, validation and test sets, metrics, and each method's settings across budgets and seeds.</p></details>`
        + `<details class="fn-details"><summary>What differs between the methods?</summary><p>Hand-crafted region features and forest fitting versus learned pixel features and gradient training. This compares practical pipelines, not isolated architectures.</p></details>`
        + `<details class="fn-details"><summary>What happens during augmentation?</summary><p>Only U-Net training receives matched flips and quarter-turn rotations, plus mild image-only brightness/contrast changes. Validation and test inputs are never augmented.</p></details>`
        + `<details class="fn-details"><summary>How is leakage controlled?</summary><p>Only training images supply features, class weights and subsets. Validation selects the U-Net checkpoint. The ${data.protocol.trained_runs} trained runs and the overlap analysis were frozen before their test evaluation.</p></details>`);
    return html;
}


function classTable(run) {
    const rows = run.metrics.per_class.map(c => `<tr><td>${esc(c.name)}</td><td>${pct(c.iou)}</td><td>${pct(c.dice)}</td><td>${pct(c.recall)}</td><td>${num(c.support)}</td></tr>`).join('');
    return `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Class</th><th>IoU</th><th>Dice</th><th>Recall</th><th>Truth pixels</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}


function inspectedRun() {
    const runs = bundle().runs;
    return runs.find(r => r.id === state.metricRun) || runs[runs.length - 1];
}

function confusionChart(options) {
    return FN_CHARTS.confusion(inspectedRun().metrics.confusion_matrix, CLASS_SHORT, options);
}


function runInspector() {
    const note = 'Rows are ground truth; each row sums to 100%. Pooled pixel counts, padding excluded. Outlined cells are correct predictions.';
    return `<div id="fn-class-metrics">${classTable(inspectedRun())}</div><div style="margin-top:20px">`
        + chartBlock('Confusion matrix', confusionChart, { heading: 'h3', svg: false, note })
        + `</div>`;
}

function overlapStrataCard() {
    const overlap = overlapInfo();
    if (!overlap || !data.summary[0]?.overlap_low) return '';
    const targets = overlap.target_images;
    const rows = bundle().summary.map(item => {
        const low = item.overlap_low.flooded_miou;
        const high = item.overlap_high.flooded_miou;
        return `<tr><td>${esc(METHOD[item.method].short)}</td><td>${item.budget}%</td><td>${meanSd(low)}</td><td>${meanSd(high)}</td><td>${points(high.mean - low.mean)}</td></tr>`;
    }).join('');
    const clean = summaryFor('unet', 100)?.clean.flooded_miou;
    return card('Does scene overlap inflate the scores?', `<p>${num(overlap.summary.test.overlapping_train)} of ${num(overlap.summary.test.images)} test images, and ${num(targets.overlapping)} of the ${num(targets.images)} that contain flooded buildings or roads, share ground with a training frame. A test set without overlap would hold almost no flooded targets, so this check splits the ${num(targets.images)} target images at their median overlap (${pct(targets.split_overlap, 0)}) and scores each half separately.</p>`
        + `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Method</th><th>Labels</th><th>Lower-overlap half (${targets.low_images} images)</th><th>Higher-overlap half (${targets.high_images} images)</th><th>Higher − lower</th></tr></thead><tbody>${rows}</tbody></table></div>`
        + `<p class="fn-figure-note">Flooded-class mIoU, mean ± SD across seeds, pooled over the target images in each half. Scores on target images are higher than on the full split because the full split also counts false alarms in the ${num(data.dataset.counts.test - targets.images)} scenes without flooded infrastructure.</p>`
        + `<div class="fn-callout">This is a weak test: most images in the lower-overlap half still share a large part of their ground with training (anything below ${pct(targets.split_overlap, 0)}), and only ${targets.images - targets.overlapping} target images have no overlap at all. It can show whether more shared ground means higher scores, but it cannot show how the models would score on a new flood area. For transparency, the ${num(overlap.summary.test.clean)} test images with no overlap (only ${targets.clean_with_building} with flooded buildings and ${targets.clean_with_road} with flooded roads) give the original U-Net (100%) ${meanSd(clean)} flooded mIoU, because false alarms across many target-free scenes weigh heavily against very few target pixels. They are listed in the all-runs CSV but not used as a leakage estimate.</div>`);
}

function resultsTakeaways() {
    const best = [...bundle().summary].filter(s => s.budget === 100).sort((a, b) => b.full.flooded_miou.mean - a.full.flooded_miou.mean)[0];
    return takeaways([
        best ? `Best at 100% labels: <strong>${esc(METHOD[best.method].name)}</strong>, ${meanSd(best.full.flooded_miou)}.` : '',
        `More labels: U-Net ${points(effectFor('unet')?.full.mean)}, SLIC + RF ${points(effectFor('rf')?.full.mean)} (25% → 100%, seed-matched).`
    ]);
}

function resultsChart() {
    const groups = [25, 100].map(budget => ({
        label: `${budget}% labels`,
        series: METHODS.map(method => {
            const effect = effectFor(method)?.full.mean;
            return {
                name: METHOD[method].short, colour: METHOD[method].colour, stat: summaryFor(method, budget)?.full.flooded_miou || { mean: null, values: [] },
                note: budget === 100 && effect != null ? `${points(effect)}` : null
            };
        })
    }));
    return vis('How much do more labels help?', options => FN_CHARTS.rangeBars(groups, { ...options, seeds: data.protocol.train_seeds, label: 'Flooded-class mean IoU by method and budget' }), `${testLabel()}. The 25% subset is nested inside the full training set. Notes above the 100% bars give the mean seed-matched change from 25%.`);
}

function budgetCard() {
    const rows = METHODS.map(method => {
        const e = effectFor(method)?.full;
        return `<tr><td>${methodPill(method)}<br>${esc(METHOD[method].name)}</td><td class="fn-score">${e?.mean == null ? 'N/A' : points(e.mean)}<small>${e ? e.values.map(points).join(' · ') : ''}</small></td></tr>`;
    }).join('');
    return card('Budget effect, seed by seed', `<p>Change in flooded-class mIoU from 25% to 100% labels, comparing runs with the same seed (${FN_MATH.inline('budget')}).</p><div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Method</th><th>Mean change (seeds ${data.protocol.train_seeds.join(', ')})</th></tr></thead><tbody>${rows}</tbody></table></div>`);
}

function runMetrics(run) {
    const cm = run.metrics.confusion_matrix;
    const total = cm.flat().reduce((a, b) => a + b, 0);
    const support = cm.map(row => row.reduce((a, b) => a + b, 0));
    const predicted = cm[0].map((_, j) => cm.reduce((sum, row) => sum + row[j], 0));
    const ratio = (n, d) => (d ? n / d : null);
    const cls = [0, 1, 2].map(c => {
        const tp = cm[c][c];
        return { iou: ratio(tp, support[c] + predicted[c] - tp), precision: ratio(tp, predicted[c]), recall: ratio(tp, support[c]), dice: ratio(2 * tp, support[c] + predicted[c]) };
    });
    const out = {
        minority: mean([cls[1].iou, cls[2].iou]), majority: cls[0].iou, miou: mean(cls.map(c => c.iou)),
        weighted: cls.reduce((sum, c, i) => sum + support[i] / total * (c.iou ?? 0), 0),
        accuracy: [0, 1, 2].reduce((sum, c) => sum + cm[c][c], 0) / total
    };
    ['building', 'road'].forEach((name, k) => Object.entries(cls[k + 1]).forEach(([key, value]) => { out[`${name}_${key}`] = value; }));
    return out;
}

function seedStat(values) {
    const defined = values.filter(v => v != null);
    const m = mean(defined);
    const sd = defined.length > 1 ? Math.sqrt(defined.reduce((a, v) => a + (v - m) ** 2, 0) / (defined.length - 1)) : null;
    return { mean: m, sd, values };
}

function configMetrics(method, budget, source) {
    const per = runsFor(method, budget, source).map(runMetrics);
    return Object.fromEntries(Object.keys(per[0]).map(key => [key, seedStat(per.map(m => m[key]))]));
}

function trivialMetrics() {
    const n = data.dataset.processed_distributions.test.pixel_counts;
    const share = n[0] / n.reduce((a, b) => a + b, 0);
    return { minority: 0, majority: share, miou: share / 3, weighted: share * share, accuracy: share };
}

function metricsCard() {
    const label = (method, budget) => `<td>${esc(METHOD[method].short)}</td><td>${budget}%</td>`;
    const cell = (stat, primary) => `<td${primary ? ' class="fn-score"' : ''}>${meanSd(stat)}${primary ? `<small>seeds: ${seedValues(stat)}</small>` : ''}</td>`;
    const rowsA = [];
    const rowsB = [];
    for (const method of METHODS) {
        for (const budget of data.protocol.budgets) {
            const m = configMetrics(method, budget, 'before');
            rowsA.push(`<tr>${label(method, budget)}${cell(m.minority, true)}${cell(m.majority)}${cell(m.miou)}${cell(m.weighted)}${cell(m.accuracy)}</tr>`);
            rowsB.push(`<tr>${label(method, budget)}${['building', 'road'].map(c => ['iou', 'precision', 'recall', 'dice'].map(k => cell(m[`${c}_${k}`])).join('')).join('')}</tr>`);
        }
    }
    const t = trivialMetrics();
    rowsA.push(`<tr class="fn-reference-row"><td colspan="2">Reference: label every pixel “Other”</td><td>${pct(t.minority)}</td><td>${pct(t.majority)}</td><td>${pct(t.miou)}</td><td>${pct(t.weighted)}</td><td>${pct(t.accuracy)}</td></tr>`);
    return card('Every metric, every configuration',
        `<p>${esc(testLabel())}; mean ± standard deviation over seeds ${data.protocol.train_seeds.join(', ')}; each run scored on its pooled pixel counts, padding excluded. <strong>Minority IoU is the primary metric</strong>; the others are reported for completeness.</p>`
        + `<h3 class="fn-vis-title">Overall</h3><div class="fn-table-wrap"><table class="fn-table fn-metrics-table"><thead><tr><th>Method</th><th>Labels</th><th>Minority IoU ★<br><small>flooded building + road</small></th><th>Majority IoU<br><small>Other</small></th><th>3-class mIoU</th><th>Weighted mIoU<br><small>by pixel share</small></th><th>Pixel accuracy</th></tr></thead><tbody>${rowsA.join('')}</tbody></table></div>`
        + `<p class="fn-figure-note">The reference row shows why majority IoU, weighted mIoU and pixel accuracy are not used to rank the models: a model that never finds a single flooded pixel already scores ${pct(t.accuracy)} accuracy and ${pct(t.weighted)} weighted mIoU, because “Other” is ${pct(t.accuracy)} of the test pixels.</p>`
        + `<h3 class="fn-vis-title">Per target class</h3><div class="fn-table-wrap"><table class="fn-table fn-metrics-table fn-class-table"><thead><tr><th rowspan="2">Method</th><th rowspan="2">Labels</th><th colspan="4">Flooded building</th><th colspan="4">Flooded road</th></tr><tr><th>IoU</th><th>Precision</th><th>Recall</th><th>Dice</th><th>IoU</th><th>Precision</th><th>Recall</th><th>Dice</th></tr></thead><tbody>${rowsB.join('')}</tbody></table></div>`
        + `<p class="fn-figure-note">Precision: share of pixels called flooded that really are (low = false alarms). Recall: share of truly flooded pixels that were found (low = misses).</p>`);
}

function metricGuideCard() {
    const rows = [
        ['Minority IoU ★', 'fmiou', 'The primary metric. How well the two rare flood classes are found and outlined; ignores “Other”.'],
        ['Flooded building IoU, flooded road IoU', 'iou', 'The same, per target class.'],
        ['Majority IoU', 'majority', 'How well the background is segmented. Always high, because “Other” is most of every image.'],
        ['3-class mIoU', 'miou3', 'Average over all three classes; the easy “Other” class lifts it.'],
        ['Weighted mIoU', 'fwiou', 'Each class weighted by its share of pixels, so it is almost the majority IoU; flattering for weak models.'],
        ['Pixel accuracy', 'acc', 'Share of all pixels labelled correctly; a model that only predicts “Other” scores very high.'],
        ['Precision', 'precision', 'Of the pixels called flooded, the share that really are: low precision means false alarms.'],
        ['Recall', 'recall', 'Of the truly flooded pixels, the share that were found: low recall means misses.'],
        ['Dice', 'dice', 'Overlap like IoU but counting the overlap twice; Dice = 2·IoU / (1 + IoU), so it ranks models the same way.']
    ];
    return card('What each metric means', `<div class="fn-table-wrap"><table class="fn-table fn-guide-table"><thead><tr><th>Metric</th><th>Formula</th><th>What it tells you</th></tr></thead><tbody>`
        + rows.map(([name, key, text]) => `<tr><td><strong>${esc(name)}</strong></td><td>${FN_MATH.inline(key)}</td><td>${esc(text)}</td></tr>`).join('')
        + `</tbody></table></div><p class="fn-figure-note">${FN_MATH.tex(String.raw`TP_c,\ FP_c,\ FN_c`)}: pooled true positives, false positives and false negatives of class ${FN_MATH.tex('c')} over the test images; ${FN_MATH.tex('n_c')}: true pixels of class ${FN_MATH.tex('c')}; ${FN_MATH.tex('N')}: all scored pixels.</p>`);
}

function deltaStat(method, budget, base, other, key) {
    const baseRuns = runsFor(method, budget, base);
    return seedStat(runsFor(method, budget, other).map(r => {
        const match = baseRuns.find(x => x.seed === r.seed);
        return match ? runMetrics(r)[key] - runMetrics(match)[key] : null;
    }));
}

function shiftTable(base, other, methods = METHODS) {
    const rows = methods.flatMap(method => data.protocol.budgets.map(budget => {
        if (!runsFor(method, budget, other).length || !runsFor(method, budget, base).length) return '';
        const c = comparisonFor(method, budget, other);
        const b = configMetrics(method, budget, base);
        const o = configMetrics(method, budget, other);
        const cell = (key, primary) => {
            const d = deltaStat(method, budget, base, other, key);
            return `<td${primary ? ' class="fn-score"' : ''}>${pct(b[key].mean)} → <strong>${pct(o[key].mean)}</strong>${primary ? `<small>Δ ${points(d.mean)} (${d.values.map(points).join(', ')})</small>` : `<span class="fn-seed-values"><br>Δ ${points(d.mean)}</span>`}</td>`;
        };
        return `<tr><td><span class="fn-dot" style="background:${METHOD[method].colour}"></span>${esc(METHOD[method].short)}</td><td>${budget}%</td>${cell('minority', true)}<td class="fn-nowrap">${c ? interval(c.ci) : 'N/A'}</td><td>${c ? decisionPill(c.decision) : ''}</td>${cell('majority')}${cell('weighted')}</tr>`;
    })).join('');
    return `<div class="fn-table-wrap"><table class="fn-table fn-metrics-table fn-shift-table"><thead><tr><th>Method</th><th>Labels</th><th>Minority IoU ★<br><small>first version → current</small></th><th>95% CI</th><th>Decision</th><th>Majority IoU<br><small>Other</small></th><th>Weighted mIoU<br><small>by pixel share</small></th></tr></thead><tbody>${rows}</tbody></table></div>`
        + `<p class="fn-figure-note">Mean over seeds ${data.protocol.train_seeds.join(', ')}; the number under each pair is the mean seed-matched change. The interval and decision (pre-registered paired bootstrap, ${num(5000)} resamples) apply to the primary metric, minority IoU; the other columns are descriptive.</p>`;
}

function firstVersionCard() {
    const n = runsFor('rf', 100, 'original')[0]?.region_class_counts;
    const u = runsFor('rf', 100)[0]?.region_class_counts;
    return card('First version of the Random Forest (kept for reference)', `<p>The first results did not treat the two models the same way. The U-Net used capped inverse-frequency class weights on every training pixel. The Random Forest used sklearn's uncapped “balanced” weights and a ${num(data.protocol.rf_max_regions)}-region training cap${n && u ? `, which at 100% labels kept only ${num(n[0])} of ${num(u[0])} “Other” regions` : ''}. Comparing the two models then mixes up the model with its imbalance handling.</p>`
        + `<p>The pipeline now gives the forest the U-Net's own class weights and every training region. The U-Net runs did not change; only the six forests were retrained. The first-version forest stays here for reference:</p>`
        + shiftTable('original', 'before', ['rf']));
}

function resultsPage() {
    if (!data.summary.length) return header('results', 'The held-out comparison.') + notice() + empty();
    let html = header('results', 'The held-out comparison.', 'Primary metric: mean IoU of flooded building and flooded road (minority IoU), pooled over the test images. Each configuration is run with three seeds.');
    html += notice();
    html += resultsTakeaways();
    html += metricsCard();
    html += firstVersionCard();
    html += metricGuideCard();
    html += resultsChart();
    html += budgetCard();
    html += overlapStrataCard();
    const runs = bundle().runs;
    if (!runs.some(r => r.id === state.metricRun)) state.metricRun = runs[runs.length - 1].id;
    const options = runs.map(r => `<option value="${esc(r.id)}" ${r.id === state.metricRun ? 'selected' : ''}>${esc(METHOD[r.method].short)} · ${r.budget}% · seed ${r.seed}</option>`).join('');
    html += `<section class="fn-card"><div class="fn-card-top"><h2>Inspect a single run</h2><div class="fn-control"><label for="fn-metric-run">Run</label><select id="fn-metric-run">${options}</select></div></div>${runInspector()}</section>`;
    html += card('Download the numbers', `<p>Per-seed and per-image scores of the first-version runs, as CSV (the U-Net rows match this page; the forest rows are the first version).</p><div class="fn-downloads">${download('summary', 'Seed summary CSV')}${download('aggregate', 'All runs CSV')}${download('per_image', 'Per-image metrics CSV')}</div>`);
    return html;
}


function trainingPage() {
    const nets = data.runs.filter(r => r.method === 'unet');
    let html = header('training', 'Quality has a computational cost.', 'The traditional baseline runs on the CPU; every U-Net run trains on the GPU. Timings describe this laptop.');
    html += notice();
    if (!nets.length) return html + empty('Training curves await completed U-Net runs');
    
    const seeds = data.protocol.train_seeds;
    const lossSeed = state.lossSeed ?? seeds[0];
    html += `<div class="fn-grid fn-two-col">${[25, 100].map(budget => {
        const series = runsFor('unet', budget).map((r, i) => ({
            label: `seed ${r.seed}`, colour: SEED_COLOURS[i % SEED_COLOURS.length],
            values: r.history.map(e => e.val_flooded_miou), marker: r.best_epoch - 1,
            markerLabel: `s${r.seed} · epoch ${r.best_epoch} · ${pct(r.history[r.best_epoch - 1]?.val_flooded_miou)}`
        }));
        const note = 'Validation flooded mIoU per epoch; notes give each seed\'s selected checkpoint. Its epoch-to-epoch swings are why each configuration has three seeds.';
        return vis(`U-Net · ${budget}% labels: validation score`, options => FN_CHARTS.lines(series, { ...options, max: 1, percent: true, label: `Validation flooded mIoU, ${budget}% labels`, note: 'Circle = selected checkpoint' }), note);
    }).join('')}</div>`;
    const seedButtons = seeds.map(s => `<button type="button" class="${s === lossSeed ? 'on' : ''}" data-loss-seed="${s}" aria-pressed="${s === lossSeed}">Seed ${s}</button>`).join('');
    html += `<section class="fn-card"><div class="fn-card-top"><h2>Training and validation loss</h2><div class="fn-seg" role="group" aria-label="Seed">${seedButtons}</div></div><div class="fn-grid fn-two-col">${[25, 100].map(budget => {
        const series = [];
        let vline = null;
        let lowestPoint = null;
        const run = runsFor('unet', budget).find(r => r.seed === lossSeed);
        if (run) {
            const validation = run.history.map(e => e.val_loss);
            series.push({ label: 'training', colour: '#4f46e5', values: run.history.map(e => e.train_loss), dash: false });
            series.push({ label: 'validation', colour: '#d97706', values: validation, dash: true });
            vline = { index: run.best_epoch - 1, label: `selected epoch ${run.best_epoch}` };
            const lowest = validation.indexOf(Math.min(...validation));
            lowestPoint = { series: series.length - 1, index: lowest, text: `min val loss ${validation[lowest].toFixed(3)}` };
        }
        if (!series.length) return empty('Run not exported');
        const render = options => FN_CHARTS.lines(series, { ...options, label: `Loss curves, ${budget}% labels, seed ${lossSeed}`, vline, points: lowestPoint ? [lowestPoint] : [] });
        return `<div>${chartBlock(`${budget}% labels · seed ${lossSeed}`, render, { heading: 'h3' })}<p class="fn-small fn-muted">The checkpoint is chosen by validation flooded mIoU, not by loss.</p></div>`;
    }).join('')}</div></section>`;
    const rows = bundle().summary.map(item => {
        const s = summaryFor(item.method, item.budget);
        const runs = runsFor(item.method, item.budget);
        const peak = Math.max(...runs.map(r => r.peak_gpu_allocated_bytes ?? -1));
        const device = runs[0]?.device || 'N/A';
        const training = `${seconds(s.training_seconds.mean)}<span class="fn-seed-values"><br>${seedValues(s.training_seconds, seconds)}</span>`;
        const epochs = s.epochs?.mean != null ? `${s.epochs.mean.toFixed(1)} (${s.epochs.min}–${s.epochs.max})` : 'N/A';
        return `<tr><td>${esc(METHOD[item.method].short)} · ${item.budget}%</td><td>${training}</td>`
            + `<td>${(s.inference_median_seconds.mean * 1000).toFixed(1)} ms</td><td>${epochs}</td>`
            + `<td>${peak < 0 ? 'N/A' : `${(peak / 2 ** 20).toFixed(0)} MiB`}</td><td>${esc(device)}${runs[0]?.precision ? `<br><small>${esc(runs[0].precision.replace('torch.', ''))}</small>` : ''}</td></tr>`;
    }).join('');
    html += card('Measured resources', `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Configuration</th><th>Training time (mean)</th><th>Median inference</th><th>Epochs</th><th>Peak GPU allocated</th><th>Device</th></tr></thead><tbody>${rows}</tbody></table></div>`);
    let timing = `<p>RF training time covers forest fitting. U-Net training time includes per-epoch validation and checkpointing. Shared preprocessing and RF feature extraction are recorded separately.</p>`
        + `<p>Inference is one image at a time, after warm-up. It starts from the processed RGB in memory and ends with a CPU mask. RF includes fresh SLIC segmentation and feature extraction (single-threaded forest prediction); U-Net includes normalisation, GPU transfer, the forward pass and argmax. Decoding the original photo is excluded for both.</p>`;
    timing += `<p>The retrained forests were trained while other runs shared the CPU and GPU, so their timings are less precise than those of the first-version runs.</p>`;
    if (data.dataset) timing += `<p>Shared preparation: <strong>${seconds(data.dataset.preprocessing_seconds)}</strong>; RF feature extraction: <strong>${seconds(data.dataset.feature_preparation_seconds)}</strong>.</p>`;
    html += card('What is included in the timings?', timing);
    return html;
}


function explorerPage() {
    let html = header('explorer', 'Look beyond the average.', `Every image comes from the held-out test split. The same image and budget are shown for both approaches.`);
    if (!data.samples.length) return html + notice() + empty('Prediction gallery pending', 'The gallery appears after data preparation and final evaluation.');
    const presence = [['all', 'All images'], ['building', 'Flooded building'], ['road', 'Flooded road'], ['both', 'Both targets'], ['neither', 'Neither target']];
    const overlap = [['all', 'All test images'], ['clean', 'No training overlap'], ['overlapping', 'Overlaps training']];
    const sorts = [['id', 'Image ID'], ['worst-unet', 'U-Net: lowest target IoU'], ['worst-rf', 'RF: lowest target IoU'], ['coverage', 'Target pixel share'], ['overlap', 'Largest training overlap']];
    if (!sorts.some(([v]) => v === state.sort)) state.sort = 'id';
    const select = (id, label, options, value) => `<div class="fn-control"><label for="${id}">${label}</label><select id="${id}">${options.map(([v, l]) => `<option value="${v}" ${v === value ? 'selected' : ''}>${l}</option>`).join('')}</select></div>`;
    html += `<section class="fn-card"><div class="fn-controls">`
        + `<div class="fn-control"><span class="fn-control-label">Training budget</span>${budgetToggle()}</div>`
        + select('fn-presence', 'Ground-truth targets', presence, state.presence)
        + (overlapInfo() ? select('fn-overlap', 'Training overlap', overlap, state.overlap) : '')
        + select('fn-sort', 'Sort by', sorts, state.sort)
        + `<div class="fn-control fn-control-grow"><label for="fn-search">Find image</label><input id="fn-search" type="search" placeholder="Image ID…" value="${esc(state.search)}"></div></div>`
        + `<div class="fn-overlay-controls"><label><input id="fn-overlay" type="checkbox" ${state.overlay ? 'checked' : ''}>Overlay on RGB</label>`
        + `<label for="fn-opacity">Opacity <input id="fn-opacity" type="range" min="0" max="1" step="0.05" value="${state.opacity}"><output id="fn-opacity-value">${Math.round(state.opacity * 100)}%</output></label>`
        + data.classes.map(c => `<label><input type="checkbox" data-class="${c.id}" ${state.visible[c.id] ? 'checked' : ''}><i style="display:inline-block;width:10px;height:10px;border-radius:3px;background:${c.colour}"></i>${esc(c.name)}</label>`).join('')
        + `</div><div id="fn-viewer" aria-live="polite"></div>`
        + `<p class="fn-figure-note">Predictions come from the seed-${data.protocol.explorer_seed} replicate; per-seed scores for every image are in the per-image CSV. Masks are at the processed resolution without padding. Target pixel share is an image statistic, not physical area.</p></section>`
        + `<section class="fn-card"><div class="fn-card-top"><h2>Held-out image library</h2><span id="fn-sample-count" class="fn-pill"></span></div><div id="fn-gallery" class="fn-sample-grid"></div><div id="fn-pager" class="fn-pager"></div></section>`;
    return html;
}

function filteredSamples() {
    const search = state.search.trim().toLowerCase();
    const presence = {
        all: () => true, building: s => s.presence[0], road: s => s.presence[1],
        both: s => s.presence.every(Boolean), neither: s => s.presence.every(x => !x)
    }[state.presence];
    const overlap = {
        all: () => true, clean: s => !s.overlapping_train_ids?.length, overlapping: s => s.overlapping_train_ids?.length > 0
    }[state.overlap];
    const output = data.samples.filter(s => s.image_id.toLowerCase().includes(search) && presence(s) && overlap(s));
    const byId = (a, b) => a.id.localeCompare(b.id, undefined, { numeric: true });
    if (state.sort.startsWith('worst-')) {
        const run = explorerRun(state.sort.slice(6), state.budget);
        const score = s => (s.presence.some(Boolean) ? sampleMetrics(s, run?.id)?.flooded_miou : null) ?? 2;
        output.sort((a, b) => score(a) - score(b) || byId(a, b));
    } else if (state.sort === 'coverage') {
        output.sort((a, b) => b.target_fraction - a.target_fraction || byId(a, b));
    } else if (state.sort === 'overlap') {
        output.sort((a, b) => (b.train_overlap ?? 0) - (a.train_overlap ?? 0) || byId(a, b));
    } else {
        output.sort(byId);
    }
    return output;
}

function targetText(sample) {
    if (!sample.presence.some(Boolean)) return 'Neither target';
    return sample.presence.map((x, i) => (x ? ['Building', 'Road'][i] : '')).filter(Boolean).join(' + ');
}

function explorerFigures() {
    const panels = [{ title: 'Original RGB', kind: 'rgb' }, { title: 'Ground truth', kind: 'truth' }];
    for (const method of METHODS) panels.push({ title: METHOD[method].short, kind: method, method, run: explorerRun(method, state.budget) });
    return panels;
}

function updateExplorer() {
    if (!$('#fn-viewer')) return;
    const samples = filteredSamples();
    const pages = Math.max(1, Math.ceil(samples.length / PAGE_SIZE));
    state.page = Math.min(state.page, pages - 1);
    if (!samples.some(s => s.id === state.selected)) state.selected = samples[0]?.id ?? null;
    const selected = samples.find(s => s.id === state.selected);
    $('#fn-sample-count').textContent = `${samples.length} / ${data.samples.length} images`;
    $('#fn-gallery').innerHTML = samples.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE).map(s => `<button type="button" class="fn-sample-card ${s.id === state.selected ? 'selected' : ''}" data-sample="${esc(s.id)}" aria-label="Inspect test image ${esc(s.image_id)}">`
        + `<img src="${esc(s.rgb)}" loading="lazy" alt="Aerial image ${esc(s.image_id)}"><span>${esc(s.image_id)}<small>${targetText(s)}</small></span></button>`).join('');
    $('#fn-pager').innerHTML = `<button type="button" class="fn-btn" data-page-step="-1" ${state.page === 0 ? 'disabled' : ''}>← Previous</button><span>Page ${state.page + 1} / ${pages}</span>`
        + `<button type="button" class="fn-btn" data-page-step="1" ${state.page + 1 >= pages ? 'disabled' : ''}>Next →</button>`;
    if (!selected) {
        $('#fn-viewer').innerHTML = empty('No matching images', 'Change the filters or the image-ID search.');
        return;
    }
    const overlapBadge = selected.overlapping_train_ids == null ? ''
        : selected.overlapping_train_ids.length
            ? `<span class="fn-pill warn" title="Verified overlap with ${esc(selected.overlapping_train_ids.join(', '))}">Overlaps training · ${pct(selected.train_overlap, 0)}</span>`
            : '<span class="fn-pill good">No training overlap</span>';
    const figures = explorerFigures();
    const sharp = FN_VIEWER.hiresUrl(selected.id) ? 'sharp 1024 px photo' : '256 px photo';
    $('#fn-viewer').innerHTML = `<div class="fn-sample-title"><h2>Image ${plain(selected.image_id)}</h2><div class="fn-sample-meta"><span class="fn-pill">${targetText(selected)}</span>${overlapBadge}<span>${pct(selected.target_fraction)} target pixels · ${selected.geometry.width}×${selected.geometry.height} input</span></div></div>`
        + `<div class="fn-comparison">${figures.map(f => {
            const iou = f.run ? sampleMetrics(selected, f.run.id)?.flooded_miou : null;
            const score = f.run ? `<span class="fn-figure-score">IoU ${pct(iou)}</span>` : '';
            const caption = [`Image ${selected.image_id}`, f.run ? `${state.budget}% labels · seed ${f.run.seed} · flooded IoU ${pct(iou)}` : '', sharp].filter(Boolean).join(' · ');
            const attrs = { group: 'explorer', title: f.title, caption, render: 'explorer', kind: f.kind };
            const body = f.kind === 'rgb' ? zoomable(`<img src="${esc(selected.rgb)}" alt="RGB image ${esc(selected.image_id)}">`, `${f.title}, image ${selected.image_id}`, attrs)
                : f.kind === 'truth' || f.run ? zoomable(`<canvas data-mask-kind="${f.kind}" aria-label="${esc(f.title)} for image ${esc(selected.image_id)}"></canvas>`, `${f.title}, image ${selected.image_id}`, { ...attrs, pixelated: '' })
                    : empty('Run not exported', '');
            return `<figure><figcaption><span class="fn-caption-name">${esc(f.title)}</span>${score}</figcaption>${body}</figure>`;
        }).join('')}</div><p class="fn-small fn-muted fn-viewer-hint">Select any panel to expand it. In the expanded view, ← / → flips between the ${figures.length} panels at the same zoom, so you can compare the same pixels.</p>`;
    drawMasks(selected);
}

function loadImage(url) {
    if (!imageCache.has(url)) {
        imageCache.set(url, new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => resolve(image);
            image.onerror = () => reject(new Error(`Cannot load ${url}`));
            image.src = url;
        }));
    }
    return imageCache.get(url);
}

function maskUrl(sample, kind) {
    if (kind === 'truth') return sample.truth;
    const run = explorerRun(kind, state.budget);
    if (!run) return null;
    return sampleData(sample, 'before')?.predictions[run.id];
}

function renderPanel(canvas, photo, mask, width, height) {
    const palette = data.classes.map(c => c.colour.match(/[0-9a-f]{2}/gi).map(x => parseInt(x, 16)));
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    context.fillStyle = '#e9eef2';
    context.fillRect(0, 0, width, height);
    if (state.overlay || !mask) {
        context.imageSmoothingEnabled = true;
        context.imageSmoothingQuality = 'high';
        context.drawImage(photo, 0, 0, width, height);
    }
    if (!mask) return canvas;
    const scratch = document.createElement('canvas');
    scratch.width = mask.naturalWidth;
    scratch.height = mask.naturalHeight;
    const scratchContext = scratch.getContext('2d');
    scratchContext.drawImage(mask, 0, 0);
    const pixels = scratchContext.getImageData(0, 0, scratch.width, scratch.height);
    for (let i = 0; i < pixels.data.length; i += 4) {
        const label = pixels.data[i];
        const colour = palette[label] || [0, 0, 0];
        const alpha = state.overlay ? state.opacity * (label === 0 ? 0.16 : 1) : 1;
        pixels.data[i] = colour[0];
        pixels.data[i + 1] = colour[1];
        pixels.data[i + 2] = colour[2];
        pixels.data[i + 3] = state.visible[label] ? Math.round(255 * alpha) : 0;
    }
    scratchContext.putImageData(pixels, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(scratch, 0, 0, width, height);
    return canvas;
}

FN_VIEWER.renderers.explorer = async trigger => {
    const sample = data.samples.find(s => s.id === state.selected);
    const kind = trigger.dataset.zoomKind;
    const sharp = FN_VIEWER.hiresUrl(sample.id);
    const photo = await loadImage(sharp || sample.rgb);
    const url = kind === 'rgb' ? null : maskUrl(sample, kind);
    const mask = url ? await loadImage(url) : null;
    const scale = sharp ? 1 : 4;
    const canvas = document.createElement('canvas');
    canvas.setAttribute('aria-label', trigger.getAttribute('aria-label').replace('Expand: ', ''));
    return renderPanel(canvas, photo, mask, photo.naturalWidth * scale, photo.naturalHeight * scale);
};

async function drawMasks(sample) {
    const token = ++drawToken;
    try {
        const rgb = await loadImage(sample.rgb);
        await Promise.all($$('canvas[data-mask-kind]').map(async canvas => {
            const url = maskUrl(sample, canvas.dataset.maskKind);
            if (!url) return;
            const mask = await loadImage(url);
            if (token !== drawToken) return;
            renderPanel(canvas, rgb, mask, rgb.naturalWidth, rgb.naturalHeight);
        }));
    } catch (error) {
        if (token === drawToken) {
            const message = document.createElement('p');
            message.className = 'fn-error';
            message.textContent = error.message;
            $('#fn-viewer')?.append(message);
        }
    }
}

function redrawSelected() {
    const sample = data.samples.find(s => s.id === state.selected);
    if (sample) drawMasks(sample);
}


function interpretation() {
    let html = header('interpretation', 'What do the findings support?', 'Read the measured trade-offs together with the limits of the dataset, the resolution and the experiment design.');
    const B = bundle('before');
    const complete = B.summary.length === METHODS.length * 2;
    if (!complete) {
        html += card('Data efficiency and method trade-offs', empty('Interpretation awaits final results', 'Findings are computed from the exported comparison.'));
    } else {
        const full = (method, budget) => summaryFor(method, budget).full;
        const gap = full('unet', 100).flooded_miou.mean - full('rf', 100).flooded_miou.mean;
        const rf = effectFor('rf');
        const unet = effectFor('unet');
        const describe = effect => {
            const e = effect.full;
            const all = e.values.every(v => v > 0) ? 'every seed improved' : e.values.every(v => v < 0) ? 'every seed got worse' : 'seeds disagree on the direction';
            return `${points(e.mean)} on average (${all}; range ${points(e.min)} to ${points(e.max)})`;
        };
        const recall = (method, key) => `${pct(full(method, 25)[key].mean, 0)} → ${pct(full(method, 100)[key].mean, 0)}`;
        const best = [...B.summary].filter(s => s.budget === 100).sort((a, b) => b.full.flooded_miou.mean - a.full.flooded_miou.mean)[0];
        const unetBeatsRf = runsFor('unet', 25).concat(runsFor('unet', 100)).every(u => B.runs.filter(r => r.method === 'rf').every(r => u.metrics.flooded_miou > r.metrics.flooded_miou));
        html += `<div class="fn-dark-card"><h3>Headline</h3><p>With all training labels, the U-Net scores <strong>${points(gap).replace('+', '')}</strong> higher flooded-class mIoU than SLIC + Random Forest on the official test split (mean of ${data.protocol.train_seeds.length} seeds)${unetBeatsRf ? '; every U-Net run beats every RF run' : ''}. The best approach at 100% labels is <strong>${esc(METHOD[best.method].name)}</strong> at ${meanSd(best.full.flooded_miou)}.` + `</p></div>`;
        const rfGains = rf.full.values.every(v => v > 0) && rf.full.mean > 0.01;
        const rfText = rfGains
            ? `With the same class weights as the U-Net and every training region, SLIC + RF also profits from the extra labels: building recall goes ${recall('rf', 'building_recall')} and road recall ${recall('rf', 'road_recall')}. Its ceiling stays low, though: superpixel colour and texture cannot separate open floodwater from a flooded road.`
            : `SLIC + RF does use the extra labels: building recall goes ${recall('rf', 'building_recall')} and road recall ${recall('rf', 'road_recall')}. It also raises more false alarms, mostly in scenes without flooded infrastructure, so IoU does not improve. Superpixel colour and texture seem to be the limit: many RF false alarms fall on open floodwater, which looks much like a flooded road.`;
        html += card('Data efficiency', `<p class="fn-insight">Going from 25% to 100% of the labels changes the U-Net by ${describe(unet)}, and SLIC + RF by ${describe(rf)}.</p>`
            + `<p>These are seed-matched differences: the same seed is trained at both budgets, so the comparison isolates the labelled-data change. The U-Net gain varies a lot between seeds, so a single run could have under- or over-stated it. Three seeds show stability, not statistical significance.</p>`
            + `<p>${rfText}</p>`
            + (`<p class="fn-small fn-muted">The first-version forest (balanced weights, ${num(data.protocol.rf_max_regions)}-region cap) changed by ${points(effectFor('rf', 'original').full.mean)} with more labels: at 100% its cap discarded most of the extra “Other” regions. See <a href="#results">Results</a>.</p>`));
        const overlap = overlapInfo();
        const strata = summaryFor('unet', 100).overlap_low && summaryFor('unet', 100).overlap_high;
        if (overlap && strata) {
            const diff = summaryFor('unet', 100).overlap_high.flooded_miou.mean - summaryFor('unet', 100).overlap_low.flooded_miou.mean;
            html += card('Does scene overlap inflate the scores?', `<p>${num(overlap.target_images.overlapping)} of the ${num(overlap.target_images.images)} test images with flooded targets share ground with a training frame, so the official split mostly tests areas the models have already seen. Results should not be read as performance on a new flood area.</p>`
                + `<p>Within the target images, the higher-overlap half scores ${points(diff)} relative to the lower-overlap half for the original U-Net (100%). ${diff <= 0.02 ? 'More shared ground did not mean higher scores, so there is no sign that the ranking or the budget effect depends on memorised scenes.' : 'More shared ground went with higher scores, so some of the measured quality may come from memorised scenes.'} The check is weak, though, because almost no flooded test scene is free of overlap.</p>`);
        }
    }
    const reps = data.representatives.map(id => data.samples.find(s => s.id === id)).filter(Boolean);
    html += card('Six representative scenes', `<p>Chosen by target presence and pixel coverage, never by prediction quality. Select a photo to expand it, or open the scene in the explorer or the step-by-step walkthrough.</p>`
        + (reps.length ? `<div class="fn-rep-grid">${reps.map(s => `<div class="fn-rep">${zoomable(`<img src="${esc(s.rgb)}" loading="lazy" alt="Representative image ${esc(s.image_id)}">`, `representative image ${s.image_id}`, { group: 'representatives', title: `Image ${s.image_id}`, hires: s.id, caption: `${targetText(s)} · ${pct(s.target_fraction)} target pixels` })}`
            + `<span>Image ${plain(s.image_id)} · ${pct(s.target_fraction)} target pixels<br><a href="#explorer" data-representative="${esc(s.id)}">Explorer →</a> <a href="#walkthrough" data-walkthrough="${esc(s.id)}">Step by step →</a></span></div>`).join('')}</div>` : empty('Representative scenes pending')));
    html += card('Failure cases deserve a closer look', `<p>In the explorer, sort by a model's lowest target IoU to examine missed or confused infrastructure.</p><a class="fn-primary-btn" href="#explorer" data-failures="unet">Inspect U-Net failure cases →</a>`);
    html += card('Limits on interpretation', `<ul>${data.limitations.map(x => `<li>${esc(x)}</li>`).join('')}${'<li>The forest was revised after its first test evaluation so that both models handle the imbalance the same way; only the forest changed, the rule was fixed in advance, and the first version is kept and reported.</li>'}</ul><p>SDG 11 and SDG 13 motivate the study. Real-world benefit for responders would need separate validation.</p>`);
    return html;
}

function reproducibility() {
    let html = header('reproducibility', 'Every result has a trail.', 'Saved configurations, fixed subset IDs, model hashes and evaluation artifacts connect this page to the underlying experiment.');
    html += card('Download the experiment records', `<div class="fn-downloads">${download('summary', 'Seed summary CSV')}${download('aggregate', 'All runs CSV')}${download('per_image', 'Per-image metrics CSV')}${download('overlap', 'Verified overlap pairs CSV')}${download('budgets', 'Exact budget image IDs')}${download('experiment', 'Frozen 12-run manifest')}</div>${Object.keys(data.downloads).length ? '' : empty('Downloads appear after export')}`);
    const p = data.protocol;
    html += card('Protocol at a glance', `<dl class="fn-definition-list"><dt>Export schema</dt><dd>Version ${data.schema_version}</dd><dt>Subset seed</dt><dd>${p.subset_seed}</dd><dt>Training seeds</dt><dd>${p.train_seeds.join(', ')}</dd>`
        + `<dt>Model input</dt><dd>${p.size}×${p.size}, aspect ratio preserved</dd><dt>Budgets</dt><dd>Nested 25% / 100%, stratified by target presence</dd>`
        + (`<dt>RF imbalance handling</dt><dd>The U-Net's capped class weights; every training region (first version: “balanced” weights, cap ${num(p.rf_max_regions)})</dd>`)
        + ('')
        + `<dt>Ignored label</dt><dd>255 (generated padding)</dd><dt>Exported</dt><dd>${plain(new Date(data.exported_at).toLocaleString())}</dd></dl>`);
    if (data.experiment_identity) {
        html += card('Frozen experiment identities', `<p class="fn-small fn-muted">12 trained runs and the overlap analysis</p><p class="fn-hash">${esc(data.experiment_identity)}</p>`
            + (`<p class="fn-small fn-muted">The six retrained forests, frozen ${plain(new Date(uniform.frozen_at).toLocaleString())}</p><p class="fn-hash">${esc(uniform.experiment_identity)}</p>`)
            + `<p>Checkpoints stay in the local experiment workspace. This static page reads only exported display files.</p>`);
    }
    const rows = bundle().runs.map(r => `<tr><td>${esc(r.id.replace(/^undersample\/uniform\//, ''))}</td><td>${esc(METHOD[r.method].short)}</td><td>${r.budget}%</td><td>${r.seed}</td><td>${pct(r.metrics.flooded_miou)}</td><td><span class="fn-hash" style="display:inline-block;padding:3px 6px">${esc(r.checkpoint_sha256.slice(0, 16))}…</span></td></tr>`).join('');
    if (rows) html += card('All runs', `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Run</th><th>Method</th><th>Labels</th><th>Seed</th><th>Flooded mIoU</th><th>Checkpoint SHA-256</th></tr></thead><tbody>${rows}</tbody></table></div>`);
    if (data.reproducibility) {
        html += card('Configuration and software', `<details class="fn-details"><summary>Saved configuration</summary><pre class="fn-code">${esc(JSON.stringify(data.reproducibility.config, null, 2))}</pre></details>`
            + `<details class="fn-details"><summary>Software versions</summary><pre class="fn-code">${esc(JSON.stringify(data.reproducibility.versions, null, 2))}</pre></details>`);
    }
    html += card('Sources and attribution', `<ul>${data.sources.map(s => `<li><a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a></li>`).join('')}</ul><p>Dataset: Rahnemoonfar et al., FloodNet, IEEE Access (2021). The project README lists every command needed to reproduce these results.</p>`);
    return html;
}


const PAGES = {
    overview, dataset: datasetPage, preprocessing: () => FN_PREP.page(), methodology, walkthrough: () => FN_WALKTHROUGH.page(),
    results: resultsPage, training: trainingPage,
    explorer: explorerPage, interpretation, reproducibility,
    summary: () => FN_PRES.summary(), problem: () => FN_PRES.problem(), data: () => FN_PRES.data_(), prepare: () => FN_PRES.prepare(), features: () => FN_PRES.features(),
    method: () => FN_PRES.method(), evaluate: () => FN_PRES.evaluate(), sustain: () => FN_PRES.sustain(), limits: () => FN_PRES.limits(), sources: () => FN_PRES.sources()
};

function renderNav() {
    const items = navItems();
    $('#fn-nav').innerHTML = navGroups().map(g => `<p class="fn-nav-label">${esc(g.group)}</p>`
        + items.filter(item => item.group === g.group).map(item => `<a class="fn-nav-btn" href="#${item.route}"><span>${item.number}</span>${esc(item.label)}</a>`).join('')).join('');
    const route = currentRoute();
    $$('.fn-nav-btn').forEach(link => link.classList.toggle('active', link.hash === `#${route}`));
}

function updateStatus() {
    const status = $('#fn-export-status');
    if (!status || !data) return;
    if (state.mode === 'presentation') {
        status.textContent = 'SLIC + RF and U-Net';
        status.parentElement.classList.add('complete');
        return;
    }
    status.textContent = data.status === 'complete'
        ? `${data.runs.length} / ${data.protocol.expected_runs} evaluations complete`
        : 'Experiment in progress';
    status.parentElement.classList.toggle('complete', data.status === 'complete');
}


function syncViewButtons() {
    $$('[data-mode]').forEach(button => {
        const on = button.dataset.mode === state.mode;
        button.classList.toggle('on', on);
        button.setAttribute('aria-pressed', String(on));
    });
    const note = $('#fn-view-note');
    if (note) {
        note.textContent = state.mode === 'presentation'
            ? 'Presentation: the report deliverables only, simplified, for SLIC + RF and U-Net.'
            : `Full details: every result. ${beforeNote()} Three classes, 256 px.`;
    }
    const copy = $('#fn-sidebar-copy');
    if (copy) copy.textContent = `Two approaches. Two data budgets. Three seeds. One shared protocol.`;
    renderNav();
    updateStatus();
}

function render(scrollTop = true) {
    const owner = modeOf(currentRoute());
    if (!owner) history.replaceState(null, '', `#${homeRoute()}`);
    else if (owner !== state.mode) {
        state.mode = owner;
        saveMode();
        syncViewButtons();
    }
    const route = currentRoute();
    const page = PAGES[route] || overview;
    chartRenderers.clear();
    FN_CHARTS.hideTip();
    $('#fn-page').innerHTML = page();
    $$('.fn-nav-btn').forEach(link => {
        const active = link.hash === `#${route}`;
        link.classList.toggle('active', active);
        if (active) link.setAttribute('aria-current', 'page');
        else link.removeAttribute('aria-current');
    });
    closeSidebar();
    if (route === 'explorer') updateExplorer();
    if (route === 'walkthrough') FN_WALKTHROUGH.mount();
    if (route === 'preprocessing') FN_PREP.mount();
    if (scrollTop) window.scrollTo({ top: 0, behavior: 'instant' });
}

function readable(node) {
    const copy = node.cloneNode(true);
    copy.querySelectorAll('.fn-tex').forEach(t => t.replaceWith(t.dataset.plain));
    return copy.textContent.replace(/\s+/g, ' ').trim();
}

function buildToc() {
    const toc = $('#fn-page [data-toc]');
    if (!toc) return;
    const headings = $$('#fn-page section.fn-card > h2, #fn-page section.fn-card > .fn-card-top > h2');
    const titles = headings.map(readable);
    const signature = titles.join('|');
    if (toc.dataset.signature === signature) return;
    toc.dataset.signature = signature;
    toc.hidden = headings.length < 5;
    if (toc.hidden) return;
    headings.forEach((h, i) => { h.id = `fn-section-${i}`; });
    toc.innerHTML = `<span class="fn-toc-label">On this page</span>` + titles.map((title, i) => `<button type="button" class="fn-toc-link" data-toc-target="fn-section-${i}">${esc(title)}</button>`).join('');
}

let postQueued = false;
function postRender() {
    postQueued = false;
    buildToc();
    FN_MATH.typeset($('#fn-page'));
    $$('dialog.fn-lightbox[open]').forEach(dialog => FN_MATH.typeset(dialog));
}

new MutationObserver(records => {
    if (records.every(r => r.target.closest?.('.fn-tooltip'))) return;
    if (!postQueued) {
        postQueued = true;
        requestAnimationFrame(postRender);
    }
}).observe(document.body, { childList: true, subtree: true });

function closeSidebar() {
    $('#fn-sidebar').classList.remove('open');
    $('.fn-mobile-menu').setAttribute('aria-expanded', 'false');
}

function downloadChart(button) {
    const scope = button.closest('[data-chart-block], .fn-lightbox, .fn-card');
    const svg = $('svg.fn-chart', scope);
    if (!svg) return;
    const copy = svg.cloneNode(true);
    copy.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    copy.querySelectorAll('.fn-off, .fn-crosshair, .fn-hit').forEach(node => node.remove());
    if (scope.classList.contains('fn-notes-off')) copy.querySelectorAll('.fn-note').forEach(node => node.remove());
    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
    style.textContent = 'text{font-family:Arial,sans-serif;font-size:11px;fill:#64748b}.fn-chart-value{fill:#0f172a;font-weight:700}'
        + '.fn-axis{stroke:#cbd5e1}.fn-note{fill:#334155;font-weight:700;font-size:11px}.fn-note-tag{fill:#fff;font-weight:800;font-size:10px}'
        + '.fn-leader,.fn-vline{stroke:#94a3b8;stroke-width:1}.fn-emphasis .fn-chart-value{fill:#0f172a}';
    copy.prepend(style);
    const blob = new Blob([new XMLSerializer().serializeToString(copy)], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'floodnet-chart.svg';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function saveMode() {
    try { localStorage.setItem('fn-mode', state.mode); } catch { }
}

function loadMode() {
    try {
        if (localStorage.getItem('fn-mode') === 'presentation') state.mode = 'presentation';
    } catch { }
}

document.addEventListener('click', event => {
    const target = event.target;
    if (target.closest('.fn-skip')) {
        event.preventDefault();
        $('#fn-main').focus();
        return;
    }
    const toggle = target.closest('.fn-mobile-menu');
    if (toggle) {
        const open = $('#fn-sidebar').classList.toggle('open');
        toggle.setAttribute('aria-expanded', String(open));
        return;
    }
    const mode = target.closest('[data-mode]');
    if (mode) {
        if (mode.dataset.mode !== state.mode) {
            state.mode = mode.dataset.mode;
            saveMode();
            syncViewButtons();
            location.hash = homeRoute();
        }
        closeSidebar();
        return;
    }
    const tocLink = target.closest('[data-toc-target]');
    if (tocLink) {
        document.getElementById(tocLink.dataset.tocTarget)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return;
    }
    const budget = target.closest('[data-budget]');
    if (budget) {
        state.budget = Number(budget.dataset.budget);
        state.page = 0;
        $$('[data-budget]').forEach(b => {
            const on = Number(b.dataset.budget) === state.budget;
            b.classList.toggle('on', on);
            b.setAttribute('aria-pressed', String(on));
        });
        updateExplorer();
        if (location.hash === '#walkthrough') FN_WALKTHROUGH.update();
        return;
    }
    const lossSeed = target.closest('[data-loss-seed]');
    if (lossSeed) {
        state.lossSeed = Number(lossSeed.dataset.lossSeed);
        render(false);
        return;
    }
    const sample = target.closest('[data-sample]');
    if (sample) {
        state.selected = sample.dataset.sample;
        updateExplorer();
        $('#fn-viewer').scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
    }
    const step = target.closest('[data-page-step]');
    if (step) {
        state.page += Number(step.dataset.pageStep);
        updateExplorer();
        return;
    }
    const representative = target.closest('[data-representative]');
    if (representative) {
        Object.assign(state, { selected: representative.dataset.representative, search: '', presence: 'all', overlap: 'all', sort: 'id', page: 0 });
        return;
    }
    const failures = target.closest('[data-failures]');
    if (failures) {
        Object.assign(state, { sort: `worst-${failures.dataset.failures}`, presence: 'all', overlap: 'all', search: '', page: 0 });
        return;
    }
    const walk = target.closest('[data-walkthrough]');
    if (walk) {
        FN_WALKTHROUGH.state.example = walk.dataset.walkthrough;
        return;
    }
    const expand = target.closest('[data-expand-chart]');
    if (expand) {
        const chart = chartRenderers.get(expand.dataset.expandChart);
        if (chart) FN_VIEWER.openChart(chart.title, chart.render({ large: true }), expand);
        return;
    }
    const chart = target.closest('[data-action="download-chart"]');
    if (chart) downloadChart(chart);
});

document.addEventListener('change', event => {
    const target = event.target;
    if (target.id === 'fn-metric-run') {
        state.metricRun = target.value;
        render(false);
        return;
    }
    const filters = { 'fn-presence': 'presence', 'fn-overlap': 'overlap', 'fn-sort': 'sort' };
    if (filters[target.id]) {
        state[filters[target.id]] = target.value;
        state.page = 0;
        updateExplorer();
    } else if (target.id === 'fn-overlay') {
        state.overlay = target.checked;
        redrawSelected();
    } else if (target.dataset.class != null) {
        state.visible[Number(target.dataset.class)] = target.checked;
        redrawSelected();
    }
});

document.addEventListener('input', event => {
    if (event.target.id === 'fn-search') {
        state.search = event.target.value;
        state.page = 0;
        updateExplorer();
    } else if (event.target.id === 'fn-opacity') {
        state.opacity = Number(event.target.value);
        $('#fn-opacity-value').textContent = `${Math.round(state.opacity * 100)}%`;
        redrawSelected();
    }
});

window.addEventListener('hashchange', () => {
    if (data) render();
});

(async () => {
    const status = $('#fn-export-status');
    renderNav();
    loadMode();
    try {
        const hires = FN_VIEWER.loadHires();
        const response = await fetch('data/results.json', { cache: 'no-store' });
        const uniformResponse = await fetch('data/uniform.json', { cache: 'no-store' });
        const presentation = FN_PRES.load();
        await hires;
        if (!response.ok) throw new Error(`Results export returned HTTP ${response.status}`);
        data = await response.json();
        if (data.schema_version !== 3 || !Array.isArray(data.runs) || !Array.isArray(data.samples)) {
            throw new Error('Unsupported or invalid results export');
        }
        if (!uniformResponse.ok) throw new Error(`Results export returned HTTP ${uniformResponse.status}`);
        uniform = await uniformResponse.json();
        await presentation;
        state.metricRun = data.runs[data.runs.length - 1]?.id ?? null;
        syncViewButtons();
        render();
    } catch (error) {
        status.textContent = 'Export unavailable';
        $('#fn-page').innerHTML = header('FloodNet / Results explorer', 'The export is not available yet.')
            + empty('Serve this app over HTTP', `${error.message}. Run the documented export-web command, then start the local HTTP server. No results have been fabricated.`);
    }
})();
