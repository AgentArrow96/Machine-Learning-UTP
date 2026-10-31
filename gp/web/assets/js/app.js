'use strict';

const { esc, pct } = FN_CHARTS;

const METHOD = {
    rf: { name: 'SLIC + Random Forest', short: 'SLIC + RF', colour: '#059669', kind: 'Traditional · CPU' },
    unet: { name: 'ResNet-18 U-Net', short: 'U-Net', colour: '#4f46e5', kind: 'Deep learning · CUDA' },
    hybrid: { name: 'U-Net + DenseCRF', short: 'U-Net + CRF', colour: '#c2410c', kind: 'Hybrid · GPU + CPU' }
};
const METHODS = Object.keys(METHOD);
const SEED_COLOURS = ['#0891b2', '#d97706', '#7c3aed', '#e11d48'];
const CLASS_SHORT = ['Other', 'Building', 'Road'];
const PAGE_SIZE = 24;
const STATE_COLOUR = { before: '#94a3b8', after: '#0891b2' };
const STATE_NAME = { before: 'Before', after: 'After' };
const COMPONENT_NAME = { aux10: '10-class supervision', recipe: 'training recipe', tta: 'test-time augmentation', res512: '512-px input', rf_context: 'context features', encoder: 'ResNet-34 encoder' };

const NAV = [
    { group: 'START', items: [['overview', 'Overview']] },
    { group: 'THE DATA', items: [['dataset', 'Dataset'], ['preprocessing', 'Data preprocessing']] },
    { group: 'THE MODELS', items: [['methodology', 'Methods & equations'], ['walkthrough', 'How the models work']] },
    { group: 'THE RESULTS', items: [['results', 'Results'], ['changes', 'What “After” changed'], ['explorer', 'Prediction explorer'], ['improve', 'Improvements'], ['imbalance', 'Class imbalance']] },
    { group: 'NOTES', items: [['training', 'Training & compute'], ['interpretation', 'Interpretation'], ['reproducibility', 'Reproducibility']] }
];
const AFTER_ONLY = new Set(['changes', 'improve']);
const VIEW_PAGES = new Set(['overview', 'results', 'explorer', 'training', 'interpretation', 'reproducibility']);

let data = null;
let after = null;
let drawToken = 0;
const imageCache = new Map();
const chartRenderers = new Map();
const state = {
    view: 'compare', budget: 100, presence: 'all', overlap: 'all', sort: 'id', search: '',
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
    return source === 'after' ? after : data;
}

function compareMode() {
    return state.view === 'compare' && !!after;
}

function showAfter() {
    return !!after && state.view !== 'before';
}

function navItems() {
    const items = NAV.flatMap(g => g.items.filter(([route]) => showAfter() || !AFTER_ONLY.has(route)).map(([route, label]) => ({ route, label, group: g.group })));
    items.forEach((item, i) => { item.number = String(i + 1).padStart(2, '0'); });
    return items;
}

function viewSource() {
    return state.view === 'after' && after ? 'after' : 'before';
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

function comparisonFor(method, budget) {
    return after?.comparisons.find(c => c.method === method && c.budget === budget);
}

function sampleMetrics(sample, runId, source = 'before') {
    return source === 'after' ? after?.samples[sample.id]?.metrics[runId] : sample.metrics[runId];
}

function overlapInfo() {
    return data.dataset?.overlap || null;
}

function testLabel() {
    return `Official test split (${num(data.dataset?.counts.test)} images)`;
}

function components(method) {
    return (after?.components[method === 'rf' ? 'rf' : 'unet'] || []).map(c => COMPONENT_NAME[c] || c);
}


function currentRoute() {
    return location.hash.slice(1) || 'overview';
}

function kicker(route) {
    const item = navItems().find(i => i.route === route);
    return item ? `${item.number} / ${item.label}` : route;
}

function viewChip(route) {
    if (!showAfter()) return '';
    if (!VIEW_PAGES.has(route)) return `<span class="fn-view-chip same">Same before and after</span>`;
    const text = { before: 'Showing: before (frozen originals)', after: 'Showing: after (combined models)', compare: 'Showing: before vs after' }[state.view];
    return `<span class="fn-view-chip ${state.view}">${text}</span>`;
}

function header(route, title, intro = '') {
    return `<header class="fn-page-header"><div class="fn-eyebrow-row"><div class="fn-eyebrow" data-plain>${esc(kicker(route))}</div>${viewChip(route)}</div><h1>${esc(title)}</h1>${intro ? `<p>${esc(intro)}</p>` : ''}</header>`
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
    return FN_IMBALANCE.decisionPill(decision);
}

function interval(ci) {
    const bare = v => points(v).replace(' pp', '');
    return `[${bare(ci[0])}, ${bare(ci[1])}] pp`;
}

function arrow(before, afterValue, format = pct) {
    return `${format(before)} → <strong>${format(afterValue)}</strong>`;
}

function stateLabel(source) {
    return `<span class="fn-state ${source}">${STATE_NAME[source]}</span>`;
}


function summaryTable() {
    if (compareMode()) return compareTable();
    const source = viewSource();
    const b = bundle(source);
    if (!b.summary.length) return empty();
    const head = '<tr><th>Method</th><th>Labels</th><th>Flooded mIoU</th><th>3-class mIoU</th><th>Building IoU</th><th>Road IoU</th></tr>';
    const rows = b.summary.map(item => {
        const s = item.full;
        const cell = key => `<td>${meanSd(s[key])}<span class="fn-seed-values"><br>${seedValues(s[key])}</span></td>`;
        return `<tr><td>${methodPill(item.method)}<br>${esc(METHOD[item.method].name)}</td><td>${item.budget}%</td>`
            + `<td class="fn-score">${meanSd(s.flooded_miou)}<small>seeds: ${seedValues(s.flooded_miou)}</small></td>`
            + `${cell('miou')}${cell('building_iou')}${cell('road_iou')}</tr>`;
    }).join('');
    const which = source === 'after' ? `Pre-registered combined models of the improvement study (${componentsText()}).` : 'Frozen original models.';
    return `<div class="fn-table-wrap"><table class="fn-table"><thead>${head}</thead><tbody>${rows}</tbody></table></div>`
        + `<p class="fn-figure-note">${which} Mean ± standard deviation across ${data.protocol.train_seeds.length} training seeds (${data.protocol.train_seeds.join(', ')}), pooled pixel counts per run. ${esc(testLabel())}. <a href="#results">Every metric (majority IoU, weighted mIoU, accuracy, precision, recall, Dice) →</a></p>`;
}

function componentsText() {
    return `RF: ${components('rf').join(' + ')}; U-Net and CRF: ${components('unet').join(' + ')}`;
}

function compareTable() {
    const rows = data.summary.map(item => {
        const a = summaryFor(item.method, item.budget, 'after');
        const c = comparisonFor(item.method, item.budget);
        return `<tr><td><span class="fn-dot" style="background:${METHOD[item.method].colour}"></span>${esc(METHOD[item.method].short)}</td><td>${item.budget}%</td>`
            + `<td>${meanSd(item.full.flooded_miou)}<span class="fn-seed-values"><br>${seedValues(item.full.flooded_miou)}</span></td>`
            + `<td class="fn-score">${meanSd(a?.full.flooded_miou)}<small>seeds: ${seedValues(a?.full.flooded_miou)}</small></td>`
            + `<td class="fn-score">${c ? points(c.mean_difference) : 'N/A'}<small>${c ? c.differences.map(points).join(' · ') : ''}</small></td>`
            + `<td class="fn-nowrap">${c ? interval(c.ci) : 'N/A'}</td><td>${c ? decisionPill(c.decision) : ''}</td></tr>`;
    }).join('');
    return `<div class="fn-table-wrap"><table class="fn-table fn-compare-table"><thead><tr><th>Method</th><th>Labels</th><th>${stateLabel('before')}</th><th>${stateLabel('after')}</th><th>Change (per seed)</th><th>95% CI</th><th>Decision</th></tr></thead><tbody>${rows}</tbody></table></div>`
        + `<p class="fn-figure-note">Flooded-class mIoU on the ${esc(testLabel().toLowerCase())}, mean ± SD over seeds ${data.protocol.train_seeds.join(', ')}. After = the pre-registered combined models (${componentsText()}). Change = seed-matched after − before; the interval and decision come from the pre-registered paired bootstrap. <a href="#changes">What exactly changed →</a></p>`;
}


function overviewTakeaways() {
    if (compareMode() || viewSource() === 'after') {
        return takeaways(METHODS.map(method => {
            const parts = [25, 100].map(budget => {
                const c = comparisonFor(method, budget);
                return c ? `${budget}%: ${pct(mean(c.baseline_scores))} → <strong>${pct(mean(c.variant_scores))}</strong> (${points(c.mean_difference)}, ${esc(c.decision)})` : '';
            });
            return `<strong>${esc(METHOD[method].short)}</strong>: ${parts.join('; ')}.`;
        }).concat(['Flooded-class mIoU on the held-out test split; "helps" means the 95% interval excludes zero and every seed improved.']), 'Before → after, at a glance');
    }
    const best = [...data.summary].filter(s => s.budget === 100).sort((a, b) => b.full.flooded_miou.mean - a.full.flooded_miou.mean)[0];
    const unet = effectFor('unet')?.full.mean;
    return takeaways([
        best ? `Best original model at 100% labels: <strong>${esc(METHOD[best.method].name)}</strong>, ${meanSd(best.full.flooded_miou)} flooded mIoU.` : '',
        unet != null ? `Four times the labels change the U-Net by ${points(unet)} on average.` : '',
        `SLIC + RF stays far behind (about ${pct(summaryFor('rf', 100)?.full.flooded_miou.mean, 0)}): colour and texture alone confuse floodwater with flooded roads.`
    ]);
}

function overview() {
    const sample = data.samples.find(s => s.id === data.representatives[0]) || data.samples[0];
    const total = data.dataset ? Object.values(data.dataset.counts).reduce((a, b) => a + b, 0) : null;
    const source = viewSource();
    const best = [...bundle(source).summary].filter(s => s.full.flooded_miou.mean != null).sort((a, b) => b.full.flooded_miou.mean - a.full.flooded_miou.mean)[0];
    const overlap = overlapInfo();
    const testOverlap = overlap ? `${num(overlap.summary.test.overlapping_train)} / ${num(overlap.summary.test.images)}` : '—';
    const hero = `<div class="fn-hero"><div class="fn-hero-copy"><div class="fn-tag">FLOODNET · A COMPARATIVE STUDY</div>`
        + `<h1>Reading the flood.<br><em>One pixel at a time.</em></h1>`
        + `<p>How well can traditional machine learning, a pretrained U-Net and a CRF-refined U-Net find flooded buildings and roads in post-hurricane aerial images, with all the labels or only a quarter of them${showAfter() ? ', and how much did a pre-registered round of improvements raise them' : ''}?</p>`
        + `<div class="fn-actions"><a class="fn-primary-btn" href="#results">Explore the findings →</a><a class="fn-secondary-btn" href="#explorer">Open the prediction explorer</a></div></div>`
        + `<div class="fn-hero-visual">${sample ? zoomable(`<img src="${esc(sample.rgb)}" alt="Held-out FloodNet aerial image ${esc(sample.image_id)}">`, `held-out image ${sample.image_id}`, { title: `Held-out test image ${sample.image_id}`, hires: sample.id, caption: 'Official test split' }) + `<span class="fn-hero-stamp" data-plain>HELD-OUT IMAGE ${esc(sample.image_id)}</span>` : ''}</div></div>`;
    let tiles;
    if (compareMode()) {
        const gains = after.comparisons.filter(c => c.decision === 'helps').sort((a, b) => b.mean_difference - a.mean_difference);
        const top = gains[0];
        const bestAfter = [...after.summary].sort((a, b) => b.full.flooded_miou.mean - a.full.flooded_miou.mean)[0];
        const bestBefore = summaryFor(bestAfter.method, bestAfter.budget);
        tiles = metricCards([
            { value: num(total), label: 'Audited image–mask pairs', note: 'Official train / validation / test' },
            { value: `${num(data.runs.length)} + ${num(after.runs.length)}`, label: 'Original + improved runs', note: '3 methods × 2 budgets × 3 seeds, each' },
            { value: top ? points(top.mean_difference) : '—', label: 'Largest clear gain', note: top ? `${METHOD[top.method].short} · ${top.budget}% · ${pct(mean(top.baseline_scores))} → ${pct(mean(top.variant_scores))}` : '' },
            { value: `${pct(bestBefore.full.flooded_miou.mean)} → ${pct(bestAfter.full.flooded_miou.mean)}`, label: 'Best flooded-class mIoU', note: `${METHOD[bestAfter.method].short} · ${bestAfter.budget}% · before → after` }
        ]);
    } else {
        tiles = metricCards([
            { value: num(total), label: 'Audited image–mask pairs', note: 'Official train / validation / test' },
            { value: `${num(data.protocol.trained_runs)} + ${num(data.protocol.hybrid_runs)}`, label: 'Trained models + CRF-refined runs', note: `2 trained methods × 2 budgets × ${data.protocol.train_seeds.length} seeds; the hybrid adds no training` },
            { value: best ? meanSd(best.full.flooded_miou) : '—', label: `Best flooded-class mIoU${showAfter() ? ` (${STATE_NAME[source].toLowerCase()})` : ''}`, note: best ? `${METHOD[best.method].short} · ${best.budget}% · mean ± SD` : 'Awaiting evaluation' },
            { value: testOverlap, label: 'Test images overlapping a train frame', note: 'Verified by keypoint matching' }
        ]);
    }
    const intro = `<div class="fn-grid fn-two-col">`
        + card('A small experiment, a specific question', `<p>The primary score is the mean IoU of the two target classes, flooded buildings and flooded roads (${FN_MATH.inline('fmiou')}). Image IDs, input resolution and scoring rules stay fixed while the method and the labelled-data budget change.</p><div class="fn-flow"><div>RGB image</div><span>→</span><div>Learn / classify</div><span>→</span><div>Pixel labels</div></div><a href="#methodology">See how the methods work →</a>`)
        + card('What this study can tell us', `<p>Measured segmentation quality, missed target regions, data-efficiency trade-offs, ${showAfter() ? 'what a pre-registered round of improvements changed, ' : ''}and the compute cost on one laptop.</p><p>It does not establish road safety, flood depth, emergency-response benefit or performance on a different flood event.</p><a href="#interpretation">Read the practical limits →</a>`)
        + `</div>`;
    const comparison = card(compareMode() ? 'The held-out comparison, before and after' : 'The held-out comparison', summaryTable());
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
        + `${legend()}<div class="fn-callout">“Other” includes floodwater. This study detects flooded infrastructure; it does not outline all water.${showAfter() ? ' The improvement study later used the ten original labels as extra training signal for the Random Forest (see <a href="#improve">Improvements</a>).' : ''}</div>`);
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

function improvedLine(method) {
    if (!showAfter()) return '';
    return `<p class="fn-improved"><span class="fn-state after">After</span> ${esc(components(method).join(' + '))}. <a href="#improve">Why →</a></p>`;
}

function methodology() {
    let html = header('methodology', 'Three approaches. Shared rules.', `The study compares practical segmentation pipelines under matched image budgets and identical evaluation${showAfter() ? ', then improves each one in a single pre-registered step' : ''}.`);
    html += takeaways([
        `Every method sees the same images, resolution (256×256), budgets (25% / 100%) and seeds (${data.protocol.train_seeds.join(', ')}); only the method changes.`,
        `The primary score is ${FN_MATH.inline('fmiou')}, from pooled pixel counts.`,
        showAfter() ? `The improved ("after") models add one validated set of changes per method; their components were fixed on validation before the test split was opened.` : ''
    ]);
    html += card('The shared preparation path', `<div class="fn-flow"><div>Audit originals</div><span>→</span><div>Remap labels</div><span>→</span><div>Resize + pad</div><span>→</span><div>Cache once</div><span>→</span><div>25% / 100%</div></div>`
        + `<p>Aspect ratio is preserved inside a 256×256 frame. Masks use nearest-neighbour resizing, and padded pixels are ignored in training and scoring. One stratified 25% subset (seed ${data.protocol.subset_seed}) is shared by every method and seed. Validation and test sets never change.</p>`);
    const rf = `<section class="fn-card fn-method-card"><span class="fn-pill rf">Traditional · CPU</span><h2 style="margin-top:12px">SLIC + Random Forest</h2>`
        + `<p>Neighbouring pixels are grouped into about 250 superpixels. Each region is described by colour and texture, then classified by a class-balanced forest.</p>`
        + `<div class="fn-method-path"><span>RGB</span>→<span>SLIC regions</span>→<span>22 features</span>→<span>Forest</span></div>`
        + `<dl class="fn-definition-list"><dt>Colour</dt><dd>RGB/HSV mean and standard deviation</dd><dt>Texture</dt><dd>10-bin uniform LBP histogram</dd>`
        + `<dt>Region label</dt><dd>Majority annotation, padding excluded</dd><dt>Forest</dt><dd>150 trees · maximum depth 20 · balanced weights</dd>`
        + `<dt>Training cap</dt><dd>${num(data.protocol.rf_max_regions)} regions, keeping every flooded region</dd>`
        + `<dt>Regions used</dt><dd>25%: ${regionCounts(25)}<br>100%: ${regionCounts(100)}<br><small class="fn-muted">Other / building / road</small></dd></dl>`
        + improvedLine('rf') + `</section>`;
    const unet = `<section class="fn-card fn-method-card unet"><span class="fn-pill unet">Deep learning · CUDA</span><h2 style="margin-top:12px">Lightweight U-Net</h2>`
        + `<p>A pretrained ResNet-18 encoder extracts features. A decoder with skip connections restores spatial detail and outputs three scores per pixel.</p>`
        + `<div class="fn-method-path"><span>RGB</span>→<span>Encoder</span>→<span>Decoder + skips</span>→<span>3 logits</span></div>`
        + `<dl class="fn-definition-list"><dt>Pretraining</dt><dd>ImageNet ResNet-18</dd><dt>Decoder</dt><dd>128 → 64 → 32 → 16 → 8</dd>`
        + `<dt>Loss</dt><dd>Class-weighted cross-entropy</dd><dt>Optimiser</dt><dd>AdamW · LR 1e-4 · WD 1e-4 · batch 4</dd>`
        + `<dt>Selection</dt><dd>Best validation flooded mIoU, patience 6, up to 30 epochs</dd></dl>`
        + `<p class="fn-small">BF16 mixed precision on the GPU. Training stops with an error instead of falling back to the CPU.</p>` + improvedLine('unet') + `</section>`;
    const crf = data.hybrid?.selected;
    const hybrid = `<section class="fn-card fn-method-card hybrid"><span class="fn-pill hybrid">Hybrid · GPU + CPU</span><h2 style="margin-top:12px">U-Net + DenseCRF</h2>`
        + `<p>No new network is trained. Each saved U-Net's class probabilities are refined by a fully connected conditional random field, which encourages pixels that are close together and similar in colour to share a label.</p>`
        + `<div class="fn-method-path"><span>U-Net probabilities</span>→<span>DenseCRF</span>→<span>Refined labels</span></div>`
        + `<dl class="fn-definition-list"><dt>Base model</dt><dd>The same U-Net checkpoint as each U-Net run</dd>`
        + `<dt>Inference</dt><dd>${crf ? `${crf.iterations} mean-field iterations` : 'N/A'}</dd>`
        + `<dt>Spatial kernel</dt><dd>${crf ? `sxy ${crf.gaussian.sxy} · weight ${crf.gaussian.compat}` : 'N/A'}</dd>`
        + `<dt>Appearance kernel</dt><dd>${crf ? `sxy ${crf.bilateral.sxy} · srgb ${crf.bilateral.srgb} · weight ${crf.bilateral.compat}` : 'N/A'}</dd>`
        + `<dt>Chosen on</dt><dd>Validation only: ${data.hybrid ? data.hybrid.grid.length : 'N/A'} settings, one shared by every budget and seed</dd></dl>`
        + improvedLine('hybrid') + `</section>`;
    html += `<div class="fn-grid fn-three-col">${rf}${unet}${hybrid}</div>`;
    html += `<div class="fn-callout">Want to see these steps on a real test image? <a href="#walkthrough">Open the step-by-step walkthrough →</a> It replays each model on six representative scenes, and every final mask it shows is checked against the frozen test prediction.</div>`;
    html += card('Notation and equations', `<p>Every formula used on this site, written once. ${FN_MATH.tex('c')} is a class, ${FN_MATH.tex(String.raw`TP_c,\ FP_c,\ FN_c`)} are pooled pixel counts over the test images and ${FN_MATH.tex('s')} is a training seed.</p>`
        + `<h3 class="fn-eq-group">Metrics</h3>${FN_MATH.equations(['fmiou', 'iou', 'majority', 'miou3', 'fwiou', 'acc', 'precision', 'recall', 'dice'])}`
        + `<h3 class="fn-eq-group">Data and training</h3>${FN_MATH.equations(['norm', 'weights', 'ce'])}`
        + `<h3 class="fn-eq-group">Refinement</h3>${FN_MATH.equations(['crf'])}`
        + (showAfter() ? `<h3 class="fn-eq-group">Improvement study</h3>${FN_MATH.equations(['aux', 'cosine', 'ema', 'tta', 'area'])}` : '')
        + `<h3 class="fn-eq-group">Statistics</h3>${FN_MATH.equations(showAfter() ? ['budget', 'change', 'bootstrap', 'decision'] : ['budget', 'bootstrap', 'decision'])}`);
    html += `<div class="fn-dark-card"><h3>Three seeds, one subset</h3><p>Every configuration is trained with seeds <strong>${data.protocol.train_seeds.join(', ')}</strong>. The seed changes forest sampling, network initialisation, data order and augmentation; the 25% image subset stays the same. Results report the mean, standard deviation and range, so a lucky or unlucky run cannot decide the comparison.</p></div>`;
    html += card('Fair comparison and model selection', `<details class="fn-details" open><summary>What is held constant?</summary><p>Image IDs within each budget, input resolution, target labels, validation and test sets, metrics, and each method's settings across budgets and seeds.</p></details>`
        + `<details class="fn-details"><summary>What differs between the methods?</summary><p>Hand-crafted region features and forest fitting versus learned pixel features and gradient training. This compares practical pipelines, not isolated architectures.</p></details>`
        + `<details class="fn-details"><summary>What happens during augmentation?</summary><p>Only U-Net training receives matched flips and quarter-turn rotations, plus mild image-only brightness/contrast changes. Validation and test inputs are never augmented${showAfter() ? ' (the improved U-Net averages flipped copies at test time instead)' : ''}.</p></details>`
        + `<details class="fn-details"><summary>How is leakage controlled?</summary><p>Only training images supply features, class weights and subsets. Validation selects the U-Net checkpoint${showAfter() ? ', the CRF setting and the improved models\' components' : ' and the CRF setting'}. The ${data.protocol.trained_runs} trained runs and the overlap analysis were frozen before their test evaluation; the hybrid${showAfter() ? ' and the improvement study were each' : ' was'} frozen separately before ${showAfter() ? 'their own' : 'its own'} single test evaluation.</p></details>`
        + `<details class="fn-details"><summary>Why is the hybrid not retrained?</summary><p>The methodology defines the hybrid as post-processing: it reuses each U-Net checkpoint unchanged, so any difference from the U-Net comes from the CRF alone.</p></details>`);
    return html;
}


function classTable(run) {
    const rows = run.metrics.per_class.map(c => `<tr><td>${esc(c.name)}</td><td>${pct(c.iou)}</td><td>${pct(c.dice)}</td><td>${pct(c.recall)}</td><td>${num(c.support)}</td></tr>`).join('');
    return `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Class</th><th>IoU</th><th>Dice</th><th>Recall</th><th>Truth pixels</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function inspectorSource() {
    return compareMode() ? 'before' : viewSource();
}

function inspectedRun() {
    const runs = bundle(inspectorSource()).runs;
    return runs.find(r => r.id === state.metricRun) || runs[runs.length - 1];
}

function matchedAfterRun(run) {
    return after?.runs.find(r => r.method === run.method && r.budget === run.budget && r.seed === run.seed);
}

function confusionChart(options) {
    return FN_CHARTS.confusion(inspectedRun().metrics.confusion_matrix, CLASS_SHORT, options);
}

function afterConfusionChart(options) {
    return FN_CHARTS.confusion(matchedAfterRun(inspectedRun()).metrics.confusion_matrix, CLASS_SHORT, options);
}

function runInspector() {
    const note = 'Rows are ground truth; each row sums to 100%. Pooled pixel counts, padding excluded. Outlined cells are correct predictions.';
    if (compareMode()) {
        const run = inspectedRun();
        return `<div class="fn-grid fn-two-col fn-inspector-pair"><div><h3 class="fn-vis-title">${stateLabel('before')} ${esc(METHOD[run.method].short)} · ${run.budget}% · seed ${run.seed}</h3><div id="fn-class-metrics">${classTable(run)}</div>`
            + `<div style="margin-top:16px">${chartBlock('Confusion matrix · before', confusionChart, { heading: 'h3', svg: false, note })}</div></div>`
            + `<div><h3 class="fn-vis-title">${stateLabel('after')} ${esc(METHOD[run.method].short)} · ${run.budget}% · seed ${run.seed}</h3><div id="fn-class-metrics-after">${classTable(matchedAfterRun(run))}</div>`
            + `<div style="margin-top:16px">${chartBlock('Confusion matrix · after', afterConfusionChart, { heading: 'h3', svg: false, note })}</div></div></div>`;
    }
    return `<div id="fn-class-metrics">${classTable(inspectedRun())}</div><div style="margin-top:20px">`
        + chartBlock('Confusion matrix', confusionChart, { heading: 'h3', svg: false, note })
        + `</div>`;
}

function overlapStrataCard() {
    const overlap = overlapInfo();
    if (!overlap || !data.summary[0]?.overlap_low) return '';
    const targets = overlap.target_images;
    const sources = compareMode() ? ['before', 'after'] : [viewSource()];
    const rows = sources.flatMap(source => bundle(source).summary.map(item => {
        const low = item.overlap_low.flooded_miou;
        const high = item.overlap_high.flooded_miou;
        return `<tr><td>${sources.length > 1 ? stateLabel(source) : ''} ${esc(METHOD[item.method].short)}</td><td>${item.budget}%</td><td>${meanSd(low)}</td><td>${meanSd(high)}</td><td>${points(high.mean - low.mean)}</td></tr>`;
    })).join('');
    const clean = summaryFor('unet', 100)?.clean.flooded_miou;
    return card('Does scene overlap inflate the scores?', `<p>${num(overlap.summary.test.overlapping_train)} of ${num(overlap.summary.test.images)} test images, and ${num(targets.overlapping)} of the ${num(targets.images)} that contain flooded buildings or roads, share ground with a training frame. A test set without overlap would hold almost no flooded targets, so this check splits the ${num(targets.images)} target images at their median overlap (${pct(targets.split_overlap, 0)}) and scores each half separately.</p>`
        + `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Method</th><th>Labels</th><th>Lower-overlap half (${targets.low_images} images)</th><th>Higher-overlap half (${targets.high_images} images)</th><th>Higher − lower</th></tr></thead><tbody>${rows}</tbody></table></div>`
        + `<p class="fn-figure-note">Flooded-class mIoU, mean ± SD across seeds, pooled over the target images in each half. Scores on target images are higher than on the full split because the full split also counts false alarms in the ${num(data.dataset.counts.test - targets.images)} scenes without flooded infrastructure.</p>`
        + `<div class="fn-callout">This is a weak test: most images in the lower-overlap half still share a large part of their ground with training (anything below ${pct(targets.split_overlap, 0)}), and only ${targets.images - targets.overlapping} target images have no overlap at all. It can show whether more shared ground means higher scores, but it cannot show how the models would score on a new flood area. For transparency, the ${num(overlap.summary.test.clean)} test images with no overlap (only ${targets.clean_with_building} with flooded buildings and ${targets.clean_with_road} with flooded roads) give the original U-Net (100%) ${meanSd(clean)} flooded mIoU, because false alarms across many target-free scenes weigh heavily against very few target pixels. They are listed in the all-runs CSV but not used as a leakage estimate.</div>`);
}

function crfCard() {
    const hybrid = data.hybrid;
    if (!hybrid || !data.crf_effects.length) return '';
    const sources = compareMode() ? ['before', 'after'] : [viewSource()];
    const rows = sources.flatMap(source => bundle(source).crf_effects.map(e => {
        const cell = key => `<td>${points(e.full[key].mean)}<span class="fn-seed-values"><br>${e.full[key].values.map(points).join(' · ')}</span></td>`;
        return `<tr><td>${sources.length > 1 ? stateLabel(source) : ''} ${e.budget}%</td><td class="fn-score">${points(e.full.flooded_miou.mean)}<small>seeds: ${e.full.flooded_miou.values.map(points).join(' · ')}</small></td>${cell('building_iou')}${cell('road_iou')}${cell('building_recall')}${cell('road_recall')}</tr>`;
    })).join('');
    const label = params => `sxy ${params.bilateral.sxy} · srgb ${params.bilateral.srgb} · weight ${params.bilateral.compat}`;
    const grid = hybrid.grid.map(row => `<tr${row.selected ? ' class="fn-selected-row"' : ''}><td>${esc(row.round)}</td><td>${label(row.params)}${row.selected ? ' <span class="fn-pill good">selected</span>' : ''}</td><td>${points(row.mean_gain)}</td><td>${points(row.min_gain)} to ${points(row.max_gain)}</td></tr>`).join('');
    const chosen = hybrid.grid.find(row => row.selected);
    const verdict = hybrid.adopted
        ? `Validation <strong>adopted</strong> refinement: the selected setting improved ${chosen.min_gain > 0 ? 'every' : 'the average'} U-Net run on validation (${points(hybrid.selected_mean_gain)} on average).`
        : `Validation <strong>rejected</strong> refinement: no setting improved the U-Net on validation, so the hybrid is reported only for completeness.`;
    const sxys = hybrid.grid.map(row => row.params.bilateral.sxy);
    const srgbs = hybrid.grid.map(row => row.params.bilateral.srgb);
    const atEdge = [Math.min(...sxys), Math.max(...sxys)].includes(hybrid.selected.bilateral.sxy)
        || [Math.min(...srgbs), Math.max(...srgbs)].includes(hybrid.selected.bilateral.srgb);
    const edgeNote = atEdge ? ' The chosen setting lies at the edge of the pre-set grid; values beyond it were not tried, to keep tuning minimal.' : '';
    const afterNote = showAfter() ? ' The improved hybrid reuses the same setting on the improved U-Net (kernel widths doubled for its 512-px input).' : '';
    return `<section class="fn-card"><h2>Does CRF refinement help?</h2>`
        + `<p>Each hybrid run refines the exact U-Net checkpoint with the same seed and budget, so the change below is caused by the CRF alone. Values are the test-set change from that U-Net run to its refined version.</p>`
        + `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Labels</th><th>Flooded mIoU change</th><th>Building IoU</th><th>Road IoU</th><th>Building recall</th><th>Road recall</th></tr></thead><tbody>${rows}</tbody></table></div>`
        + `<div class="fn-callout">${verdict} One setting is used for both budgets and all seeds.${edgeNote}${afterNote}</div>`
        + `<details class="fn-details"><summary>Validation tuning grid (${hybrid.grid.length} settings)</summary><div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Round</th><th>Appearance kernel</th><th>Mean validation change</th><th>Range across the 6 U-Net runs</th></tr></thead><tbody>${grid}</tbody></table></div>`
        + `<p class="fn-figure-note">Validation flooded-class mIoU change over each run's own U-Net. The test split was not used for tuning. Spatial kernel fixed at sxy ${hybrid.selected.gaussian.sxy}, weight ${hybrid.selected.gaussian.compat}; ${hybrid.selected.iterations} iterations.</p></details>`
        + `<div class="fn-downloads">${download('crf_tuning', 'CRF tuning CSV')}${download('hybrid_experiment', 'Hybrid freeze manifest')}</div></section>`;
}

function resultsTakeaways() {
    if (compareMode()) {
        const ordered = [...after.comparisons].sort((a, b) => METHODS.indexOf(a.method) - METHODS.indexOf(b.method) || a.budget - b.budget);
        const helps = ordered.filter(c => c.decision === 'helps');
        const unclear = ordered.filter(c => c.decision !== 'helps');
        return takeaways([
            `${helps.length} of ${after.comparisons.length} method × budget comparisons clearly improved: ${helps.map(c => `${esc(METHOD[c.method].short)} ${c.budget}% (${points(c.mean_difference)})`).join(', ')}.`,
            unclear.length ? `No clear change for ${unclear.map(c => `${esc(METHOD[c.method].short)} ${c.budget}% (${points(c.mean_difference)})`).join(', ')}: the seeds disagree.` : '',
            `The ranking holds: the U-Net family stays far ahead of the Random Forest, which nevertheless gained the most.`
        ]);
    }
    const source = viewSource();
    const best = [...bundle(source).summary].filter(s => s.budget === 100).sort((a, b) => b.full.flooded_miou.mean - a.full.flooded_miou.mean)[0];
    const crf = bundle(source).crf_effects.find(e => e.budget === 100);
    return takeaways([
        best ? `Best at 100% labels${showAfter() ? ` (${STATE_NAME[source].toLowerCase()})` : ''}: <strong>${esc(METHOD[best.method].name)}</strong>, ${meanSd(best.full.flooded_miou)}.` : '',
        `More labels: U-Net ${points(effectFor('unet', source)?.full.mean)}, SLIC + RF ${points(effectFor('rf', source)?.full.mean)} (25% → 100%, seed-matched).`,
        crf ? `DenseCRF refinement at 100%: ${points(crf.full.flooded_miou.mean)} over its own U-Net.` : ''
    ]);
}

function resultsChart() {
    if (compareMode()) {
        const groups = METHODS.flatMap(method => [25, 100].map(budget => {
            const c = comparisonFor(method, budget);
            return {
                label: `${METHOD[method].short} · ${budget}%`,
                series: ['before', 'after'].map(source => ({
                    name: STATE_NAME[source], colour: STATE_COLOUR[source], stat: summaryFor(method, budget, source)?.full.flooded_miou || { mean: null, values: [] },
                    note: source === 'after' && c ? points(c.mean_difference) : null
                }))
            };
        }));
        return vis('Before and after, by method and budget', options => FN_CHARTS.rangeBars(groups, { ...options, large: true, seeds: data.protocol.train_seeds, label: 'Flooded-class mean IoU before and after the improvement study' }),
            `${testLabel()}. Grey = frozen original, teal = pre-registered combined model; notes above the teal bars give the seed-matched change.`);
    }
    const source = viewSource();
    const groups = [25, 100].map(budget => ({
        label: `${budget}% labels`,
        series: METHODS.map(method => {
            const effect = effectFor(method, source)?.full.mean;
            return {
                name: METHOD[method].short, colour: METHOD[method].colour, stat: summaryFor(method, budget, source)?.full.flooded_miou || { mean: null, values: [] },
                note: budget === 100 && effect != null ? `${points(effect)}` : null
            };
        })
    }));
    return vis('How much do more labels help?', options => FN_CHARTS.rangeBars(groups, { ...options, seeds: data.protocol.train_seeds, label: 'Flooded-class mean IoU by method and budget' }), `${testLabel()}. The 25% subset is nested inside the full training set. Notes above the 100% bars give the mean seed-matched change from 25%.`);
}

function budgetCard() {
    const sources = compareMode() ? ['before', 'after'] : [viewSource()];
    const rows = METHODS.map(method => `<tr><td>${methodPill(method)}<br>${esc(METHOD[method].name)}</td>${sources.map(source => {
        const e = effectFor(method, source)?.full;
        return `<td class="fn-score">${e?.mean == null ? 'N/A' : points(e.mean)}<small>${e ? e.values.map(points).join(' · ') : ''}</small></td>`;
    }).join('')}</tr>`).join('');
    return card('Budget effect, seed by seed', `<p>Change in flooded-class mIoU from 25% to 100% labels, comparing runs with the same seed (${FN_MATH.inline('budget')}).</p><div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Method</th>${sources.map(s => `<th>${sources.length > 1 ? stateLabel(s) : ''} Mean change (seeds ${data.protocol.train_seeds.join(', ')})</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div>`
        + (sources.length > 1 ? `<p class="fn-figure-note">After the improvements the 25% models gained more than the 100% models, so the extra labels are worth less than before.</p>` : ''));
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
    const sources = compareMode() ? ['before', 'after'] : [viewSource()];
    const label = (method, budget, source) => `<td>${sources.length > 1 ? stateLabel(source) : ''} ${esc(METHOD[method].short)}</td><td>${budget}%</td>`;
    const cell = (stat, primary) => `<td${primary ? ' class="fn-score"' : ''}>${meanSd(stat)}${primary ? `<small>seeds: ${seedValues(stat)}</small>` : ''}</td>`;
    const rowsA = [];
    const rowsB = [];
    for (const method of METHODS) {
        for (const budget of data.protocol.budgets) {
            for (const source of sources) {
                const m = configMetrics(method, budget, source);
                rowsA.push(`<tr>${label(method, budget, source)}${cell(m.minority, true)}${cell(m.majority)}${cell(m.miou)}${cell(m.weighted)}${cell(m.accuracy)}</tr>`);
                rowsB.push(`<tr>${label(method, budget, source)}${['building', 'road'].map(c => ['iou', 'precision', 'recall', 'dice'].map(k => cell(m[`${c}_${k}`])).join('')).join('')}</tr>`);
            }
        }
    }
    const t = trivialMetrics();
    rowsA.push(`<tr class="fn-reference-row"><td colspan="2">Reference: label every pixel “Other”</td><td>${pct(t.minority)}</td><td>${pct(t.majority)}</td><td>${pct(t.miou)}</td><td>${pct(t.weighted)}</td><td>${pct(t.accuracy)}</td></tr>`);
    return card(compareMode() ? 'Every metric, before and after' : 'Every metric, every configuration',
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
    if (compareMode()) rows.push(['Change (Δ), 95% interval, decision', 'change', 'Seed-matched after − before; the interval and the pre-registered decision come from a paired bootstrap over the test images.']);
    return card('What each metric means', `<div class="fn-table-wrap"><table class="fn-table fn-guide-table"><thead><tr><th>Metric</th><th>Formula</th><th>What it tells you</th></tr></thead><tbody>`
        + rows.map(([name, key, text]) => `<tr><td><strong>${esc(name)}</strong></td><td>${FN_MATH.inline(key)}</td><td>${esc(text)}</td></tr>`).join('')
        + `</tbody></table></div><p class="fn-figure-note">${FN_MATH.tex(String.raw`TP_c,\ FP_c,\ FN_c`)}: pooled true positives, false positives and false negatives of class ${FN_MATH.tex('c')} over the test images; ${FN_MATH.tex('n_c')}: true pixels of class ${FN_MATH.tex('c')}; ${FN_MATH.tex('N')}: all scored pixels.</p>`);
}

function resultsPage() {
    if (!data.summary.length) return header('results', 'The held-out comparison.') + notice() + empty();
    let html = header('results', compareMode() ? 'Before and after, on held-out images.' : 'The held-out comparison.', 'Primary metric: mean IoU of flooded building and flooded road, pooled over the test images. Each configuration is run with three seeds.');
    html += notice();
    html += resultsTakeaways();
    if (compareMode()) html += card('All configurations, before → after', summaryTable());
    html += metricsCard();
    html += metricGuideCard();
    html += resultsChart();
    html += budgetCard();
    html += crfCard();
    html += overlapStrataCard();
    const runs = bundle(inspectorSource()).runs;
    if (!runs.some(r => r.id === state.metricRun)) state.metricRun = runs[runs.length - 1].id;
    const options = runs.map(r => `<option value="${esc(r.id)}" ${r.id === state.metricRun ? 'selected' : ''}>${esc(METHOD[r.method].short)} · ${r.budget}% · seed ${r.seed}</option>`).join('');
    html += `<section class="fn-card"><div class="fn-card-top"><h2>Inspect a single run</h2><div class="fn-control"><label for="fn-metric-run">Run</label><select id="fn-metric-run">${options}</select></div></div>${runInspector()}</section>`;
    html += card('Download the numbers', `<p>Per-seed and per-image scores of the original models, as CSV.</p><div class="fn-downloads">${download('summary', 'Seed summary CSV')}${download('aggregate', 'All runs CSV')}${download('per_image', 'Per-image metrics CSV')}</div>`);
    return html;
}


function trainingPage() {
    const sources = compareMode() ? ['before', 'after'] : [viewSource()];
    const nets = data.runs.filter(r => r.method === 'unet');
    let html = header('training', 'Quality has a computational cost.', 'The traditional baseline runs on the CPU; every U-Net run trains on the GPU. Timings describe this laptop.');
    html += notice();
    if (!nets.length) return html + empty('Training curves await completed U-Net runs');
    if (after && sources.includes('after')) {
        const t = s => summaryFor('unet', 100, s)?.training_seconds.mean;
        const i = s => summaryFor('unet', 100, s)?.inference_median_seconds.mean;
        html += takeaways([
            `The improved U-Net (100%) trains in ${seconds(t('after'))} instead of ${seconds(t('before'))}: 512-px inputs and up to 60 epochs.`,
            `Its prediction takes ${(i('after') * 1000).toFixed(0)} ms instead of ${(i('before') * 1000).toFixed(0)} ms per image, because test-time augmentation runs 8 forward passes.`,
            `The improved Random Forest stays cheap to train (${seconds(summaryFor('rf', 100, 'after')?.training_seconds.mean)}), but its context features and 512-px superpixels make inference slower.`
        ]);
    }
    const seeds = data.protocol.train_seeds;
    const lossSeed = state.lossSeed ?? seeds[0];
    html += `<div class="fn-grid fn-two-col">${[25, 100].map(budget => {
        const series = sources.flatMap(source => runsFor('unet', budget, source).map((r, i) => ({
            label: `${sources.length > 1 ? `${STATE_NAME[source]} · ` : ''}seed ${r.seed}`, colour: SEED_COLOURS[i % SEED_COLOURS.length], dash: sources.length > 1 && source === 'before',
            values: r.history.map(e => e.val_flooded_miou), marker: r.best_epoch - 1,
            markerLabel: sources.length > 1 && source === 'before' ? null : `s${r.seed} · epoch ${r.best_epoch} · ${pct(r.history[r.best_epoch - 1]?.val_flooded_miou)}`
        })));
        const note = sources.length > 1 ? 'Dashed = before (30 epochs max), solid = after (recipe: up to 60 epochs, cosine LR, weight averaging); circle = selected checkpoint.'
            : 'Validation flooded mIoU per epoch; notes give each seed\'s selected checkpoint. Its epoch-to-epoch swings are why each configuration has three seeds.';
        return vis(`U-Net · ${budget}% labels: validation score`, options => FN_CHARTS.lines(series, { ...options, max: 1, percent: true, label: `Validation flooded mIoU, ${budget}% labels`, note: 'Circle = selected checkpoint' }), note);
    }).join('')}</div>`;
    const seedButtons = seeds.map(s => `<button type="button" class="${s === lossSeed ? 'on' : ''}" data-loss-seed="${s}" aria-pressed="${s === lossSeed}">Seed ${s}</button>`).join('');
    html += `<section class="fn-card"><div class="fn-card-top"><h2>Training and validation loss</h2><div class="fn-seg" role="group" aria-label="Seed">${seedButtons}</div></div><div class="fn-grid fn-two-col">${[25, 100].map(budget => {
        const series = [];
        let vline = null;
        let lowestPoint = null;
        sources.forEach(source => {
            const run = runsFor('unet', budget, source).find(r => r.seed === lossSeed);
            if (!run) return;
            const prefix = sources.length > 1 ? `${STATE_NAME[source]} · ` : '';
            const validation = run.history.map(e => e.val_loss);
            series.push({ label: `${prefix}training`, colour: source === 'after' ? '#0891b2' : '#4f46e5', values: run.history.map(e => e.train_loss), dash: sources.length > 1 && source === 'before' });
            series.push({ label: `${prefix}validation`, colour: source === 'after' ? '#0e7490' : '#d97706', values: validation, dash: true });
            if (source === sources[sources.length - 1]) {
                vline = { index: run.best_epoch - 1, label: `selected epoch ${run.best_epoch}` };
                const lowest = validation.indexOf(Math.min(...validation));
                lowestPoint = { series: series.length - 1, index: lowest, text: `min val loss ${validation[lowest].toFixed(3)}` };
            }
        });
        if (!series.length) return empty('Run not exported');
        const render = options => FN_CHARTS.lines(series, { ...options, label: `Loss curves, ${budget}% labels, seed ${lossSeed}`, vline, points: lowestPoint ? [lowestPoint] : [] });
        return `<div>${chartBlock(`${budget}% labels · seed ${lossSeed}`, render, { heading: 'h3' })}<p class="fn-small fn-muted">The checkpoint is chosen by validation flooded mIoU, not by loss.${sources.length > 1 ? ' The improved loss also contains no auxiliary term (10-class supervision was not selected).' : ''}</p></div>`;
    }).join('')}</div></section>`;
    const rows = data.summary.flatMap(item => sources.map(source => {
        const s = summaryFor(item.method, item.budget, source);
        const runs = runsFor(item.method, item.budget, source);
        const peak = Math.max(...runs.map(r => r.peak_gpu_allocated_bytes ?? -1));
        const device = item.method === 'hybrid' ? `${runs[0]?.device || 'GPU'} + CPU CRF` : runs[0]?.device || 'N/A';
        const training = item.method === 'hybrid' ? 'No training<span class="fn-seed-values"><br>reuses the U-Net checkpoint</span>'
            : `${seconds(s.training_seconds.mean)}<span class="fn-seed-values"><br>${seedValues(s.training_seconds, seconds)}</span>`;
        const epochs = s.epochs?.mean != null ? `${s.epochs.mean.toFixed(1)} (${s.epochs.min}–${s.epochs.max})` : 'N/A';
        return `<tr><td>${sources.length > 1 ? stateLabel(source) : ''} ${esc(METHOD[item.method].short)} · ${item.budget}%</td><td>${training}</td>`
            + `<td>${(s.inference_median_seconds.mean * 1000).toFixed(1)} ms</td><td>${epochs}</td>`
            + `<td>${peak < 0 || item.method === 'hybrid' ? 'N/A' : `${(peak / 2 ** 20).toFixed(0)} MiB`}</td><td>${esc(device)}${runs[0]?.precision ? `<br><small>${esc(runs[0].precision.replace('torch.', ''))}</small>` : ''}</td></tr>`;
    })).join('');
    html += card('Measured resources', `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Configuration</th><th>Training time (mean)</th><th>Median inference</th><th>Epochs</th><th>Peak GPU allocated</th><th>Device</th></tr></thead><tbody>${rows}</tbody></table></div>`);
    let timing = `<p>RF training time covers forest fitting. U-Net training time includes per-epoch validation and checkpointing. Shared preprocessing and RF feature extraction are recorded separately.</p>`
        + `<p>Inference is one image at a time, after warm-up. It starts from the processed RGB in memory and ends with a CPU mask. RF includes fresh SLIC segmentation and feature extraction (single-threaded forest prediction); U-Net includes normalisation, GPU transfer, the forward pass and argmax; U-Net + CRF adds the softmax and a single-threaded DenseCRF on the CPU. Decoding the original photo is excluded for all three.</p>`;
    if (showAfter()) timing += `<p>The improved models' timings include their 512-px input, the 8 forward passes of test-time augmentation and the 512 → 256 px averaging (${FN_MATH.inline('area')}).</p>`;
    if (data.dataset) timing += `<p>Shared preparation: <strong>${seconds(data.dataset.preprocessing_seconds)}</strong>; RF feature extraction: <strong>${seconds(data.dataset.feature_preparation_seconds)}</strong>${data.hybrid ? `; CRF validation tuning (all settings, all six U-Net runs): <strong>${seconds(data.hybrid.tuning_seconds)}</strong>` : ''}.</p>`;
    html += card('What is included in the timings?', timing);
    return html;
}


function panelSource(kind) {
    return kind.endsWith('@after') ? 'after' : 'before';
}

function panelMethod(kind) {
    return kind.split('@')[0];
}

function explorerSources() {
    return compareMode() ? ['before', 'after'] : [viewSource()];
}

function explorerPage() {
    let html = header('explorer', 'Look beyond the average.', `Every image comes from the held-out test split. The same image and budget are shown for all three approaches${showAfter() ? ', before and after the improvements' : ''}.`);
    if (!data.samples.length) return html + notice() + empty('Prediction gallery pending', 'The gallery appears after data preparation and final evaluation.');
    const presence = [['all', 'All images'], ['building', 'Flooded building'], ['road', 'Flooded road'], ['both', 'Both targets'], ['neither', 'Neither target']];
    const overlap = [['all', 'All test images'], ['clean', 'No training overlap'], ['overlapping', 'Overlaps training']];
    const sorts = [['id', 'Image ID'], ['worst-unet', 'U-Net: lowest target IoU'], ['worst-hybrid', 'U-Net + CRF: lowest target IoU'], ['worst-rf', 'RF: lowest target IoU'], ['coverage', 'Target pixel share'], ['overlap', 'Largest training overlap']];
    if (compareMode()) sorts.push(...METHODS.flatMap(m => [[`gain-${m}`, `${METHOD[m].short}: largest gain`], [`loss-${m}`, `${METHOD[m].short}: largest loss`]]));
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
        + `<p class="fn-figure-note">Predictions come from the seed-${data.protocol.explorer_seed} replicate${showAfter() ? ' of the original and the improved models' : ''}; per-seed scores for every image are in the per-image CSV. Masks are at the processed resolution without padding. Target pixel share is an image statistic, not physical area.</p></section>`
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
    const source = explorerSources()[explorerSources().length - 1];
    if (state.sort.startsWith('worst-')) {
        const run = explorerRun(state.sort.slice(6), state.budget, source);
        const score = s => (s.presence.some(Boolean) ? sampleMetrics(s, run?.id, source)?.flooded_miou : null) ?? 2;
        output.sort((a, b) => score(a) - score(b) || byId(a, b));
    } else if (state.sort.startsWith('gain-') || state.sort.startsWith('loss-')) {
        const method = state.sort.slice(5);
        const b = explorerRun(method, state.budget, 'before');
        const a = explorerRun(method, state.budget, 'after');
        const delta = s => {
            const x = sampleMetrics(s, b?.id, 'before')?.flooded_miou;
            const y = sampleMetrics(s, a?.id, 'after')?.flooded_miou;
            return s.presence.some(Boolean) && x != null && y != null ? y - x : null;
        };
        const sign = state.sort.startsWith('gain-') ? -1 : 1;
        output.sort((p, q) => (delta(p) == null) - (delta(q) == null) || sign * ((delta(p) ?? 0) - (delta(q) ?? 0)) || byId(p, q));
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
    const sources = explorerSources();
    const panels = [{ title: 'Original RGB', kind: 'rgb' }, { title: 'Ground truth', kind: 'truth' }];
    for (const method of METHODS) {
        for (const source of sources) {
            panels.push({
                title: sources.length > 1 ? `${METHOD[method].short} · ${source}` : METHOD[method].short,
                kind: source === 'after' ? `${method}@after` : method, source, method, run: explorerRun(method, state.budget, source)
            });
        }
    }
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
    const compare = explorerSources().length > 1;
    const sharp = FN_VIEWER.hiresUrl(selected.id) ? 'sharp 1024 px photo' : '256 px photo';
    $('#fn-viewer').innerHTML = `<div class="fn-sample-title"><h2>Image ${plain(selected.image_id)}</h2><div class="fn-sample-meta"><span class="fn-pill">${targetText(selected)}</span>${overlapBadge}<span>${pct(selected.target_fraction)} target pixels · ${selected.geometry.width}×${selected.geometry.height} input</span></div></div>`
        + `<div class="fn-comparison${compare ? ' fn-comparison-compare' : ''}">${figures.map(f => {
            const iou = f.run ? sampleMetrics(selected, f.run.id, f.source)?.flooded_miou : null;
            let delta = '';
            if (compare && f.source === 'after' && f.run) {
                const base = sampleMetrics(selected, explorerRun(f.method, state.budget, 'before')?.id, 'before')?.flooded_miou;
                if (iou != null && base != null) delta = ` <span class="fn-delta ${iou > base ? 'up' : iou < base ? 'down' : ''}">${points(iou - base)}</span>`;
            }
            const score = f.run ? `<span class="fn-figure-score">IoU ${pct(iou)}${delta}</span>` : '';
            const caption = [`Image ${selected.image_id}`, f.run ? `${state.budget}% labels · seed ${f.run.seed} · flooded IoU ${pct(iou)}` : '', sharp].filter(Boolean).join(' · ');
            const attrs = { group: 'explorer', title: f.title, caption, render: 'explorer', kind: f.kind };
            const body = f.kind === 'rgb' ? zoomable(`<img src="${esc(selected.rgb)}" alt="RGB image ${esc(selected.image_id)}">`, `${f.title}, image ${selected.image_id}`, attrs)
                : f.kind === 'truth' || f.run ? zoomable(`<canvas data-mask-kind="${f.kind}" aria-label="${esc(f.title)} for image ${esc(selected.image_id)}"></canvas>`, `${f.title}, image ${selected.image_id}`, { ...attrs, pixelated: '' })
                    : empty('Run not exported', '');
            const tag = f.source && compare ? `<span class="fn-state ${f.source}">${STATE_NAME[f.source]}</span>` : '';
            return `<figure class="${f.source ? `fn-panel-${f.source}` : ''}"><figcaption><span class="fn-caption-name">${tag}${esc(compare && f.method ? METHOD[f.method].short : f.title)}</span>${score}</figcaption>${body}</figure>`;
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
    const source = panelSource(kind);
    const run = explorerRun(panelMethod(kind), state.budget, source);
    if (!run) return null;
    return source === 'after' ? after.samples[sample.id]?.predictions[run.id] : sample.predictions[run.id];
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


function improvementCard() {
    if (!showAfter()) return '';
    const c = after.comparisons;
    const line = method => {
        const [a, b] = [25, 100].map(budget => c.find(x => x.method === method && x.budget === budget));
        return `<li><strong>${esc(METHOD[method].name)}</strong> (${esc(components(method).join(' + '))}): ${a.budget}% ${pct(mean(a.baseline_scores))} → ${pct(mean(a.variant_scores))} (${points(a.mean_difference)}, ${esc(a.decision)}); ${b.budget}% ${pct(mean(b.baseline_scores))} → ${pct(mean(b.variant_scores))} (${points(b.mean_difference)}, ${esc(b.decision)}).</li>`;
    };
    return card('What the improvement study changed', `<p class="fn-insight">The error analysis found that most false alarms fall on classes the 3-class task merges into "Other" (open water, non-flooded buildings), and that the Random Forest judges each superpixel in isolation. One pre-registered set of changes per method, chosen on validation, gave:</p>`
        + `<ul>${METHODS.map(line).join('')}</ul>`
        + `<p>The Random Forest's gain is large and consistent across seeds: representation (neighbour context and the ten original labels) was its limit. The U-Net family improved clearly at 25% labels; at 100% the combined U-Net helped two seeds and lost on the third, so the pre-registered rule reports no clear effect. The training recipe alone was the only change that clearly helped the U-Net and U-Net + CRF at 100% (see <a href="#improve">Improvements</a>); that is evidence for a follow-up, not a model selected on the test split.</p>`);
}

function interpretation() {
    let html = header('interpretation', 'What do the findings support?', 'Read the measured trade-offs together with the limits of the dataset, the resolution and the experiment design.');
    const complete = data.summary.length === METHODS.length * 2;
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
        const best = [...data.summary].filter(s => s.budget === 100).sort((a, b) => b.full.flooded_miou.mean - a.full.flooded_miou.mean)[0];
        const unetBeatsRf = runsFor('unet', 25).concat(runsFor('unet', 100)).every(u => data.runs.filter(r => r.method === 'rf').every(r => u.metrics.flooded_miou > r.metrics.flooded_miou));
        const bestAfter = showAfter() ? [...after.summary].filter(s => s.budget === 100).sort((a, b) => b.full.flooded_miou.mean - a.full.flooded_miou.mean)[0] : null;
        html += `<div class="fn-dark-card"><h3>Headline</h3><p>With all training labels, the original U-Net scores <strong>${points(gap).replace('+', '')}</strong> higher flooded-class mIoU than SLIC + Random Forest on the official test split (mean of ${data.protocol.train_seeds.length} seeds)${unetBeatsRf ? '; every U-Net run beats every RF run' : ''}. The best original approach at 100% labels is <strong>${esc(METHOD[best.method].name)}</strong> at ${meanSd(best.full.flooded_miou)}.`
            + (bestAfter ? ` After the improvement study the best at 100% is still ${esc(METHOD[bestAfter.method].name)} (${meanSd(bestAfter.full.flooded_miou)}); the largest gains were at 25% labels and for the Random Forest.` : '') + `</p></div>`;
        html += improvementCard();
        html += card('Data efficiency', `<p class="fn-insight">Going from 25% to 100% of the labels changes the original U-Net by ${describe(unet)}, and SLIC + RF by ${describe(rf)}.</p>`
            + `<p>These are seed-matched differences: the same seed is trained at both budgets, so the comparison isolates the labelled-data change. The U-Net gain varies a lot between seeds, so a single run could have under- or over-stated it. Three seeds show stability, not statistical significance.</p>`
            + `<p>SLIC + RF does use the extra labels: building recall goes ${recall('rf', 'building_recall')} and road recall ${recall('rf', 'road_recall')}. It also raises more false alarms, mostly in scenes without flooded infrastructure, so IoU does not improve. Superpixel colour and texture seem to be the limit: many RF false alarms fall on open floodwater, which looks much like a flooded road.${showAfter() ? ' Giving each superpixel its neighbours\' features and the ten original labels is what lifted it most in the improvement study.' : ''}</p>`);
        if (data.crf_effects.length && data.hybrid) {
            const effect = budget => data.crf_effects.find(e => e.budget === budget).full;
            const consistent = data.crf_effects.every(e => e.full.flooded_miou.values.every(v => v > 0)) ? 'improved every U-Net run at both budgets'
                : data.crf_effects.every(e => e.full.flooded_miou.values.every(v => v < 0)) ? 'lowered every U-Net run at both budgets' : 'did not change every U-Net run in the same direction';
            const classes = budget => `building IoU ${points(effect(budget).building_iou.mean)}, road IoU ${points(effect(budget).road_iou.mean)}`;
            html += card('Does CRF refinement help?', `<p class="fn-insight">On the test split, DenseCRF refinement changes the original flooded-class mIoU by ${points(effect(25).flooded_miou.mean)} at 25% labels and ${points(effect(100).flooded_miou.mean)} at 100%; it ${consistent}.</p>`
                + `<p>Per class: ${classes(25)} at 25%; ${classes(100)} at 100%. The setting was chosen on validation (mean gain ${points(data.hybrid.selected_mean_gain)}) before the test split was scored, and validation ${data.hybrid.adopted ? 'adopted' : 'rejected'} refinement. The CRF only reuses each U-Net's own probabilities and the image colours, so these changes come from sharper, colour-aware boundaries rather than new learning. The cost is roughly ${(summaryFor('hybrid', 100).inference_median_seconds.mean * 1000 - summaryFor('unet', 100).inference_median_seconds.mean * 1000).toFixed(0)} ms more CPU time per image.`
                + (showAfter() ? ` On the improved U-Net (smoother, test-time-augmented probabilities) the CRF adds less: ${points(after.crf_effects.find(e => e.budget === 100).full.flooded_miou.mean)} at 100%.` : '') + `</p>`);
        }
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
    html += card('Failure cases deserve a closer look', `<p>In the explorer, sort by a model's lowest target IoU to examine missed or confused infrastructure${showAfter() ? ', or, in the Compare view, by its largest gain or loss after the improvements' : ''}.</p><a class="fn-primary-btn" href="#explorer" data-failures="unet">Inspect U-Net failure cases →</a>`);
    html += card('Limits on interpretation', `<ul>${data.limitations.map(x => `<li>${esc(x)}</li>`).join('')}${showAfter() ? '<li>The improvement study was designed after the original test results had been seen. Its choices were fixed in advance or made on validation, and its test split was scored once after its own freeze; seven changes across several models are still many comparisons.</li>' : ''}</ul><p>SDG 11 and SDG 13 motivate the study. Real-world benefit for responders would need separate validation.</p>`);
    return html;
}

function reproducibility() {
    let html = header('reproducibility', 'Every result has a trail.', 'Saved configurations, fixed subset IDs, model hashes and evaluation artifacts connect this page to the underlying experiment.');
    html += card('Download the experiment records', `<div class="fn-downloads">${download('summary', 'Seed summary CSV')}${download('aggregate', 'All runs CSV')}${download('per_image', 'Per-image metrics CSV')}${download('overlap', 'Verified overlap pairs CSV')}${download('crf_tuning', 'CRF tuning CSV')}${download('budgets', 'Exact budget image IDs')}${download('experiment', 'Frozen 12-run manifest')}${download('hybrid_experiment', 'Hybrid freeze manifest')}</div>${Object.keys(data.downloads).length ? '' : empty('Downloads appear after export')}`);
    const p = data.protocol;
    html += card('Protocol at a glance', `<dl class="fn-definition-list"><dt>Export schema</dt><dd>Version ${data.schema_version}</dd><dt>Subset seed</dt><dd>${p.subset_seed}</dd><dt>Training seeds</dt><dd>${p.train_seeds.join(', ')}</dd>`
        + `<dt>Model input</dt><dd>${p.size}×${p.size}, aspect ratio preserved${showAfter() ? ' (improved U-Net and RF: 512×512, scored on the same 256 px grid)' : ''}</dd><dt>Budgets</dt><dd>Nested 25% / 100%, stratified by target presence</dd>`
        + `<dt>RF region sampling</dt><dd>${p.rf_region_sampling === 'keep_targets' ? 'Keep every flooded region' : 'Uniform'}, cap ${num(p.rf_max_regions)}</dd>`
        + (data.hybrid ? `<dt>Hybrid CRF</dt><dd>${esc(data.hybrid.backend)}; selected on validation (${data.hybrid.adopted ? 'adopted' : 'rejected'})</dd>` : '')
        + `<dt>Ignored label</dt><dd>255 (generated padding)</dd><dt>Exported</dt><dd>${plain(new Date(data.exported_at).toLocaleString())}</dd></dl>`);
    if (data.experiment_identity) {
        html += card('Frozen experiment identities', `<p class="fn-small fn-muted">12 trained runs and the overlap analysis</p><p class="fn-hash">${esc(data.experiment_identity)}</p>`
            + (data.hybrid_experiment_identity ? `<p class="fn-small fn-muted">Hybrid runs and their CRF selection (frozen after validation tuning, before their test evaluation)</p><p class="fn-hash">${esc(data.hybrid_experiment_identity)}</p>` : '')
            + (showAfter() ? `<p class="fn-small fn-muted">Improvement study (96 runs; the 18 combined runs are the "after" models), frozen ${plain(new Date(after.frozen_at).toLocaleString())}</p><p class="fn-hash">${esc(after.experiment_identity)}</p>` : '')
            + `<p>Checkpoints stay in the local experiment workspace. This static page reads only exported display files.</p>`);
    }
    const sources = compareMode() ? ['before', 'after'] : [viewSource()];
    const rows = sources.flatMap(source => bundle(source).runs.map(r => `<tr><td>${sources.length > 1 ? stateLabel(source) : ''} ${esc(r.id)}</td><td>${esc(METHOD[r.method].short)}</td><td>${r.budget}%</td><td>${r.seed}</td><td>${pct(r.metrics.flooded_miou)}</td><td><span class="fn-hash" style="display:inline-block;padding:3px 6px">${esc(r.checkpoint_sha256.slice(0, 16))}…</span></td></tr>`)).join('');
    if (rows) html += card('All runs', `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Run</th><th>Method</th><th>Labels</th><th>Seed</th><th>Flooded mIoU</th><th>Checkpoint SHA-256</th></tr></thead><tbody>${rows}</tbody></table></div><p class="fn-figure-note">A hybrid run lists the checkpoint of the U-Net run it refines; it trains no weights of its own.</p>`);
    if (data.reproducibility) {
        html += card('Configuration and software', `<details class="fn-details"><summary>Saved configuration</summary><pre class="fn-code">${esc(JSON.stringify(data.reproducibility.config, null, 2))}</pre></details>`
            + (data.reproducibility.crf ? `<details class="fn-details"><summary>Selected CRF settings</summary><pre class="fn-code">${esc(JSON.stringify(data.reproducibility.crf, null, 2))}</pre></details>` : '')
            + `<details class="fn-details"><summary>Software versions</summary><pre class="fn-code">${esc(JSON.stringify(data.reproducibility.versions, null, 2))}</pre></details>`);
    }
    html += card('Sources and attribution', `<ul>${data.sources.map(s => `<li><a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a></li>`).join('')}</ul><p>Dataset: Rahnemoonfar et al., FloodNet, IEEE Access (2021). The project README lists every command needed to reproduce these results.</p>`);
    return html;
}


function errorShare(method, budget, which, target, name) {
    return after.errors_seed42[which === 'before' ? 'baseline' : 'combined']?.[`${method}_${budget}`]?.[target]?.[name] ?? 0;
}

function single(variant, method, budget) {
    return after.single_changes.find(x => x.variant === variant && x.method === method && x.budget === budget)?.mean_difference;
}

function classChange(c, key) {
    return mean(c.variant_classes[key]) - mean(c.baseline_classes[key]);
}

function resultTile(method, budget) {
    const c = comparisonFor(method, budget);
    if (!c) return '';
    return `<div class="fn-result-tile"><span class="fn-result-label">${budget}% labels</span>`
        + `<strong>${pct(mean(c.baseline_scores))} → ${pct(mean(c.variant_scores))}</strong>`
        + `<span>${points(c.mean_difference)} · 95% CI ${interval(c.ci)}</span>${decisionPill(c.decision)}</div>`;
}

function methodologyTable(method) {
    const rows = after.methodology[method].map(r => `<tr class="${r.changed ? 'fn-changed' : 'fn-unchanged'}"><th scope="row">${esc(r.aspect)}${r.changed ? ' <span class="fn-pill good">changed</span>' : ''}</th>`
        + `<td>${esc(r.before)}</td><td>${r.changed ? `<strong>${esc(r.after)}</strong>` : '<span class="fn-muted">same</span>'}</td></tr>`).join('');
    return `<div class="fn-table-wrap"><table class="fn-table fn-method-diff"><thead><tr><th>Step</th><th>${stateLabel('before')} Original pipeline</th><th>${stateLabel('after')} Improved pipeline</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function largestDrop(runs) {
    return Math.max(...runs.map(r => {
        const v = r.history.map(e => e.val_flooded_miou);
        return Math.max(0, ...v.slice(1).map((x, i) => v[i] - x));
    }));
}


const FN_TEACH = {
    neighbours() {
        const cells = [];
        for (let y = 0; y < 5; y++) {
            for (let x = 0; x < 5; x++) {
                const d = Math.abs(x - 2) + Math.abs(y - 2);
                cells.push(`<span class="${d === 0 ? 'centre' : d === 1 ? 'touch' : x > 0 && x < 4 && y > 0 && y < 4 ? 'coarse' : ''}"></span>`);
            }
        }
        return `<figure class="fn-teach"><div class="fn-teach-cells">${cells.join('')}</div>`
            + `<figcaption><b class="k centre"></b>the region being classified <b class="k touch"></b>its touching neighbours (their average is added) <b class="k coarse"></b>the larger coarse region it lies in (its colour is added). Schematic: real superpixels have irregular shapes.</figcaption></figure>`;
    },

    labels() {
        const other = ['background', 'non-flooded building', 'non-flooded road', 'water', 'tree', 'vehicle', 'pool', 'grass'];
        return `<figure class="fn-teach"><div class="fn-teach-labels"><div><p>3 classes (before)</p><div class="fn-chip other">Other<small>${other.join(' · ')}</small></div><div class="fn-chip b">flooded building</div><div class="fn-chip r">flooded road</div></div>`
            + `<span class="fn-teach-arrow">→</span><div><p>10 classes (after, for learning)</p>${other.map(o => `<div class="fn-chip${o === 'water' ? ' w' : ''}">${o}</div>`).join('')}<div class="fn-chip b">flooded building</div><div class="fn-chip r">flooded road</div></div></div>`
            + `<figcaption>With three classes, water is "the same thing" as grass and ordinary roads. With ten, the model is told explicitly that water is not a road.</figcaption></figure>`;
    },

    resolution() {
        const grid = (n, size) => {
            let cells = '';
            const s = size / n;
            for (let y = 0; y < n; y++) {
                for (let x = 0; x < n; x++) {
                    const cx = (x + .5) / n;
                    const cy = (y + .5) / n;
                    const road = Math.abs(cy - (0.15 + 0.7 * cx)) < 0.07 + 0.5 / n;
                    cells += `<rect x="${x * s}" y="${y * s}" width="${s}" height="${s}" fill="${road ? '#0891b2' : '#e2e8f0'}" stroke="white" stroke-width="${n > 10 ? .5 : 1}"/>`;
                }
            }
            return `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" role="img" aria-label="${n} by ${n} pixels">${cells}</svg>`;
        };
        return `<figure class="fn-teach"><div class="fn-teach-pair"><div>${grid(8, 112)}<small>same ground, 256 px</small></div><div>${grid(16, 112)}<small>same ground, 512 px</small></div></div>`
            + `<figcaption>A thin road at the lower resolution becomes a few blocky pixels; at twice the resolution its shape and edges survive.</figcaption></figure>`;
    },

    schedule() {
        const w = 240, h = 92, n = 60;
        const x = i => 8 + i / (n - 1) * (w - 16);
        const y = v => 10 + (1 - v) * (h - 30);
        const cosine = Array.from({ length: n }, (_, i) => `${x(i)},${y(0.01 + 0.99 * (1 + Math.cos(Math.PI * i / (n - 1))) / 2)}`).join(' ');
        return `<figure class="fn-teach"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Constant learning rate versus cosine decay">`
            + `<polyline fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="5 4" points="${x(0)},${y(1)} ${x(29)},${y(1)}"/>`
            + `<polyline fill="none" stroke="#0891b2" stroke-width="2.4" points="${cosine}"/>`
            + `<text x="${w - 6}" y="${y(1) + 3}" text-anchor="end" class="fn-teach-text">before: constant</text>`
            + `<text x="${w - 6}" y="${h - 6}" text-anchor="end" class="fn-teach-text accent">after: big steps first, tiny last</text></svg>`
            + `<figcaption>Step size (learning rate) over training.</figcaption></figure>`;
    },

    curves(budget, seed) {
        const b = runsFor('unet', budget).find(r => r.seed === seed);
        const a = runsFor('unet', budget, 'after').find(r => r.seed === seed);
        if (!b || !a) return '';
        const w = 260, h = 110;
        const n = Math.max(b.history.length, a.history.length);
        const x = i => 8 + i / (n - 1) * (w - 16);
        const y = v => 8 + (1 - v / 0.8) * (h - 26);
        const line = (run, colour, dash) => `<polyline fill="none" stroke="${colour}" stroke-width="2" ${dash ? 'stroke-dasharray="4 3"' : ''} points="${run.history.map((e, i) => `${x(i)},${y(e.val_flooded_miou)}`).join(' ')}"/>`;
        return `<figure class="fn-teach"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Validation score per epoch, original and improved U-Net">${line(b, '#94a3b8', true)}${line(a, '#0891b2', false)}`
            + `<text x="8" y="${h - 4}" class="fn-teach-text">epoch 1</text><text x="${w - 8}" y="${h - 4}" text-anchor="end" class="fn-teach-text">epoch ${n}</text></svg>`
            + `<figcaption>Real validation flooded mIoU per epoch, U-Net at ${budget}% labels, seed ${seed}: grey dashed = original, teal = improved. The original jumps; the improved one settles.</figcaption></figure>`;
    },

    views() {
        const views = [0, 90, 180, 270].flatMap(r => [`rotate(${r}deg)`, `rotate(${r}deg) scaleX(-1)`]);
        return `<figure class="fn-teach"><div class="fn-teach-views">${views.map(t => `<span><i style="transform:${t}">F</i></span>`).join('')}</div>`
            + `<figcaption>The 8 ways to turn or mirror an image (shown with the letter F). Each is predicted, turned back, and the 8 predictions are averaged.</figcaption></figure>`;
    },

    bottleneck() {
        const boxes = (n, cls) => Array.from({ length: n }, () => `<span class="${cls}"></span>`).join('');
        return `<figure class="fn-teach"><div class="fn-teach-bottleneck"><div><p>last layer: 8 channels</p>${boxes(8, 'ch')}</div><span class="fn-teach-arrow">→</span>`
            + `<div><p>3 class scores</p>${boxes(3, 'out3')}<p>+ 10 auxiliary scores</p>${boxes(10, 'out10')}</div></div>`
            + `<figcaption>The extra 10-class head had to share the same 8 numbers per pixel with the 3-class head.</figcaption></figure>`;
    },

    neighbourhood() {
        const panel = (n, r, label) => {
            const s = 112 / n;
            let cells = '';
            for (let i = 0; i <= n; i++) cells += `<line x1="${i * s}" y1="0" x2="${i * s}" y2="112" stroke="#e2e8f0"/><line x1="0" y1="${i * s}" x2="112" y2="${i * s}" stroke="#e2e8f0"/>`;
            return `<div><svg viewBox="0 0 112 112" width="112" height="112" role="img" aria-label="${label}">${cells}<circle cx="56" cy="56" r="${r}" fill="rgba(8,145,178,.15)" stroke="#0891b2" stroke-width="2"/><circle cx="56" cy="56" r="3" fill="#0f172a"/></svg><small>${label}</small></div>`;
        };
        return `<figure class="fn-teach"><div class="fn-teach-pair">${panel(8, 30, 'radius sxy at 256 px')}${panel(16, 30, 'radius 2 × sxy at 512 px')}</div>`
            + `<figcaption>The CRF measures distance in pixels. At 512 px a pixel covers a quarter of the ground, so doubling the distances keeps the same patch of ground.</figcaption></figure>`;
    }
};

function idea(number, { variant, method, title, logic, how, eq, figure, happened, lesson }) {
    const kept = after.components[method === 'rf' ? 'rf' : 'unet'].includes(variant);
    const alone = [25, 100].map(budget => {
        const c = after.single_changes.find(x => x.variant === variant && x.method === method && x.budget === budget);
        return c ? `${budget}%: <strong>${points(c.mean_difference)}</strong> ${decisionPill(c.decision)}` : '';
    }).join(' &nbsp; ');
    const evidence = after.combine.evidence[`${variant}/${method === 'rf' ? 'rf' : 'unet'}`];
    const validation = evidence ? Object.entries(evidence.differences).sort(([a], [b]) => a - b).map(([b, d]) => `${b}%: ${d.map(points).join(', ')}`).join('; ') : '';
    return `<article class="fn-idea ${kept ? 'kept' : 'dropped'}"><header><span class="fn-idea-num">Idea ${number}</span><h3>${esc(title)}</h3>`
        + `<span class="fn-pill ${kept ? 'good' : 'warn'}">${kept ? '✓ kept' : '✗ left out'}</span></header>`
        + `<div class="fn-idea-body"><div>`
        + `<p class="fn-idea-label">The logic</p><p>${logic}</p>`
        + `<p class="fn-idea-label">How it works</p><ol>${how.map(step => `<li>${step}</li>`).join('')}</ol>${eq ? FN_MATH.equations(eq) : ''}`
        + `<p class="fn-idea-label">What happened</p><p>${happened}</p>`
        + `<p class="fn-idea-result">On its own, on the test split: ${alone}</p>`
        + (validation ? `<p class="fn-small fn-muted">Validation, per seed (this decided ${kept ? 'to keep it' : 'to leave it out'}): ${validation}.</p>` : '')
        + (lesson ? `<p class="fn-idea-lesson">${lesson}</p>` : '')
        + `</div><div class="fn-idea-figures">${figure || ''}</div></div></article>`;
}

function chapter(method, title, intro, ideas, together) {
    return `<section class="fn-card fn-chapter ${method}"><div class="fn-card-top"><h2>${esc(title)}</h2>${methodPill(method)}</div>`
        + `<p class="fn-chapter-step">At first</p>${intro}`
        + (ideas.length ? `<p class="fn-chapter-step">What we tried</p><div class="fn-ideas">${ideas.join('')}</div>` : '')
        + `<p class="fn-chapter-step">Putting it together</p><div class="fn-result-row">${resultTile(method, 25)}${resultTile(method, 100)}</div>${together}`
        + `<details class="fn-details"><summary>The full pipeline, step by step (before vs after)</summary>${methodologyTable(method)}</details></section>`;
}

function changesPage() {
    let html = header('changes', 'How the models got better, step by step.', 'What went wrong at first, which ideas were tried, the logic behind each one, how it works, and what it did. The "After" models are the ideas that survived, combined.');
    if (!after) return html + empty('Improvement study not exported', 'Run "floodnet improve export-after" to compare the original and improved models.');
    const s = after.study;
    const c = (m, b) => comparisonFor(m, b);
    const before = m => runsFor(m, 100);
    const dropBefore = largestDrop(before('unet'));
    const dropAfter = largestDrop(runsFor('unet', 100, 'after'));
    const crfBefore = data.crf_effects.find(e => e.budget === 100)?.full.flooded_miou.mean;
    const crfAfter = after.crf_effects.find(e => e.budget === 100)?.full.flooded_miou.mean;
    const rfWater = which => errorShare('rf', 100, which, 'road', 'water');
    const unetWater = which => errorShare('unet', 100, which, 'road', 'water');
    const unetBuilding = which => errorShare('unet', 100, which, 'building', 'building non-flooded');

    html += takeaways([
        `<strong>At first</strong>, the models confused things that only look alike: open floodwater with flooded roads, dry buildings with flooded ones. The Random Forest reached only about ${pct(summaryFor('rf', 100).full.flooded_miou.mean, 0)} flooded mIoU, and the U-Net's training was jumpy.`,
        `<strong>We tried ${s.changes_tested} ideas</strong>, each on its own first, and kept only those that helped on validation images. Then we trained the kept ideas together and scored the held-out test split once.`,
        `<strong>The result</strong>: the Random Forest gained about ${points(mean([c('rf', 25).mean_difference, c('rf', 100).mean_difference]))} at both budgets; the U-Net and U-Net + CRF gained clearly with 25% of the labels (${points(c('unet', 25).mean_difference)}, ${points(c('hybrid', 25).mean_difference)}), but not clearly with all labels.`
    ], 'The story in one minute');

    html += card('At first: where the models went wrong', `<p>Every model answers one question per pixel: is this a <em>flooded building</em>, a <em>flooded road</em>, or <em>Other</em>? Looking at the original models' mistakes on the test images showed a pattern:</p>`
        + `<ul><li><strong>They mistook look-alikes.</strong> Brown floodwater looks like a flooded road, and an ordinary roof looks like a flooded one. Of the U-Net's flooded-road false alarms (100% labels), ${pct(unetWater('before'), 0)} were open water; ${pct(unetBuilding('before'), 0)} of its flooded-building false alarms were buildings that were not flooded. For the Random Forest, ${pct(rfWater('before'), 0)} of its road false alarms were water.</li>`
        + `<li><strong>The task hid the difference.</strong> The three-class labels lump water, grass, trees, ordinary roads and dry buildings into one class, "Other". The models were never told that water and grass are different things, so they never learned the line that mattered most.</li>`
        + `<li><strong>The Random Forest looked through a keyhole.</strong> It judged each superpixel only by its own colour and texture, with no idea what surrounded it.</li>`
        + `<li><strong>The U-Net's training was jumpy.</strong> Its validation score fell by up to ${points(dropBefore).replace('+', '')} from one epoch to the next, so which epoch got saved was partly luck.</li>`
        + `<li><strong>Detail was thrown away.</strong> The 4000 × 3000 photos were shrunk to 256 px, so thin roads and edges became a few blocky pixels.</li></ul>`
        + `<p>More data would not have fixed this: most test scenes already overlap training flights. The ideas below target these specific problems instead.</p>`);

    html += card('How we decided what to keep (a fair test)', `<div class="fn-flow fn-flow-steps">`
        + `<div><b>1</b>Write down the ideas and the rules before training</div><span>→</span>`
        + `<div><b>2</b>Try each idea alone: ${num(s.component_runs)} runs</div><span>→</span>`
        + `<div><b>3</b>Keep it only if it helps on validation</div><span>→</span>`
        + `<div><b>4</b>Train the kept ideas together: ${num(after.runs.length)} runs</div><span>→</span>`
        + `<div><b>5</b>Freeze everything, then open the test split once</div></div>`
        + `<p>An idea was kept if, on the validation images, it raised flooded mIoU on average at both budgets and for at least 2 of the 3 seeds at each budget. The test split stayed sealed until the end, so the final numbers are not tuned to it. Each "before vs after" comparison uses the same images, budgets and seeds, and counts as "helps" only if its 95% interval is above zero and every seed improved.</p>`);

    const rfIdeas = [
        idea(1, {
            variant: 'rf_context', method: 'rf', title: 'Let each region see its neighbours',
            logic: 'Up close, a patch of brown floodwater and a patch of flooded road can look identical. People tell them apart by the surroundings: a road is a long strip between houses and lawns, open water is a wide area surrounded by more water. The original forest saw only the patch itself, like judging a jigsaw piece without the picture on the box.',
            how: ['Split the photo into superpixels as before and compute the same 22 colour and texture numbers for each.',
                'For every superpixel, average those 22 numbers over the superpixels that touch it, and add them as 22 new features.',
                'Add where it sits in the frame (centre y, x) and how big it is (share of the image).',
                'Split the photo a second time into about 60 large regions, and add the average colour of the large region it lies in.',
                'Train the same forest on these 50 numbers instead of 22.'],
            figure: FN_TEACH.neighbours(),
            happened: `This was the biggest single gain for the forest, at both budgets and for every seed. Its precision rose most: it stopped calling isolated brown patches "flooded road".`
        }),
        idea(2, {
            variant: 'aux10', method: 'rf', title: 'Give it the full answer key (10 classes)',
            logic: 'With three classes, the forest is told that water and grass are "the same thing", so it cannot learn the boundary that matters most: water versus flooded road. The original FloodNet labels already separate ten classes; using them costs nothing.',
            how: ['Label each training superpixel with the majority of the ten original classes inside it (water, grass, trees, ordinary road, flooded road, …).',
                'Train the forest to tell all ten apart.',
                'At prediction time, map its answer back: flooded building stays flooded building, flooded road stays flooded road, everything else becomes "Other". Scoring is unchanged.'],
            figure: FN_TEACH.labels(),
            happened: `It helped at both budgets and for every seed. Flooded-road false alarms on open water dropped because water is now a class of its own.`
        }),
        idea(3, {
            variant: 'res512', method: 'rf', title: 'Look at the photo in more detail (512 px)',
            logic: 'The photos are 4000 × 3000 pixels; at 256 px most detail is gone. With twice the resolution, each superpixel holds four times as many pixels, so its colour and texture numbers are less noisy and its outline follows real edges better.',
            how: ['Prepare the same images at 512 × 512 px instead of 256 × 256.',
                'Keep the same number of superpixels (250), now each about four times larger in pixels.',
                'Predict at 512 px, then average the prediction back onto the same 256 px grid, so it is scored exactly like the original.'],
            eq: ['area'], figure: FN_TEACH.resolution(),
            happened: 'A small but consistent gain at both budgets.'
        })
    ];
    html += chapter('rf', `SLIC + Random Forest: ${pct(mean(c('rf', 100).baseline_scores))} → ${pct(mean(c('rf', 100).variant_scores))}`,
        `<p>The forest was the weakest model: about ${pct(mean(c('rf', 25).baseline_scores), 0)} flooded mIoU, and more labels did not help it (it simply learned more look-alikes). ${pct(rfWater('before'), 0)} of its flooded-road false alarms were open water. Its two blind spots were the labels (water hidden in "Other") and the keyhole view of one superpixel at a time.</p>`,
        rfIdeas,
        `<p>All three ideas were kept and work together: about ${points(mean([c('rf', 25).mean_difference, c('rf', 100).mean_difference]))} at both budgets, more than any single idea, and every seed improved. Precision rose by ${points(classChange(c('rf', 100), 'building_precision'))} (building) and ${points(classChange(c('rf', 100), 'road_precision'))} (road) at 100%, and flooded-road false alarms on open water fell from ${pct(rfWater('before'), 0)} to ${pct(rfWater('after'), 0)}. The lesson: for the forest, <strong>a better description of each region mattered far more than more data</strong>.</p>`);

    const unetIdeas = [
        idea(1, {
            variant: 'recipe', method: 'unet', title: 'Train more calmly (learning-rate decay + weight averaging)',
            logic: `Training is like walking downhill in fog with a fixed stride: near the bottom, big strides keep overshooting the valley, so you end up somewhere on its walls. That is why the original validation score fell by up to ${points(dropBefore).replace('+', '')} between consecutive epochs. Two fixes: take smaller steps as you get close (decay the learning rate), and report the average of your last positions rather than the last one (weight averaging).`,
            how: ['Start with the same step size (1e-4) and lower it smoothly along a cosine curve to 1e-6 by the last epoch.',
                'After every training step, update a running average of the weights: 99.9% of the old average plus 0.1% of the new weights.',
                'Validate and save the averaged weights, not the raw ones.',
                'Allow up to 60 epochs (instead of 30) and wait 12 epochs without improvement before stopping (instead of 6), because the slower, steadier training needs the time.'],
            eq: ['cosine', 'ema'], figure: FN_TEACH.schedule() + FN_TEACH.curves(100, 42),
            happened: `Training settled: the largest one-epoch drop fell from ${points(dropBefore).replace('+', '')} to ${points(dropAfter).replace('+', '')} (100% labels). It helped the U-Net clearly at both budgets.`
        }),
        idea(2, {
            variant: 'tta', method: 'unet', title: 'Look at each photo 8 ways and average',
            logic: 'An aerial photo has no "up": a flooded road is a flooded road whichever way the drone faced. Yet a network can make small mistakes that depend on orientation. Asking the same question eight ways and averaging the answers cancels those random slips, like asking eight people and going with the consensus.',
            how: ['Make the 8 versions of the photo: 4 rotations, each with and without a mirror flip.',
                'Predict class probabilities for each version.',
                'Turn each prediction back to the original orientation.',
                'Average the 8 probability maps and pick the most likely class per pixel. No retraining is needed.'],
            eq: ['tta'], figure: FN_TEACH.views(),
            happened: 'A reliable gain at both budgets, for free in training time, but every prediction now costs 8 forward passes.'
        }),
        idea(3, {
            variant: 'res512', method: 'unet', title: 'Look at the photo in more detail (512 px)',
            logic: 'Thin roads and the exact edge of a flooded area are a few pixels wide at 256 px. At twice the resolution the network sees their shape, so boundaries, where most of its errors were, should improve.',
            how: ['Prepare the same images at 512 × 512 px.',
                'Train the same U-Net on them (4× the pixels, so each epoch takes longer).',
                'Average the 512 px prediction back onto the 256 px grid for scoring.'],
            figure: FN_TEACH.resolution(),
            happened: 'A large gain at 25% labels; at 100% the seeds disagreed (one seed lost several points). It passed the validation rule, so it was kept, but it is the reason the combined U-Net is not clearly better at 100%.'
        }),
        idea(4, {
            variant: 'aux10', method: 'unet', title: 'Also ask the network to name all 10 classes',
            logic: 'The same idea that worked for the forest: if the network must also say "water" or "grass", it has to learn those differences, and the flooded-road decision should benefit.',
            how: ['Add 10 extra outputs per pixel to the U-Net, one per original FloodNet class.',
                'Train on both: loss = the usual 3-class loss + 0.5 × the 10-class loss.',
                'Predict with the 3-class outputs only.'],
            eq: ['aux'], figure: FN_TEACH.bottleneck(),
            happened: `Mixed for the U-Net (the seeds disagreed) and harmful with the CRF (${points(single('aux10', 'hybrid', 100))} for the U-Net + CRF at 100%). The network\'s last layer has only 8 numbers per pixel, and the 13 outputs had to share them. The 3-class answer stayed mostly right, but its confidence collapsed towards one third per class, and the CRF, which trusts confident pixels, then washed the flooded areas away.`,
            lesson: 'Lesson: the idea is sound, but it needs its own, wider output branch. Left for future work.'
        }),
        idea(5, {
            variant: 'encoder', method: 'unet', title: 'Use a bigger network (ResNet-34)',
            logic: 'A deeper encoder can learn more complex patterns. If the U-Net were limited by its size, a bigger one should help.',
            how: ['Swap the ResNet-18 encoder for the deeper ResNet-34 (also pretrained on ImageNet).', 'Keep everything else the same.'],
            happened: `It was worse at both budgets. With ${num(data.dataset.counts.train)} training images, the extra capacity found no new signal. The problem was never the size of the network; it was the labels, the training stability and the detail.`,
            lesson: 'Lesson: bigger is not better when the bottleneck is the task and the training, not the model.'
        })
    ];
    html += chapter('unet', `ResNet-18 U-Net: ${pct(mean(c('unet', 25).baseline_scores))} → ${pct(mean(c('unet', 25).variant_scores))} at 25% labels`,
        `<p>The U-Net was already much better than the forest, but its training was jumpy (validation dropped by up to ${points(dropBefore).replace('+', '')} between epochs), it saw only 256 px, and ${pct(unetWater('before'), 0)} of its flooded-road false alarms were open water. Five ideas were tried.</p>`,
        unetIdeas,
        `<p>Three ideas were kept: the calmer training, test-time augmentation and 512 px input. Together they gave ${points(c('unet', 25).mean_difference)} at 25% labels, with every seed improving (${c('unet', 25).differences.map(points).join(', ')}). At 100% the gain (${points(c('unet', 100).mean_difference)}) is not clear: two seeds improved, one got worse (${c('unet', 100).differences.map(points).join(', ')}). The 512 px part is the likely cause: on their own, the calmer training and test-time augmentation each helped clearly at 100%, while 512 px did not. ${unetWater('after') < unetWater('before') ? `Flooded-road false alarms on open water fell from ${pct(unetWater('before'), 0)} to ${pct(unetWater('after'), 0)}` : `One error did not go away: at 100% labels the share of flooded-road false alarms on open water stayed high (${pct(unetWater('before'), 0)} → ${pct(unetWater('after'), 0)}, seed 42), because the 10-class idea that targeted it was left out for the U-Net`}.</p>`
        + `<p>The lesson: with fewer labels, steadier training and more careful prediction make the biggest difference; with all labels, the original U-Net was already close to what this setup can do.</p>`);

    html += chapter('hybrid', `U-Net + CRF: ${pct(mean(c('hybrid', 25).baseline_scores))} → ${pct(mean(c('hybrid', 25).variant_scores))} at 25% labels`,
        `<p>The CRF is a clean-up step: it takes the U-Net's probabilities and makes neighbouring pixels of similar colour agree, which sharpens ragged edges. It learns nothing itself, so it can only be as good as the U-Net under it. Before, it added ${points(crfBefore)} on top of its U-Net (100% labels).</p>`
        + `<p class="fn-chapter-step">What we changed</p><article class="fn-idea kept"><header><span class="fn-idea-num">Adjustment</span><h3>Keep the same CRF, give it the better U-Net, and rescale it</h3><span class="fn-pill good">applied</span></header><div class="fn-idea-body"><div>`
        + `<p class="fn-idea-label">The logic</p><p>No new idea was tested for the CRF. It automatically benefits from a better U-Net. One detail needs care: the CRF measures "nearby" in pixels. At 512 px, the same patch of ground spans twice as many pixels in each direction, so its distances must double to mean the same thing.</p>`
        + `<p class="fn-idea-label">How it works</p><ol><li>Take the improved U-Net's probabilities (already averaged over the 8 views).</li><li>Run the same DenseCRF with the same colour setting and weights, but with its pixel distances doubled (spatial 3 → 6 px, appearance 20 → 40 px).</li><li>No new tuning: the setting chosen on validation for the original study is reused.</li></ol>${FN_MATH.equations(['crf'])}`
        + `<p class="fn-idea-label">What happened</p><p>The CRF's own contribution shrank from ${points(crfBefore)} to ${points(crfAfter)} (100% labels), because the improved U-Net's predictions are already smoother: test-time augmentation and weight averaging do part of the clean-up the CRF used to do.</p>`
        + `</div><div class="fn-idea-figures">${FN_TEACH.neighbourhood()}</div></div></article>`,
        [],
        `<p>The hybrid follows its U-Net: ${points(c('hybrid', 25).mean_difference)} at 25% labels (every seed improved), ${points(c('hybrid', 100).mean_difference)} at 100% (no clear change). The lesson: post-processing helps most when the model under it is rough. Improve the model, and the clean-up step has less to do.</p>`);

    html += card('What we learned', `<ol class="fn-lessons">`
        + `<li><strong>Find out where the errors come from before changing the model.</strong> The error analysis pointed straight at the fixes that worked.</li>`
        + `<li><strong>How the task and the inputs are described can matter more than the model.</strong> The Random Forest gained about ${points(mean([c('rf', 25).mean_difference, c('rf', 100).mean_difference]))} from better features and labels, with the same forest.</li>`
        + `<li><strong>Stable training beats a bigger network.</strong> The calmer training helped; ResNet-34 hurt.</li>`
        + `<li><strong>An idea that works for one model can fail for another.</strong> The ten classes helped the forest and hurt the U-Net + CRF, for a reason that can be fixed.</li>`
        + `<li><strong>Test fairly, and report what did not work.</strong> Ideas were chosen on validation and the test split was opened once. At 100% labels the U-Net family did not clearly improve, and that is stated, not hidden.</li></ol>`);

    html += card('What did not change', `<ul><li>The data: the same audited images, 256 px ground truth, three target classes, 25% subset and seeds (${data.protocol.train_seeds.join(', ')}).</li>`
        + `<li>The scoring: every prediction, including the 512 px ones, is scored on the same 256 px test grid against the same masks, with the same flooded mIoU.</li>`
        + `<li>The protocol: the ideas and rules were written down on ${plain(new Date(s.preregistered_at).toLocaleDateString())} before training, choices were made on validation, and the test split was scored once after a freeze.</li></ul>`);

    html += card('What it cost', (() => {
        const rows = METHODS.map(method => [25, 100].map(budget => {
            const b = summaryFor(method, budget);
            const a = summaryFor(method, budget, 'after');
            const train = method === 'hybrid' ? 'no training (reuses the U-Net)' : `${seconds(b.training_seconds.mean)} → ${seconds(a.training_seconds.mean)}`;
            return `<tr><td>${esc(METHOD[method].short)} · ${budget}%</td><td>${train}</td><td>${(b.inference_median_seconds.mean * 1000).toFixed(1)} ms → ${(a.inference_median_seconds.mean * 1000).toFixed(1)} ms</td></tr>`;
        }).join('')).join('');
        return `<p>Better scores were paid for in compute: 512 px inputs, longer training and 8 forward passes per image.</p><div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Model</th><th>Training time (mean of 3 seeds)</th><th>Median prediction time per image</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    })());
    return html;
}


const PAGES = {
    overview, dataset: datasetPage, preprocessing: () => FN_PREP.page(), methodology, walkthrough: () => FN_WALKTHROUGH.page(),
    results: resultsPage, changes: changesPage, imbalance: () => FN_IMBALANCE.page(), improve: () => FN_IMPROVE.page(), training: trainingPage,
    explorer: explorerPage, interpretation, reproducibility
};

function renderNav() {
    const items = navItems();
    $('#fn-nav').innerHTML = NAV.map(g => `<p class="fn-nav-label">${esc(g.group)}</p>`
        + items.filter(item => item.group === g.group).map(item => `<a class="fn-nav-btn" href="#${item.route}"><span>${item.number}</span>${esc(item.label)}</a>`).join('')).join('');
    const route = currentRoute();
    $$('.fn-nav-btn').forEach(link => link.classList.toggle('active', link.hash === `#${route}`));
}

function updateStatus() {
    const status = $('#fn-export-status');
    if (!status || !data) return;
    status.textContent = data.status === 'complete'
        ? `${data.runs.length} / ${data.protocol.expected_runs} evaluations complete${showAfter() ? ` · ${after.runs.length} improved` : ''}`
        : 'Experiment in progress';
    status.parentElement.classList.toggle('complete', data.status === 'complete');
}

function syncViewButtons() {
    $$('[data-view]').forEach(button => {
        const on = button.dataset.view === state.view;
        button.classList.toggle('on', on);
        button.setAttribute('aria-pressed', String(on));
        if (button.dataset.view !== 'before') button.disabled = !after;
    });
    const note = $('#fn-view-note');
    if (note) {
        note.innerHTML = !after ? 'The improvement-study export is not available; showing the original models.'
            : state.view === 'before' ? 'Before: the original models, exactly as in the proposed methodology (three classes, 256 px).'
                : `${esc({ after: 'After: the improvement study\'s pre-registered combined models.', compare: 'Compare: originals and improved models side by side.' }[state.view])} <a href="#changes">What does “After” change? →</a>`;
    }
    const copy = $('#fn-sidebar-copy');
    if (copy) copy.textContent = `Three approaches. Two data budgets. Three seeds. One shared protocol${showAfter() ? ', then one pre-registered round of improvements' : ''}.`;
    renderNav();
    updateStatus();
}

function render(scrollTop = true) {
    if (AFTER_ONLY.has(currentRoute()) && !showAfter()) history.replaceState(null, '', '#overview');
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
    if (route === 'imbalance') FN_IMBALANCE.mount();
    if (route === 'improve') FN_IMPROVE.mount();
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

function saveView() {
    try { localStorage.setItem('fn-view', state.view); } catch { }
}

function loadView() {
    try {
        const saved = localStorage.getItem('fn-view');
        if (['before', 'after', 'compare'].includes(saved)) state.view = saved;
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
    const view = target.closest('[data-view]');
    if (view) {
        if (view.disabled) return;
        state.view = view.dataset.view;
        saveView();
        syncViewButtons();
        render(false);
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
    loadView();
    try {
        const hires = FN_VIEWER.loadHires();
        const response = await fetch('data/results.json', { cache: 'no-store' });
        const afterResponse = await fetch('data/after.json', { cache: 'no-store' }).catch(() => null);
        await hires;
        if (!response.ok) throw new Error(`Results export returned HTTP ${response.status}`);
        data = await response.json();
        if (data.schema_version !== 3 || !Array.isArray(data.runs) || !Array.isArray(data.samples)) {
            throw new Error('Unsupported or invalid results export');
        }
        if (afterResponse?.ok) {
            const candidate = await afterResponse.json();
            if (candidate.schema_version === 1 && candidate.status === 'complete' && Array.isArray(candidate.runs)) after = candidate;
        }
        if (!after) state.view = 'before';
        state.metricRun = data.runs[data.runs.length - 1]?.id ?? null;
        syncViewButtons();
        render();
    } catch (error) {
        status.textContent = 'Export unavailable';
        $('#fn-page').innerHTML = header('FloodNet / Results explorer', 'The export is not available yet.')
            + empty('Serve this app over HTTP', `${error.message}. Run the documented export-web command, then start the local HTTP server. No results have been fabricated.`);
    }
})();
