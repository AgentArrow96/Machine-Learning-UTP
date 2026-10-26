'use strict';

const FN_IMPROVE = {
    data: null,
    failed: null,
    budget: 100,
    method: 'unet',

    COLOURS: { baseline: '#64748b', aux10: '#2a78d6', recipe: '#eb6834', tta: '#1baf7a', res512: '#eda100', rf_context: '#e87ba4', encoder: '#008300', combined: '#6250d6' },
    NAMES: { baseline: 'Original (frozen)', aux10: '10-class supervision', recipe: 'Training recipe', tta: 'Test-time augmentation', res512: '512-px input', rf_context: 'RF context features', encoder: 'ResNet-34 encoder', combined: 'Combined winners' },
    ORDER: ['aux10', 'recipe', 'tta', 'res512', 'rf_context', 'encoder', 'combined'],
    METHODS: { rf: 'SLIC + Random Forest', unet: 'U-Net', hybrid: 'U-Net + CRF' },
    SHORT: { rf: 'RF', unet: 'U-Net', hybrid: 'U-Net + CRF' },
    WHAT: {
        aux10: ['The U-Net also predicts all 10 original classes (water, non-flooded road, non-flooded building, …) as a second output; the RF learns the 10 classes and its answer is mapped back to 3.', 'GPU (U-Net) · CPU (RF, CRF)', 'Water and non-flooded buildings hidden inside "Other"'],
        recipe: ['Cosine learning-rate decay (1e-4 → 1e-6), up to 60 epochs with patience 12, and an exponential moving average of the weights (0.999) used for validation and the final model.', 'GPU · CPU (CRF)', 'Validation IoU swinging 5–6 points between epochs'],
        tta: ['No retraining: each frozen U-Net predicts the 8 flips/rotations of an image and the class probabilities are averaged.', 'GPU · CPU (CRF)', 'Unstable predictions along target boundaries'],
        res512: ['Images kept at 512 px instead of 256 px (new cache built from the same originals); predictions are scored on the same 256-px grid.', 'GPU (U-Net) · CPU (cache, SLIC, RF, CRF)', 'Detail lost by downsampling, boundary errors'],
        rf_context: ['Each superpixel also sees the average features of its neighbours, its position and area, and the colour of a coarser segmentation (50 numbers instead of 22).', 'CPU only', 'RF judging each superpixel in isolation (precision 0.23–0.25)'],
        encoder: ['ResNet-34 instead of ResNet-18 as the U-Net encoder (ImageNet weights).', 'GPU · CPU (CRF)', 'Model capacity'],
        combined: ['The components that helped on validation (rule below), trained together.', 'GPU and CPU', 'All of the above']
    },

    page() {
        return header('improve', 'Where the errors are, and what fixes them.', 'An error analysis of the frozen models, and a pre-registered test of seven changes meant to raise the scores of the Random Forest, the U-Net and the CRF hybrid.')
            + `<div id="fn-improve"><div class="fn-loading">Reading the improvement-study export…</div></div>`;
    },

    async mount() {
        const self = FN_IMPROVE;
        if (!self.data && !self.failed) {
            try {
                const response = await fetch('data/improve.json', { cache: 'no-store' });
                if (!response.ok) throw new Error(`Improvement export returned HTTP ${response.status}`);
                self.data = await response.json();
            } catch (error) {
                self.failed = error.message;
            }
        }
        self.update();
    },

    pp: v => FN_IMBALANCE.pp(v),
    pending: text => FN_IMBALANCE.pending(text),
    decisionPill: d => FN_IMBALANCE.decisionPill(d),
    mean: values => values.filter(v => v != null).reduce((a, b) => a + b, 0) / values.filter(v => v != null).length,

    toggle(kind) {
        const self = FN_IMPROVE;
        if (kind === 'budget') return [25, 100].map(b => `<button type="button" class="${self.budget === b ? 'on' : ''}" data-imp-budget="${b}" aria-pressed="${self.budget === b}">${b}% labels</button>`).join('');
        return Object.keys(self.SHORT).map(m => `<button type="button" class="${self.method === m ? 'on' : ''}" data-imp-method="${m}" aria-pressed="${self.method === m}">${self.SHORT[m]}</button>`).join('');
    },

    verdict() {
        const self = FN_IMPROVE;
        const d = self.data;
        if (d.status !== 'complete') {
            const p = d.progress;
            return `<div class="fn-dark-card"><h3>Study in progress</h3><p><strong>${num(p.runs_validated)} of ${num(p.runs_planned)}</strong> runs trained and validated. The test split stays sealed until every run is frozen, so <strong>no verdict yet</strong>: sections marked ${self.pending()} fill in automatically when the study finishes. Nothing on this page is an invented placeholder number.</p>`
                + `<div class="fn-progress" role="progressbar" aria-valuemin="0" aria-valuemax="${p.runs_planned}" aria-valuenow="${p.runs_validated}" aria-label="Runs validated"><span style="width:${(100 * p.runs_validated / p.runs_planned).toFixed(1)}%"></span></div></div>`;
        }
        const c = d.comparisons;
        const count = decision => c.filter(x => x.decision === decision).length;
        const hurts = c.filter(x => x.decision === 'hurts').map(x => `${self.NAMES[x.variant]} → ${self.SHORT[x.method]} ${x.budget}%`);
        return `<div class="fn-dark-card"><h3>Verdict (from the pre-registered rule)</h3><p><strong>${count('helps')}</strong> of ${c.length} comparisons clearly helped, <strong>${count('hurts')}</strong> clearly hurt (${esc(hurts.join('; '))}) and ${count('no clear effect')} showed no clear effect.</p>`
            + `<p>The table below gives the scores now. Every model gained most from the <strong>combined</strong> changes at 25% labels; the Random Forest gained about 14 points at both budgets.</p></div>`;
    },

    scoreboard() {
        const self = FN_IMPROVE;
        const d = self.data;
        if (d.status !== 'complete') return '';
        const rows = Object.keys(self.SHORT).flatMap(m => d.budgets.map(b => {
            const mine = d.comparisons.filter(c => c.method === m && c.budget === b);
            const combined = mine.find(c => c.variant === 'combined');
            const original = combined ? self.mean(combined.baseline_scores) : self.mean(d.baseline_test[`${m}_${b}`].flooded_miou);
            const helped = self.sorted(mine.filter(c => c.decision === 'helps' && c.variant !== 'combined'))
                .map(c => `${esc(self.NAMES[c.variant])} ${pct(self.mean(c.variant_scores))} <span class="fn-muted">(${self.pp(c.mean_difference)})</span>`).join('<br>');
            return `<tr><td>${esc(self.METHODS[m])}</td><td>${b}%</td><td>${pct(original)}</td>`
                + `<td class="fn-score">${combined ? `${pct(self.mean(combined.variant_scores))}<small>${self.pp(combined.mean_difference)}</small>` : 'not trained'}</td><td>${combined ? self.decisionPill(combined.decision) : ''}</td><td>${helped || '<span class="fn-muted">none</span>'}</td></tr>`;
        })).join('');
        return card('Scores now (test flooded mIoU, mean of 3 seeds)', `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Model</th><th>Labels</th><th>Original</th><th>Combined model</th><th>Decision</th><th>Single changes that clearly helped</th></tr></thead><tbody>${rows}</tbody></table></div>`
            + `<p class="fn-small fn-muted">The combined model's components were fixed on validation before the test split was opened. At 100% labels its U-Net and CRF gains are not clear on test: the 512-px component helped seeds 42 and 43 but cost seed 44 several points, so the seeds disagree. Choosing a single change from this table after the fact would be a choice made on test data; treat the single-change column as evidence, not as a tuned model.</p>`);
    },

    sorted(comparisons) {
        const self = FN_IMPROVE;
        const methods = Object.keys(self.SHORT);
        return [...comparisons].sort((a, b) => self.ORDER.indexOf(a.variant) - self.ORDER.indexOf(b.variant) || methods.indexOf(a.method) - methods.indexOf(b.method) || a.budget - b.budget);
    },

    errors() {
        const self = FN_IMPROVE;
        const key = `${self.method}_${self.budget}`;
        const e = self.data.errors_seed42.baseline[key];
        const confusers = { building: ['building non-flooded'], road: ['water', 'road non-flooded'] };
        const chart = (target, label) => {
            const items = Object.entries(e[target]).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([name, share]) => ({
                label: name, value: share, emphasis: confusers[target].includes(name),
                colour: confusers[target].includes(name) ? '#eb6834' : '#94a3b8',
                tip: [pct(share), `${label} false alarms on "${name}"`, `${self.METHODS[self.method]} · ${self.budget}% · seed 42 · test`]
            }));
            return chartBlock(`${label} false alarms`, options => FN_CHARTS.bars(items, { ...options, max: 1, left: 170, label: `${label} false alarms by original class`, note: 'share of false-alarm pixels · orange = merged into "Other"' }), { heading: 'h3' });
        };
        return `<section class="fn-card"><div class="fn-card-top"><h2>Where the errors are</h2><div class="fn-seg" role="group" aria-label="Model">${self.toggle('method')}</div><div class="fn-seg" role="group" aria-label="Training budget">${self.toggle('budget')}</div></div>`
            + `<p>Every pixel the frozen model wrongly called a flooded building or road, broken down by its <strong>original 10-class label</strong> (seed 42, test split, already scored for the Results page). The 3-class task merges water, non-flooded roads and non-flooded buildings into "Other", and those are exactly what the models mistake for flood targets. Almost all misses go the other way: 97–98% of missed target pixels are predicted "Other", not the other target class, and 43% of wrong pixels lie within 3 px of a target boundary.</p>`
            + `<div class="fn-chart-pair">${chart('building', 'Flooded building')}${chart('road', 'Flooded road')}</div>`
            + `<div class="fn-callout">So the problem is less "too little data" (${num(overlapInfo()?.summary.test.overlapping_train)} of ${num(data.dataset?.counts.test)} test scenes already overlap training flights) than <strong>how the task is represented</strong>: the information that separates water from a flooded road exists in the 10-class labels but was thrown away. Experiment 1 gives it back.</div></section>`;
    },

    experiments() {
        const self = FN_IMPROVE;
        const d = self.data;
        const rows = self.ORDER.filter(v => d.variants[v]).map((v, i) => `<tr><td><span class="fn-swatch" style="background:${self.COLOURS[v]}"></span>${i + 1}. ${esc(self.NAMES[v])}</td><td>${esc(self.WHAT[v][0])}</td><td>${esc(self.WHAT[v][2])}</td><td>${d.variants[v].methods.map(m => self.SHORT[m]).join(', ')}</td><td>${esc(self.WHAT[v][1])}</td></tr>`).join('');
        return `<section class="fn-card"><h2>Seven changes, one at a time</h2><p>Each change is trained with the same budgets (25% / 100% of labels) and seeds (${d.seeds.join(', ')}) as the frozen models, and compared with the frozen model of the same budget and seed. Everything else is unchanged: ${esc(d.config.unchanged.join('; '))}.</p>`
            + `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Change</th><th>What it does</th><th>Error it targets</th><th>Models</th><th>Device</th></tr></thead><tbody>${rows}</tbody></table></div>`
            + `<details class="fn-details"><summary>The changes as equations</summary>${FN_MATH.equations(['aux', 'cosine', 'ema', 'tta', 'area'])}</details>`
            + `<details class="fn-details"><summary>Scoring and the CRF at 512 px</summary><p>${esc(d.config.scoring)}</p><p>${esc(d.config.hybrid)}</p><p>${esc(d.config.slic_at_512)}</p></details>`
            + `<p class="fn-small fn-muted">Hardware: NVIDIA RTX 5080 Laptop GPU (16 GB) for U-Net training and inference (CUDA only); Intel Core Ultra 9 275HX (8 performance + 16 efficiency cores, 32 GB RAM) for SLIC, Random Forests, DenseCRF and statistics. CPU-only work ran alongside GPU training.</p></section>`;
    },

    combine() {
        const self = FN_IMPROVE;
        const c = self.data.combine;
        if (!c) return card('Which changes join the combined model?', `<p>${esc(self.data.config.combine_rule)}</p><p>${self.pending('Decided on validation once the component runs finish')}</p>`);
        const rows = Object.entries(c.evidence).map(([key, e]) => {
            const [variant, method] = key.split('/');
            const cells = Object.entries(e.differences).map(([b, diffs]) => `${b}%: ${diffs.map(self.pp).join(', ')}`).join('<br>');
            return `<tr><td>${esc(self.NAMES[variant])}</td><td>${self.SHORT[method]}</td><td class="fn-small">${cells}</td><td>${e.passes ? '<span class="fn-pill good">joins</span>' : '<span class="fn-pill">left out</span>'}</td></tr>`;
        }).join('');
        const list = items => (items.length ? items.map(v => self.NAMES[v]).join(' + ') : 'none (the combined model would equal the original, so it is not trained)');
        return card('Which changes join the combined model?', `<p><strong>U-Net and CRF hybrid:</strong> ${esc(list(c.components.unet))}<br><strong>Random Forest:</strong> ${esc(list(c.components.rf))}</p><p>${esc(c.rule)}</p>`
            + `<details class="fn-details"><summary>Validation evidence for every component</summary><div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Change</th><th>Decided on</th><th>Validation change per seed</th><th>Result</th></tr></thead><tbody>${rows}</tbody></table></div></details>`
            + `<p class="fn-small fn-muted">${esc(c.based_on)}. These combined models are the "after" models everywhere else in the app (switch in the sidebar).</p>`);
    },

    preview() {
        const self = FN_IMPROVE;
        const d = self.data;
        let html = `<section class="fn-card fn-preview"><div class="fn-card-top"><h2>Validation preview (not the result)</h2><div class="fn-seg" role="group" aria-label="Training budget">${self.toggle('budget')}</div></div>`
            + `<div class="fn-callout"><strong>The test split is still sealed.</strong> These are changes on the ${num(data.dataset?.counts.val)} <em>validation</em> images for runs finished so far, against the original model with the same seed. They can be optimistic and are not used for the decision; the pre-registered verdict appears here once all ${num(d.progress.runs_planned)} runs are frozen and the test split is scored once.</div>`;
        for (const [method, name] of Object.entries(self.METHODS)) {
            const rows = d.validation_preview.filter(p => p.method === method && p.budget === self.budget);
            const ready = rows.filter(p => p.differences.length);
            const waiting = rows.filter(p => !p.differences.length).map(p => self.NAMES[p.variant]);
            const chartRows = ready.map(p => {
                const mean = self.mean(p.differences);
                return { label: `${self.NAMES[p.variant]}${p.complete ? '' : ` (${p.differences.length}/3 seeds)`}`, value: mean, low: Math.min(...p.differences), high: Math.max(...p.differences),
                    colour: self.COLOURS[p.variant], tip: [self.pp(mean), `${self.NAMES[p.variant]} · ${name} · ${self.budget}% · validation`, `Seeds ${p.seeds.join(', ')}: ${p.differences.map(self.pp).join(', ')}`] };
            });
            html += chartRows.length
                ? chartBlock(`${name} · ${self.budget}% · validation`, options => FN_CHARTS.forest(chartRows, { ...options, label: `Validation change in flooded mIoU, ${name}`, legend: 'dot = mean change over finished seeds · line = range of the seeds (not a confidence interval)' }), { heading: 'h3' })
                : `<h3 class="fn-vis-title">${esc(name)} · ${self.budget}%</h3>`;
            if (waiting.length) html += `<p class="fn-small">${self.pending('Not trained yet')} ${esc(waiting.join(', '))}</p>`;
        }
        return html + `</section>`;
    },

    results() {
        const self = FN_IMPROVE;
        const d = self.data;
        if (d.status !== 'complete') return self.preview();
        let html = `<section class="fn-card"><div class="fn-card-top"><h2>What changed on the held-out test split?</h2><div class="fn-seg" role="group" aria-label="Training budget">${self.toggle('budget')}</div></div>`
            + `<p>Each row is the change in flooded-class mIoU against the frozen original model with the same budget and seed (mean of 3 seeds), with its 95% bootstrap interval over the ${num(data.dataset?.counts.test)} test images. The label applies the pre-registered rule.</p>`
            + `<details class="fn-details"><summary>The statistics, as equations</summary>${FN_MATH.equations(['change', 'bootstrap', 'decision'])}</details>`;
        for (const [method, name] of Object.entries(self.METHODS)) {
            const rows = self.sorted(d.comparisons.filter(c => c.method === method && c.budget === self.budget)).map(c => ({
                label: self.NAMES[c.variant], value: c.mean_difference, low: c.ci[0], high: c.ci[1], colour: self.COLOURS[c.variant],
                note: c.decision === 'no clear effect' ? '' : c.decision,
                tip: [self.pp(c.mean_difference), `${self.NAMES[c.variant]} · ${name} · ${self.budget}%`, `95% CI ${self.pp(c.ci[0])} to ${self.pp(c.ci[1])}`, `Seeds: ${c.differences.map(self.pp).join(', ')}`, `Decision: ${c.decision}`]
            }));
            if (rows.length) html += chartBlock(`${name} · ${self.budget}% labels`, options => FN_CHARTS.forest(rows, { ...options, label: `Change in flooded mIoU, ${name}`, legend: 'dot = mean change over 3 seeds · line = 95% bootstrap interval · notes = pre-registered decision' }), { heading: 'h3' });
        }
        const table = self.sorted(d.comparisons.filter(c => c.budget === self.budget)).map(c => `<tr><td>${esc(self.NAMES[c.variant])}</td><td>${esc(self.SHORT[c.method])}</td>`
            + `<td>${pct(self.mean(c.baseline_scores))}</td><td>${pct(self.mean(c.variant_scores))}</td>`
            + `<td class="fn-score">${self.pp(c.mean_difference)}<small>${c.differences.map(self.pp).join(' · ')}</small></td><td>${self.pp(c.ci[0])} to ${self.pp(c.ci[1])}</td><td>${self.decisionPill(c.decision)}</td></tr>`).join('');
        return html + `<h3 class="fn-vis-title" style="margin:18px 0 10px">All comparisons at ${self.budget}%</h3><div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Change</th><th>Model</th><th>Original</th><th>With change</th><th>Change (seeds)</th><th>95% CI</th><th>Decision</th></tr></thead><tbody>${table}</tbody></table></div></section>`;
    },

    confusion() {
        const self = FN_IMPROVE;
        const d = self.data;
        if (d.status !== 'complete') {
            return card('Did it fix the targeted errors?', `<p>${self.pending()} For every change, the share of road false alarms on water and of building false alarms on non-flooded buildings, before and after, from the test evaluation after the freeze.</p>`);
        }
        const share = (e, target, name) => (e && e[target] ? e[target][name] || 0 : null);
        const rows = self.sorted(d.comparisons.filter(c => c.budget === self.budget)).map(c => {
            const key = `${c.method}_${c.budget}`;
            const before = d.errors_seed42.baseline[key];
            const after = (d.errors_seed42[c.variant] || {})[key];
            const cell = (target, name) => `${pct(share(before, target, name))} → ${pct(share(after, target, name))}`;
            return `<tr><td>${esc(self.NAMES[c.variant])}</td><td>${esc(self.SHORT[c.method])}</td><td>${cell('road', 'water')}</td><td>${cell('building', 'building non-flooded')}</td>`
                + `<td>${self.pp(self.mean(c.variant_classes.building_precision) - self.mean(c.baseline_classes.building_precision))} / ${self.pp(self.mean(c.variant_classes.road_precision) - self.mean(c.baseline_classes.road_precision))}</td>`
                + `<td>${self.pp(self.mean(c.variant_classes.building_recall) - self.mean(c.baseline_classes.building_recall))} / ${self.pp(self.mean(c.variant_classes.road_recall) - self.mean(c.baseline_classes.road_recall))}</td></tr>`;
        }).join('');
        return card(`Did it fix the targeted errors? (${self.budget}%)`, `<p>Shares of false alarms on the two main confusers (seed 42, before → after), and the change in precision and recall (building / road, mean of 3 seeds). The training recipe and the Random Forest's 10-class labels moved road false alarms off open water the most.</p><details class="fn-details" open><summary>Every change</summary><div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Change</th><th>Model</th><th>Road false alarms on water</th><th>Building false alarms on non-flooded buildings</th><th>Precision change</th><th>Recall change</th></tr></thead><tbody>${rows}</tbody></table></div></details>`);
    },

    costs() {
        const self = FN_IMPROVE;
        const d = self.data;
        const rows = d.budgets.flatMap(b => ['rf', 'unet'].map(m => {
            const base = d.costs[`baseline_${m}_${b}`];
            const cells = self.ORDER.filter(v => d.variants[v] && d.variants[v].methods.includes(m) && v !== 'tta')
                .map(v => `${esc(self.NAMES[v])}: ${d.costs[`${v}_${m}_${b}`] ? seconds(self.mean(d.costs[`${v}_${m}_${b}`])) : self.pending()}`).join('<br>');
            return `<tr><td>${self.SHORT[m]} · ${b}%</td><td>${seconds(self.mean(base))}</td><td>${cells}</td></tr>`;
        })).join('');
        return card('What it cost', `<details class="fn-details"><summary>Training time of every change</summary><div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Model</th><th>Original training</th><th>With each change (mean of 3 seeds)</th></tr></thead><tbody>${rows}</tbody></table></div></details><p class="fn-small fn-muted">Test-time augmentation needs no training but makes every prediction 8 forward passes. CPU jobs ran alongside GPU training, so times may be slightly inflated; training time is not used in any decision.</p>`);
    },

    limits() {
        const self = FN_IMPROVE;
        const d = self.data;
        const when = value => (value ? esc(new Date(value).toLocaleString()) : self.pending());
        return card('Integrity and limits', `<ul><li>Pre-registered ${when(d.preregistered_at)}; all new runs frozen ${when(d.frozen_at)}; first test evaluation ${when(d.first_test_evaluation)}.</li>`
            + `<li>This study was added after the original and class-imbalance test results had been seen. Every choice was fixed in advance or made on validation; the test split is scored once, after the freeze.</li>`
            + `<li>The error analysis that motivated the changes used the frozen models' test predictions, so the choice of <em>what to try</em> was informed by test errors; no setting was tuned on them.</li>`
            + `<li>Seven changes × several models × two budgets are many comparisons; a single "helps" should be read with that in mind.</li>`
            + `<li>Most test scenes overlap training flights, as on the Results page; these comparisons share that limitation.</li></ul>`);
    },

    update() {
        const self = FN_IMPROVE;
        const root = $('#fn-improve');
        if (!root) return;
        if (self.failed) {
            root.innerHTML = empty('Improvement study not exported yet', `${self.failed}. Run "floodnet improve export". No numbers are substituted meanwhile.`);
            return;
        }
        root.innerHTML = self.verdict() + self.scoreboard() + self.errors() + self.experiments() + self.combine() + self.results() + self.confusion() + self.costs() + self.limits();
    }
};

document.addEventListener('click', event => {
    const button = event.target.closest('[data-imp-budget], [data-imp-method]');
    if (!button || !FN_IMPROVE.data) return;
    if (button.dataset.impBudget) FN_IMPROVE.budget = Number(button.dataset.impBudget);
    if (button.dataset.impMethod) FN_IMPROVE.method = button.dataset.impMethod;
    chartRenderers.clear();
    FN_IMPROVE.update();
});
