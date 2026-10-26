'use strict';

const FN_IMBALANCE = {
    data: null,
    failed: null,
    budget: 100,

    COLOURS: { baseline: '#64748b', oversample: '#2a78d6', chessmix: '#eb6834', rgan: '#1baf7a', oasis: '#eda100', spade: '#e87ba4', dice: '#008300', smote: '#6250d6', rgan_tabular: '#e34948' },
    NAMES: { baseline: 'Original (frozen)', oversample: 'Oversampling (control)', chessmix: 'ChessMix', rgan: 'R-GAN (image)', oasis: 'OASIS', spade: 'SPADE', dice: 'Dice + CE loss', smote: 'SMOTE', rgan_tabular: 'R-GAN (tabular)' },

    page() {
        return header('imbalance', 'Rare targets, measured remedies.', 'Flooded buildings and roads are a few percent of all pixels. This page shows how the original pipeline handled that, and what changed when the training data were rebalanced with oversampling, patch mixing or GAN-generated scenes.')
            + `<div id="fn-imbalance"><div class="fn-loading">Reading the class-imbalance export…</div></div>`;
    },

    async mount() {
        const self = FN_IMBALANCE;
        if (!self.data && !self.failed) {
            try {
                const response = await fetch('data/imbalance.json', { cache: 'no-store' });
                if (!response.ok) throw new Error(`Class-imbalance export returned HTTP ${response.status}`);
                self.data = await response.json();
            } catch (error) {
                self.failed = error.message;
            }
        }
        self.update();
    },

    pp(v) {
        return v == null ? 'N/A' : `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v * 100).toFixed(1)} pp`;
    },

    decisionPill(decision) {
        const kind = decision === 'helps' ? 'good' : decision === 'hurts' ? 'warn' : '';
        return `<span class="fn-pill ${kind}">${decision === 'helps' ? '▲ helps' : decision === 'hurts' ? '▼ hurts' : '● no clear effect'}</span>`;
    },

    original() {
        const o = FN_IMBALANCE.data.original;
        const w = budget => o.unet_class_weights[budget].map(v => v.toFixed(2)).join(' / ');
        const r = budget => o.rf_region_counts[budget].map(num).join(' / ');
        return card('What the original pipeline already did', `<p>The imbalance has two sides: <strong>pixels</strong> (flooded building ${pct(o.train_pixel_shares[1])}, flooded road ${pct(o.train_pixel_shares[2])} of training pixels) and <strong>images</strong> (${num(o.images_without_targets)} of ${num(o.train_images)} training images contain neither target). Four measures were already in place, none of them synthetic data:</p>`
            + `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Where</th><th>Measure</th><th>Numbers</th></tr></thead><tbody>`
            + `<tr><td>U-Net loss</td><td>Inverse-frequency class weights from each budget's training images, clipped to [${o.weight_clip.join(', ')}]: ${FN_MATH.inline('weights')}</td><td>25%: ${w('25')}<br>100%: ${w('100')}<br><small class="fn-muted">Other / building / road. Unclipped, building would weigh ≈ ${Math.round(o.unclipped_building_over_other['100'])}× Other; the clip keeps it ≈ ${Math.round(o.unet_class_weights['100'][1] / o.unet_class_weights['100'][0])}×.</small></td></tr>`
            + `<tr><td>Random Forest</td><td>Balanced class weights; under the region cap every flooded region is kept and only "Other" is thinned</td><td>25%: ${r('25')}<br>100%: ${r('100')}<br><small class="fn-muted">training regions, Other / building / road</small></td></tr>`
            + `<tr><td>Data</td><td>The 25% subset is stratified by target presence</td><td>Same strata shares as 100% (<a href="#preprocessing">Data preprocessing, step 9</a>)</td></tr>`
            + `<tr><td>Metric</td><td>Flooded-class mIoU, never pixel accuracy</td><td>"Other" is ${pct(o.train_pixel_shares[0], 0)} of pixels</td></tr></tbody></table></div>`);
    },

    review() {
        const rows = FN_IMBALANCE.data.review.map(m => `<tr><td><a href="${esc(m.url)}" target="_blank" rel="noopener noreferrer">${esc(m.name)}</a><br><small class="fn-muted">${esc(m.cite)}</small></td><td>${esc(m.family)}</td><td>${esc(m.what)}</td><td>${esc(m.fit)}</td><td>${m.run.startsWith('yes') ? `<span class="fn-pill good">${esc(m.run)}</span>` : `<span class="fn-pill">${esc(m.run)}</span>`}</td></tr>`).join('');
        return `<section class="fn-card"><h2>Methods reviewed</h2><p>The "R-GAN" you asked about is the <strong>Advanced R-GAN</strong> (a regularised GAN for minority rows in fraud data). Other GAN families target visual imbalance more directly, so they were reviewed too. Only papers whose full text was unreachable are marked: the Advanced R-GAN and the temporal-graph-network paper were read from their abstracts, so their architectural details here are assumptions.</p>`
            + `<details class="fn-details"><summary>Show all ${FN_IMBALANCE.data.review.length} reviewed methods</summary><div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Method</th><th>Family</th><th>What it does</th><th>Fit for this study</th><th>Run here?</th></tr></thead><tbody>${rows}</tbody></table></div></details>`
            + `<div class="fn-callout">Lesson from the temporal-graph-network fraud paper: richer relations between entities improved detection. Its counterpart here would be giving the Random Forest the context of neighbouring superpixels (a region graph); that is recorded as future work, not run.</div></section>`;
    },

    howItWorks() {
        const d = FN_IMBALANCE.data;
        const s = d.config.image_gan;
        return card('How the GAN variants make new training data', `<div class="fn-flow"><div>Budget layouts<br><small>10-class</small></div><span>→</span><div>Paste extra flooded<br>buildings / roads</div><span>→</span><div>Generator renders<br>a photo</div><span>→</span><div>Exact labels<br>= remapped layout</div><span>→</span><div>+${pct(d.config.synthetic_fraction, 0)} images<br>for training</div></div>`
            + `<p>A GAN that only copies the training distribution would keep the targets rare, so each new scene starts from a real layout of the same budget with ${d.config.layouts.paste_min}–${d.config.layouts.paste_max} extra flooded buildings or roads pasted in from other layouts. The generator then paints a photo for that layout, so the label of every synthetic pixel is known exactly. All three GANs paint the <em>same</em> layouts, train for the same ${num(s.iterations['100'])} (100%) / ${num(s.iterations['25'])} (25%) iterations on that budget's labelled images only, and are picked by validation KID.</p>`
            + `<dl class="fn-definition-list"><dt>R-GAN</dt><dd>pix2pix-style generator with <strong>spectral normalisation</strong>; PatchGAN discriminator with <strong>CELU</strong>; <strong>similarity-measure loss</strong> (feature cosine distance) + relativistic-average adversarial loss + L1 + feature matching. The three bold parts come from the Advanced R-GAN abstract (implementation assumed).</dd>`
            + `<dt>OASIS</dt><dd>SPADE generator with 3D noise; the discriminator <em>segments</em> the image into the 11 layout classes + "fake" with a class-balanced loss, so rare classes keep getting gradient; LabelMix.</dd>`
            + `<dt>SPADE</dt><dd>Spatially-adaptive normalisation; 2-scale PatchGAN, hinge loss, feature matching, perceptual loss (ResNet-18, offline).</dd>`
            + `<dt>R-GAN (tabular)</dt><dd>The paper's own setting: it generates the Random Forest's 22-number superpixel rows (colour statistics in range, texture histogram summing to 1).</dd>`
            + `<dt>Controls</dt><dd>Oversampling repeats real target images; ChessMix builds rare-class patch mosaics; SMOTE interpolates region rows; Dice + CE changes only the U-Net loss.</dd></dl>`);
    },

    pending(text = 'Pending') {
        return `<span class="fn-pill fn-pending">${esc(text)}</span>`;
    },

    gallery() {
        const self = FN_IMBALANCE;
        const g = self.data.gallery;
        const kinds = ['rgan', 'oasis', 'spade'];
        const figure = (url, label, group, caption) => `<figure>${zoomable(`<img src="${esc(url)}" alt="${esc(label)}" loading="lazy">`, label, { group, title: label, caption })}<figcaption>${esc(caption)}</figcaption></figure>`;
        const tile = name => `<figure><div class="fn-pending-tile">${esc(name)}<br><small>still training</small></div><figcaption>${esc(name)}</figcaption></figure>`;
        const rows = g.layouts.map((l, i) => `<div class="fn-gan-row">${figure(l.labels, `Layout ${i + 1}`, `gan-${i}`, `Layout (base ${l.base.replace('train_', '')} + ${l.pasted_from.length} pasted)`)}`
            + kinds.map(k => (g.renderings[k] ? figure(g.renderings[k][i], `${self.NAMES[k]}, layout ${i + 1}`, `gan-${i}`, self.NAMES[k]) : tile(self.NAMES[k]))).join('') + `</div>`).join('');
        const chess = g.chessmix.map((c, i) => figure(c.image, `ChessMix mosaic ${i + 1}`, 'chessmix', 'Mosaic') + figure(c.labels, `ChessMix labels ${i + 1}`, 'chessmix', 'Labels (grey squares are ignored)')).join('');
        return `<section class="fn-card"><h2>Same layouts, three generators</h2><p>The first four synthetic layouts of the 100% budget as painted by each GAN (select any image to compare them at the same zoom with ← / →). Orange = flooded building, cyan = flooded road in the layout.</p><div class="fn-gan-grid">${rows}</div>`
            + `<h3 class="fn-vis-title" style="margin:18px 0 10px">ChessMix mosaics</h3><div class="fn-prep-grid">${chess}</div></section>`;
    },

    quality() {
        const self = FN_IMBALANCE;
        const d = self.data;
        const rows = ['rgan', 'oasis', 'spade'].flatMap(k => [25, 100].map(b => {
            const gan = d.gans[`${k}_${b}`];
            if (!gan) return `<tr><td>${esc(self.NAMES[k])}</td><td>${b}%</td><td colspan="7">${self.pending('Training or quality check not finished yet')}</td></tr>`;
            const q = gan.quality;
            return `<tr><td>${esc(self.NAMES[k])}</td><td>${b}%</td><td>${q.kid.generated_vs_real_val.mean.toFixed(4)}</td><td>${q.kid.real_train_vs_real_val.mean.toFixed(4)}</td><td>${q.kid.mean_colour_painting_vs_real_val.mean.toFixed(4)}</td>`
                + `<td>${pct(q.mask_consistency.flooded_miou_on_generated_val)} <small class="fn-muted">vs ${pct(q.mask_consistency.flooded_miou_on_real_val)} real</small></td>`
                + `<td>${pct(q.memorisation.synthetic_above_real_p99)} ${q.memorisation.pass ? '<span class="fn-pill good">ok</span>' : '<span class="fn-pill warn">check</span>'}</td>`
                + `<td>${num(gan.selected_iteration)} / ${num(gan.iterations)}</td><td>${seconds(gan.active_seconds ?? gan.training_seconds)}${gan.pauses?.length ? `<small class="fn-muted"><br>${gan.pauses.length} pause${gan.pauses.length > 1 ? 's' : ''} excluded (${seconds(gan.pauses.reduce((a, x) => a + x.seconds, 0))})</small>` : ''}</td></tr>`;
        })).join('');
        const done = ['rgan', 'oasis', 'spade'].filter(k => d.gans[`${k}_100`]);
        const series = done.map(k => {
            const gan = d.gans[`${k}_100`];
            const chosen = gan.history.find(h => h.iteration === gan.selected_iteration);
            return { label: self.NAMES[k], colour: self.COLOURS[k], values: gan.history.map(h => h.kid), marker: gan.history.indexOf(chosen), markerLabel: `${self.NAMES[k]}: ${chosen.kid.toFixed(3)}` };
        });
        const step = d.config.image_gan.checkpoint_every;
        return `<section class="fn-card"><h2>Are the synthetic images any good?</h2><p>Checked on validation images only, before any model was trained on synthetic data. <strong>KID</strong> compares image features with real photos (lower = closer; the real-vs-real value is the floor, a flat colour painting of the layout is a useless generator). <strong>Mask consistency</strong> asks the frozen U-Net to segment generated validation scenes: if it finds the flooded areas where the layout put them, the GAN painted recognisable targets. <strong>Memorisation</strong> is the share of synthetic images closer to a training photo than 99% of real validation photos are.</p>`
            + `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>GAN</th><th>Budget</th><th>KID vs real val</th><th>Real train vs val (floor)</th><th>Colour painting (ceiling)</th><th>U-Net flooded IoU on generated val</th><th>Memorisation</th><th>Selected iteration</th><th>Active training time</th></tr></thead><tbody>${rows}</tbody></table></div>`
            + (series.length ? chartBlock('Validation KID during training (100% budget)', options => FN_CHARTS.lines(series, { ...options, xName: 'Checkpoint', xStart: 1, label: 'Validation KID per checkpoint', note: `one checkpoint every ${num(step)} iterations; circle = selected${done.length < 3 ? '; generators still training are not shown yet' : ''}` }), { heading: 'h3' }) : '')
            + `</section>`;
    },

    budgetToggle() {
        const self = FN_IMBALANCE;
        return [25, 100].map(b => `<button type="button" class="${self.budget === b ? 'on' : ''}" data-imb-budget="${b}" aria-pressed="${self.budget === b}">${b}% labels</button>`).join('');
    },

    preview() {
        const self = FN_IMBALANCE;
        const d = self.data;
        const methods = [['rf', 'SLIC + Random Forest'], ['unet', 'U-Net'], ['hybrid', 'U-Net + CRF']];
        let html = `<section class="fn-card fn-preview"><div class="fn-card-top"><h2>Validation preview (not the result)</h2><div class="fn-seg" role="group" aria-label="Training budget">${self.budgetToggle()}</div></div>`
            + `<div class="fn-callout"><strong>The test split is still sealed.</strong> These are changes on the ${num(data.dataset?.counts.val)} <em>validation</em> images for runs finished so far, against the original model with the same seed. They can be optimistic and are not used for the decision; the pre-registered verdict appears here once all ${num(d.progress.runs_planned)} runs are frozen and the test split is scored once.</div>`;
        for (const [method, name] of methods) {
            const rows = d.validation_preview.filter(p => p.method === method && p.budget === self.budget);
            const ready = rows.filter(p => p.differences.length);
            const waiting = rows.filter(p => !p.differences.length).map(p => self.NAMES[p.variant]);
            const chartRows = ready.map(p => {
                const mean = p.differences.reduce((a, b) => a + b, 0) / p.differences.length;
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
        const self = FN_IMBALANCE;
        const d = self.data;
        if (d.status !== 'complete') return self.preview();
        const methods = [['rf', 'SLIC + Random Forest'], ['unet', 'U-Net'], ['hybrid', 'U-Net + CRF']];
        let html = `<section class="fn-card"><div class="fn-card-top"><h2>What changed on the held-out test split?</h2><div class="fn-seg" role="group" aria-label="Training budget">${self.budgetToggle()}</div></div>`
            + `<p>Each row is the change in flooded-class mIoU against the frozen original model with the same budget and seed (mean of 3 seeds), with its 95% bootstrap interval over the 448 test images. The label applies the pre-registered rule: <em>${esc(d.config.decision_rule)}</em>.</p>`;
        for (const [method, name] of methods) {
            const rows = d.comparisons.filter(c => c.method === method && c.budget === self.budget).map(c => ({
                label: self.NAMES[c.variant], value: c.mean_difference, low: c.ci[0], high: c.ci[1], colour: self.COLOURS[c.variant],
                note: c.decision === 'no clear effect' ? '' : c.decision,
                tip: [self.pp(c.mean_difference), `${self.NAMES[c.variant]} · ${name} · ${self.budget}%`, `95% CI ${self.pp(c.ci[0])} to ${self.pp(c.ci[1])}`, `Seeds: ${c.differences.map(self.pp).join(', ')}`, `Decision: ${c.decision}`]
            }));
            html += chartBlock(`${name} · ${self.budget}% labels`, options => FN_CHARTS.forest(rows, { ...options, label: `Change in flooded mIoU, ${name}`, legend: 'dot = mean change over 3 seeds · line = 95% bootstrap interval · notes = pre-registered decision' }), { heading: 'h3' });
        }
        const table = d.comparisons.filter(c => c.budget === self.budget).map(c => `<tr><td>${esc(self.NAMES[c.variant])}</td><td>${esc({ rf: 'RF', unet: 'U-Net', hybrid: 'U-Net + CRF' }[c.method])}</td>`
            + `<td>${pct(c.baseline_scores.reduce((a, b) => a + b, 0) / c.baseline_scores.length)}</td><td>${pct(c.variant_scores.reduce((a, b) => a + b, 0) / c.variant_scores.length)}</td>`
            + `<td class="fn-score">${self.pp(c.mean_difference)}<small>${c.differences.map(self.pp).join(' · ')}</small></td><td>${self.pp(c.ci[0])} to ${self.pp(c.ci[1])}</td><td>${self.decisionPill(c.decision)}</td></tr>`).join('');
        html += `<h3 class="fn-vis-title" style="margin:18px 0 10px">All comparisons at ${self.budget}%</h3><div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Variant</th><th>Model</th><th>Original</th><th>With remedy</th><th>Change (seeds)</th><th>95% CI</th><th>Decision</th></tr></thead><tbody>${table}</tbody></table></div></section>`;
        return html;
    },

    perClass() {
        const self = FN_IMBALANCE;
        if (self.data.status !== 'complete') {
            return card('Found more, or just guessed more?', `<p>${self.pending()} Per-class recall and precision changes (flooded pixels found vs false alarms) are computed from the test evaluation, after the freeze.</p>`);
        }
        const mean = values => values.filter(v => v != null).reduce((a, b) => a + b, 0) / values.filter(v => v != null).length;
        const rows = self.data.comparisons.filter(c => c.budget === self.budget).map(c => {
            const delta = key => self.pp(mean(c.variant_classes[key]) - mean(c.baseline_classes[key]));
            return `<tr><td>${esc(self.NAMES[c.variant])}</td><td>${esc({ rf: 'RF', unet: 'U-Net', hybrid: 'U-Net + CRF' }[c.method])}</td><td>${delta('building_recall')}</td><td>${delta('building_precision')}</td><td>${delta('road_recall')}</td><td>${delta('road_precision')}</td></tr>`;
        }).join('');
        return card(`Found more, or just guessed more? (${self.budget}%)`, `<p>Rebalancing usually raises <strong>recall</strong> (more flooded pixels found) and can lower <strong>precision</strong> (more false alarms). Changes against the original model, mean of 3 seeds:</p><div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Variant</th><th>Model</th><th>Building recall</th><th>Building precision</th><th>Road recall</th><th>Road precision</th></tr></thead><tbody>${rows}</tbody></table></div>`);
    },

    verdict() {
        const self = FN_IMBALANCE;
        const d = self.data;
        if (d.status !== 'complete') {
            const p = d.progress;
            return `<div class="fn-dark-card"><h3>Study in progress</h3><p><strong>${num(p.runs_validated)} of ${num(p.runs_planned)}</strong> runs trained and validated · <strong>${p.gans_done} of ${p.gans_planned}</strong> generators trained and checked. The test split stays sealed until every run is frozen, so <strong>no verdict yet</strong>: sections marked ${self.pending()} fill in automatically when the study finishes. Nothing on this page is an invented placeholder number.</p>`
                + `<div class="fn-progress" role="progressbar" aria-valuemin="0" aria-valuemax="${p.runs_planned}" aria-valuenow="${p.runs_validated}" aria-label="Runs validated"><span style="width:${(100 * p.runs_validated / p.runs_planned).toFixed(1)}%"></span></div></div>`;
        }
        const c = d.comparisons;
        const list = decision => c.filter(x => x.decision === decision).map(x => `${self.NAMES[x.variant]} → ${({ rf: 'RF', unet: 'U-Net', hybrid: 'U-Net + CRF' })[x.method]} ${x.budget}% (${self.pp(x.mean_difference)})`);
        const helps = list('helps');
        const hurts = list('hurts');
        const gans = c.filter(x => ['rgan', 'oasis', 'spade'].includes(x.variant));
        const best = [...c].sort((a, b) => b.mean_difference - a.mean_difference)[0];
        return `<div class="fn-dark-card"><h3>Verdict (from the pre-registered rule)</h3><p><strong>${helps.length}</strong> of ${c.length} comparisons clearly helped${helps.length ? `: ${esc(helps.join('; '))}` : ''}. <strong>${hurts.length}</strong> clearly hurt${hurts.length ? `: ${esc(hurts.join('; '))}` : ''}. The rest showed no clear effect.</p>`
            + `<p>Image GANs: ${gans.filter(x => x.decision === 'helps').length} of ${gans.length} GAN comparisons helped, ${gans.filter(x => x.decision === 'hurts').length} hurt. Largest mean change overall: ${esc(self.NAMES[best.variant])} on ${esc({ rf: 'RF', unet: 'U-Net', hybrid: 'U-Net + CRF' }[best.method])} at ${best.budget}% (${self.pp(best.mean_difference)}).</p></div>`;
    },

    costs() {
        const self = FN_IMBALANCE;
        const d = self.data;
        const avg = values => values.reduce((a, b) => a + b, 0) / values.length;
        const rows = [25, 100].flatMap(b => ['rf', 'unet'].map(m => {
            const base = d.costs[`baseline_${m}_${b}`];
            const cells = Object.keys(d.variants).filter(v => d.variants[v].methods.includes(m))
                .map(v => `${esc(self.NAMES[v])}: ${d.costs[`${v}_${m}_${b}`] ? seconds(avg(d.costs[`${v}_${m}_${b}`])) : self.pending()}`).join('<br>');
            return `<tr><td>${m === 'rf' ? 'RF' : 'U-Net'} · ${b}%</td><td>${seconds(avg(base))}</td><td>${cells}</td></tr>`;
        })).join('');
        const gan = ['rgan', 'oasis', 'spade'].map(k => `${esc(self.NAMES[k])}: ${[25, 100].map(b => (d.gans[`${k}_${b}`] ? `${seconds(d.gans[`${k}_${b}`].active_seconds ?? d.gans[`${k}_${b}`].training_seconds)} (${b}%)` : `${self.pending()} (${b}%)`)).join(' + ')}`).join('<br>');
        return card('What it cost', `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Model</th><th>Original training</th><th>With each remedy (mean of 3 seeds)</th></tr></thead><tbody>${rows}</tbody></table></div><p><strong>GAN training (once per budget, active time):</strong><br>${gan}</p><p class="fn-small fn-muted">GAN times exclude the hours the laptop spent in Windows standby or stalled: each GAN's own progress log is used, and any gap longer than three times its typical 500-iteration step counts as one typical step. Iterations redone after a crash or a requested pause are real work and are included. The R-GAN, SPADE and SMOTE Random Forests were trained on the CPU while OASIS used the GPU, so their times may be slightly inflated.</p>`);
    },

    limits() {
        const self = FN_IMBALANCE;
        const d = self.data;
        const when = value => (value ? esc(new Date(value).toLocaleString()) : self.pending());
        return card('Integrity and limits', `<ul><li>Pre-registered ${when(d.preregistered_at)}; all new runs frozen ${when(d.frozen_at)}; first test evaluation ${when(d.first_test_evaluation)}.</li>`
            + `<li>This study was added after the original test results had been seen. Every choice was fixed in advance or made on validation; the test split is scored once, after the freeze.</li>`
            + `<li>Every generator, layout, mosaic, duplicate and synthetic row was built only from that budget's training images.</li>`
            + `<li>One generator per budget and family (seed 42); generator-to-generator variance is not measured.</li>`
            + `<li>Advanced R-GAN details were implemented from its abstract; SPADE's perceptual loss uses ResNet-18 instead of VGG-19; OASIS's discriminator needed normalised residual blocks to train stably (all logged in the task ledger).</li>`
            + `<li>The R-GAN at 25% overfitted quickly: its best validation checkpoint was the first one saved (2,000 iterations), so an earlier optimum cannot be excluded.</li>`
            + `<li>Most test scenes overlap training frames, as on the Results page; these comparisons share that limitation.</li></ul>`);
    },

    update() {
        const self = FN_IMBALANCE;
        const root = $('#fn-imbalance');
        if (!root) return;
        if (self.failed) {
            root.innerHTML = empty('Class-imbalance study not exported yet', `${self.failed}. Run "floodnet imbalance export". No numbers are substituted meanwhile.`);
            return;
        }
        root.innerHTML = self.verdict() + self.original() + self.review() + self.howItWorks() + self.gallery() + self.quality() + self.results() + self.perClass() + self.costs() + self.limits();
    }
};

document.addEventListener('click', event => {
    const button = event.target.closest('[data-imb-budget]');
    if (!button || !FN_IMBALANCE.data) return;
    FN_IMBALANCE.budget = Number(button.dataset.imbBudget);
    chartRenderers.clear();
    FN_IMBALANCE.update();
});
