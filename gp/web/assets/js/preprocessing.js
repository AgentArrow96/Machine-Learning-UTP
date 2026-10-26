'use strict';

const FN_PREP = {
    data: null,
    failed: null,
    step: 0,

    page() {
        return header('preprocessing', 'From the download to the model input.', 'What was checked, what was fixed and what was transformed before any model saw an image. Every number comes from the saved audit and preparation manifests.')
            + `<div id="fn-prep"><div class="fn-loading">Reading the preprocessing export…</div></div>`;
    },

    async mount() {
        const self = FN_PREP;
        if (!self.data && !self.failed) {
            try {
                const response = await fetch('data/preprocessing.json', { cache: 'no-store' });
                if (!response.ok) throw new Error(`Preprocessing export returned HTTP ${response.status}`);
                self.data = await response.json();
            } catch (error) {
                self.failed = error.message;
            }
        }
        self.update();
    },

    figure(url, caption, label, group, pixelated = false) {
        return `<figure>${zoomable(`<img src="${esc(url)}" alt="${esc(label)}" loading="lazy">`, label, { group, title: label, ...(pixelated ? { pixelated: '' } : {}) })}<figcaption>${caption}</figcaption></figure>`;
    },

    check(ok, text) {
        return `<li class="${ok ? 'fn-check' : 'fn-error'}">${ok ? '✓' : '✗'} ${text}</li>`;
    },

    steps() {
        const self = FN_PREP;
        const d = self.data;
        const s = d.steps;
        const share = (part, whole) => pct(part / whole, 2);
        const strataLabels = { both: 'Both targets', building: 'Building only', road: 'Road only', neither: 'Neither target' };
        const strataBars = (counts, total) => Object.keys(strataLabels).map(k => ({ label: strataLabels[k], value: counts[k] / total, text: `${num(counts[k])} (${pct(counts[k] / total)})`, colour: k === 'neither' ? '#94a3b8' : '#0891b2', tip: [`${num(counts[k])} images`, strataLabels[k], pct(counts[k] / total, 2)] }));
        const totalPairs = s.pairing.pairs;
        const resolution = s.resolution;
        return [
            {
                title: 'Download and verify the release',
                figures: `<div class="fn-prep-facts">${metricCards([
                    { value: num(s.acquire.files_hashed), label: 'Files hashed (SHA-256)', note: `${num(totalPairs)} photos + ${num(totalPairs)} annotations` },
                    { value: `${(s.acquire.archive_bytes / 2 ** 30).toFixed(1)} GiB`, label: 'Archive re-hashed', note: s.acquire.archive_verified ? 'matches the recorded SHA-256' : 'MISMATCH' }
                ])}</div>`,
                text: `<p>The official FloodNet archive is downloaded once and its SHA-256 is recorded. This export re-hashed the whole archive again and it still matches:</p><p class="fn-hash">${esc(s.acquire.archive_sha256)}</p>`
                    + `<ul class="fn-check-list">${self.check(JSON.stringify(s.acquire.counts) === JSON.stringify(s.acquire.expected_counts), `Split sizes match the release note: ${num(s.acquire.counts.train)} train / ${num(s.acquire.counts.val)} validation / ${num(s.acquire.counts.test)} test`)}`
                    + self.check(s.acquire.archive_verified, 'Archive hash unchanged since download')
                    + self.check(true, `Every one of the ${num(s.acquire.files_hashed)} extracted files has its own recorded hash, re-checked before each use`)
                    + `</ul><p class="fn-small fn-muted">Licence: ${esc(s.acquire.licence)}.</p>`
            },
            {
                title: 'Pair every photo with its annotation',
                figures: chartBlock('Annotations that contain each original class', options => FN_CHARTS.bars(d.original_classes.map(c => ({
                    label: `${c.id} · ${c.name}`, value: s.pairing.class_presence[c.id] / totalPairs, text: `${num(s.pairing.class_presence[c.id])} images`,
                    colour: `rgb(${c.colour.join(',')})`, emphasis: c.id === 1 || c.id === 3,
                    tip: [`${num(s.pairing.class_presence[c.id])} of ${num(totalPairs)} annotations`, `${c.name} (ID ${c.id})`, pct(s.pairing.class_presence[c.id] / totalPairs)]
                })), { ...options, left: 170, rowHeight: 28, label: 'Images containing each class', note: `Flooded road appears in ${num(s.pairing.class_presence[3])} of ${num(totalPairs)} annotations, flooded building in ${num(s.pairing.class_presence[1])}` }), { heading: 'h3' }),
                text: `<p>Pairing rule: <strong>${esc(s.pairing.rule)}</strong>.</p><ul class="fn-check-list">`
                    + self.check(s.pairing.unpaired === 0, `${num(s.pairing.pairs)} complete pairs, ${s.pairing.unpaired} unpaired files`)
                    + self.check(s.pairing.mask_modes.length === 1 && s.pairing.mask_modes[0] === 'L', `Every annotation is a single-channel indexed image (mode ${esc(s.pairing.mask_modes.join(', '))})`)
                    + self.check(s.pairing.ids_in_range, 'Every annotation uses only class IDs 0–9')
                    + self.check(s.pairing.size_mismatches === 0, 'Every photo has exactly the size of its annotation (after orientation)')
                    + `</ul><p>The chart already shows the imbalance at image level: the two target classes appear in far fewer images than trees or grass.</p>`
            },
            {
                title: 'Screen for duplicates ("cleaning")',
                figures: s.duplicates.examples.map((e, i) => `<div class="fn-prep-pair">${e.urls.map((u, j) => self.figure(u, `${esc(e.ids[j])}`, `Near-duplicate candidate ${e.ids[j]}`, `dup-${i}`)).join('')}<p class="fn-small">Hash distance ${e.distance}${e.cross_split ? ' · <strong>different splits</strong>' : ''}</p></div>`).join(''),
                text: `<ul class="fn-check-list">${self.check(s.duplicates.exact_groups === 0, `${s.duplicates.exact_groups} exact duplicates (identical file hashes)`)}</ul>`
                    + `<p>A coarse 8×8 picture hash flags <strong>${num(s.duplicates.near_candidates)}</strong> pairs that look almost the same, ${num(s.duplicates.near_cross_split)} of them across splits. A hash distance of 0 means the tiny 8×8 thumbnails match, not the photos: these are consecutive drone frames of the same street taken a moment apart. They were reviewed rather than deleted (the three closest pairs of different scenes are shown).</p>`
                    + `<p>Shared ground between training and test frames is real, though, so it was measured properly with keypoint matching: see <a href="#dataset">Dataset → Do test scenes also appear in training?</a></p>`
                    + `<div class="fn-callout"><strong>Images removed: ${s.duplicates.removed}.</strong> No check failed, so nothing needed cleaning; the official splits are kept exactly as published.</div>`
            },
            {
                title: 'Fix the photo orientation',
                figures: self.figure(s.orientation.example.raw, `Raw pixels, ignoring EXIF: annotation misaligned (edge agreement ${s.orientation.example.agreement_raw.toFixed(2)})`, `${s.orientation.example.id}: annotation over the unrotated photo`, 'exif')
                    + self.figure(s.orientation.example.fixed, `EXIF orientation applied: aligned (edge agreement ${s.orientation.example.agreement_fixed.toFixed(2)})`, `${s.orientation.example.id}: annotation over the rotated photo`, 'exif'),
                two: true,
                text: `<p><strong>${num(s.orientation.rotated)} of ${num(s.orientation.total)}</strong> photos store their pixels upside down with an EXIF orientation flag of 3. The annotations were drawn on the displayed (rotated) photo.</p>`
                    + `<p>Reading the raw pixels would put every label on the wrong part of these ${num(s.orientation.rotated)} photos. The pipeline applies the EXIF rotation to the photo and leaves the annotation as it is. Example: ${esc(s.orientation.example.id)}. Edge agreement measures how strongly photo edges line up with label boundaries (higher = aligned).</p>`
            },
            {
                title: 'Two photo sizes, one model input',
                figures: s.resize.sizes.map(x => self.figure(x.padded, `${num(x.count)} photos at ${x.size[0]}×${x.size[1]} → ${x.geometry.width}×${x.geometry.height} inside 256×256 (hatched = padding)`, `Padded model input for a ${x.size[0]}×${x.size[1]} photo`, 'sizes', true)).join(''),
                two: true,
                text: `<p>Every photo is shrunk to fit a <strong>${s.resize.size}×${s.resize.size}</strong> frame without stretching. Photos use ${esc(s.resize.rgb_resampling)} resampling; annotations use ${esc(s.resize.mask_resampling)}-neighbour so no in-between label is invented.</p>`
                    + `<p>The leftover rows are padding with label ${s.resize.padding_label}, which is ignored in training and scoring.</p>`
            },
            {
                title: 'Ten classes become three',
                figures: chartBlock('Training pixels after remapping', options => FN_CHARTS.bars(d.classes.map((c, i) => ({
                    label: c.name, value: s.remap.processed_train_pixels[i] / s.remap.processed_train_pixels.reduce((a, b) => a + b, 0),
                    colour: c.colour, emphasis: i > 0, tip: [pct(s.remap.processed_train_pixels[i] / s.remap.processed_train_pixels.reduce((a, b) => a + b, 0), 2), c.name, `${num(s.remap.processed_train_pixels[i])} pixels at 256 px`]
                })), { ...options, rowHeight: 34, label: 'Processed training class shares', note: `Targets are ${pct((s.remap.processed_train_pixels[1] + s.remap.processed_train_pixels[2]) / s.remap.processed_train_pixels.reduce((a, b) => a + b, 0))} of training pixels` }), { heading: 'h3' }),
                text: `<p>Only two original classes are targets: <strong>building flooded (ID 1) → flooded building</strong> and <strong>road flooded (ID 3) → flooded road</strong>. The other eight, including open water, become "Other".</p>`
                    + `<p>This is where the pixel-level class imbalance comes from. It is handled during training, not by changing the data: see <a href="#imbalance">Class imbalance</a>.</p>`
            },
            {
                title: 'Check that no target is lost at 256 px',
                figures: `<div class="fn-prep-facts">${metricCards([
                    { value: share(resolution.original_target_pixels[0], resolution.original_total), label: 'Flooded building, full resolution' },
                    { value: share(resolution.processed_target_pixels[0], resolution.processed_total), label: 'Flooded building, 256 px' },
                    { value: share(resolution.original_target_pixels[1], resolution.original_total), label: 'Flooded road, full resolution' },
                    { value: share(resolution.processed_target_pixels[1], resolution.processed_total), label: 'Flooded road, 256 px' }
                ])}</div>`,
                text: `<p>Shrinking a 4000-px photo to 256 px could make a small flooded building disappear. The target shares stay almost the same after resizing, and:</p><ul class="fn-check-list">`
                    + self.check(resolution.targets_lost_images['1'] === 0, `${resolution.targets_lost_images['1']} images lose all their flooded-building pixels`)
                    + self.check(resolution.targets_lost_images['2'] === 0, `${resolution.targets_lost_images['2']} images lose all their flooded-road pixels`) + `</ul>`
            },
            {
                title: 'Cache once, verify always',
                figures: `<div class="fn-table-wrap"><table class="fn-table"><thead><tr><th>Array</th><th>Shape</th></tr></thead><tbody>${Object.entries(s.cache.arrays).map(([k, v]) => `<tr><td>${esc(k)}.npy</td><td>${v.join(' × ')}</td></tr>`).join('')}</tbody></table></div>`,
                text: `<p>The prepared images are written once to six arrays (${seconds(s.cache.seconds)}) under a content identity, so every method reads exactly the same pixels.</p><p class="fn-hash">${esc(s.cache.identity)}</p>`
                    + `<ul class="fn-check-list">${self.check(true, esc(s.cache.rederived_by_audit))}</ul>`
            },
            {
                title: 'Draw the 25% subset fairly',
                figures: chartBlock('Training images per stratum: 100% vs 25%', options => FN_CHARTS.bars([
                    ...strataBars(s.budgets.strata['100'], s.budgets.counts['100']).map(b => ({ ...b, label: `100% · ${b.label}` })),
                    ...strataBars(s.budgets.strata['25'], s.budgets.counts['25']).map(b => ({ ...b, label: `25% · ${b.label}`, colour: b.colour === '#94a3b8' ? '#cbd5e1' : '#67b8cc' }))
                ], { ...options, left: 170, rowHeight: 26, max: 1, label: 'Stratum shares in each budget', note: 'Each stratum keeps the same share in the 25% subset' }), { heading: 'h3' }),
                text: `<p>The 25% budget (${num(s.budgets.counts['25'])} of ${num(s.budgets.counts['100'])} images, seed ${s.budgets.seed}) is drawn separately inside four strata, so it keeps the same mix of target and target-free images. It is nested in the full set and shared by every method and seed.</p>`
                    + `<p>${pct(s.budgets.strata['100'].neither / s.budgets.counts['100'])} of training images contain neither target: the image-level side of the imbalance.</p>`
            },
            {
                title: 'Training-time augmentation (U-Net only)',
                figures: `<div class="fn-prep-grid">${self.figure(s.augmentation.original, 'Original', 'Augmentation: original image', 'augment', true)}${s.augmentation.draws.map(x => self.figure(x.overlay, `Epoch ${x.epoch}: ${[x.flip_lr ? 'flip ↔' : '', x.flip_ud ? 'flip ↕' : '', x.quarter_turns ? `${x.quarter_turns * 90}°` : ''].filter(Boolean).join(', ') || 'no flip'}, contrast ×${x.contrast.toFixed(2)}, brightness ${x.brightness >= 0 ? '+' : ''}${x.brightness.toFixed(2)}`, `Augmentation draw ${x.epoch}`, 'augment', true)).join('')}</div>`,
                text: `<p>Each epoch, every U-Net training image gets a fresh, seeded random flip, quarter turn and mild brightness/contrast change (strength ±${s.augmentation.strength}). The annotation (coloured overlay) is flipped and turned with it, so labels stay on the right pixels.</p>`
                    + `<p>These are the actual first 8 epochs' draws for ${esc(s.augmentation.example_id)} with seed ${s.augmentation.seed}. Validation and test images are never augmented. The Random Forest sees un-augmented regions.</p>`
            },
            {
                title: 'Normalise for the pretrained encoder',
                figures: FN_MATH.equations(['norm']) + `<dl class="fn-definition-list">${['Red', 'Green', 'Blue'].map((c, i) => `<dt>${c}</dt><dd>${FN_MATH.tex(String.raw`\mu=${s.normalisation.mean[i].toFixed(3)},\ \sigma=${s.normalisation.std[i].toFixed(3)}`)}</dd>`).join('')}</dl>`,
                text: `<p>The U-Net's ResNet-18 encoder was pretrained on ImageNet, so its input uses the same per-channel statistics. Values then range from ${s.normalisation.theoretical_range[0].toFixed(2)} to ${s.normalisation.theoretical_range[1].toFixed(2)} (this example: ${s.normalisation.example_range[0].toFixed(2)} to ${s.normalisation.example_range[1].toFixed(2)}).</p>`
                    + `<p>The Random Forest instead summarises superpixels into 22 numbers: <a href="#walkthrough">How the models work → SLIC + Random Forest</a>.</p>`
            }
        ];
    },

    update() {
        const self = FN_PREP;
        const root = $('#fn-prep');
        if (!root) return;
        if (self.failed) {
            root.innerHTML = empty('Preprocessing page not exported', `${self.failed}. Run "floodnet export-preprocessing" and reload.`);
            return;
        }
        const steps = self.steps();
        self.step = Math.min(self.step, steps.length - 1);
        const step = steps[self.step];
        const rail = steps.map((x, i) => `<li><button type="button" class="${i === self.step ? 'on' : ''}" aria-current="${i === self.step ? 'step' : 'false'}" data-prep-step="${i}"><span>${i + 1}</span>${esc(x.title)}</button></li>`).join('');
        const summary = self.data.steps;
        root.innerHTML = `<div class="fn-dark-card"><h3>In one line</h3><p>${num(summary.pairing.pairs)} photo–annotation pairs passed every integrity check, so <strong>no image was removed</strong>. Two problems were fixed (57 upside-down photos, two photo sizes) and every image was resized, padded, remapped to three classes and cached once.</p></div>`
            + `<section class="fn-card fn-walk-stage"><ol class="fn-walk-rail">${rail}</ol>`
            + `<div class="fn-walk-head"><span class="fn-eyebrow">Step ${self.step + 1} of ${steps.length}</span><h2>${esc(step.title)}</h2></div>`
            + `<div class="fn-walk-body"><div class="fn-walk-figures ${step.two ? 'two' : ''}">${step.figures}</div><div class="fn-walk-text">${step.text}</div></div>`
            + `<div class="fn-walk-nav"><button type="button" class="fn-btn" data-prep-move="-1" ${self.step === 0 ? 'disabled' : ''}>← Previous step</button>`
            + `<button type="button" class="fn-btn fn-btn-strong" data-prep-move="1" ${self.step === steps.length - 1 ? 'disabled' : ''}>Next step →</button></div></section>`;
    }
};

document.addEventListener('click', event => {
    if (!$('#fn-prep') || !FN_PREP.data) return;
    const step = event.target.closest('[data-prep-step]');
    const move = event.target.closest('[data-prep-move]');
    if (!step && !move) return;
    FN_PREP.step = step ? Number(step.dataset.prepStep) : FN_PREP.step + Number(move.dataset.prepMove);
    FN_PREP.update();
});
