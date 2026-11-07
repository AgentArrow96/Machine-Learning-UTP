'use strict';

const FN_WALKTHROUGH = {
    data: null,
    failed: null,
    state: { example: null, tab: 'prep', steps: { prep: 0, rf: 0, unet: 0 }, region: null, readout: null },
    pixels: new Map(),

    TABS: [['prep', 'Preparation'], ['rf', 'SLIC + Random Forest'], ['unet', 'U-Net']],

    page() {
        return header('walkthrough', 'Step by step, on a real test image.', 'Pick a held-out scene and follow what each approach does to it, from the raw photo to the final mask. Every picture and number is recomputed from the frozen models.')
            + `<div id="fn-walk"><div class="fn-loading">Reading the walkthrough export…</div></div>`;
    },

    async mount() {
        const self = FN_WALKTHROUGH;
        if (!self.data && !self.failed) {
            try {
                const response = await fetch('data/walkthrough.json', { cache: 'no-store' });
                if (!response.ok) throw new Error(`Walkthrough export returned HTTP ${response.status}`);
                self.data = await response.json();
            } catch (error) {
                self.failed = error.message;
            }
        }
        self.update();
    },

    example() {
        const d = FN_WALKTHROUGH.data;
        return d.examples.find(e => e.id === FN_WALKTHROUGH.state.example) || d.examples[0];
    },


    update() {
        const self = FN_WALKTHROUGH;
        const root = $('#fn-walk');
        if (!root) return;
        if (self.failed) {
            root.innerHTML = empty('Walkthrough not exported', `${self.failed}. Run "floodnet export-walkthrough" (needs the GPU) and reload.`);
            return;
        }
        const s = self.state;
        const e = self.example();
        if (s.example !== e.id) Object.assign(s, { example: e.id, region: null, readout: null });
        const steps = self.steps(e)[s.tab];
        s.steps[s.tab] = Math.min(s.steps[s.tab], steps.length - 1);
        const step = steps[s.steps[s.tab]];
        const picker = self.data.examples.map(x => {
            const sample = data.samples.find(item => item.id === x.id);
            return `<button type="button" class="fn-walk-example ${x.id === e.id ? 'selected' : ''}" data-walk-example="${esc(x.id)}" aria-pressed="${x.id === e.id}">`
                + `<img src="${esc(x.shared.rgb)}" alt="">`
                + `<span>Image ${esc(x.image_id)}<small>${sample ? targetText(sample) : ''}</small></span></button>`;
        }).join('');
        const tabs = self.TABS.map(([id, label], i) => `<button type="button" role="tab" id="fn-walk-tab-${id}" class="${id === s.tab ? 'on' : ''}" aria-selected="${id === s.tab}" data-walk-tab="${id}"><span>${i + 1}</span>${esc(label)}</button>`).join('');
        const rail = steps.map((x, i) => `<li><button type="button" class="${i === s.steps[s.tab] ? 'on' : ''}" aria-current="${i === s.steps[s.tab] ? 'step' : 'false'}" data-walk-step="${i}"><span>${i + 1}</span>${esc(x.title)}</button></li>`).join('');
        const index = s.steps[s.tab];
        root.innerHTML = `<section class="fn-card"><div class="fn-card-top"><h2>Choose a scene</h2><div class="fn-control"><span class="fn-control-label">Training budget</span>${budgetToggle()}</div></div>`
            + `<div class="fn-walk-examples">${picker}</div><p class="fn-figure-note">${esc(self.data.selection)} Models: seed ${self.data.seed} (the explorer seed).</p></section>`
            + `<div class="fn-walk-tabs" role="tablist" aria-label="Method">${tabs}</div>`
            + `<section class="fn-card fn-walk-stage" role="tabpanel" aria-labelledby="fn-walk-tab-${s.tab}"><ol class="fn-walk-rail">${rail}</ol>`
            + `<div class="fn-walk-head"><span class="fn-eyebrow">Step ${index + 1} of ${steps.length}</span><h2>${esc(step.title)}</h2></div>`
            + `<div class="fn-walk-body"><div class="fn-walk-figures ${(step.figures.match(/<figure/g) || []).length > 1 ? 'two' : ''}">${step.figures}</div><div class="fn-walk-text">${step.text}</div></div>`
            + `<div class="fn-walk-nav"><button type="button" class="fn-btn" data-walk-move="-1" ${index === 0 ? 'disabled' : ''}>← Previous step</button>`
            + `<button type="button" class="fn-btn fn-btn-strong" data-walk-move="1">${index === steps.length - 1 ? (s.tab === 'unet' ? 'Back to preparation' : 'Next method →') : 'Next step →'}</button></div></section>`
            + self.verification(e);
        step.draw?.();
    },

    verification(e) {
        const b = e.budgets[String(state.budget)];
        const rows = [['rf', b.rf], ['unet', b.unet]].map(([method, part]) => `<li><span class="fn-pill ${method}">${esc(METHOD[method].short)}</span> `
            + `${part.check.matches_frozen_prediction ? '✓' : '✗'} final mask equals the frozen test prediction of <code>${esc(part.run_id.split('/').pop())}</code>${part.run_id.includes('/') ? ' (retrained forest)' : ''}: `
            + `${num(part.check.pixels - part.check.differing)} of ${num(part.check.pixels)} pixels identical · flooded IoU on this image ${pct(part.scores.flooded_miou)}</li>`).join('');
        return card('Is this what the models really did?', `<ul class="fn-walk-checks">${rows}</ul><p class="fn-figure-note">Checked when the walkthrough was exported: it reloads each frozen checkpoint (hash-verified), reruns the steps above and fails unless the result matches the saved test predictions pixel for pixel. The preparation step was also rerun from the raw photo and annotation and reproduced the cached 256-px arrays exactly.</p>`);
    },


    figure(inner, caption, label, attrs = {}) {
        const s = FN_WALKTHROUGH.state;
        return `<figure>${zoomable(inner, label, { group: `walk-${s.tab}`, title: label, caption: `Image ${FN_WALKTHROUGH.example().image_id} · ${state.budget}% labels`, pixelated: '', ...attrs })}<figcaption>${caption}</figcaption></figure>`;
    },

    image(url, caption, label, attrs) {
        return FN_WALKTHROUGH.figure(`<img src="${esc(url)}" alt="${esc(label)}">`, caption, label, attrs);
    },

    canvas(id, caption, label, attrs) {
        return FN_WALKTHROUGH.figure(`<canvas id="${id}" aria-label="${esc(label)}"></canvas>`, caption, label, attrs);
    },

    live(id, caption, label, hint) {
        const s = FN_WALKTHROUGH.state;
        return `<figure class="fn-walk-live"><div class="fn-walk-live-box"><canvas id="${id}" aria-label="${esc(label)}" tabindex="0"></canvas>`
            + `<button type="button" class="fn-zoom-corner" data-zoom data-zoom-target="${id}" data-zoom-group="walk-${s.tab}" data-zoom-title="${esc(label)}" data-zoom-pixelated="" aria-label="Expand: ${esc(label)}">⤢</button></div>`
            + `<figcaption>${caption}${hint ? `<small>${hint}</small>` : ''}</figcaption></figure>`;
    },

    classLegend(ids = [0, 1, 2]) {
        return `<div class="fn-legend">${ids.map(i => `<span><i style="background:${data.classes[i].colour}"></i>${esc(data.classes[i].name)}</span>`).join('')}</div>`;
    },

    async imageData(url) {
        const self = FN_WALKTHROUGH;
        if (!self.pixels.has(url)) {
            self.pixels.set(url, loadImage(url).then(image => {
                const canvas = document.createElement('canvas');
                canvas.width = image.naturalWidth;
                canvas.height = image.naturalHeight;
                const context = canvas.getContext('2d', { willReadFrequently: true });
                context.drawImage(image, 0, 0);
                return context.getImageData(0, 0, canvas.width, canvas.height);
            }));
        }
        return self.pixels.get(url);
    },

    rgbOf(hex) {
        return hex.match(/[0-9a-f]{2}/gi).map(x => parseInt(x, 16));
    },

    paint(id, width, height, fill) {
        const canvas = document.getElementById(id);
        if (!canvas) return null;
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d');
        const out = context.createImageData(width, height);
        fill(out.data);
        context.putImageData(out, 0, 0);
        return canvas;
    },

    async drawLabels(id, url, photoUrl = null, alpha = .62) {
        const self = FN_WALKTHROUGH;
        const labels = await self.imageData(url);
        const photo = photoUrl ? await self.imageData(photoUrl) : null;
        const palette = data.classes.map(c => self.rgbOf(c.colour));
        self.paint(id, labels.width, labels.height, out => {
            for (let i = 0; i < out.length; i += 4) {
                const colour = palette[labels.data[i]] || [0, 0, 0];
                const a = photo ? (labels.data[i] === 0 ? .1 : alpha) : 1;
                for (let c = 0; c < 3; c++) out[i + c] = photo ? photo.data[i + c] * (1 - a) + colour[c] * a : colour[c];
                out[i + 3] = 255;
            }
        });
    },

    async drawProbability(id, url, channel, hex) {
        const self = FN_WALKTHROUGH;
        const probs = await self.imageData(url);
        const colour = self.rgbOf(hex);
        const light = [248, 250, 252];
        self.paint(id, probs.width, probs.height, out => {
            for (let i = 0; i < out.length; i += 4) {
                const p = probs.data[i + channel] / 255;
                for (let c = 0; c < 3; c++) out[i + c] = light[c] + (colour[c] - light[c]) * p;
                out[i + 3] = 255;
            }
        });
    },


    steps(e) {
        const self = FN_WALKTHROUGH;
        const d = self.data;
        const s = self.state;
        const S = e.shared;
        const B = e.budgets[String(state.budget)];
        const g = e.geometry;
        const label = `Image ${e.image_id}`;
        const counts = S.truth_counts;
        const total = counts.reduce((a, b) => a + b, 0);
        const scoreLine = part => `Flooded IoU on this image: <strong>${pct(part.scores.flooded_miou)}</strong> (building ${pct(part.scores.building_iou)}, road ${pct(part.scores.road_iou)}).`;
        const present = d.original_classes.filter(c => S.original_shares[c.id] > 0);
        const regions = S.slic;
        const region = self.selectedRegion(e);
        const probs = B.rf.probabilities[region];
        const votes = B.rf.votes[region];
        const forestPick = probs.indexOf(Math.max(...probs));
        const votePick = votes.indexOf(Math.max(...votes));
        const featureLabel = name => name.replace('rgb_mean_', 'mean ').replace('rgb_std_', 'spread ').replace('hsv_mean_', 'mean ').replace('hsv_std_', 'spread ')
            .replace(/^lbp_(\d)$/, (m, k) => `texture bin ${k}`).replace(/ (r|g|b)$/, (m, c) => ` ${{ r: 'red', g: 'green', b: 'blue' }[c]}`).replace(/ (h|s|v)$/, (m, c) => ` ${{ h: 'hue', s: 'saturation', v: 'brightness' }[c]}`);
        const importance = d.feature_importances[String(state.budget)];
        const ranked = importance.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
        const regionButtons = [0, 1, 2].map(c => {
            const best = self.largestRegion(e, c);
            return best == null ? '' : `<button type="button" class="fn-btn" data-walk-region="${best}">Largest ${esc(data.classes[c].name.toLowerCase())} region</button>`;
        }).join('');
        return {
            prep: [
                {
                    title: 'The original photo',
                    figures: self.image(S.photo, `Original drone photo, ${S.original_size[0]}×${S.original_size[1]} px (shown at ${S.photo_size[0]} px)`, `${label}: original photo`, { hires: e.id, pixelated: null }),
                    text: `<p>Every method starts from the same released photo: <strong>${num(S.original_size[0])} × ${num(S.original_size[1])} pixels</strong> of Hurricane Harvey imagery. The photo's EXIF orientation is applied first, so it lines up with its annotation. All the integrity checks and fixes are on the <a href="#preprocessing">Data preprocessing</a> page.</p>`
                        + `<p>This scene's annotation marks <strong>${pct(S.original_shares[1])}</strong> of its pixels as flooded building and <strong>${pct(S.original_shares[3])}</strong> as flooded road.</p>`
                },
                {
                    title: 'The original annotation: ten classes',
                    figures: self.image(S.original_mask, 'Released annotation, official FloodNet colours', `${label}: original 10-class annotation`)
                        + `<div class="fn-legend">${present.map(c => `<span><i style="background:rgb(${c.colour.join(',')})"></i>${esc(c.name)}</span>`).join('')}</div>`,
                    text: `<p>Annotators labelled every pixel with one of ten classes. Only two of them are the study's targets: <strong>building flooded (ID 1)</strong> and <strong>road flooded (ID 3)</strong>.</p>`
                        + chartBlock('Share of this image\'s pixels', options => FN_CHARTS.bars(present.map(c => ({
                            label: c.name, value: S.original_shares[c.id], colour: `rgb(${c.colour.join(',')})`, emphasis: c.id === 1 || c.id === 3,
                            tip: [pct(S.original_shares[c.id], 2), `${c.name} (ID ${c.id})`]
                        })), { ...options, left: 150, rowHeight: 30, label: 'Original class shares in this image', note: `Targets: ${pct(S.original_shares[1] + S.original_shares[3])} of the image` }), { heading: 'h3' })
                },
                {
                    title: 'Resize and pad to 256 × 256',
                    figures: self.canvas('fn-walk-input', `Model input: ${g.width}×${g.height} image inside a 256×256 frame; hatched = padding`, `${label}: padded 256×256 input`),
                    text: `<p>The photo is shrunk to fit a <strong>256 × 256</strong> frame without distortion: here ${g.width} × ${g.height} pixels, placed ${g.top} px from the top. Photos use high-quality (Lanczos) resampling; the annotation uses nearest-neighbour so no new label values are invented.</p>`
                        + `<p>The leftover <strong>${num(256 * 256 - g.width * g.height)} padding pixels</strong> get the label 255 and are ignored in training and scoring.</p>`,
                    draw: async () => {
                        const input = await self.imageData(S.input);
                        self.paint('fn-walk-input', 256, 256, out => {
                            for (let y = 0; y < 256; y++) {
                                for (let x = 0; x < 256; x++) {
                                    const i = (y * 256 + x) * 4;
                                    const pad = y < g.top || y >= g.top + g.height || x < g.left || x >= g.left + g.width;
                                    const stripe = pad && (x + y) % 8 < 3;
                                    for (let c = 0; c < 3; c++) out[i + c] = stripe ? 148 : input.data[i + c];
                                    out[i + 3] = 255;
                                }
                            }
                        });
                    }
                },
                {
                    title: 'Three classes',
                    figures: self.canvas('fn-walk-truth', 'Ground truth after remapping', `${label}: three-class ground truth`) + self.classLegend(),
                    text: `<p>The ten classes collapse to three: ID 1 → <strong>flooded building</strong>, ID 3 → <strong>flooded road</strong>, everything else → <strong>Other</strong> (including open floodwater, which is not a target).</p>`
                        + `<dl class="fn-definition-list">${[0, 1, 2].map(c => `<dt>${esc(data.classes[c].name)}</dt><dd>${num(counts[c])} px · ${pct(counts[c] / total)}</dd>`).join('')}</dl>`
                        + (S.preparation_reproduced ? `<p class="fn-small fn-check">✓ Re-running this preparation from the raw photo and annotation reproduced the cached 256-px arrays exactly.</p>` : ''),
                    draw: () => self.drawLabels('fn-walk-truth', S.truth)
                }
            ],
            rf: [
                {
                    title: 'Group pixels into superpixels (SLIC)',
                    figures: self.image(regions.boundaries, `${regions.region_count} SLIC regions (white outlines)`, `${label}: SLIC superpixels`),
                    text: `<p>SLIC grows about ${regions.settings.n_segments} compact regions whose pixels are close together and similar in colour (compactness ${regions.settings.compactness}). This image gets <strong>${regions.region_count} regions</strong>.</p>`
                        + `<p>From here on the forest never looks at single pixels: it labels whole regions. That keeps it fast, but a region that mixes road and water must get one label.</p>`
                },
                {
                    title: 'Describe every region with 22 numbers',
                    figures: self.image(regions.mean_colour, 'Each region painted with its mean colour', `${label}: region mean colours`)
                        + self.image(regions.lbp, 'Local binary pattern (LBP) texture codes', `${label}: LBP texture`),
                    text: `<p>Each region becomes a row of <strong>22 features</strong>:</p><ul><li>12 colour numbers: the mean and spread of red, green, blue and of hue, saturation, brightness</li>`
                        + `<li>10 texture numbers: how often each LBP texture code occurs in the region (smooth water, rough foliage and roof edges give different mixes)</li></ul>`
                        + `<p>Shape, size and the neighbouring regions are not part of the description.</p>`
                },
                {
                    title: 'What the forest sees for one region',
                    figures: self.live('fn-walk-regions', 'Selected region outlined in yellow', `${label}: SLIC regions`, 'Click any region to inspect it')
                        + `<div class="fn-walk-quick">${regionButtons}</div>`,
                    text: `<p>Region <strong>#${region}</strong>: ${num(regions.area[region])} pixels, majority ground truth <strong>${esc(data.classes[regions.truth_label[region]].name)}</strong>.</p>`
                        + chartBlock(`Region #${region}: its 22 features`, options => FN_CHARTS.bars(d.feature_names.map((name, i) => ({
                            label: featureLabel(name), value: regions.features[region][i], text: regions.features[region][i].toFixed(3), colour: name.startsWith('lbp') ? '#64748b' : name.startsWith('hsv') ? '#0d9488' : '#4f46e5',
                            tip: [regions.features[region][i].toFixed(4), featureLabel(name), `Forest importance rank ${ranked.findIndex(r => r[1] === i) + 1} of 22`]
                        })), { ...options, max: 1, left: 150, rowHeight: 22, label: `Feature values of region ${region}` }), { heading: 'h3', note: 'Indigo = RGB, teal = HSV, grey = texture. All values lie between 0 and 1.' }),
                    draw: () => self.drawRegions(e)
                },
                {
                    title: `${B.rf.trees} trees vote`,
                    figures: self.canvas('fn-walk-p-building', 'Forest probability: flooded building, per region', `${label}: RF building probability`)
                        + self.canvas('fn-walk-p-road', 'Forest probability: flooded road, per region', `${label}: RF road probability`),
                    text: `<p>Each of the ${B.rf.trees} trees asks a chain of yes/no questions about the 22 numbers and ends in a class estimate. The forest averages them; the class with the highest average wins.</p>`
                        + `<p>Region #${region}: <strong>${pct(probs[forestPick])} ${esc(data.classes[forestPick].name.toLowerCase())}</strong>; ${votes[votePick]} of ${B.rf.trees} trees pick ${esc(data.classes[votePick].name.toLowerCase())}${forestPick !== votePick ? ' (the averaged probabilities and the plain vote disagree here; the average decides)' : ''}.</p>`
                        + chartBlock(`Region #${region}: forest output`, options => FN_CHARTS.bars([0, 1, 2].map(c => ({
                            label: data.classes[c].name, value: probs[c], colour: data.classes[c].colour, tip: [pct(probs[c]), `Average tree probability: ${data.classes[c].name}`, `${votes[c]} of ${B.rf.trees} trees pick it`]
                        })), { ...options, max: 1, left: 130, rowHeight: 30, label: 'Forest probabilities for the selected region' }), { heading: 'h3' })
                        + chartBlock(`What the ${state.budget}% forest relies on most`, options => FN_CHARTS.bars(ranked.slice(0, 8).map(([v, i]) => ({
                            label: featureLabel(d.feature_names[i]), value: v, text: pct(v), colour: '#059669', tip: [pct(v, 2), `Importance of ${featureLabel(d.feature_names[i])}`, 'Mean impurity decrease over all trees']
                        })), { ...options, left: 150, rowHeight: 26, label: 'Top eight feature importances', note: `Top 3 features carry ${pct(ranked.slice(0, 3).reduce((a, r) => a + r[0], 0))} of the importance` }), { heading: 'h3' }),
                    draw: () => self.drawRegionProbabilities(e)
                },
                {
                    title: 'Paint each region with its label',
                    figures: self.canvas('fn-walk-rf', 'SLIC + RF prediction', `${label}: SLIC + RF prediction`) + self.canvas('fn-walk-rf-truth', 'Ground truth', `${label}: ground truth`) + self.classLegend(),
                    text: `<p>Every pixel takes its region's winning class. Region edges therefore become label edges.</p><p>${scoreLine(B.rf)}</p>`,
                    draw: () => Promise.all([self.drawLabels('fn-walk-rf', B.rf.mask, S.rgb), self.drawLabels('fn-walk-rf-truth', S.truth, S.rgb)])
                }
            ],
            unet: [
                {
                    title: 'Normalise the input',
                    figures: self.image(S.rgb, 'The 256-px input (padding removed for display)', `${label}: model input`, { pixelated: null }),
                    text: `<p>The U-Net reads the whole image at once, every pixel. Each colour channel is shifted and scaled with the ImageNet statistics its encoder was pretrained with:</p>`
                        + FN_MATH.tex(String.raw`\hat x_k=\frac{x_k/255-\mu_k}{\sigma_k}`, { display: true, plain: '(value/255 − mean) ÷ std' })
                            + `<dl class="fn-definition-list">${['Red', 'Green', 'Blue'].map((c, i) => `<dt>${c}</dt><dd>${FN_MATH.tex(String.raw`\mu=${d.normalisation.mean[i].toFixed(3)},\ \sigma=${d.normalisation.std[i].toFixed(3)}`)}</dd>`).join('')}</dl>`
                        + `<p>So 0–255 pixel values become numbers between about −2.1 and 2.6.</p>`
                },
                {
                    title: 'Encoder: squeeze the image into features',
                    figures: `<div class="fn-walk-strip">${B.unet.encoder.map((x, i) => self.image(x.url, `Stage ${i + 1}: ${x.channels} channels @ ${x.resolution}×${x.resolution}`, `${label}: encoder stage ${i + 1}`)).join('')}</div>`,
                    text: `<p>A pretrained <strong>ResNet-18</strong> halves the resolution five times while growing the number of feature channels from 64 to 512. Early stages respond to edges and colours; deep stages respond to larger patterns, at 8×8 resolution.</p>`
                        + `<p>To show ${B.unet.encoder.at(-1).channels} channels in one picture, each stage is reduced to its <strong>first three principal components</strong>, shown as red, green and blue: pixels with similar colours have similar features. The colours themselves have no fixed meaning.</p>`
                },
                {
                    title: 'Decoder: rebuild the detail',
                    figures: `<div class="fn-walk-strip">${B.unet.decoder.map((x, i) => self.image(x.url, `Block ${i + 1}: ${x.channels} channels @ ${x.resolution}×${x.resolution}`, `${label}: decoder block ${i + 1}`)).join('')}</div>`,
                    text: `<p>Each decoder block doubles the resolution and joins in the encoder features of the same size through a <strong>skip connection</strong>. The deep features say <em>what</em> is there; the skips say <em>exactly where</em>.</p>`
                        + `<p>Notice the building and road outlines sharpening from 16×16 to 256×256 (channels ${B.unet.decoder.map(x => x.channels).join(' → ')}).</p>`
                },
                {
                    title: 'Three scores per pixel become probabilities',
                    figures: self.live('fn-walk-unet-building', 'P(flooded building)', `${label}: U-Net building probability`, 'Hover or tap for the values at a pixel')
                        + self.live('fn-walk-unet-road', 'P(flooded road)', `${label}: U-Net road probability`, 'Hover or tap for the values at a pixel'),
                    text: `<p>The last layer outputs three scores for every pixel; a softmax turns them into probabilities that sum to 1. Darker means more certain.</p>`
                        + `<div class="fn-walk-readout" id="fn-walk-readout">${self.readoutText()}</div>`,
                    draw: () => self.drawUnetProbabilities(e)
                },
                {
                    title: 'Pick the most likely class',
                    figures: self.canvas('fn-walk-unet', 'U-Net prediction', `${label}: U-Net prediction`) + self.canvas('fn-walk-unet-truth', 'Ground truth', `${label}: ground truth`) + self.classLegend(),
                    text: `<p>Each pixel takes the class with the highest probability. No regions are involved, so boundaries can follow roofs and road edges pixel by pixel.</p><p>${scoreLine(B.unet)}</p>`,
                    draw: () => Promise.all([self.drawLabels('fn-walk-unet', B.unet.mask, S.rgb), self.drawLabels('fn-walk-unet-truth', S.truth, S.rgb)])
                }
            ]
        };
    },


    largestRegion(e, label) {
        const r = e.shared.slic;
        let best = null;
        r.truth_label.forEach((l, i) => {
            if (l === label && (best == null || r.area[i] > r.area[best])) best = i;
        });
        return best;
    },

    selectedRegion(e) {
        const s = FN_WALKTHROUGH.state;
        if (s.region == null || s.region >= e.shared.slic.region_count) {
            s.region = [1, 2, 0].map(c => FN_WALKTHROUGH.largestRegion(e, c)).find(r => r != null) ?? 0;
        }
        return s.region;
    },

    async regionMap(e) {
        const encoded = await FN_WALKTHROUGH.imageData(e.shared.slic.regions);
        const map = new Int32Array(encoded.width * encoded.height);
        for (let i = 0; i < map.length; i++) map[i] = encoded.data[i * 4] + (encoded.data[i * 4 + 1] << 8) - 1;
        return { map, width: encoded.width, height: encoded.height };
    },

    async drawRegions(e) {
        const self = FN_WALKTHROUGH;
        const region = self.selectedRegion(e);
        const [{ map, width, height }, photo] = await Promise.all([self.regionMap(e), self.imageData(e.shared.rgb)]);
        self.paint('fn-walk-regions', width, height, out => {
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const k = y * width + x;
                    const i = k * 4;
                    const r = map[k];
                    const edge = (x > 0 && map[k - 1] !== r) || (y > 0 && map[k - width] !== r);
                    const inside = r === region;
                    const border = inside && ((x === 0 || map[k - 1] !== r) || (x === width - 1 || map[k + 1] !== r) || (y === 0 || map[k - width] !== r) || (y === height - 1 || map[k + width] !== r));
                    for (let c = 0; c < 3; c++) {
                        const v = photo.data[i + c];
                        out[i + c] = border ? [251, 191, 36][c] : inside ? v : edge ? 255 : v * .55;
                    }
                    out[i + 3] = 255;
                }
            }
        });
    },

    async drawRegionProbabilities(e) {
        const self = FN_WALKTHROUGH;
        const probs = e.budgets[String(state.budget)].rf.probabilities;
        const { map, width, height } = await self.regionMap(e);
        const light = [248, 250, 252];
        for (const [id, c] of [['fn-walk-p-building', 1], ['fn-walk-p-road', 2]]) {
            const colour = self.rgbOf(data.classes[c].colour);
            self.paint(id, width, height, out => {
                for (let k = 0; k < map.length; k++) {
                    const p = map[k] >= 0 ? probs[map[k]][c] : 0;
                    for (let ch = 0; ch < 3; ch++) out[k * 4 + ch] = light[ch] + (colour[ch] - light[ch]) * p;
                    out[k * 4 + 3] = 255;
                }
            });
        }
    },

    async drawUnetProbabilities(e) {
        const self = FN_WALKTHROUGH;
        const url = e.budgets[String(state.budget)].unet.probabilities;
        await Promise.all([self.drawProbability('fn-walk-unet-building', url, 1, data.classes[1].colour), self.drawProbability('fn-walk-unet-road', url, 2, data.classes[2].colour)]);
    },

    readoutText() {
        const r = FN_WALKTHROUGH.state.readout;
        if (!r) return '<p class="fn-small fn-muted">Point at either map to read the three probabilities at that pixel.</p>';
        return `<p class="fn-small">Pixel (${r.x}, ${r.y}): ${[0, 1, 2].map(c => `${esc(data.classes[c].name)} <strong>${pct(r.p[c])}</strong>`).join(' · ')} → <strong>${esc(data.classes[r.p.indexOf(Math.max(...r.p))].name)}</strong></p>`;
    },

    async readout(canvas, clientX, clientY) {
        const self = FN_WALKTHROUGH;
        const e = self.example();
        const probs = await self.imageData(e.budgets[String(state.budget)].unet.probabilities);
        const { x, y } = self.pixelAt(canvas, clientX, clientY);
        const i = (y * probs.width + x) * 4;
        self.state.readout = { x, y, p: [0, 1, 2].map(c => probs.data[i + c] / 255) };
        const box = $('#fn-walk-readout');
        if (box) box.innerHTML = self.readoutText();
    },

    pixelAt(canvas, clientX, clientY) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: Math.min(canvas.width - 1, Math.max(0, Math.floor((clientX - rect.left) / rect.width * canvas.width))),
            y: Math.min(canvas.height - 1, Math.max(0, Math.floor((clientY - rect.top) / rect.height * canvas.height)))
        };
    },

    move(delta) {
        const self = FN_WALKTHROUGH;
        const s = self.state;
        const count = self.steps(self.example())[s.tab].length;
        const next = s.steps[s.tab] + delta;
        if (next >= count) {
            const tabs = self.TABS.map(t => t[0]);
            s.tab = tabs[(tabs.indexOf(s.tab) + 1) % tabs.length];
            s.steps[s.tab] = 0;
        } else {
            s.steps[s.tab] = Math.max(0, next);
        }
        self.update();
        $('.fn-walk-tabs')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
};

document.addEventListener('click', event => {
    const self = FN_WALKTHROUGH;
    if (!$('#fn-walk') || !self.data) return;
    const target = event.target;
    const example = target.closest('[data-walk-example]');
    if (example) {
        self.state.example = example.dataset.walkExample;
        self.update();
        return;
    }
    const tab = target.closest('[data-walk-tab]');
    if (tab) {
        self.state.tab = tab.dataset.walkTab;
        self.update();
        $(`#fn-walk-tab-${self.state.tab}`)?.focus();
        return;
    }
    const step = target.closest('[data-walk-step]');
    if (step) {
        self.state.steps[self.state.tab] = Number(step.dataset.walkStep);
        self.update();
        return;
    }
    const move = target.closest('[data-walk-move]');
    if (move) {
        self.move(Number(move.dataset.walkMove));
        return;
    }
    const region = target.closest('[data-walk-region]');
    if (region) {
        self.state.region = Number(region.dataset.walkRegion);
        self.update();
        return;
    }
    if (target.id === 'fn-walk-regions') {
        self.regionMap(self.example()).then(({ map, width }) => {
            const { x, y } = self.pixelAt(target, event.clientX, event.clientY);
            const r = map[y * width + x];
            if (r >= 0) {
                self.state.region = r;
                self.update();
                $('#fn-walk-regions')?.focus();
            }
        });
    } else if (target.id === 'fn-walk-unet-building' || target.id === 'fn-walk-unet-road') {
        self.readout(target, event.clientX, event.clientY);
    }
});

document.addEventListener('pointermove', event => {
    const id = event.target.id;
    if ((id === 'fn-walk-unet-building' || id === 'fn-walk-unet-road') && FN_WALKTHROUGH.data) FN_WALKTHROUGH.readout(event.target, event.clientX, event.clientY);
});

document.addEventListener('keydown', event => {
    const tab = event.target.closest?.('[data-walk-tab]');
    if (!tab || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const tabs = FN_WALKTHROUGH.TABS.map(t => t[0]);
    FN_WALKTHROUGH.state.tab = tabs[(tabs.indexOf(tab.dataset.walkTab) + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
    FN_WALKTHROUGH.update();
    $(`#fn-walk-tab-${FN_WALKTHROUGH.state.tab}`)?.focus();
});
