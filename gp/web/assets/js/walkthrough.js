'use strict';

const FN_WALKTHROUGH = {
    kernelTex(sxy, srgb) {
        return FN_MATH.tex(String.raw`w=\exp\!\left(-\frac{d^2}{2\cdot${sxy}^2}-\frac{\lVert\Delta c\rVert^2}{2\cdot${srgb}^2}\right)`,
            { plain: `exp(−d²/2·${sxy}² − Δcolour²/2·${srgb}²)` });
    },

    data: null,
    failed: null,
    state: { example: null, tab: 'prep', steps: { prep: 0, rf: 0, unet: 0, crf: 0 }, region: null, iteration: null, kernel: null, readout: null },
    pixels: new Map(),

    TABS: [['prep', 'Preparation'], ['rf', 'SLIC + Random Forest'], ['unet', 'U-Net'], ['crf', 'U-Net + CRF']],

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
        if (s.example !== e.id) Object.assign(s, { example: e.id, region: null, kernel: null, iteration: null, readout: null });
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
            + `<button type="button" class="fn-btn fn-btn-strong" data-walk-move="1">${index === steps.length - 1 ? (s.tab === 'crf' ? 'Back to preparation' : 'Next method →') : 'Next step →'}</button></div></section>`
            + self.verification(e);
        step.draw?.();
    },

    verification(e) {
        const b = e.budgets[String(state.budget)];
        const rows = [['rf', b.rf], ['unet', b.unet], ['hybrid', b.hybrid]].map(([method, part]) => `<li><span class="fn-pill ${method}">${esc(METHOD[method].short)}</span> `
            + `${part.check.matches_frozen_prediction ? '✓' : '✗'} final mask equals the frozen test prediction of <code>${esc(part.run_id.split('/').pop())}</code>${part.run_id.includes('/') ? ' (Before, retrained forest)' : ''}: `
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
        const crf = d.crf;
        const iteration = s.iteration ?? crf.iterations;
        const lowConfidence = self.lowConfidence;
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
            ],
            crf: [
                {
                    title: 'Start from the U-Net\'s probabilities',
                    figures: self.canvas('fn-walk-uncertain', 'U-Net uncertainty: darker = less sure', `${label}: U-Net uncertainty`),
                    text: `<p>The hybrid trains nothing new. It takes the same U-Net's probabilities and turns each into a cost, ${FN_MATH.tex(String.raw`\psi_u(y_i)=-\log p_i(y_i)`, { plain: '−log p' })}: cheap for confident labels, expensive for unlikely ones.</p>`
                        + `<p id="fn-walk-uncertain-text">${lowConfidence == null ? 'Counting uncertain pixels…' : `${pct(lowConfidence)} of this image's pixels have a top probability below 90%. Those are where neighbours can change the label.`}</p>`,
                    draw: () => self.drawUncertainty(e)
                },
                {
                    title: 'Connect every pixel to similar neighbours',
                    figures: self.live('fn-walk-kernel', 'Appearance-kernel weights from the selected pixel (bright = strong pull)', `${label}: CRF appearance kernel`, 'Click any pixel to move the centre'),
                    text: `<p>The CRF adds two kinds of agreement costs between pixels (frozen settings, chosen on validation):</p>`
                        + `<dl class="fn-definition-list"><dt>Spatial</dt><dd>neighbours within about ${crf.gaussian.sxy} px should agree (weight ${crf.gaussian.compat}); removes isolated specks</dd>`
                        + `<dt>Appearance</dt><dd>pixels within about ${crf.bilateral.sxy} px <em>and</em> within about ${crf.bilateral.srgb} colour levels should agree (weight ${crf.bilateral.compat}); snaps labels to colour edges</dd></dl>`
                        + `<p id="fn-walk-kernel-text">The picture shows the appearance weight ${self.kernelTex(crf.bilateral.sxy, crf.bilateral.srgb)} from the chosen pixel, computed here in your browser.</p>`,
                    draw: () => self.drawKernel(e)
                },
                {
                    title: 'Mean-field iterations',
                    figures: self.canvas('fn-walk-iteration', `Labels after iteration ${iteration} (red outline = differs from the U-Net)`, `${label}: CRF iteration ${iteration}`)
                        + `<label class="fn-walk-slider">Iteration <input type="range" id="fn-walk-iteration-input" min="0" max="${crf.iterations}" step="1" value="${iteration}"><output>${iteration} of ${crf.iterations}</output></label>`,
                    text: `<p>Each iteration, every pixel re-weighs its label using its own cost plus the current labels of the pixels it is connected to. ${crf.iterations} iterations are used.</p>`
                        + `<p>After iteration <strong>${iteration}</strong>: ${num(B.hybrid.changed_from_previous[iteration])} pixels changed since the previous iteration; <strong>${num(B.hybrid.changed_from_unet[iteration])}</strong> now differ from the U-Net.</p>`
                        + chartBlock('Pixels changed per iteration', options => FN_CHARTS.lines([
                            { label: 'Changed since previous iteration', colour: '#c2410c', values: B.hybrid.changed_from_previous },
                            { label: 'Differ from the U-Net', colour: '#4f46e5', values: B.hybrid.changed_from_unet, dash: true }
                        ], { compact: true, ...options, integer: true, xName: 'Iteration', xStart: 0, label: 'Pixels changed per mean-field iteration',
                            points: [{ series: 1, index: crf.iterations, text: `${num(B.hybrid.changed_from_unet.at(-1))} px differ at the end` }] }), { heading: 'h3' }),
                    draw: () => self.drawIteration(e, iteration)
                },
                {
                    title: 'What the CRF fixed and what it broke',
                    figures: self.canvas('fn-walk-change', 'Green = fixed, red = broken, grey = changed but still wrong', `${label}: CRF changes against the ground truth`)
                        + self.canvas('fn-walk-hybrid', 'U-Net + CRF prediction', `${label}: U-Net + CRF prediction`)
                        + `<div class="fn-legend"><span><i style="background:#059669"></i>Fixed (was wrong, now right)</span><span><i style="background:#b91c1c"></i>Broken (was right, now wrong)</span><span><i style="background:#64748b"></i>Changed, still wrong</span></div>`,
                    text: `<p>Compared with the ground truth, the CRF <strong>fixed ${num(B.hybrid.fixed)}</strong> pixels and <strong>broke ${num(B.hybrid.broken)}</strong> on this image.</p>`
                        + `<p>Flooded IoU: U-Net ${pct(B.unet.scores.flooded_miou)} → U-Net + CRF <strong>${pct(B.hybrid.scores.flooded_miou)}</strong> (${points(B.hybrid.scores.flooded_miou - B.unet.scores.flooded_miou)}). Across the whole test split the CRF's effect is on the Results page.</p>`,
                    draw: () => Promise.all([self.drawChanges(e), self.drawLabels('fn-walk-hybrid', B.hybrid.mask, S.rgb)])
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

    async drawUncertainty(e) {
        const self = FN_WALKTHROUGH;
        const probs = await self.imageData(e.budgets[String(state.budget)].unet.probabilities);
        let low = 0;
        self.paint('fn-walk-uncertain', probs.width, probs.height, out => {
            for (let i = 0; i < out.length; i += 4) {
                const top = Math.max(probs.data[i], probs.data[i + 1], probs.data[i + 2]) / 255;
                if (top < .9) low++;
                const v = 255 * Math.min(1, Math.max(0, (top - 1 / 3) / (2 / 3)));
                out[i] = out[i + 1] = out[i + 2] = v;
                out[i + 3] = 255;
            }
        });
        self.lowConfidence = low / (probs.width * probs.height);
        const text = $('#fn-walk-uncertain-text');
        if (text) text.textContent = `${pct(self.lowConfidence)} of this image's pixels have a top probability below 90%. Those are where neighbours can change the label.`;
    },

    async drawKernel(e) {
        const self = FN_WALKTHROUGH;
        const photo = await self.imageData(e.shared.rgb);
        const { sxy, srgb } = self.data.crf.bilateral;
        const width = photo.width;
        const height = photo.height;
        if (!self.state.kernel) self.state.kernel = await self.defaultKernelPixel(e);
        const { x: x0, y: y0 } = self.state.kernel;
        const i0 = (y0 * width + x0) * 4;
        let strong = 0;
        let noticeable = 0;
        self.paint('fn-walk-kernel', width, height, out => {
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const i = (y * width + x) * 4;
                    const d2 = (x - x0) ** 2 + (y - y0) ** 2;
                    let c2 = 0;
                    for (let c = 0; c < 3; c++) c2 += (photo.data[i + c] - photo.data[i0 + c]) ** 2;
                    const w = Math.exp(-d2 / (2 * sxy * sxy) - c2 / (2 * srgb * srgb));
                    if (w > .1) strong++;
                    if (w > .01) noticeable++;
                    const shown = Math.sqrt(w);
                    const marker = Math.abs(x - x0) + Math.abs(y - y0) <= 2 && (x === x0 || y === y0);
                    for (let c = 0; c < 3; c++) out[i + c] = marker ? [225, 29, 72][c] : photo.data[i + c] * .42 * (1 - shown) + [251, 191, 36][c] * shown;
                    out[i + 3] = 255;
                }
            }
        });
        const text = $('#fn-walk-kernel-text');
        if (text) text.innerHTML = `From pixel (${x0}, ${y0}) (red cross): ${num(strong)} pixels have an appearance weight above 0.1 and ${num(noticeable)} above 0.01. Pixels of a different colour get almost no pull, even when they are close. Weight ${FN_WALKTHROUGH.kernelTex(sxy, srgb)}, computed here in your browser; brightness shows ${FN_MATH.tex(String.raw`\sqrt{w}`, { plain: '√weight' })} so weak pulls stay visible.`;
    },

    async defaultKernelPixel(e) {
        const self = FN_WALKTHROUGH;
        const b = e.budgets[String(state.budget)];
        const [unet, hybrid, photo] = await Promise.all([self.imageData(b.unet.mask), self.imageData(b.hybrid.mask), self.imageData(e.shared.rgb)]);
        const { sxy, srgb } = self.data.crf.bilateral;
        const width = unet.width;
        const height = unet.height;
        const changed = [];
        for (let k = 0; k < width * height; k++) if (unet.data[k * 4] !== hybrid.data[k * 4]) changed.push(k);
        if (!changed.length) return { x: Math.floor(width / 2), y: Math.floor(height / 2) };
        const sample = changed.filter((_, i) => i % Math.ceil(changed.length / 200) === 0);
        const reach = 2 * sxy;
        let best = sample[0];
        let bestPull = -1;
        for (const k of sample) {
            const x0 = k % width;
            const y0 = Math.floor(k / width);
            let pull = 0;
            for (let y = Math.max(0, y0 - reach); y < Math.min(height, y0 + reach); y += 2) {
                for (let x = Math.max(0, x0 - reach); x < Math.min(width, x0 + reach); x += 2) {
                    const i = (y * width + x) * 4;
                    let c2 = 0;
                    for (let c = 0; c < 3; c++) c2 += (photo.data[i + c] - photo.data[k * 4 + c]) ** 2;
                    pull += Math.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sxy * sxy) - c2 / (2 * srgb * srgb));
                }
            }
            if (pull > bestPull) {
                bestPull = pull;
                best = k;
            }
        }
        return { x: best % width, y: Math.floor(best / width) };
    },

    async drawIteration(e, iteration) {
        const self = FN_WALKTHROUGH;
        const b = e.budgets[String(state.budget)];
        const [labels, base, photo] = await Promise.all([self.imageData(b.hybrid.iterations[iteration]), self.imageData(b.unet.mask), self.imageData(e.shared.rgb)]);
        const palette = data.classes.map(c => self.rgbOf(c.colour));
        const width = labels.width;
        self.paint('fn-walk-iteration', width, labels.height, out => {
            for (let k = 0; k < width * labels.height; k++) {
                const i = k * 4;
                const label = labels.data[i];
                const differs = label !== base.data[i];
                const a = label === 0 ? .1 : .62;
                for (let c = 0; c < 3; c++) out[i + c] = differs ? [225, 29, 72][c] : photo.data[i + c] * (1 - a) + palette[label][c] * a;
                out[i + 3] = 255;
            }
        });
    },

    async drawChanges(e) {
        const self = FN_WALKTHROUGH;
        const b = e.budgets[String(state.budget)];
        const [truth, base, final, photo] = await Promise.all([self.imageData(e.shared.truth), self.imageData(b.unet.mask), self.imageData(b.hybrid.mask), self.imageData(e.shared.rgb)]);
        self.paint('fn-walk-change', truth.width, truth.height, out => {
            for (let i = 0; i < out.length; i += 4) {
                const t = truth.data[i];
                const before = base.data[i];
                const after = final.data[i];
                const colour = before === after ? null : after === t ? [5, 150, 105] : before === t ? [185, 28, 28] : [100, 116, 139];
                for (let c = 0; c < 3; c++) out[i + c] = colour ? colour[c] : photo.data[i + c] * .45 + 140;
                out[i + 3] = 255;
            }
        });
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
    } else if (target.id === 'fn-walk-kernel') {
        self.state.kernel = self.pixelAt(target, event.clientX, event.clientY);
        self.drawKernel(self.example());
    } else if (target.id === 'fn-walk-unet-building' || target.id === 'fn-walk-unet-road') {
        self.readout(target, event.clientX, event.clientY);
    }
});

document.addEventListener('pointermove', event => {
    const id = event.target.id;
    if ((id === 'fn-walk-unet-building' || id === 'fn-walk-unet-road') && FN_WALKTHROUGH.data) FN_WALKTHROUGH.readout(event.target, event.clientX, event.clientY);
});

document.addEventListener('input', event => {
    if (event.target.id !== 'fn-walk-iteration-input') return;
    FN_WALKTHROUGH.state.iteration = Number(event.target.value);
    FN_WALKTHROUGH.update();
    $('#fn-walk-iteration-input')?.focus();
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
