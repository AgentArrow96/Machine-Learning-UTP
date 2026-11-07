'use strict';

const FN_PRES = {
    data: null,
    MODELS: ['rf', 'unet'],
    NAV: [
        { group: 'OVERVIEW', items: [['summary', 'Summary', 'S']] },
        { group: 'THE REPORT', items: [['problem', 'Define the problem', '01'], ['data', 'Understand the data', '02'], ['prepare', 'Prepare the data', '03'], ['features', 'Engineer the features', '04'], ['method', 'Explain the ML method', '05'], ['evaluate', 'Evaluate and interpret', '06'], ['sustain', 'Connect to sustainability', '07'], ['limits', 'Conclude with limits', '08']] },
        { group: 'BACK MATTER', items: [['sources', 'Sources and code', 'R']] }
    ],

    routes() {
        return FN_PRES.NAV.flatMap(g => g.items.map(i => i[0]));
    },

    async load() {
        const response = await fetch('data/presentation.json', { cache: 'no-store' });
        if (!response.ok) throw new Error(`Presentation export returned HTTP ${response.status}`);
        FN_PRES.data = await response.json();
    },

    m(method, budget) {
        return configMetrics(method, budget, 'before');
    },

    v(method, budget, key, digits = 1) {
        return pct(FN_PRES.m(method, budget)[key].mean, digits);
    },

    effect(method, key) {
        const rows = [25, 100].map(b => runsFor(method, b).map(runMetrics));
        return mean(rows[1].map((r, i) => r[key] - rows[0][i][key]));
    },

    miss(method, budget, c) {
        const total = runsFor(method, budget).reduce((sum, r) => {
            const row = r.metrics.confusion_matrix[c];
            return [sum[0] + row[c], sum[1] + row.reduce((a, b) => a + b, 0)];
        }, [0, 0]);
        return 1 - total[0] / total[1];
    },

    alarm(method, target, name, digits = 0) {
        return pct(FN_PRES.data.false_alarms[`${method}_100`][target][name] ?? 0, digits);
    },

    figure(file, title, caption, width = '') {
        const url = `data/presentation/${file}`;
        return `<figure class="fn-pres-figure${width ? ` ${width}` : ''}">${zoomable(`<img src="${url}" alt="${esc(title)}">`, title, { group: 'presentation', title, caption })}<figcaption>${esc(caption)}</figcaption></figure>`;
    },

    table(head, rows, extra = '') {
        return `<div class="fn-table-wrap"><table class="fn-table ${extra}"><thead><tr>${head.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
    },

    pager(route) {
        const routes = FN_PRES.routes();
        const items = navItems();
        const i = routes.indexOf(route);
        const link = (r, text, cls) => (r ? `<a class="fn-pres-link ${cls}" href="#${r}">${text}</a>` : '<span></span>');
        const name = r => items.find(x => x.route === r)?.label;
        return `<nav class="fn-pres-pager" aria-label="Presentation pages">${link(routes[i - 1], `<small>Previous</small>${esc(name(routes[i - 1]) || '')}`, 'prev')}${link(routes[i + 1], `<small>Next</small>${esc(name(routes[i + 1]) || '')}`, 'next')}</nav>`;
    },

    summary() {
        return header('summary', 'Reading the flood, one pixel at a time.', 'A comparative study of flooded infrastructure segmentation using traditional and deep learning approaches.')
            + metricCards([
                { value: FN_PRES.v('unet', 100, 'minority'), label: 'U-Net minority IoU', note: '100% labels, flooded building and road' },
                { value: FN_PRES.v('rf', 100, 'minority'), label: 'SLIC + RF minority IoU', note: '100% labels, the traditional baseline' },
                { value: `${num(data.dataset.counts.test)}`, label: 'Held-out test images', note: 'Official FloodNet split' },
                { value: 'SDG 11, 13', label: 'Sustainability focus', note: 'Sustainable Living challenge' }
            ])
            + takeaways([
                `<strong>The question:</strong> after a flood, which roads and buildings are affected? We compared a traditional pipeline (SLIC superpixels with a Random Forest) and a deep learning pipeline (U-Net with a pretrained ResNet-18) on post-Hurricane Harvey aerial images.`,
                `<strong>The result:</strong> with all labels, the U-Net reaches ${FN_PRES.v('unet', 100, 'minority')} minority IoU against ${FN_PRES.v('rf', 100, 'minority')} for SLIC + RF. Every U-Net run beats every forest run.`,
                `<strong>Insight over pure accuracy:</strong> flooded pixels are rare, so a model that never finds a flood already scores ${pct(trivialMetrics().accuracy)} accuracy. The forest scores about the same (${FN_PRES.v('rf', 100, 'accuracy')}) while missing ${pct(FN_PRES.miss('rf', 100, 2), 0)} of flooded road pixels.`,
                `<strong>What more labels buy:</strong> four times the labels lift U-Net flooded road IoU by ${points(FN_PRES.effect('unet', 'road_iou'))}, but flooded building IoU by only ${points(FN_PRES.effect('unet', 'building_iou'))}.`,
                `<strong>Recommendation:</strong> use the U-Net for a first-pass map that people check, and test it on a new flood before any real use.`
            ], 'Executive summary')
            + card('The team', `<p>TEB3123 Machine Learning, Group 3: Ammar Qayyim bin Mohd Adam, Ariff bin Azemi, Muhammad Haikal bin Man Ghazali, Wan Aisar binti Wan Aziz and Lim Jia Chyuen.</p><p>Dataset: FloodNet-Supervised v1.0. UTP Grand Challenge: Sustainable Living.</p>`);
    },

    problem() {
        return header('problem', 'Define the problem', 'Where is the flooded infrastructure, and which model should find it?')
            + `<div class="fn-grid fn-two-col">`
            + card('Context', `<p>Floods accounted for 41% of recorded disaster events between 2000 and 2019, affecting about 1.6 billion people (UNDRR, 2020). After a flood, response teams need to know which roads and buildings are affected, but analysing large areas of aerial imagery by hand is slow (Rahnemoonfar et al., 2021).</p>`)
            + card('Objectives', `<ol><li>Build a system that segments flooded buildings and roads in aerial photos, with SLIC + Random Forest and a U-Net.</li><li>Evaluate segmentation quality and compute cost.</li><li>Turn the results into insights for response planning, in line with SDG 11 and SDG 13.</li></ol>`)
            + `</div>`
            + card('Stakeholders', FN_PRES.table(['Who', 'Decision they face', 'What they get'], [
                ['Disaster management authorities', 'Where to direct assessment first', 'A map of likely flooded buildings and roads, for people to check'],
                ['Emergency response teams', 'Which roads and neighbourhoods appear affected', 'Photo and predicted mask, side by side'],
                ['Infrastructure and municipal planners', 'Which assets to inspect', 'How well each type of infrastructure is detected'],
                ['Researchers and data teams', 'Which model to use, how many images to label', 'Measured trade-offs between models, labels and compute']
            ]))
            + card('Scope', `<p><strong>In scope:</strong> three classes (flooded building, flooded road, other), two models, 25% and 100% of the labels, three training seeds, and evaluation at 256 × 256 pixels on the official test split.</p><p><strong>Out of scope:</strong> mapping all floodwater, flood depth, physical area, road passability, real-time use and other flood events. This is a research prototype.</p>`)
            + FN_PRES.pager('problem');
    },

    data_() {
        const train = data.dataset.processed_distributions.train;
        const total = train.pixel_counts.reduce((a, b) => a + b, 0);
        const overlap = overlapInfo();
        const chart = options => FN_CHARTS.bars(data.classes.map((c, i) => ({
            label: c.name, value: train.pixel_counts[i] / total, colour: c.colour, emphasis: i > 0,
            tip: [pct(train.pixel_counts[i] / total, 2), c.name, `${num(train.pixel_counts[i])} pixels at 256 px`]
        })), { ...options, label: 'Training pixel shares', note: 'Flooded targets are a tiny share of all pixels' });
        return header('data', 'Understand the data', 'FloodNet-Supervised v1.0: aerial photos taken after Hurricane Harvey, labelled pixel by pixel.')
            + metricCards([
                { value: num(data.dataset.counts.train), label: 'Training images' },
                { value: num(data.dataset.counts.val), label: 'Validation images' },
                { value: num(data.dataset.counts.test), label: 'Test images' },
                { value: '10 to 3', label: 'Classes', note: 'Original labels to study classes' }
            ])
            + card('Source and variables', `<p>Released by the Bina Lab, University of Maryland, Baltimore County, under the CDLA-Permissive 1.0 licence (Rahnemoonfar et al., 2021). The inputs are the red, green and blue values of each pixel (0 to 255). The target is the class of each pixel, stored in an annotation mask.</p>`
                + FN_PRES.figure('samples.webp', 'Sample FloodNet scenes', 'Sample scenes with their photo, label mask and overlay.', 'fn-pres-narrow'))
            + `<div class="fn-grid fn-two-col">`
            + card('Data quality', `<ul><li><strong>Missing:</strong> all 2,343 images pair with a mask; none are missing.</li><li><strong>Duplicates:</strong> no exact duplicates, but consecutive frames overlap.</li><li><strong>Orientation:</strong> 57 photos needed rotating to match their masks.</li><li><strong>Noise:</strong> shadows and glare vary, and floodwater can look like a flooded road.</li></ul>`)
            + card('Key patterns', `<ul><li><strong>Targets are rare:</strong> only ${pct((train.pixel_counts[1] + train.pixel_counts[2]) / total)} of training pixels are flooded infrastructure.</li><li><strong>Most images have no target:</strong> ${num(train.image_presence[1])} of ${num(train.image_presence[0])} training images hold a flooded building, and ${num(train.image_presence[2])} a flooded road.</li><li><strong>Scenes repeat:</strong> ${num(overlap.summary.test.overlapping_train)} of ${num(overlap.summary.test.images)} test images share ground with a training image.</li></ul>`)
            + `</div>`
            + vis('Training pixel share by class', chart, 'After remapping to three classes and resizing to 256 px; padding excluded.')
            + FN_PRES.pager('data');
    },

    prepare() {
        const w = runsFor('unet', 100)[0].class_weights.map(x => x.toFixed(2)).join(' / ');
        return header('prepare', 'Prepare the data', 'Every photo and mask goes through the same steps, and no image is removed.')
            + card('What we did and why', FN_PRES.table(['Step', 'What we did', 'Why'], [
                ['<strong>Missing values</strong>', 'None to fill. Padding pixels get an ignore label (255).', 'Padding is not real data, so it is never trained on or scored.'],
                ['<strong>Outliers</strong>', 'Kept glare and shadows. Corrected 57 rotated photos.', 'They are real flight conditions; removing them would make the test too easy.'],
                ['<strong>Encoding</strong>', 'Mask ID 1 becomes flooded building, ID 3 flooded road, and all others “Other”.', 'The goal is flooded infrastructure, not every object in the image.'],
                ['<strong>Scaling</strong>', 'Resized to 256 × 256, keeping the aspect ratio. U-Net inputs are normalised with ImageNet statistics.', 'Fits our computing resources and what the pretrained encoder expects. No target disappeared at this size.'],
                ['<strong>Data budgets</strong>', `Full training set and a stratified 25% subset (${num(data.dataset.budget_counts['25'])} images).`, 'Tests what happens when labels are scarce.'],
                ['<strong>Class imbalance</strong>', `Both models use the same capped class weights (${w} for Other / building / road at 100%).`, 'So a difference between models comes from the model, not from how it handles imbalance.']
            ]))
            + FN_MATH.equations(['weights'])
            + FN_PRES.pager('prepare');
    },

    features() {
        const importance = FN_PRES.data.feature_importance['100'].slice(0, 6);
        const nice = f => f.replace('hsv_mean_', 'HSV mean ').replace('hsv_std_', 'HSV SD ').replace('rgb_mean_', 'RGB mean ').replace('rgb_std_', 'RGB SD ').replace('lbp_', 'LBP bin ');
        const chart = options => FN_CHARTS.bars(importance.map(([f, v]) => ({
            label: nice(f), value: v, colour: METHOD.rf.colour, tip: [v.toFixed(3), nice(f), 'Forest feature importance']
        })), { ...options, left: 110, label: 'Top forest feature importances', note: 'Colour matters most, led by mean hue and saturation' });
        return header('features', 'Engineer the features', 'The forest uses inputs we designed; the U-Net learns its own.')
            + `<div class="fn-grid fn-two-col">`
            + card('SLIC + Random Forest: designed', `<p>SLIC groups each image into about 250 superpixels. Each one becomes a row with 22 features:</p>${FN_PRES.table(['Features', 'Why'], [
                ['RGB mean and SD (6)', 'Water, grass and roofs differ in colour'],
                ['HSV mean and SD (6)', 'Separates muddy water from vegetation'],
                ['Texture, LBP histogram (10)', 'Separates smooth water from rough foliage']
            ])}<p class="fn-figure-note">Limit: each superpixel is judged alone, with no view of its surroundings.</p>`)
            + card('U-Net: learned', `<p>The encoder learns its own colour, texture and shape patterns from the pixels. Three input decisions shape this:</p>${FN_PRES.table(['Decision', 'Why'], [
                ['ImageNet-pretrained ResNet-18', 'Only 361 or 1,445 images are available to learn from'],
                ['Matched flips and rotations', 'An aerial photo has no fixed “up”'],
                ['Mild brightness and contrast changes', 'Lighting varies across the flight']
            ])}`)
            + `</div>`
            + vis('What the forest relies on', chart, 'SLIC + RF at 100% labels, seed 42.')
            + FN_PRES.pager('features');
    },

    method() {
        return header('method', 'Explain the ML method', 'Supervised semantic segmentation: give every pixel a class.')
            + card('The two algorithms', `<p><strong>SLIC + Random Forest</strong> (the traditional baseline) classifies each superpixel from its colour and texture with 150 trees.</p>${FN_PRES.figure('rf_pipeline.webp', 'SLIC + Random Forest pipeline', 'The traditional baseline: resize, group pixels, describe regions, classify.')}`
                + `<p><strong>U-Net</strong> (deep learning) shrinks the image to learn what is present, then rebuilds it to find where, using skip connections to recover edges.</p>${FN_PRES.figure('unet.webp', 'U-Net architecture', 'U-Net with a pretrained ResNet-18 encoder.', 'fn-pres-narrow')}`)
            + `<div class="fn-grid fn-three-col">`
            + card('Data splits', `<p>The official split: ${num(data.dataset.counts.train)} training, ${num(data.dataset.counts.val)} validation and ${num(data.dataset.counts.test)} test images. Each model is trained at both label budgets with seeds ${data.protocol.train_seeds.join(', ')}.</p>`)
            + card('Tuning', `<p>No hyperparameter search, so the comparison stays simple. Settings were fixed in advance. The only choice made from data is the U-Net's best epoch, picked on validation (up to 30 epochs, patience 6).</p>`)
            + card('Baseline', `<p>SLIC + RF is the traditional baseline. A second reference, labelling every pixel “Other”, shows what a model that never finds a flood would score.</p>`)
            + `</div>`
            + FN_PRES.pager('method');
    },

    evaluate() {
        const models = FN_PRES.MODELS;
        const budgets = data.protocol.budgets;
        const t = trivialMetrics();
        const rows = models.flatMap(m => budgets.map(b => {
            const x = FN_PRES.m(m, b);
            return [`<span class="fn-dot" style="background:${METHOD[m].colour}"></span>${esc(METHOD[m].short)}`, `${b}%`, `<strong>${meanSd(x.minority)}</strong>`, meanSd(x.majority), meanSd(x.weighted), meanSd(x.accuracy)];
        }));
        rows.push(['<em>Label everything “Other”</em>', '', pct(t.minority), pct(t.majority), pct(t.weighted), pct(t.accuracy)]);
        const series = key => budgets.map(b => ({
            label: `${b}% labels`,
            series: models.map(m => ({ name: METHOD[m].short, colour: METHOD[m].colour, stat: summaryFor(m, b).full[key], note: null }))
        }));
        const bars = key => options => FN_CHARTS.rangeBars(series(key), { ...options, seeds: data.protocol.train_seeds, label: key });
        const pooled = (m, b) => runsFor(m, b).reduce((sum, r) => sum.map((row, i) => row.map((v, j) => v + r.metrics.confusion_matrix[i][j])), [[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
        const alarms = options => FN_CHARTS.bars(['water', 'background', 'grass', 'tree'].flatMap(name => models.map(m => {
            const v = FN_PRES.data.false_alarms[`${m}_100`].road[name] ?? 0;
            return { label: `${name[0].toUpperCase()}${name.slice(1)}, ${METHOD[m].short}`, value: v, colour: METHOD[m].colour, tip: [pct(v, 0), `${METHOD[m].short} false flooded-road pixels on ${name}`, '100% labels, mean of 3 seeds'] };
        })), { ...options, left: 130, label: 'Where false flooded-road pixels really are', note: 'Open water is the biggest source for both models' });
        return header('evaluate', 'Evaluate and interpret', 'Results on the 448 held-out test images, three seeds each.')
            + takeaways([
                `<strong>The U-Net is clearly better.</strong> ${FN_PRES.v('unet', 100, 'minority')} against ${FN_PRES.v('rf', 100, 'minority')} minority IoU at 100% labels, and every U-Net run beats every forest run.`,
                `<strong>Accuracy hides the problem.</strong> The forest's ${FN_PRES.v('rf', 100, 'accuracy')} accuracy equals that of a model that never predicts a flood.`,
                `<strong>Extra labels are not equally valuable.</strong> They lift U-Net road IoU by ${points(FN_PRES.effect('unet', 'road_iou'))} and building IoU by ${points(FN_PRES.effect('unet', 'building_iou'))}. The road gain comes from precision: fewer false alarms, not more finds.`,
                `<strong>Look-alikes cause the errors.</strong> ${FN_PRES.alarm('unet', 'road', 'water')} of the U-Net's false flooded-road pixels are open water, and ${FN_PRES.alarm('unet', 'building', 'building non-flooded')} of its false flooded-building pixels are dry buildings.`
            ], 'Key insights')
            + card('Metrics', `${FN_PRES.table(['Model', 'Labels', 'Minority IoU', 'Majority IoU', 'Weighted mIoU', 'Pixel accuracy'], rows, 'fn-metrics-table')}<p class="fn-figure-note">Mean ± SD over seeds. <strong>Minority IoU</strong> (flooded building and road) is the primary metric, because it is the only one here that a flood-blind model cannot score well on.</p>${FN_MATH.equations(['iou', 'fmiou'])}`)
            + `<div class="fn-grid fn-two-col">`
            + vis('Minority IoU', bars('flooded_miou'), 'Dots show each seed.')
            + vis('Flooded road IoU', bars('road_iou'), 'The biggest gain from more labels.')
            + vis('Flooded building IoU', bars('building_iou'), 'Little change from more labels.')
            + vis('False flooded-road pixels by true class', alarms, 'At 100% labels, mean of three seeds.')
            + `</div>`
            + `<div class="fn-grid fn-two-col">`
            + models.map(m => `<section class="fn-card">${chartBlock(`Errors: ${METHOD[m].short}, 100% labels`, options => FN_CHARTS.confusion(pooled(m, 100), CLASS_SHORT, options), { svg: false, note: 'Rows are the true class and each row sums to 100%. Pooled over seeds.' })}</section>`).join('')
            + `</div>`
            + card('Visual results', `<p>Six test scenes chosen by target content, not by quality. Amber is flooded building and teal is flooded road. The U-Net follows the real shapes, while the forest gives blocky patches and marks parts of open water as road. In some scenes (6412, 7291) both models largely miss the labelled flood.</p>${FN_PRES.figure('scenes.webp', 'Six test scenes', 'Photo, ground truth, SLIC + RF and U-Net, at 100% labels, seed 42. Scores are minority IoU on that image.')}`)
            + card('Cost', FN_PRES.table(['Model', 'Labels', 'Training time', 'Prediction per image', 'Peak GPU memory'], models.flatMap(m => budgets.map(b => {
                const s = summaryFor(m, b);
                const peak = Math.max(...runsFor(m, b).map(r => r.peak_gpu_allocated_bytes ?? -1));
                return [esc(METHOD[m].short), `${b}%`, seconds(s.training_seconds.mean), `${(s.inference_median_seconds.mean * 1000).toFixed(1)} ms`, peak < 0 ? 'CPU only' : `${(peak / 2 ** 20).toFixed(0)} MiB`];
            }))) + `<p class="fn-figure-note">The forest trains faster, but the U-Net predicts about nine times faster, because the forest computes superpixels and features for every image on the CPU.</p>`)
            + FN_PRES.pager('evaluate');
    },

    sustain() {
        const v = FN_PRES.v;
        return header('sustain', 'Connect to sustainability', 'What should people do with these results, and what does that rest on?')
            + card('Recommended actions', FN_PRES.table(['Action', 'For whom', 'Evidence', 'Assumptions and limits'], [
                ['<strong>Use the U-Net</strong> for flood mapping when a GPU is available', 'Data teams, authorities', `${v('unet', 100, 'minority')} vs ${v('rf', 100, 'minority')} minority IoU`, 'Scores come from one event, mostly on ground seen in training'],
                ['<strong>Keep people in the loop:</strong> a first-pass map, not automatic dispatch', 'Authorities, response teams', `Finds ${v('unet', 100, 'building_recall')} of building and ${v('unet', 100, 'road_recall')} of road pixels; ${v('unet', 100, 'building_precision')} and ${v('unet', 100, 'road_precision')} of flagged pixels are correct`, 'Someone can review the map. Pixels do not measure depth or passability'],
                ['<strong>Spend scarce labelling on roads</strong>, especially near water', 'Annotation teams', `More labels: road IoU ${points(FN_PRES.effect('unet', 'road_iou'))}, building IoU ${points(FN_PRES.effect('unet', 'building_iou'))}`, 'One subset and three seeds; targeted labelling itself was not tested'],
                ['<strong>Judge tools by per-class IoU</strong>, not accuracy', 'Authorities, procurers', `A flood-blind model scores ${pct(trivialMetrics().accuracy)} accuracy`, 'Applies whenever flooded pixels are rare'],
                ['<strong>Test on a new flood</strong> before any real use', 'Authorities, researchers', `${num(overlapInfo().summary.test.overlapping_train)} of ${num(overlapInfo().summary.test.images)} test images overlap training`, 'A labelled set from a new event is needed']
            ]))
            + card('Justified benefits', FN_PRES.table(['Benefit', 'SDG', 'Why we expect it', 'Not measured'], [
                ['Faster first look at a flooded area', 'SDG 11', `About ${(summaryFor('unet', 100).inference_median_seconds.mean * 1000).toFixed(1)} ms per image for the U-Net`, 'Time saved in a real response'],
                ['More focused inspection', 'SDG 11', 'Most flooded infrastructure pixels are flagged', 'Whether crews arrive sooner; lives or costs saved'],
                ['Less annotation for future flood data', 'SDG 13', 'Evidence on where more labels help', 'Annotation hours; other regions'],
                ['Modest computing demand', 'SDG 9 (indirect)', 'One laptop; 512 MiB peak GPU memory', 'Energy use']
            ]) + `<div class="fn-callout">We make no claim about lives saved or damage avoided; that needs a field study with responders. Errors have a cost too: a false alarm wastes a trip, and a missed flooded road could send a team into one.</div>`)
            + FN_PRES.pager('sustain');
    },

    limits() {
        const first = comparisonFor('rf', 100);
        return header('limits', 'Conclude with limits', 'What we can claim, and what we cannot.')
            + card('Limitations', `<ul><li><strong>One event, one region.</strong> A research prototype, not an operational service.</li><li><strong>Repeated ground.</strong> ${num(overlapInfo().target_images.overlapping)} of ${num(overlapInfo().target_images.images)} test images with flooded targets share ground with training, so scores do not predict a new flood.</li><li><strong>Three seeds, one 25% subset.</strong> This shows stability, not statistical significance.</li><li><strong>Low resolution and pixels only.</strong> At 256 px, thin edges can vanish, and pixels do not measure depth or road passability.</li><li><strong>The forest was revised after its first test evaluation</strong> so that both models handle imbalance alike (${pct(mean(first.baseline_scores))} to ${pct(mean(first.variant_scores))} at 100% labels). The change was not tuned on the test split, and the first version is reported, but this is not a fully blind result.</li><li><strong>No hyperparameter search,</strong> so neither score is necessarily a model's best.</li></ul>`)
            + `<div class="fn-grid fn-two-col">`
            + card('Conclusions', `<ol><li>Both pipelines produce pixel-level masks of flooded buildings and roads.</li><li>The U-Net clearly outperforms SLIC + RF (${FN_PRES.v('unet', 100, 'minority')} vs ${FN_PRES.v('rf', 100, 'minority')}), costs more to train and predicts faster.</li><li>Accuracy cannot guide this decision. Errors centre on look-alikes, and extra labels matter more for roads than buildings.</li></ol>`)
            + card('Future work', `<ul><li>Test on a flood with no overlap with training.</li><li>Train with the original ten labels, so models learn that water is not a road.</li><li>Give the forest context from neighbouring superpixels.</li><li>Stabilise U-Net training.</li><li>Try targeted labelling of road and water scenes.</li><li>Evaluate with responders.</li></ul>`)
            + `</div>`
            + FN_PRES.pager('limits');
    },

    sources() {
        const refs = [
            'Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., & Süsstrunk, S. (2012). SLIC superpixels compared to state-of-the-art superpixel methods. IEEE TPAMI, 34(11), 2274–2282.',
            'Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.',
            'He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR, 770–778.',
            'Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE TPAMI, 24(7), 971–987.',
            'Rahnemoonfar, M., Chowdhury, T., Sarkar, A., Varshney, D., Yari, M., & Murphy, R. R. (2021). FloodNet: A high resolution aerial imagery dataset for post flood scene understanding. IEEE Access, 9, 89644–89654.',
            'Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI, 234–241.',
            'United Nations. (n.d.). Goal 11 and Goal 13. https://sdgs.un.org/goals',
            'United Nations Office for Disaster Risk Reduction. (2020). The human cost of disasters: An overview of the last 20 years (2000–2019).'
        ];
        return header('sources', 'Sources and code', 'Where the evidence comes from, and how to reproduce it.')
            + card('Implementation', `<p>Code: <a href="https://github.com/AgentArrow96/Machine-Learning-UTP" target="_blank" rel="noopener noreferrer">github.com/AgentArrow96/Machine-Learning-UTP</a> (folder <code>gp/</code>). The README lists each stage.</p><ol><li>Set up the environment: <code>scripts/setup.ps1</code></li><li>Download the data: <code>python -m floodnet download</code></li><li>Run the experiment: <code>python scripts/run_experiment.py</code></li></ol><p>Needs Windows with PowerShell, Python 3.12 and an NVIDIA GPU. Switch this app to <strong>Full details</strong> for every metric, all test images and the step-by-step model walkthrough.</p>`)
            + card('References', `<ul>${refs.map(r => `<li>${esc(r)}</li>`).join('')}</ul>`)
            + FN_PRES.pager('sources');
    }
};
