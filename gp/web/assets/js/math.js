'use strict';

const FN_MATH = {
    cache: new Map(),

    ready() {
        return typeof katex !== 'undefined';
    },

    tex(expr, { display = false, plain = expr } = {}) {
        const key = `${display ? 'D' : 'I'}${expr}`;
        if (!FN_MATH.ready()) return `<span class="fn-tex-fallback">${FN_CHARTS.esc(plain)}</span>`;
        if (!FN_MATH.cache.has(key)) {
            FN_MATH.cache.set(key, katex.renderToString(expr, { displayMode: display, throwOnError: false, output: 'htmlAndMathml', strict: 'ignore' }));
        }
        return `<span class="fn-tex${display ? ' fn-tex-display' : ''}" data-plain="${FN_CHARTS.esc(plain)}">${FN_MATH.cache.get(key)}</span>`;
    },


    TOKEN: /(?<![\w.\-/#:])(±\s?|[+\-−])?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(e[-−]?\d+)?(\s?%|\s?pp\b|\s(?:ms|s|min|h|MiB|MB|GB|KB|px)\b)?(?:\s?×\s?(\d+(?:\.\d+)?)(\s?px\b)?)?(?![\w\-:])/g,

    numberTex(text) {
        return text.replace(/,/g, '{,}');
    },

    tokenTex(match, sign = '', number, exponent, unit, times, timesUnit) {
        let tex = '';
        const s = sign.trim();
        if (s === '±') tex += '\\pm ';
        else if (s === '+') tex += '+';
        else if (s) tex += '-';
        if (exponent) {
            const power = exponent.slice(1).replace('−', '-');
            tex += number === '1' ? `10^{${power}}` : `${FN_MATH.numberTex(number)}\\times10^{${power}}`;
        } else {
            tex += FN_MATH.numberTex(number);
        }
        const u = (unit || '').trim();
        if (u === '%') tex += '\\%';
        else if (u) tex += `\\,\\text{${u}}`;
        if (times) tex += `\\times${FN_MATH.numberTex(times)}${timesUnit ? '\\,\\text{px}' : ''}`;
        return tex;
    },

    SKIP: 'svg, script, style, code, pre, textarea, input, select, option, button, label, output, .katex, .fn-tex, .fn-tex-fallback, [data-plain], .fn-hash, .fn-nav-panel, .fn-seg',

    typeset(root) {
        if (!root || !FN_MATH.ready()) return;
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
            acceptNode(node) {
                if (!/\d/.test(node.nodeValue)) return NodeFilter.FILTER_REJECT;
                const parent = node.parentElement;
                return !parent || parent.closest(FN_MATH.SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
            }
        });
        const nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);
        for (const node of nodes) {
            const text = node.nodeValue;
            FN_MATH.TOKEN.lastIndex = 0;
            if (!FN_MATH.TOKEN.test(text)) continue;
            FN_MATH.TOKEN.lastIndex = 0;
            let html = '';
            let last = 0;
            for (const m of text.matchAll(FN_MATH.TOKEN)) {
                html += FN_CHARTS.esc(text.slice(last, m.index)) + FN_MATH.tex(FN_MATH.tokenTex(...m), { plain: m[0] });
                last = m.index + m[0].length;
            }
            html += FN_CHARTS.esc(text.slice(last));
            const span = document.createElement('span');
            span.innerHTML = html;
            node.replaceWith(...span.childNodes);
        }
    },

    observe(root) {
        let queued = false;
        const run = () => { queued = false; FN_MATH.typeset(root); };
        new MutationObserver(() => {
            if (!queued) { queued = true; requestAnimationFrame(run); }
        }).observe(root, { childList: true, subtree: true });
        run();
    },


    EQ: {
        iou: ['IoU', String.raw`\mathrm{IoU}_c=\frac{TP_c}{TP_c+FP_c+FN_c}`, 'Overlap of prediction and truth for class c, relative to their union; pooled pixel counts over the test images.'],
        fmiou: ['Flooded mIoU (primary metric)', String.raw`\mathrm{mIoU}_{\text{flood}}=\tfrac12\left(\mathrm{IoU}_{\text{building}}+\mathrm{IoU}_{\text{road}}\right)`, 'Mean IoU of the two target classes; "Other" is excluded.'],
        majority: ['Majority IoU (Other)', String.raw`\mathrm{IoU}_{\text{Other}}=\frac{TP_0}{TP_0+FP_0+FN_0}`, 'IoU of the background class "Other".'],
        miou3: ['3-class mIoU', String.raw`\mathrm{mIoU}_3=\tfrac13\left(\mathrm{IoU}_{\text{Other}}+\mathrm{IoU}_{\text{building}}+\mathrm{IoU}_{\text{road}}\right)`, 'Mean IoU over all three classes.'],
        fwiou: ['Weighted mIoU', String.raw`\mathrm{wIoU}=\sum_{c}\frac{n_c}{N}\,\mathrm{IoU}_c`, 'Each class IoU weighted by its share of true pixels (n_c of N).'],
        acc: ['Pixel accuracy', String.raw`\mathrm{Acc}=\frac{\sum_c TP_c}{N}`, 'Share of all scored pixels labelled correctly.'],
        dice: ['Dice',String.raw`\mathrm{Dice}_c=\frac{2\,TP_c}{2\,TP_c+FP_c+FN_c}`, 'Overlap relative to the combined mask size.'],
        recall: ['Recall', String.raw`\mathrm{Recall}_c=\frac{TP_c}{TP_c+FN_c}`, 'Share of annotated target pixels that were found.'],
        precision: ['Precision', String.raw`\mathrm{Precision}_c=\frac{TP_c}{TP_c+FP_c}`, 'Share of predicted target pixels that are correct (1 − false-alarm share).'],
        norm: ['Input normalisation', String.raw`\begin{aligned}\hat x_{k}&=\frac{x_{k}/255-\mu_k}{\sigma_k}\\ \mu&=(0.485,\,0.456,\,0.406),\quad \sigma=(0.229,\,0.224,\,0.225)\end{aligned}`, 'Per RGB channel k, with the ImageNet statistics the pretrained encoder expects.'],
        weights: ['Class weights', String.raw`w_c=\operatorname{clip}\!\left(\frac{N}{3\,n_c},\,0.25,\,4\right)`, 'Inverse pixel frequency in the budget\'s training images (n_c pixels of class c, N in total), clipped.'],
        ce: ['Weighted cross-entropy', String.raw`\mathcal L_{\text{CE}}=-\frac{\sum_i w_{y_i}\log p_{i,y_i}}{\sum_i w_{y_i}}`, 'Over all non-padding pixels i with label y_i and softmax probability p.'],
        aux: ['10-class auxiliary loss', String.raw`\mathcal L=\mathcal L_{\text{CE},w}^{(3)}+0.5\,\mathcal L_{\text{CE}}^{(10)}`, 'Improvement study: a second head learns the 10 original classes; only the 3-class head predicts.'],
        cosine: ['Cosine learning rate', String.raw`\begin{aligned}\eta_t&=\eta_{\min}+\tfrac12(\eta_0-\eta_{\min})\left(1+\cos\frac{\pi t}{T-1}\right)\\ \eta_0&=10^{-4},\quad \eta_{\min}=10^{-6},\quad T=60\end{aligned}`, 'Improvement study (training recipe): epoch t of at most T.'],
        ema: ['Weight averaging (EMA)', String.raw`\bar\theta\leftarrow0.999\,\bar\theta+0.001\,\theta`, 'After every optimiser step; the averaged weights are validated and saved.'],
        tta: ['Test-time augmentation', String.raw`\bar p(x)=\frac18\sum_{g\in D_4}g^{-1}\!\left(f(g(x))\right)`, 'Mean class probabilities over the 8 flips and rotations g of the image, each mapped back.'],
        area: ['512 → 256 px scoring', String.raw`\tilde p_{o}=\sum_{i}\frac{|\,o\cap i\,|}{|\,o\,|}\,p_i`, 'Each 256-px output pixel o averages the 512-px pixels i it covers (a 2×2 mean when the content is exactly twice as large).'],
        crf: ['DenseCRF energy', String.raw`\begin{aligned}E(\mathbf y)&=\sum_i -\log p_i(y_i)+\sum_{i<j}\mu(y_i,y_j)\,k(i,j)\\ k(i,j)&=w_1\exp\!\Big(-\tfrac{\lVert s_i-s_j\rVert^2}{2\theta_\gamma^2}\Big)+w_2\exp\!\Big(-\tfrac{\lVert s_i-s_j\rVert^2}{2\theta_\alpha^2}-\tfrac{\lVert c_i-c_j\rVert^2}{2\theta_\beta^2}\Big)\end{aligned}`, 'Unary term from the U-Net probabilities; a spatial kernel (θγ = sxy, w1) and a colour-aware kernel (θα = sxy, θβ = srgb, w2) pull nearby, similar pixels to one label.'],
        budget: ['Budget effect', String.raw`\Delta_{\text{budget}}=\frac13\sum_{s}\left(\mathrm{mIoU}^{100\%}_{\text{flood},s}-\mathrm{mIoU}^{25\%}_{\text{flood},s}\right)`, 'Seed-matched: the same seed s is trained at both budgets.'],
        change: ['Before → after change', String.raw`\Delta=\frac13\sum_{s\in\{42,43,44\}}\left(\mathrm{mIoU}^{\text{after}}_{\text{flood},s}-\mathrm{mIoU}^{\text{before}}_{\text{flood},s}\right)`, 'Seed-matched: each improved run is compared with the original run of the same budget and seed.'],
        bootstrap: ['95% bootstrap interval', String.raw`\mathrm{CI}_{95}=\left[\hat\Delta^{*}_{(2.5\%)},\ \hat\Delta^{*}_{(97.5\%)}\right],\quad B=5000`, 'Paired image bootstrap: the 448 test images are resampled with replacement and the seed-mean Δ recomputed.'],
        decision: ['Pre-registered decision', String.raw`\begin{aligned}\text{helps}&\iff \mathrm{CI}_{95}>0\ \wedge\ \Delta_s>0\ \forall s\\ \text{hurts}&\iff \mathrm{CI}_{95}<0\ \wedge\ \Delta_s<0\ \forall s\end{aligned}`, 'Otherwise "no clear effect".']
    },

    WIDE: new Set(['norm', 'crf', 'cosine']),

    equations(keys) {
        return `<div class="fn-equations">${keys.map(key => {
            const [name, tex, note] = FN_MATH.EQ[key];
            return `<div class="fn-equation${FN_MATH.WIDE.has(key) ? ' fn-equation-wide' : ''}"><div class="fn-equation-name">${FN_CHARTS.esc(name)}</div>${FN_MATH.tex(tex, { display: true, plain: name })}<p>${FN_CHARTS.esc(note)}</p></div>`;
        }).join('')}</div>`;
    },

    inline(key) {
        return FN_MATH.tex(FN_MATH.EQ[key][1], { plain: FN_MATH.EQ[key][0] });
    }
};
