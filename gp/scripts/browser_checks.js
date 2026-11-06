async (page) => {
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    // Optional study bundles may be absent (404); any other console error fails the run.
    page.on('console', message => { if (message.type() === 'error' && !/data\/(uniform|undersample)\.json/.test(message.location().url)) errors.push(message.text()); });
    const checks = [];
    const assert = (condition, message) => { if (!condition) throw new Error(message); checks.push(message); };
    const text = locator => locator.evaluate(node => {
        const copy = node.cloneNode(true);
        copy.querySelectorAll('.fn-tex').forEach(t => t.replaceWith(t.dataset.plain));
        return copy.textContent.replace(/\s+/g, ' ');
    });
    const json = async name => {
        const response = await page.request.get(`http://127.0.0.1:8765/data/${name}.json`);
        return response.ok() ? response.json() : null;
    };
    // Navigate with the sidebar link of a route and wait until the router has rendered it (it renders on hashchange).
    const openMenu = async () => {
        if (await page.locator('.fn-mobile-menu').isVisible() && !(await page.locator('#fn-sidebar.open').count())) await page.getByRole('button', { name: 'Open navigation' }).click();
    };
    const go = async route => {
        await openMenu();
        const link = page.locator(`#fn-nav a[href="#${route}"]`);
        const number = (await link.locator('span').innerText()).trim();
        await link.click();
        await page.waitForFunction(([r, n]) => location.hash === `#${r}` && (r === 'overview' ? !!document.querySelector('#fn-page .fn-hero') : document.querySelector('#fn-page .fn-eyebrow')?.innerText.startsWith(n)), [route, number], { timeout: 10000 });
        return number;
    };
    const routes = () => page.locator('#fn-nav .fn-nav-btn').evaluateAll(links => links.map(a => a.hash.slice(1)));
    const navText = () => page.locator('#fn-nav .fn-nav-btn').allInnerTexts();
    const setView = async (view, against) => {
        await openMenu();
        await page.locator(`[data-view="${view}"]`).click();
        await page.waitForFunction(v => document.querySelector(`[data-view="${v}"]`)?.getAttribute('aria-pressed') === 'true', view);
        if (against) {
            await openMenu();
            await page.locator(`[data-compare-with="${against}"]`).click();
            await page.waitForFunction(a => document.querySelector(`[data-compare-with="${a}"]`)?.getAttribute('aria-pressed') === 'true', against);
        }
    };
    await page.emulateMedia({ reducedMotion: 'reduce' });
    // The persistent QA profile caches index.html; a unique query loads the current page and assets.
    await page.goto(`http://127.0.0.1:8765/?qa=${Date.now()}`);
    await page.evaluate(() => { try { localStorage.clear(); } catch { } });
    await page.goto(`http://127.0.0.1:8765/?qa=${Date.now()}#overview`);
    await page.getByRole('heading', { level: 1 }).first().waitFor();
    const bundle = await json('results');
    const after = await json('after');
    const uniform = await json('uniform');
    const under = uniform ? await json('undersample') : null;
    const pct = v => `${(v * 100).toFixed(1)}%`;
    assert(await page.locator('[data-view="before"]').getAttribute('aria-pressed') === 'true', 'The app opens in the Before view');
    const dialog = page.locator('dialog.fn-lightbox');

    // Every page of every view: no overflow, no LaTeX errors, matching page numbers; Compare views also on a phone.
    const views = [['before'], ...(under ? [['undersample'], ['compare', 'undersample']] : []), ...(after ? [['after'], ['compare', 'after']] : [])];
    for (const [view, against] of views) {
        const sizes = view === 'compare' ? [{ width: 1440, height: 1000 }, { width: 390, height: 844 }] : [{ width: 1440, height: 1000 }];
        for (const size of sizes) {
            await page.setViewportSize(size);
            await setView(view, against);
            const label = `${size.width}px ${view}${against ? ` with ${against}` : ''}`;
            for (const route of await routes()) {
                await go(route);
                assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)), `${label} #${route}: no page overflow`);
                assert(await page.locator('#fn-page .katex-error').count() === 0, `${label} #${route}: no LaTeX errors`);
                if (size.width <= 1024) {
                    await page.waitForFunction(() => document.querySelector('.fn-mobile-menu')?.getAttribute('aria-expanded') === 'false', null, { timeout: 5000 });
                }
            }
        }
    }
    await page.setViewportSize({ width: 1440, height: 1000 });

    // The Before view shows only Before content.
    await setView('before');
    const beforeNav = await navText();
    assert(!beforeNav.some(t => /After|Improvements|Undersampling/.test(t)), 'Before view: the improvement and undersampling pages leave the navigation');
    assert(!(await page.locator('#fn-export-status').innerText()).match(/improved|undersampled/), 'Before view: the sidebar status mentions no other study');
    for (const route of await routes()) {
        await go(route);
        const pageText = await text(page.locator('#fn-page'));
        const leak = pageText.match(/improvement study|improved (?:model|U-Net|hybrid|Random Forest|RF)|combined model|“After”|Showing:|undersampl|blacken/i);
        assert(!leak, `Before view #${route}: no After or undersampling content${leak ? ` (found "${leak[0]}")` : ''}`);
    }
    await go('results');
    const beforeSummary = (uniform || bundle).summary;
    assert(await page.locator('.fn-metrics-table').first().locator('tbody tr').count() === beforeSummary.length + 1, 'Results lists every metric for every configuration, plus the all-"Other" reference');
    assert(await page.locator('.fn-guide-table tbody tr').count() >= 9, 'Results explains every metric with its formula');
    if (uniform) {
        const first = page.getByRole('heading', { name: 'First version of Before (kept for reference)' }).locator('..');
        assert(await first.locator('tbody tr').count() === 2, 'Before view keeps the first-version forest next to the uniform one');
        const rf = uniform.summary.find(s => s.method === 'rf' && s.budget === 100);
        assert((await text(page.locator('.fn-table').first())).includes(pct(rf.full.flooded_miou.mean)), 'Before view reports the uniform forest');
    }
    await page.goto(`http://127.0.0.1:8765/?qa=${Date.now()}#changes`);
    await page.waitForFunction(() => location.hash === '#overview');
    checks.push('Before view: a direct link to an improvement page falls back to the overview');
    if (under) {
        await page.goto(`http://127.0.0.1:8765/?qa=${Date.now()}#undersample`);
        await page.waitForFunction(() => location.hash === '#overview');
        checks.push('Before view: a direct link to the undersampling page falls back to the overview');
    }
    await page.getByRole('heading', { level: 1 }).first().waitFor();

    if (under) {
        await setView('undersample');
        const nav = await navText();
        assert(nav.some(t => t.includes('Undersampling pipeline')) && !nav.some(t => /After|Improvements/.test(t)), 'Undersampling view: its own page, no improvement-study pages');
        for (const route of await routes()) {
            await go(route);
            const leak = (await text(page.locator('#fn-page'))).match(/improvement study|improved (?:model|U-Net|hybrid|Random Forest|RF)|combined model|“After”/i);
            assert(!leak, `Undersampling view #${route}: no improvement-study content${leak ? ` (found "${leak[0]}")` : ''}`);
        }
        await go('undersample');
        assert(await page.locator('#fn-page .fn-result-tile').count() === under.comparisons.length, 'Undersampling page shows every Before → undersampling result');
        const ratioRows = page.getByRole('heading', { name: 'Choosing r on validation' }).locator('..').locator('tbody tr');
        assert(await ratioRows.count() === 4 && await page.locator('.fn-selected-cell').count() === 4, 'Ratio table: one chosen r per model family and budget');
        assert(await page.locator('.fn-equation .katex-display').count() >= 4, 'Undersampling rule and blackening shown as display equations');
        const images = page.locator('.fn-blacken-grid img');
        assert(await images.count() === 5, 'Blackening test shows the training photo, its blackened copy and three test masks');
        await images.last().scrollIntoViewIfNeeded();
        await page.waitForFunction(() => [...document.querySelectorAll('.fn-blacken-grid img')].every(i => i.complete && i.naturalWidth > 0));
        checks.push('Blackening example images load');
        const blackRows = page.getByRole('heading', { name: 'What about blackening the unneeded classes?' }).locator('..').locator('.fn-metrics-table tbody tr');
        assert(await blackRows.count() === under.blacken.comparisons.length, 'Blackening results listed for every configuration');
        const pageText = await text(page.locator('#fn-page'));
        assert(under.comparisons.every(c => pageText.includes(c.decision)), 'Undersampling page shows every pre-registered decision');
        await page.screenshot({ path: 'output/playwright/undersample-desktop.png', fullPage: true });
        await setView('compare', 'undersample');
        await go('results');
        const compareText = await text(page.locator('.fn-compare-table'));
        const u = under.summary.find(s => s.method === 'unet' && s.budget === 100);
        assert(compareText.includes(pct(u.full.flooded_miou.mean)) && under.comparisons.every(c => compareText.includes(c.decision)), 'Compare with undersampling: table shows the undersampled seed means and decisions');
        assert(await page.getByRole('heading', { name: 'Minority, majority and weighted IoU: what changed' }).count() === 1, 'Compare with undersampling: minority, majority and weighted IoU side by side');
        await go('explorer');
        await page.waitForFunction(() => document.querySelectorAll('canvas[data-mask-kind]').length > 0);
        assert(await page.locator('canvas[data-mask-kind="unet@undersample"]').count() === 1, 'Compare explorer shows the undersampled U-Net mask');
    }

    if (after) {
        await setView('compare', 'after');
        await go('changes');
        await page.locator('.fn-chapter').first().waitFor();
        assert(await page.locator('.fn-chapter').count() === bundle.protocol.methods.length, 'What-changed page has one chapter per method');
        const tried = new Set(after.single_changes.map(c => `${c.variant}/${c.method === 'rf' ? 'rf' : 'unet'}`)).size;
        assert(await page.locator('.fn-chapter .fn-idea[class*="kept"], .fn-chapter .fn-idea.dropped').count() >= tried, 'Every idea tried is explained (logic, how it works, what happened)');
        assert(await page.locator('.fn-idea.dropped').count() === 2, 'Ideas that were left out are explained too');
        const changed = Object.values(after.methodology).flat().filter(r => r.changed).length;
        assert(await page.locator('.fn-method-diff tr.fn-changed').count() === changed, 'What-changed page marks every changed pipeline step');
        assert(await page.locator('.fn-result-tile').count() === after.comparisons.length, 'What-changed page shows every before → after result');
        await page.screenshot({ path: 'output/playwright/changes-desktop.png', fullPage: true });
    }

    await setView('before');
    await go('overview');
    assert((await page.locator('.fn-table tbody tr').count()) === beforeSummary.length, 'Overview summary table lists every configuration');
    assert(await page.locator('#fn-page .katex').count() > 10, 'Numbers on the page are typeset as LaTeX');
    await page.screenshot({ path: 'output/playwright/overview-desktop.png', fullPage: true });
    await go('dataset');
    if (bundle.dataset?.overlap) {
        const pageText = await text(page.locator('#fn-page'));
        assert(pageText.includes(`${bundle.dataset.overlap.summary.test.overlapping_train} / ${bundle.dataset.overlap.summary.test.images}`), 'Dataset page shows verified test overlap counts');
        await page.screenshot({ path: 'output/playwright/dataset-desktop.png', fullPage: true });
    }
    await go('methodology');
    assert(await page.locator('.fn-method-card').count() === bundle.protocol.methods.length, 'Methodology shows one card per approach');
    assert(await page.locator('.fn-equation .katex-display').count() >= 15, 'Methods page shows the equation reference in display LaTeX');
    assert(await page.locator('[data-toc] .fn-toc-link').count() >= 5, 'Long pages get an "On this page" index');
    await page.screenshot({ path: 'output/playwright/methodology-desktop.png', fullPage: true });
    const prep = await json('preprocessing');
    if (prep) {
        const steps = prep.steps;
        await go('preprocessing');
        await page.locator('#fn-prep .fn-walk-stage').waitFor();
        assert((await text(page.locator('#fn-prep'))).includes(`${steps.pairing.pairs.toLocaleString('en-US')} photo`), 'Preprocessing page states the audited pair count');
        await page.locator('[data-prep-step="3"]').click();
        assert((await text(page.locator('#fn-prep .fn-walk-text'))).includes(`${steps.orientation.rotated} of`), 'Orientation step shows the EXIF-rotated count');
        await page.locator('#fn-prep .fn-zoom').first().click();
        await dialog.locator('.fn-lightbox-media').waitFor();
        await page.keyboard.press('ArrowRight');
        await page.keyboard.press('Escape');
        checks.push('Preprocessing figures open in the viewer and flip');
        await page.locator('[data-prep-move="1"]').click();
        assert((await text(page.locator('#fn-prep .fn-walk-head'))).toLowerCase().includes('step 5 of'), 'Preprocessing Next step works');
        await page.screenshot({ path: 'output/playwright/preprocessing-desktop.png', fullPage: true });
    }
    const imbalance = await json('imbalance');
    if (imbalance) {
        await go('imbalance');
        await page.locator('#fn-imbalance .fn-dark-card').waitFor();
        const forests = await page.locator('#fn-imbalance svg.fn-chart circle.fn-bar').count();
        assert(forests === imbalance.comparisons.filter(c => c.budget === 100).length, 'Class-imbalance forest plots show every 100% comparison');
        await page.locator('[data-imb-budget="25"]').click();
        assert(await page.locator('[data-imb-budget="25"]').getAttribute('aria-pressed') === 'true', 'Class-imbalance budget switch works');
        await page.locator('#fn-imbalance svg.fn-chart .fn-mark').first().hover();
        assert((await page.locator('.fn-tooltip').innerText()).includes('CI'), 'Forest-plot tooltip shows the interval');
        await page.locator('.fn-gan-row .fn-zoom').first().click();
        await dialog.locator('.fn-lightbox-media').waitFor();
        await page.keyboard.press('ArrowRight');
        await page.keyboard.press('Escape');
        checks.push('GAN gallery opens in the viewer and flips between generators');
        await page.screenshot({ path: 'output/playwright/imbalance-desktop.png', fullPage: true });
    }
    const improve = await json('improve');
    if (improve && after) {
        await setView('after');
        await go('improve');
        await page.locator('#fn-improve .fn-dark-card').waitFor();
        assert(await page.locator('#fn-improve .fn-chart-pair svg.fn-chart').count() === 2, 'Improvements page shows both false-alarm charts');
        await page.locator('[data-imp-method="rf"]').click();
        assert(await page.locator('[data-imp-method="rf"]').getAttribute('aria-pressed') === 'true', 'Improvements model switch works');
        await page.locator('#fn-improve .fn-chart-pair .fn-mark').first().hover();
        assert((await page.locator('.fn-tooltip').innerText()).includes('false alarms'), 'False-alarm tooltip names the original class');
        if (improve.status === 'complete') {
            const forests = await page.locator('#fn-improve svg.fn-chart circle.fn-bar').count();
            assert(forests === improve.comparisons.filter(c => c.budget === 100).length, 'Improvement forest plots show every 100% comparison');
        }
        await page.screenshot({ path: 'output/playwright/improve-desktop.png', fullPage: true });
        await setView('before');
    }
    const walk = await json('walkthrough');
    if (walk) {
        await go('walkthrough');
        await page.locator('.fn-walk-stage').waitFor();
        assert(await page.locator('[data-walk-example]').count() === walk.examples.length, 'Walkthrough offers every exported example');
        assert(await page.getByRole('tab').count() === 4, 'Walkthrough has one tab per stage');
        if (uniform) assert(walk.rf_variant === 'undersample/uniform', 'Walkthrough replays the uniform Before forests');
        await page.locator('[data-walk-move="1"]').click();
        assert((await text(page.locator('.fn-walk-head'))).toLowerCase().includes('step 2 of 4'), 'Walkthrough Next step works');
        await page.getByRole('tab', { name: /SLIC \+ Random Forest/ }).click();
        await page.locator('[data-walk-step="2"]').click();
        const before = await page.locator('.fn-walk-text').innerText();
        const box = await page.locator('#fn-walk-regions').boundingBox();
        await page.mouse.click(box.x + box.width * .15, box.y + box.height * .2);
        await page.waitForFunction(t => document.querySelector('.fn-walk-text')?.innerText !== t, before);
        checks.push('Clicking a superpixel selects it and shows its features');
        await page.getByRole('tab', { name: /U-Net \+ CRF/ }).click();
        await page.locator('[data-walk-step="1"]').click();
        const kernel = await page.locator('#fn-walk-kernel').boundingBox();
        await page.mouse.click(kernel.x + kernel.width * .3, kernel.y + kernel.height * .4);
        await page.waitForFunction(() => document.querySelector('#fn-walk-kernel-text')?.innerText.startsWith('From pixel'));
        checks.push('CRF kernel view responds to a click');
        await page.locator('[data-walk-step="2"]').click();
        await page.locator('#fn-walk-iteration-input').fill('2');
        assert((await text(page.locator('.fn-walk-slider output'))) === `2 of ${walk.crf.iterations}`, 'CRF iteration slider works');
        assert((await page.locator('.fn-walk-checks li').allInnerTexts()).every(t => t.includes('✓')), 'Walkthrough shows a frozen-prediction match for every method');
        await page.screenshot({ path: 'output/playwright/walkthrough-desktop.png', fullPage: true });
    }

    await go('results');
    await page.screenshot({ path: 'output/playwright/results-desktop.png', fullPage: true });
    const gridlines = await page.evaluate(() => [...document.querySelectorAll('svg.fn-chart line')]
        .filter(l => l.getAttribute('y1') === l.getAttribute('y2') && Math.abs(l.x2.baseVal.value - l.x1.baseVal.value) > 200 && !l.classList.contains('fn-axis')).length);
    assert(gridlines === 0, 'Charts have no horizontal gridlines');
    assert(await page.locator('svg.fn-chart .fn-note').count() > 0, 'Budget chart is annotated by default');
    await page.locator('svg.fn-chart .fn-mark').first().hover();
    await page.locator('.fn-tooltip').waitFor({ state: 'visible' });
    assert((await page.locator('.fn-tooltip').innerText()).includes('mean'), 'Chart tooltip appears on hover');
    await page.locator('[data-series-toggle="0"]').first().click();
    assert(await page.locator('svg.fn-chart [data-series="0"].fn-off').count() > 0, 'Legend entry hides its series');
    await page.locator('[data-series-toggle="0"]').first().click();
    await page.getByRole('button', { name: 'Notes on', exact: true }).first().click();
    assert(!(await page.locator('svg.fn-chart .fn-note').first().isVisible()), 'Notes toggle hides annotations');
    await page.getByRole('button', { name: 'Notes off', exact: true }).first().click();
    await page.getByRole('button', { name: 'Expand', exact: true }).first().click();
    await dialog.locator('svg.fn-chart').waitFor();
    assert(await dialog.locator('svg.fn-chart.fn-large').count() === 1, 'Expand opens a larger chart');
    await dialog.locator('svg.fn-chart .fn-mark').first().focus();
    assert(await page.locator('dialog .fn-tooltip').isVisible(), 'Tooltip works inside the expanded chart (keyboard focus)');
    await page.keyboard.press('Escape');
    assert(!(await dialog.isVisible()), 'Expanded chart closes with Escape');
    const best = beforeSummary.find(s => s.method === 'unet' && s.budget === 100);
    assert((await text(page.locator('.fn-table').first())).includes(pct(best.full.flooded_miou.mean)), 'Results table shows the official-split seed mean');
    if (after) {
        await setView('compare', 'after');
        const fullText = await text(page.locator('.fn-compare-table'));
        const improved = after.summary.find(s => s.method === 'unet' && s.budget === 100);
        assert(fullText.includes(pct(improved.full.flooded_miou.mean)), 'Compare with After: table shows the improved seed mean next to the original');
        assert(after.comparisons.every(c => fullText.includes(c.decision)), 'Compare with After: table shows every pre-registered decision');
        await setView('after');
        const afterText = await text(page.locator('.fn-table').first());
        assert(afterText.includes(pct(improved.full.flooded_miou.mean)) && !afterText.includes('Change (per seed)'), 'After view shows only the improved models');
        await setView('before');
        await page.reload();
        await page.getByRole('heading', { level: 1 }).first().waitFor();
        assert(await page.locator('[data-view="before"]').getAttribute('aria-pressed') === 'true', 'The chosen view survives a reload');
        await setView('compare', 'after');
    }
    if (best.overlap_low) {
        const strata = await text(page.getByRole('heading', { name: 'Does scene overlap inflate the scores?' }).locator('..'));
        assert(strata.includes(pct(best.overlap_low.flooded_miou.mean)) && strata.includes(pct(best.overlap_high.flooded_miou.mean)), 'Overlap-strata card shows both halves');
    }
    if (bundle.hybrid) {
        const crf = await text(page.getByRole('heading', { name: 'Does CRF refinement help?' }).locator('..'));
        const gridRows = await page.getByRole('heading', { name: 'Does CRF refinement help?' }).locator('..').locator('tr.fn-selected-row').count();
        assert(crf.includes(bundle.hybrid.adopted ? 'adopted' : 'rejected') && gridRows === 1, 'CRF card shows the verdict and one selected setting');
    }
    const runIds = await page.locator('#fn-metric-run option').evaluateAll(o => o.map(x => x.value));
    await page.locator('#fn-metric-run').selectOption(runIds[0]);
    await page.locator('#fn-class-metrics').waitFor();
    assert((await text(page.locator('#fn-class-metrics'))).includes('Flooded road'), 'Selected-run per-class table renders');
    if (after) assert(await page.locator('#fn-class-metrics-after').count() === 1, 'Compare view inspects the matching run of both versions');
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Save SVG', exact: true }).first().click();
    const download = await downloadPromise;
    await download.saveAs('output/playwright/budget-chart.svg');
    assert(download.suggestedFilename().endsWith('.svg'), 'Chart SVG download succeeds');
    await go('training');
    await page.locator(`[data-loss-seed="${bundle.protocol.train_seeds.at(-1)}"]`).click();
    assert(await page.locator(`[data-loss-seed="${bundle.protocol.train_seeds.at(-1)}"]`).getAttribute('aria-pressed') === 'true', 'Loss-curve seed switch works');
    await page.screenshot({ path: 'output/playwright/training-desktop.png', fullPage: true });
    await go('explorer');
    await page.getByRole('button', { name: '25% labels', exact: true }).click();
    assert(await page.getByRole('button', { name: '25% labels', exact: true }).getAttribute('aria-pressed') === 'true', '25% budget switch works');
    await page.locator('#fn-presence').selectOption('both');
    await page.locator('#fn-search').fill('no-such-image');
    await page.getByRole('heading', { name: 'No matching images', exact: true }).waitFor();
    checks.push('Empty filter has a clear state');
    await page.locator('#fn-search').fill('');
    await page.locator('#fn-presence').selectOption('all');
    if (bundle.dataset?.overlap) {
        await page.locator('#fn-overlap').selectOption('clean');
        const count = await text(page.locator('#fn-sample-count'));
        assert(count.startsWith(`${bundle.dataset.overlap.summary.test.clean} /`), 'Clean-only filter matches the clean test count');
        await page.locator('#fn-overlap').selectOption('all');
    }
    await page.getByRole('button', { name: 'Next →', exact: true }).click();
    assert((await text(page.locator('#fn-pager'))).includes('Page 2'), 'Gallery pagination works');
    const chosen = bundle.samples.find(s => s.id === bundle.representatives[0]) || bundle.samples[0];
    await page.locator('#fn-search').fill(chosen.image_id);
    await page.locator(`[data-sample="${chosen.id}"]`).click();
    await page.getByRole('button', { name: '100% labels', exact: true }).click();
    await page.locator('#fn-overlay').uncheck();
    await page.waitForFunction(width => [...document.querySelectorAll('canvas[data-mask-kind]')].every(c => c.width === width), chosen.geometry.width);
    assert(await page.locator('canvas[data-mask-kind="truth"]').count() === 1, 'Ground truth canvas renders');
    const panels = 1 + bundle.protocol.methods.length * (after ? 2 : 1);
    assert(await page.locator('canvas[data-mask-kind]').count() === panels, 'Every method prediction canvas renders (both versions in Compare)');
    if (after) assert(await page.locator('canvas[data-mask-kind="unet@after"]').count() === 1, 'Compare explorer shows the improved U-Net mask');
    await page.locator('[data-class="1"]').uncheck();
    await page.locator('[data-class="1"]').check();
    await page.locator('#fn-overlay').check();
    await page.locator('#fn-opacity').fill('0.8');
    assert((await page.locator('#fn-opacity-value').innerText()) === '80%', 'Overlay opacity control works');
    const truthPanel = page.locator('.fn-comparison .fn-zoom').nth(1);
    await truthPanel.click();
    await dialog.locator('.fn-lightbox-media').waitFor();
    assert((await dialog.locator('#fn-lightbox-title').innerText()) === 'Ground truth', 'Explorer panel opens in the viewer');
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(title => document.querySelector('#fn-lightbox-title')?.textContent === title, after ? 'SLIC + RF · before v1' : 'SLIC + RF');
    if (after) {
        await page.keyboard.press('ArrowRight');
        await page.waitForFunction(() => document.querySelector('#fn-lightbox-title')?.textContent === 'SLIC + RF · after');
        checks.push('Viewer flips from the first-version to the improved mask');
    }
    await page.keyboard.press('+');
    assert((await dialog.locator('.fn-lightbox-zoom').innerText()) === '150%', 'Viewer zooms and keeps flipping between panels');
    await page.screenshot({ path: 'output/playwright/viewer-desktop.png' });
    await page.keyboard.press('Escape');
    assert(await truthPanel.evaluate(node => document.activeElement === node), 'Viewer closes with Escape and returns focus');
    await page.screenshot({ path: 'output/playwright/explorer-desktop.png', fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: 'output/playwright/explorer-mobile.png', fullPage: true });
    assert(errors.length === 0, `No browser console/page errors (${errors.length}): ${errors.join(' | ')}`);
    return { status: 'passed', export_status: bundle.status, run_count: bundle.runs.length, sample_count: bundle.samples.length, checks };
}
