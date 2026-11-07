async (page) => {
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    const checks = [];
    const assert = (condition, message) => { if (!condition) throw new Error(message); checks.push(message); };
    const text = locator => locator.evaluate(node => {
        const copy = node.cloneNode(true);
        copy.querySelectorAll('.fn-tex').forEach(t => t.replaceWith(t.dataset.plain));
        return copy.textContent.replace(/\s+/g, ' ');
    });
    const base = 'http://127.0.0.1:8765';
    const json = async name => (await page.request.get(`${base}/data/${name}.json`)).json();
    const openMenu = async () => {
        if (await page.locator('.fn-mobile-menu').isVisible() && !(await page.locator('#fn-sidebar.open').count())) await page.getByRole('button', { name: 'Open navigation' }).click();
    };
    const go = async route => {
        await openMenu();
        const link = page.locator(`#fn-nav a[href="#${route}"]`);
        const number = (await link.locator('span').innerText()).trim();
        await link.click();
        await page.waitForFunction(([r, n]) => location.hash === `#${r}` && (r === 'overview' ? !!document.querySelector('#fn-page .fn-hero') : document.querySelector('#fn-page .fn-eyebrow')?.innerText.startsWith(n)), [route, number], { timeout: 10000 });
    };
    const routes = () => page.locator('#fn-nav .fn-nav-btn').evaluateAll(links => links.map(a => a.hash.slice(1)));
    const setMode = async mode => {
        await openMenu();
        await page.locator(`[data-mode="${mode}"]`).click();
        await page.waitForFunction(m => document.querySelector(`[data-mode="${m}"]`)?.getAttribute('aria-pressed') === 'true', mode);
        await page.getByRole('heading', { level: 1 }).first().waitFor();
    };
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto(`${base}/app.html?qa=${Date.now()}`);
    await page.evaluate(() => { try { localStorage.clear(); } catch { } });
    await page.goto(`${base}/app.html?qa=${Date.now()}#overview`);
    await page.getByRole('heading', { level: 1 }).first().waitFor();
    const bundle = await json('results');
    const uniform = await json('uniform');
    const pct = v => `${(v * 100).toFixed(1)}%`;
    const dialog = page.locator('dialog.fn-lightbox');
    assert(await page.locator('[data-mode="details"]').getAttribute('aria-pressed') === 'true', 'The app opens in Full details mode');
    assert(await page.locator('[data-mode]').count() === 2, 'The sidebar offers exactly two modes');

    const OTHER_STUDIES = /improvement study|improved (?:model|U-Net|hybrid|Random Forest|RF)|combined model|“After”|\bBefore v1\b|undersampl|blacken|R-GAN|oversampl|class-imbalance study/i;
    const pages = {};
    for (const mode of ['details', 'presentation']) {
        for (const size of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
            await page.setViewportSize(size);
            await setMode(mode);
            pages[mode] = await routes();
            for (const route of pages[mode]) {
                await go(route);
                const label = `${mode} ${size.width}px #${route}`;
                await page.waitForTimeout(80);
                assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)), `${label}: no page overflow`);
                assert(await page.locator('#fn-page .katex-error').count() === 0, `${label}: no LaTeX errors`);
                if (size.width > 1000) {
                    const all = await text(page.locator('#fn-page'));
                    const leak = all.match(OTHER_STUDIES);
                    assert(!leak, `${label}: no other study's content${leak ? ` (found "${leak[0]}")` : ''}`);
                    const crf = all.match(/CRF|hybrid/i);
                    assert(!crf, `${label}: SLIC + RF and U-Net only${crf ? ` (found "${crf[0]}")` : ''}`);
                }
            }
        }
    }
    await page.setViewportSize({ width: 1440, height: 1000 });
    assert(pages.details.join() === 'overview,dataset,preprocessing,methodology,walkthrough,results,explorer,training,interpretation,reproducibility', 'Full details lists exactly the Before pages');
    assert(pages.presentation.join() === 'summary,problem,data,prepare,features,method,evaluate,sustain,limits,sources', 'Presentation lists the summary, the eight deliverables and the sources');

    const allRoutes = [...pages.details, ...pages.presentation];
    for (const mode of ['details', 'presentation']) {
        await setMode(mode);
        for (const route of pages[mode]) {
            await go(route);
            const label = `${mode} #${route}`;
            const bad = await page.evaluate(known => [...document.querySelectorAll('#fn-page a[href^="#"]')]
                .map(a => a.getAttribute('href').slice(1))
                .filter(h => h && !known.includes(h) && !document.getElementById(h)), allRoutes);
            assert(bad.length === 0, `${label}: every in-app link points to a page or an anchor${bad.length ? ` (broken: ${bad.join(', ')})` : ''}`);
            for (const opener of ['#fn-page .fn-zoom', '#fn-page [data-expand-chart]']) {
                const trigger = page.locator(opener).first();
                if (!(await trigger.count())) continue;
                await trigger.scrollIntoViewIfNeeded();
                await trigger.click();
                await page.locator('.fn-lightbox[open]').waitFor();
                await page.locator('.fn-lightbox [data-viewer="close"]').click();
                await page.locator('.fn-lightbox[open]').waitFor({ state: 'detached' }).catch(() => { });
                await page.waitForTimeout(80);
                const after = await page.evaluate(() => ({ hash: location.hash, mode: document.querySelector('button[data-mode="presentation"]').getAttribute('aria-pressed') === 'true' ? 'presentation' : 'details' }));
                assert(after.hash === `#${route}` && after.mode === mode, `${label}: closing the ${opener.includes('zoom') ? 'image' : 'chart'} viewer stays on the page and in ${mode} mode (now ${after.hash}, ${after.mode})`);
            }
        }
    }

    const presentation = async () => {
        await setMode('presentation');
        await go('summary');
        const u100 = uniform.summary.find(s => s.method === 'unet' && s.budget === 100);
        const r100 = uniform.summary.find(s => s.method === 'rf' && s.budget === 100);
        const summary = await text(page.locator('#fn-page'));
        assert(summary.includes(pct(u100.full.flooded_miou.mean)) && summary.includes(pct(r100.full.flooded_miou.mean)), 'Presentation summary states the headline scores from the Before data');
        assert(!/\b2200\d{4}\b/.test(summary), 'No student IDs on the public site');
        await go('evaluate');
        const table = await text(page.locator('#fn-page .fn-metrics-table'));
        for (const method of ['rf', 'unet']) {
            for (const budget of [25, 100]) {
                const s = uniform.summary.find(x => x.method === method && x.budget === budget);
                assert(table.includes(pct(s.full.flooded_miou.mean)), `Presentation metrics table: ${method} ${budget}% equals the Before data`);
            }
        }
        assert(await page.locator('#fn-page svg.fn-chart').count() >= 4, 'Evaluate page has its charts');
        assert((await page.locator('#fn-page .fn-table-wrap').count()) >= 2, 'Evaluate page has metrics and cost tables');
        await page.locator('.fn-pres-figure .fn-zoom').first().click();
        await dialog.locator('.fn-lightbox-media').waitFor();
        await page.keyboard.press('Escape');
        checks.push('Presentation figure opens in the viewer');
        await page.waitForFunction(() => [...document.querySelectorAll('#fn-page img')].every(i => i.complete && i.naturalWidth > 0));
        checks.push('Presentation images load');
        await page.locator('.fn-pres-link.next').click();
        await page.waitForFunction(() => document.querySelector('#fn-page h1')?.innerText.includes('Connect to sustainability'));
        checks.push('Next link moves to the following deliverable');
        await page.locator('.fn-pres-link.prev').click();
        await page.waitForFunction(() => location.hash === '#evaluate');
        checks.push('Previous link moves back');
        await page.screenshot({ path: 'output/playwright/presentation-desktop.png', fullPage: true });
        await page.reload();
        await page.getByRole('heading', { level: 1 }).first().waitFor();
        assert(await page.locator('[data-mode="presentation"]').getAttribute('aria-pressed') === 'true', 'The chosen mode survives a reload');
    };
    await presentation();
    await page.goto(`${base}/app.html?qa=${Date.now()}#overview`);
    await page.waitForFunction(() => document.querySelector('[data-mode="details"]')?.getAttribute('aria-pressed') === 'true');
    checks.push('A direct link to a Full details page switches to that mode');
    await page.goto(`${base}/app.html?qa=${Date.now()}#evaluate`);
    await page.waitForFunction(() => document.querySelector('[data-mode="presentation"]')?.getAttribute('aria-pressed') === 'true');
    checks.push('A direct link to a Presentation page switches to that mode');
    await setMode('details');
    for (const removed of ['changes', 'improve', 'undersample', 'imbalance']) {
        await page.goto(`${base}/app.html?qa=${Date.now()}#${removed}`);
        await page.waitForFunction(() => location.hash === '#overview');
    }
    checks.push('Links to removed pages fall back to the overview');
    for (const removed of ['after', 'undersample', 'improve', 'imbalance']) {
        assert((await page.request.get(`${base}/data/${removed}.json`)).status() === 404, `data/${removed}.json is gone`);
    }

    await page.goto(`${base}/app.html?qa=${Date.now()}#overview`);
    await page.getByRole('heading', { level: 1 }).first().waitFor();
    assert((await page.locator('.fn-table tbody tr').count()) === uniform.summary.length, 'Overview summary table lists every configuration');
    assert(await page.locator('#fn-page .katex').count() > 10, 'Numbers on the page are typeset as LaTeX');
    await page.screenshot({ path: 'output/playwright/overview-desktop.png', fullPage: true });
    await go('dataset');
    assert((await text(page.locator('#fn-page'))).includes(`${bundle.dataset.overlap.summary.test.overlapping_train} / ${bundle.dataset.overlap.summary.test.images}`), 'Dataset page shows verified test overlap counts');
    await go('methodology');
    assert(await page.locator('.fn-method-card').count() === bundle.protocol.methods.length, 'Methodology shows one card per approach');
    assert(await page.locator('.fn-equation .katex-display').count() >= 15, 'Methods page shows the equation reference in display LaTeX');
    assert(await page.locator('[data-toc] .fn-toc-link').count() >= 5, 'Long pages get an "On this page" index');
    const prep = await json('preprocessing');
    await go('preprocessing');
    await page.locator('#fn-prep .fn-walk-stage').waitFor();
    assert((await text(page.locator('#fn-prep'))).includes(`${prep.steps.pairing.pairs.toLocaleString('en-US')} photo`), 'Preprocessing page states the audited pair count');
    await page.locator('[data-prep-step="3"]').click();
    assert((await text(page.locator('#fn-prep .fn-walk-text'))).includes(`${prep.steps.orientation.rotated} of`), 'Orientation step shows the EXIF-rotated count');
    await page.locator('#fn-prep .fn-zoom').first().click();
    await dialog.locator('.fn-lightbox-media').waitFor();
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('Escape');
    checks.push('Preprocessing figures open in the viewer and flip');
    const walk = await json('walkthrough');
    await go('walkthrough');
    await page.locator('.fn-walk-stage').waitFor();
    assert(await page.locator('[data-walk-example]').count() === walk.examples.length, 'Walkthrough offers every exported example');
    assert(walk.rf_variant === 'undersample/uniform', 'Walkthrough replays the retrained forests');
    assert((await page.locator('[data-walk-tab]').evaluateAll(t => t.map(x => x.dataset.walkTab))).join() === 'prep,rf,unet', 'Walkthrough has only the preparation, forest and U-Net tabs');
    assert(walk.examples.every(e => Object.keys(e.budgets).every(b => Object.keys(e.budgets[b]).sort().join() === 'rf,unet')) && !('crf' in walk), 'Walkthrough export holds only the forest and the U-Net');
    await page.locator('[data-walk-move="1"]').click();
    assert((await text(page.locator('.fn-walk-head'))).toLowerCase().includes('step 2 of 4'), 'Walkthrough Next step works');
    const walkChecks = await page.locator('.fn-walk-checks li').allInnerTexts();
    assert(walkChecks.length === 2 && walkChecks.every(t => t.includes('✓')), 'Walkthrough shows a frozen-prediction match for both methods');
    await go('results');
    await page.screenshot({ path: 'output/playwright/results-desktop.png', fullPage: true });
    assert(await page.locator('.fn-metrics-table').first().locator('tbody tr').count() === uniform.summary.length + 1, 'Results lists every metric for every configuration, plus the all-"Other" reference');
    assert(await page.locator('.fn-guide-table tbody tr').count() >= 9, 'Results explains every metric with its formula');
    const first = page.getByRole('heading', { name: 'First version of the Random Forest (kept for reference)' }).locator('..');
    assert(await first.locator('tbody tr').count() === 2, 'The first-version forest is kept next to the current one');
    assert(await page.locator('svg.fn-chart .fn-note').count() > 0, 'Budget chart is annotated by default');
    await page.locator('svg.fn-chart .fn-mark').first().hover();
    await page.locator('.fn-tooltip').waitFor({ state: 'visible' });
    assert((await page.locator('.fn-tooltip').innerText()).includes('mean'), 'Chart tooltip appears on hover');
    await page.getByRole('button', { name: 'Expand', exact: true }).first().click();
    await dialog.locator('svg.fn-chart').waitFor();
    assert(await dialog.locator('svg.fn-chart.fn-large').count() === 1, 'Expand opens a larger chart');
    await page.keyboard.press('Escape');
    const best = uniform.summary.find(s => s.method === 'unet' && s.budget === 100);
    const strata = await text(page.getByRole('heading', { name: 'Does scene overlap inflate the scores?' }).locator('..'));
    assert(strata.includes(pct(best.overlap_low.flooded_miou.mean)) && strata.includes(pct(best.overlap_high.flooded_miou.mean)), 'Overlap-strata card shows both halves');
    assert(await page.getByRole('heading', { name: 'Does CRF refinement help?' }).count() === 0, 'Results has no CRF card');
    const runIds = await page.locator('#fn-metric-run option').evaluateAll(o => o.map(x => x.value));
    assert(runIds.length === uniform.runs.length, 'Run inspector lists every run');
    await page.locator('#fn-metric-run').selectOption(runIds[0]);
    await page.locator('#fn-class-metrics').waitFor();
    assert((await text(page.locator('#fn-class-metrics'))).includes('Flooded road'), 'Selected-run per-class table renders');
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Save SVG', exact: true }).first().click();
    const download = await downloadPromise;
    await download.saveAs('output/playwright/budget-chart.svg');
    assert(download.suggestedFilename().endsWith('.svg'), 'Chart SVG download succeeds');
    await go('training');
    await page.locator(`[data-loss-seed="${bundle.protocol.train_seeds.at(-1)}"]`).click();
    assert(await page.locator(`[data-loss-seed="${bundle.protocol.train_seeds.at(-1)}"]`).getAttribute('aria-pressed') === 'true', 'Loss-curve seed switch works');
    await go('explorer');
    await page.getByRole('button', { name: '25% labels', exact: true }).click();
    assert(await page.getByRole('button', { name: '25% labels', exact: true }).getAttribute('aria-pressed') === 'true', '25% budget switch works');
    await page.locator('#fn-search').fill('no-such-image');
    await page.getByRole('heading', { name: 'No matching images', exact: true }).waitFor();
    await page.locator('#fn-search').fill('');
    checks.push('Empty filter has a clear state');
    const chosen = bundle.samples.find(s => s.id === bundle.representatives[0]);
    await page.locator('#fn-search').fill(chosen.image_id);
    await page.locator(`[data-sample="${chosen.id}"]`).click();
    await page.getByRole('button', { name: '100% labels', exact: true }).click();
    await page.locator('#fn-overlay').uncheck();
    await page.waitForFunction(width => [...document.querySelectorAll('canvas[data-mask-kind]')].every(c => c.width === width), chosen.geometry.width);
    assert(bundle.protocol.methods.join() === 'rf,unet' && uniform.runs.every(r => r.method !== 'hybrid'), 'The exports hold only the forest and the U-Net');
    assert(await page.locator('canvas[data-mask-kind]').count() === 1 + bundle.protocol.methods.length, 'Every method prediction canvas renders');
    await page.locator('#fn-overlay').check();
    const truthPanel = page.locator('.fn-comparison .fn-zoom').nth(1);
    await truthPanel.click();
    await dialog.locator('.fn-lightbox-media').waitFor();
    assert((await dialog.locator('#fn-lightbox-title').innerText()) === 'Ground truth', 'Explorer panel opens in the viewer');
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => document.querySelector('#fn-lightbox-title')?.textContent === 'SLIC + RF');
    await page.keyboard.press('Escape');
    checks.push('Viewer flips between explorer panels and closes');
    const story = async () => {
        await page.goto(`${base}/?qa=${Date.now()}`);
        await page.locator('#lp-title').waitFor();
        await page.evaluate(() => Promise.all([...document.images].map(i => { i.loading = 'eager'; return i.complete ? 0 : new Promise(r => { i.onload = i.onerror = r; }); })));
    };
    const summary = (m, b) => uniform.summary.find(s => s.method === m && s.budget === b);
    for (const size of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
        await page.setViewportSize(size);
        await story();
        const label = `Landing ${size.width}px`;
        assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)), `${label}: no page overflow`);
        assert(await page.evaluate(() => [...document.images].every(i => i.naturalWidth > 0)), `${label}: every image loads`);
    }
    await page.setViewportSize({ width: 1440, height: 1000 });
    const storyText = await text(page.locator('body'));
    assert(!/CRF|hybrid/i.test(storyText) && !/\b\d{8}\b/.test(storyText), 'Landing: SLIC + RF and U-Net only, no student IDs');
    const value = async key => (await page.locator(`[data-check="${key}"]`).first().innerText()).replace('%', '');
    for (const [m, b] of [['unet', 100], ['rf', 100], ['unet', 25], ['rf', 25]]) {
        assert(await value(`${m}-${b}`) === (summary(m, b).full.flooded_miou.mean * 100).toFixed(1), `Landing: ${m} ${b}% score matches uniform.json`);
    }
    assert(await value('road-gain') === ((summary('unet', 100).full.road_iou.mean - summary('unet', 25).full.road_iou.mean) * 100).toFixed(1), 'Landing: road gain matches uniform.json');
    assert(await value('unet-ms') === (summary('unet', 100).inference_median_seconds.mean * 1000).toFixed(1), 'Landing: U-Net speed matches uniform.json');
    assert(await value('overlap') === String(bundle.dataset.overlap.summary.test.overlapping_train), 'Landing: overlap count matches results.json');
    await page.locator('[data-rail-next="why-rail"]').click();
    await page.waitForFunction(() => document.getElementById('why-rail').scrollLeft > 0);
    checks.push('Landing: carousel next button scrolls the cards');
    await page.locator('#lp-theme').click();
    assert(await page.evaluate(() => document.documentElement.dataset.theme) === 'light', 'Landing: theme toggle switches to light');
    await story();
    assert(await page.evaluate(() => document.documentElement.dataset.theme) === 'light', 'Landing: chosen theme survives a reload');
    await page.locator('#lp-theme').click();
    await page.locator('.lp-cta a[href="app.html#summary"]').click();
    await page.waitForFunction(() => location.pathname.endsWith('app.html') && document.querySelector('[data-mode="presentation"]')?.getAttribute('aria-pressed') === 'true');
    checks.push('Landing: Presentation button opens the dashboard in Presentation mode');
    await story();
    await page.locator('.lp-cta a[href="app.html#overview"]').click();
    await page.waitForFunction(() => location.pathname.endsWith('app.html') && document.querySelector('[data-mode="details"]')?.getAttribute('aria-pressed') === 'true');
    checks.push('Landing: Full results button opens the dashboard in Full details mode');
    await page.goto(`${base}/?qa=${Date.now()}#results`);
    await page.waitForFunction(() => location.pathname.endsWith('app.html') && location.hash === '#results');
    checks.push('Landing: an old dashboard link (/#results) forwards to app.html#results');
    await page.locator('.fn-story-link').first().click();
    await page.locator('#lp-title').waitFor();
    checks.push('Dashboard links back to the project story');
    assert(errors.length === 0, `No browser console/page errors (${errors.length}): ${errors.join(' | ')}`);
    return { status: 'passed', export_status: bundle.status, run_count: bundle.runs.length, sample_count: bundle.samples.length, checks };
}
