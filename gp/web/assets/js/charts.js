const FN_CHARTS = {
    esc(value) {
        return String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    },

    pct(value, digits = 1) {
        return value == null ? 'N/A' : `${(value * 100).toFixed(digits)}%`;
    },

    tip(lines, focusable = true) {
        return ` data-tip="${FN_CHARTS.esc([].concat(lines).filter(Boolean).join('\n'))}"${focusable ? ' tabindex="0"' : ''}`;
    },

    legend(entries, extra = '') {
        const esc = FN_CHARTS.esc;
        const items = entries.map((e, i) => `<button type="button" class="fn-legend-item" data-series-toggle="${i}" aria-pressed="true">`
            + `<i class="${e.line ? 'fn-key-line' : ''}" style="background:${e.colour}${e.dash ? ';background:repeating-linear-gradient(90deg,' + e.colour + ' 0 5px,transparent 5px 8px)' : ''}"></i>${esc(e.name)}</button>`).join('');
        return `<div class="fn-legend">${items}${extra ? `<span class="fn-legend-note">${extra}</span>` : ''}</div>`;
    },

    box(svg, legend = '') {
        return `<div class="fn-chart-box">${svg}${legend}</div>`;
    },

    bars(items, options = {}) {
        const esc = FN_CHARTS.esc;
        const max = options.max || Math.max(...items.map(i => i.value), 1e-9);
        const width = options.large ? 760 : 520;
        const left = options.left || 150;
        const rowHeight = options.rowHeight || 40;
        const barHeight = Math.round(rowHeight * .6);
        const noteHeight = options.note ? 30 : 0;
        const height = items.length * rowHeight + 12 + noteHeight;
        const track = width - left - 76;
        const rows = items.map((item, i) => {
            const y = i * rowHeight + 8;
            const barWidth = Math.max(0, track * Math.min(item.value, max) / max);
            const text = item.text ?? FN_CHARTS.pct(item.value);
            const baseline = y + barHeight / 2 + 4;
            return `<g class="fn-mark${item.emphasis ? ' fn-emphasis' : ''}"${FN_CHARTS.tip(item.tip || [text, item.label])}>`
                + `<rect class="fn-hit" x="0" y="${y - (rowHeight - barHeight) / 2}" width="${width}" height="${rowHeight}" fill="transparent"/>`
                + `<text x="0" y="${baseline}">${esc(item.label)}</text>`
                + `<rect x="${left}" y="${y}" width="${track}" height="${barHeight}" rx="4" fill="#f1f5f9"/>`
                + `<rect class="fn-bar" x="${left}" y="${y}" width="${barWidth}" height="${barHeight}" rx="4" fill="${item.colour || '#0891b2'}"/>`
                + `<text class="fn-chart-value" x="${left + barWidth + 8}" y="${baseline}">${esc(text)}</text></g>`;
        }).join('');
        const note = options.note
            ? `<text class="fn-note" x="${left}" y="${items.length * rowHeight + 30}">${esc(options.note)}</text>` : '';
        return FN_CHARTS.box(`<svg class="fn-chart fn-chart-bars${options.large ? ' fn-large' : ''}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(options.label || 'Bar chart')}">${rows}${note}</svg>`);
    },

    rangeBars(groups, options = {}) {
        const esc = FN_CHARTS.esc;
        const width = options.large ? 900 : 650;
        const height = options.large ? 380 : 320;
        const left = 44;
        const right = 12;
        const top = 54;
        const bottom = 40;
        const maxY = options.max || 1;
        const plotH = height - top - bottom;
        const groupW = (width - left - right) / groups.length;
        const seriesCount = Math.max(...groups.map(g => g.series.length));
        const barW = Math.min(64, groupW / (seriesCount + 1));
        const y = v => top + plotH - plotH * v / maxY;
        const best = Math.max(...groups.flatMap(g => g.series.map(s => s.stat.mean ?? -1)));
        let svg = `<line class="fn-axis" x1="${left}" x2="${width - right}" y1="${y(0)}" y2="${y(0)}"/>`;
        for (const tick of [0, .25, .5, .75, 1]) {
            svg += `<text x="0" y="${y(maxY * tick) + 4}">${Math.round(maxY * tick * 100)}%</text>`;
        }
        groups.forEach((group, g) => {
            const centre = left + groupW * (g + .5);
            group.series.forEach((series, s) => {
                const x = centre + (s - (group.series.length - 1) / 2) * (barW + 10) - barW / 2;
                const stat = series.stat;
                if (stat.mean == null) return;
                const mid = x + barW / 2;
                const seeds = stat.values.map((v, i) => `${options.seeds ? `Seed ${options.seeds[i]}` : `Run ${i + 1}`}: ${FN_CHARTS.pct(v)}`);
                svg += `<g class="fn-mark" data-series="${s}"${FN_CHARTS.tip([`${FN_CHARTS.pct(stat.mean)} mean`, `${series.name} · ${group.label}`, `Range ${FN_CHARTS.pct(stat.min)}–${FN_CHARTS.pct(stat.max)}`, ...seeds])}>`;
                svg += `<rect class="fn-hit" x="${x - 5}" y="${top - 10}" width="${barW + 10}" height="${plotH + 10}" fill="transparent"/>`;
                svg += `<rect class="fn-bar" x="${x}" y="${y(stat.mean)}" width="${barW}" height="${y(0) - y(stat.mean)}" rx="4" fill="${series.colour}"/>`;
                svg += `<line x1="${mid}" x2="${mid}" y1="${y(stat.min)}" y2="${y(stat.max)}" stroke="#0f172a" stroke-width="1.6"/>`;
                svg += `<line x1="${mid - 7}" x2="${mid + 7}" y1="${y(stat.min)}" y2="${y(stat.min)}" stroke="#0f172a" stroke-width="1.6"/>`;
                svg += `<line x1="${mid - 7}" x2="${mid + 7}" y1="${y(stat.max)}" y2="${y(stat.max)}" stroke="#0f172a" stroke-width="1.6"/>`;
                stat.values.forEach((value, i) => {
                    const jitter = (i - (stat.values.length - 1) / 2) * 7;
                    svg += `<circle cx="${mid + jitter}" cy="${y(value)}" r="4" fill="white" stroke="#0f172a" stroke-width="1.2"/>`;
                });
                svg += `<text class="fn-chart-value" x="${mid}" y="${y(stat.max) - 8}" text-anchor="middle">${FN_CHARTS.pct(stat.mean)}</text>`;
                let lift = 22;
                if (series.note) {
                    svg += `<text class="fn-note" x="${mid}" y="${y(stat.max) - lift}" text-anchor="middle">${esc(series.note)}</text>`;
                    lift += 14;
                }
                if (stat.mean === best) {
                    svg += `<g class="fn-note"><rect x="${mid - 19}" y="${y(stat.max) - lift - 11}" width="38" height="15" rx="7" fill="#0f172a"/>`
                        + `<text class="fn-note-tag" x="${mid}" y="${y(stat.max) - lift}" text-anchor="middle">Best</text></g>`;
                }
                svg += `</g>`;
            });
            svg += `<text x="${centre}" y="${height - 14}" text-anchor="middle">${esc(group.label)}</text>`;
        });
        const legend = FN_CHARTS.legend(groups[0]?.series.map(s => ({ name: s.name, colour: s.colour })) || [], 'Bar = mean · whisker = min–max · dots = individual seeds');
        return FN_CHARTS.box(`<svg class="fn-chart${options.large ? ' fn-large' : ''}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(options.label || 'Grouped bar chart')}">${svg}</svg>`, legend);
    },

    lines(series, options = {}) {
        const esc = FN_CHARTS.esc;
        const width = options.large ? 900 : options.compact ? 470 : 650;
        const height = options.large ? 340 : options.compact ? 240 : 270;
        const left = 44;
        const right = options.large ? 150 : 128;
        const top = 22;
        const bottom = 34;
        const all = series.flatMap(s => s.values);
        const maxY = options.max || Math.max(...all, 1e-9) * 1.08;
        const maxX = Math.max(...series.map(s => s.values.length), 2) - 1;
        const px = i => left + i / maxX * (width - left - right);
        const py = v => height - bottom - v / maxY * (height - top - bottom);
        const format = v => (v == null ? 'N/A' : options.percent ? FN_CHARTS.pct(v) : options.integer ? Math.round(v).toLocaleString('en-US') : v.toFixed(3));
        let svg = `<line class="fn-axis" x1="${left}" x2="${width - right}" y1="${py(0)}" y2="${py(0)}"/>`;
        for (const tick of [0, .25, .5, .75, 1]) {
            const value = maxY * tick;
            svg += `<text x="0" y="${py(value) + 4}">${options.percent ? `${Math.round(value * 100)}%` : options.integer ? Math.round(value).toLocaleString('en-US') : value.toFixed(2)}</text>`;
        }
        if (options.vline) {
            const x = px(options.vline.index);
            svg += `<g class="fn-note"><line class="fn-vline" x1="${x}" x2="${x}" y1="${top - 6}" y2="${py(0)}"/>`
                + `<text class="fn-note" x="${x + 5}" y="${top + 4}">${esc(options.vline.label)}</text></g>`;
        }
        series.forEach((s, k) => {
            const points = s.values.map((v, i) => `${px(i)},${py(v)}`).join(' ');
            svg += `<g data-series="${k}"><polyline fill="none" stroke="${s.colour}" stroke-width="2" ${s.dash ? 'stroke-dasharray="6 4"' : ''} points="${points}"/>`;
            if (s.marker != null && s.values[s.marker] != null) {
                svg += `<circle cx="${px(s.marker)}" cy="${py(s.values[s.marker])}" r="5" fill="white" stroke="${s.colour}" stroke-width="2.4"/>`;
            }
            svg += `</g>`;
        });
        const labels = [];
        series.forEach((s, k) => {
            if (s.marker != null && s.markerLabel && s.values[s.marker] != null) labels.push({ k, index: s.marker, value: s.values[s.marker], text: s.markerLabel, colour: s.colour });
        });
        for (const p of options.points || []) {
            const s = series[p.series];
            if (s?.values[p.index] != null) labels.push({ k: p.series, index: p.index, value: s.values[p.index], text: p.text, colour: s.colour, point: true });
        }
        labels.forEach(l => { l.x = px(l.index); l.y = py(l.value); l.ty = l.y; });
        labels.sort((a, b) => a.ty - b.ty);
        for (let i = 1; i < labels.length; i++) labels[i].ty = Math.max(labels[i].ty, labels[i - 1].ty + 14);
        const overflow = labels.length ? labels[labels.length - 1].ty - (py(0) - 4) : 0;
        if (overflow > 0) labels.forEach(l => { l.ty -= overflow; });
        for (const l of labels) {
            const tx = width - right + 8;
            svg += `<g class="fn-note" data-series="${l.k}">`
                + (l.point ? `<circle cx="${l.x}" cy="${l.y}" r="4" fill="${l.colour}"/>` : '')
                + `<polyline class="fn-leader" fill="none" points="${l.x + 6},${l.y} ${tx - 4},${l.ty - 4}"/>`
                + `<text class="fn-note" x="${tx}" y="${l.ty}">${esc(l.text)}</text></g>`;
        }
        const xName = options.xName || 'Epoch';
        const first = options.xStart ?? 1;
        svg += `<text x="${left}" y="${height - 8}">${esc(xName)} ${first}</text><text x="${width - right}" y="${height - 8}" text-anchor="end">${esc(xName)} ${maxX + first}</text>`;
        svg += `<line class="fn-crosshair" x1="0" x2="0" y1="${top - 6}" y2="${py(0)}"/>`;
        const step = (width - left - right) / maxX;
        for (let i = 0; i <= maxX; i++) {
            const rows = series.map(s => `${s.label}: ${format(s.values[i])}`);
            svg += `<rect class="fn-hit fn-column" data-x="${px(i)}" x="${px(i) - step / 2}" y="${top - 6}" width="${step}" height="${py(0) - top + 6}" fill="transparent"${FN_CHARTS.tip([`${xName} ${i + first}`, ...rows], false)}/>`;
        }
        const legend = FN_CHARTS.legend(series.map(s => ({ name: s.label, colour: s.colour, line: true, dash: s.dash })), options.note ? FN_CHARTS.esc(options.note) : '');
        return FN_CHARTS.box(`<svg class="fn-chart fn-lines${options.large ? ' fn-large' : ''}" viewBox="0 0 ${width} ${height}" role="img" tabindex="0" aria-label="${esc(options.label || 'Line chart')}. Use the arrow keys to read each epoch.">${svg}</svg>`, legend);
    },

    forest(rows, options = {}) {
        const esc = FN_CHARTS.esc;
        const width = options.large ? 900 : 650;
        const left = options.left || 170;
        const right = options.large ? 170 : 150;
        const rowHeight = 30;
        const top = 22;
        const height = rows.length * rowHeight + top + 26;
        const extent = Math.max(.02, ...rows.flatMap(r => [Math.abs(r.low ?? 0), Math.abs(r.high ?? 0), Math.abs(r.value ?? 0)])) * 1.1;
        const x = v => left + (v + extent) / (2 * extent) * (width - left - right);
        const pp = v => `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v * 100).toFixed(1)} pp`;
        let svg = `<line class="fn-axis" x1="${x(0)}" x2="${x(0)}" y1="${top - 8}" y2="${height - 22}"/>`
            + `<text x="${x(0)}" y="${top - 10}" text-anchor="middle">no change</text>`
            + `<text x="${x(-extent)}" y="${height - 6}">${pp(-extent)}</text><text x="${x(extent)}" y="${height - 6}" text-anchor="end">${pp(extent)}</text>`;
        rows.forEach((r, i) => {
            const y = top + i * rowHeight + rowHeight / 2;
            svg += `<g class="fn-mark"${FN_CHARTS.tip(r.tip || [pp(r.value), r.label, `95% CI ${pp(r.low)} to ${pp(r.high)}`])}>`
                + `<rect class="fn-hit" x="0" y="${y - rowHeight / 2}" width="${width}" height="${rowHeight}" fill="transparent"/>`
                + `<text x="0" y="${y + 4}">${esc(r.label)}</text>`
                + `<line x1="${x(r.low)}" x2="${x(r.high)}" y1="${y}" y2="${y}" stroke="${r.colour}" stroke-width="2.5"/>`
                + `<circle class="fn-bar" cx="${x(r.value)}" cy="${y}" r="5.5" fill="${r.colour}" stroke="white" stroke-width="2"/>`
                + `<text class="fn-chart-value" x="${width - right + 10}" y="${y + 4}">${esc(pp(r.value))}</text>`
                + (r.note ? `<text class="fn-note" x="${width - right + 70}" y="${y + 4}">${esc(r.note)}</text>` : '')
                + `</g>`;
        });
        return FN_CHARTS.box(`<svg class="fn-chart${options.large ? ' fn-large' : ''}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(options.label || 'Differences with confidence intervals')}">${svg}</svg>`,
            options.legend ? `<div class="fn-legend"><span class="fn-legend-note">${esc(options.legend)}</span></div>` : '');
    },

    confusion(matrix, names, options = {}) {
        const esc = FN_CHARTS.esc;
        const lower = name => name.toLowerCase();
        let html = `<div class="fn-matrix${options.large ? ' fn-large' : ''}"><div class="fn-matrix-head">Truth ↓ / Pred →</div>`;
        html += names.map(n => `<div class="fn-matrix-head">${esc(n)}</div>`).join('');
        matrix.forEach((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1;
            const errors = row.map((v, j) => (j === i ? -1 : v));
            const worst = i > 0 && Math.max(...errors) > 0 ? errors.indexOf(Math.max(...errors)) : -1;
            html += `<div class="fn-matrix-head">${esc(names[i])}</div>`;
            html += row.map((v, j) => {
                const share = v / total;
                const colour = share > .5 ? '#fff' : '#0f172a';
                const sentence = i === j ? `${FN_CHARTS.pct(share)} of true ${lower(names[i])} pixels were predicted correctly`
                    : `${FN_CHARTS.pct(share)} of true ${lower(names[i])} pixels were predicted ${names[j]}`;
                const tag = j === worst ? `<em class="fn-note fn-cell-tag">largest error</em>` : '';
                return `<div class="fn-cell${i === j ? ' fn-diag' : ''}" style="background:rgba(8,145,178,${(.06 + .86 * share).toFixed(3)});color:${colour}"${FN_CHARTS.tip([`${(share * 100).toFixed(1)}%`, sentence, `${v.toLocaleString('en-US')} of ${total.toLocaleString('en-US')} pixels`])}>${(share * 100).toFixed(1)}%<small>${v.toLocaleString('en-US')} px</small>${tag}</div>`;
            }).join('');
        });
        return `${html}</div>`;
    },


    tooltip: null,

    showTip(target, clientX, clientY) {
        let tip = FN_CHARTS.tooltip;
        if (!tip) {
            tip = FN_CHARTS.tooltip = document.createElement('div');
            tip.className = 'fn-tooltip';
            tip.setAttribute('role', 'status');
        }
        const host = target.closest('dialog[open]') || document.body;
        if (tip.parentNode !== host) host.append(tip);
        tip.replaceChildren();
        target.dataset.tip.split('\n').forEach((line, i) => {
            const row = document.createElement(i === 0 ? 'strong' : 'span');
            row.textContent = line;
            tip.append(row);
        });
        if (typeof FN_MATH !== 'undefined') FN_MATH.typeset(tip);
        tip.hidden = false;
        const box = target.getBoundingClientRect();
        const x = clientX ?? box.left + box.width / 2;
        const y = clientY ?? box.top;
        const width = tip.offsetWidth;
        const height = tip.offsetHeight;
        tip.style.left = `${Math.min(Math.max(8, x + 14), innerWidth - width - 8)}px`;
        tip.style.top = `${y - height - 12 < 8 ? y + 18 : y - height - 12}px`;
        const column = target.closest('.fn-column');
        if (column) {
            const line = column.ownerSVGElement.querySelector('.fn-crosshair');
            line.setAttribute('x1', column.dataset.x);
            line.setAttribute('x2', column.dataset.x);
            line.classList.add('on');
        }
    },

    hideTip() {
        if (FN_CHARTS.tooltip) FN_CHARTS.tooltip.hidden = true;
        document.querySelectorAll('.fn-crosshair.on').forEach(line => line.classList.remove('on'));
    },

    bind() {
        document.addEventListener('pointermove', event => {
            const target = event.target.closest?.('[data-tip]');
            if (target) FN_CHARTS.showTip(target, event.clientX, event.clientY);
            else FN_CHARTS.hideTip();
        });
        document.addEventListener('pointerleave', () => FN_CHARTS.hideTip());
        document.addEventListener('focusin', event => {
            const target = event.target.closest?.('[data-tip]');
            if (target) FN_CHARTS.showTip(target);
            else FN_CHARTS.hideTip();
        });
        document.addEventListener('keydown', event => {
            const svg = event.target.closest?.('svg.fn-lines');
            if (!svg || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
            event.preventDefault();
            const columns = [...svg.querySelectorAll('.fn-column')];
            const now = Number(svg.dataset.epoch ?? -1);
            const next = event.key === 'Home' ? 0 : event.key === 'End' ? columns.length - 1
                : Math.min(columns.length - 1, Math.max(0, now + (event.key === 'ArrowRight' ? 1 : -1)));
            svg.dataset.epoch = next;
            const box = columns[next].getBoundingClientRect();
            FN_CHARTS.showTip(columns[next], box.left + box.width / 2, box.top + 20);
        });
        document.addEventListener('focusout', event => {
            if (event.target.matches?.('svg.fn-lines')) delete event.target.dataset.epoch;
        });
        document.addEventListener('click', event => {
            const toggle = event.target.closest('[data-series-toggle]');
            if (toggle) {
                const on = toggle.getAttribute('aria-pressed') !== 'true';
                toggle.setAttribute('aria-pressed', String(on));
                toggle.closest('.fn-chart-box').querySelectorAll(`[data-series="${toggle.dataset.seriesToggle}"]`)
                    .forEach(mark => mark.classList.toggle('fn-off', !on));
                return;
            }
            const notes = event.target.closest('[data-action="toggle-notes"]');
            if (notes) {
                const card = notes.closest('[data-chart-block], .fn-lightbox');
                const off = card.classList.toggle('fn-notes-off');
                notes.setAttribute('aria-pressed', String(!off));
                notes.textContent = off ? 'Notes off' : 'Notes on';
            }
        });
    }
};

FN_CHARTS.bind();
