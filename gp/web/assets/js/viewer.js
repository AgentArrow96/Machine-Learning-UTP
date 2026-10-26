'use strict';

const FN_VIEWER = {
    dialog: null,
    hires: {},
    renderers: {},
    items: [],
    index: 0,
    view: { zoom: 1, cx: .5, cy: .5 },
    content: null,
    trigger: null,
    token: 0,

    async loadHires() {
        try {
            const response = await fetch('data/hires/index.json', { cache: 'no-store' });
            if (response.ok) FN_VIEWER.hires = (await response.json()).images || {};
        } catch {
            FN_VIEWER.hires = {};
        }
    },

    hiresUrl(id) {
        return id && FN_VIEWER.hires[id] ? FN_VIEWER.hires[id].url : null;
    },

    build() {
        if (FN_VIEWER.dialog) return FN_VIEWER.dialog;
        const dialog = document.createElement('dialog');
        dialog.className = 'fn-lightbox';
        dialog.setAttribute('aria-labelledby', 'fn-lightbox-title');
        dialog.innerHTML = `<div class="fn-lightbox-bar"><div><h2 id="fn-lightbox-title"></h2><p class="fn-lightbox-caption"></p></div>`
            + `<div class="fn-lightbox-tools" data-image-tools><button type="button" class="fn-btn" data-viewer="out" aria-label="Zoom out">−</button>`
            + `<output class="fn-lightbox-zoom">100%</output><button type="button" class="fn-btn" data-viewer="in" aria-label="Zoom in">+</button>`
            + `<button type="button" class="fn-btn" data-viewer="fit">Fit</button></div>`
            + `<div class="fn-lightbox-tools" data-chart-tools hidden><button type="button" class="fn-btn" data-action="toggle-notes" aria-pressed="true">Notes on</button>`
            + `<button type="button" class="fn-btn" data-action="download-chart">Save SVG</button></div>`
            + `<button type="button" class="fn-btn fn-lightbox-close" data-viewer="close" aria-label="Close">✕ Close</button></div>`
            + `<div class="fn-lightbox-stage"><div class="fn-lightbox-content"></div>`
            + `<button type="button" class="fn-lightbox-nav prev" data-viewer="prev" aria-label="Previous panel">‹</button>`
            + `<button type="button" class="fn-lightbox-nav next" data-viewer="next" aria-label="Next panel">›</button></div>`
            + `<p class="fn-lightbox-hint"></p>`;
        document.body.append(dialog);
        FN_VIEWER.dialog = dialog;
        const stage = dialog.querySelector('.fn-lightbox-stage');
        stage.addEventListener('wheel', event => {
            if (!FN_VIEWER.content || dialog.dataset.mode !== 'image') return;
            event.preventDefault();
            FN_VIEWER.zoomAt(event.deltaY < 0 ? 1.25 : 0.8, event.clientX, event.clientY);
        }, { passive: false });
        let drag = null;
        stage.addEventListener('pointerdown', event => {
            if (dialog.dataset.mode !== 'image' || event.target.closest('button')) return;
            drag = { x: event.clientX, y: event.clientY, cx: FN_VIEWER.view.cx, cy: FN_VIEWER.view.cy };
            stage.setPointerCapture(event.pointerId);
            stage.classList.add('dragging');
        });
        stage.addEventListener('pointermove', event => {
            if (!drag) return;
            const { scale, width, height } = FN_VIEWER.geometry();
            FN_VIEWER.view.cx = drag.cx - (event.clientX - drag.x) / (scale * width);
            FN_VIEWER.view.cy = drag.cy - (event.clientY - drag.y) / (scale * height);
            FN_VIEWER.apply();
        });
        const stop = () => { drag = null; stage.classList.remove('dragging'); };
        stage.addEventListener('pointerup', stop);
        stage.addEventListener('pointercancel', stop);
        stage.addEventListener('dblclick', event => {
            if (dialog.dataset.mode !== 'image') return;
            FN_VIEWER.zoomAt(FN_VIEWER.view.zoom > 1.01 ? 1 / FN_VIEWER.view.zoom : 3 / FN_VIEWER.view.zoom, event.clientX, event.clientY);
        });
        dialog.addEventListener('click', event => {
            const action = event.target.closest('[data-viewer]')?.dataset.viewer;
            if (action === 'close') dialog.close();
            else if (action === 'in') FN_VIEWER.zoomAt(1.5);
            else if (action === 'out') FN_VIEWER.zoomAt(1 / 1.5);
            else if (action === 'fit') FN_VIEWER.reset();
            else if (action === 'prev' || action === 'next') FN_VIEWER.show(FN_VIEWER.index + (action === 'next' ? 1 : -1));
        });
        dialog.addEventListener('keydown', event => {
            if (dialog.dataset.mode !== 'image' || event.target.closest('input, select')) return;
            const keys = { ArrowLeft: () => FN_VIEWER.show(FN_VIEWER.index - 1), ArrowRight: () => FN_VIEWER.show(FN_VIEWER.index + 1),
                '+': () => FN_VIEWER.zoomAt(1.5), '=': () => FN_VIEWER.zoomAt(1.5), '-': () => FN_VIEWER.zoomAt(1 / 1.5), 0: () => FN_VIEWER.reset() };
            if (keys[event.key]) {
                event.preventDefault();
                keys[event.key]();
            }
        });
        dialog.addEventListener('close', () => {
            dialog.querySelector('.fn-lightbox-content').replaceChildren();
            FN_VIEWER.content = null;
            FN_CHARTS.hideTip();
            FN_VIEWER.trigger?.focus();
        });
        new ResizeObserver(() => FN_VIEWER.apply()).observe(stage);
        return dialog;
    },


    itemsFor(trigger) {
        const group = trigger.dataset.zoomGroup;
        return group ? [...document.querySelectorAll(`[data-zoom][data-zoom-group="${CSS.escape(group)}"]`)] : [trigger];
    },

    openImage(trigger) {
        const dialog = FN_VIEWER.build();
        dialog.dataset.mode = 'image';
        dialog.classList.remove('fn-notes-off');
        FN_VIEWER.trigger = trigger;
        FN_VIEWER.items = FN_VIEWER.itemsFor(trigger);
        FN_VIEWER.view = { zoom: 1, cx: .5, cy: .5 };
        dialog.querySelector('[data-image-tools]').hidden = false;
        dialog.querySelector('[data-chart-tools]').hidden = true;
        const many = FN_VIEWER.items.length > 1;
        dialog.querySelectorAll('.fn-lightbox-nav').forEach(button => { button.hidden = !many; });
        dialog.querySelector('.fn-lightbox-hint').textContent = `Scroll or use + / − to zoom · drag to pan · double-click to zoom in${many ? ' · ← / → flips panels and keeps the same view' : ''} · Esc closes`;
        if (!dialog.open) dialog.showModal();
        FN_VIEWER.show(FN_VIEWER.items.indexOf(trigger));
    },

    async show(index) {
        const items = FN_VIEWER.items;
        if (!items.length) return;
        FN_VIEWER.index = (index + items.length) % items.length;
        const item = items[FN_VIEWER.index];
        const dialog = FN_VIEWER.dialog;
        const token = ++FN_VIEWER.token;
        dialog.querySelector('#fn-lightbox-title').textContent = item.dataset.zoomTitle || item.getAttribute('aria-label') || 'Image';
        dialog.querySelector('.fn-lightbox-caption').textContent = [item.dataset.zoomCaption, items.length > 1 ? `${FN_VIEWER.index + 1} of ${items.length}` : ''].filter(Boolean).join(' · ');
        let element;
        try {
            element = await FN_VIEWER.render(item);
        } catch (error) {
            element = document.createElement('p');
            element.className = 'fn-error';
            element.textContent = error.message;
        }
        if (token !== FN_VIEWER.token) return;
        element.classList.add('fn-lightbox-media');
        element.draggable = false;
        if (item.dataset.zoomPixelated != null) element.classList.add('pixelated');
        dialog.querySelector('.fn-lightbox-content').replaceChildren(element);
        FN_VIEWER.content = element;
        FN_VIEWER.apply();
    },

    async render(item) {
        const renderer = item.dataset.zoomRender && FN_VIEWER.renderers[item.dataset.zoomRender];
        if (renderer) return renderer(item);
        const source = (item.dataset.zoomTarget && document.getElementById(item.dataset.zoomTarget)) || item.querySelector('img, canvas') || item;
        if (source.tagName === 'CANVAS') {
            const copy = document.createElement('canvas');
            copy.width = source.width;
            copy.height = source.height;
            copy.getContext('2d').drawImage(source, 0, 0);
            copy.setAttribute('aria-label', source.getAttribute('aria-label') || '');
            return copy;
        }
        const image = new Image();
        image.alt = source.alt || '';
        image.src = FN_VIEWER.hiresUrl(item.dataset.zoomHires) || item.dataset.zoomSrc || source.currentSrc || source.src;
        await image.decode();
        return image;
    },

    size(element) {
        return { width: element.naturalWidth || element.width || 1, height: element.naturalHeight || element.height || 1 };
    },

    geometry() {
        const stage = FN_VIEWER.dialog.querySelector('.fn-lightbox-stage');
        const { width, height } = FN_VIEWER.content ? FN_VIEWER.size(FN_VIEWER.content) : { width: 1, height: 1 };
        const fit = Math.min(stage.clientWidth / width, stage.clientHeight / height);
        return { stage, width, height, fit, scale: fit * FN_VIEWER.view.zoom };
    },

    apply() {
        const content = FN_VIEWER.content;
        if (!content || FN_VIEWER.dialog?.dataset.mode !== 'image') return;
        const { stage, width, height, scale } = FN_VIEWER.geometry();
        const view = FN_VIEWER.view;
        const halfW = stage.clientWidth / (2 * scale * width);
        const halfH = stage.clientHeight / (2 * scale * height);
        view.cx = halfW >= .5 ? .5 : Math.min(1 - halfW, Math.max(halfW, view.cx));
        view.cy = halfH >= .5 ? .5 : Math.min(1 - halfH, Math.max(halfH, view.cy));
        const x = stage.clientWidth / 2 - view.cx * width * scale;
        const y = stage.clientHeight / 2 - view.cy * height * scale;
        content.style.width = `${width}px`;
        content.style.height = `${height}px`;
        content.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
        FN_VIEWER.dialog.querySelector('.fn-lightbox-zoom').textContent = `${Math.round(view.zoom * 100)}%`;
    },

    zoomAt(factor, clientX, clientY) {
        if (!FN_VIEWER.content) return;
        const { stage, width, height, scale } = FN_VIEWER.geometry();
        const view = FN_VIEWER.view;
        const zoom = Math.min(16, Math.max(1, view.zoom * factor));
        const box = stage.getBoundingClientRect();
        const ox = (clientX ?? box.left + box.width / 2) - box.left - stage.clientWidth / 2;
        const oy = (clientY ?? box.top + box.height / 2) - box.top - stage.clientHeight / 2;
        const px = view.cx + ox / (scale * width);
        const py = view.cy + oy / (scale * height);
        const next = scale * zoom / view.zoom;
        view.zoom = zoom;
        view.cx = px - ox / (next * width);
        view.cy = py - oy / (next * height);
        FN_VIEWER.apply();
    },

    reset() {
        FN_VIEWER.view = { zoom: 1, cx: .5, cy: .5 };
        FN_VIEWER.apply();
    },


    openChart(title, html, trigger) {
        const dialog = FN_VIEWER.build();
        dialog.dataset.mode = 'chart';
        dialog.classList.remove('fn-notes-off');
        FN_VIEWER.trigger = trigger;
        FN_VIEWER.content = null;
        dialog.querySelector('#fn-lightbox-title').textContent = title;
        dialog.querySelector('.fn-lightbox-caption').textContent = 'Hover or focus any mark for its values · click a legend entry to hide or show a series';
        dialog.querySelector('[data-image-tools]').hidden = true;
        const tools = dialog.querySelector('[data-chart-tools]');
        tools.hidden = false;
        const notes = tools.querySelector('[data-action="toggle-notes"]');
        notes.setAttribute('aria-pressed', 'true');
        notes.textContent = 'Notes on';
        dialog.querySelectorAll('.fn-lightbox-nav').forEach(button => { button.hidden = true; });
        dialog.querySelector('.fn-lightbox-hint').textContent = 'Esc closes';
        const body = document.createElement('div');
        body.className = 'fn-lightbox-chart';
        body.innerHTML = html;
        dialog.querySelector('.fn-lightbox-content').replaceChildren(body);
        if (!dialog.open) dialog.showModal();
        dialog.querySelector('.fn-lightbox-close').focus();
    }
};

document.addEventListener('click', event => {
    const trigger = event.target.closest('[data-zoom]');
    if (trigger && !trigger.closest('dialog')) {
        event.preventDefault();
        FN_VIEWER.openImage(trigger);
    }
});
