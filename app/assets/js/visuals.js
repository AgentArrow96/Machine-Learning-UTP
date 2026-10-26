// Visual helpers
const ML_VIS = {};
const V = {
    FONT: "Kalam, 'Segoe Print', 'Comic Sans MS', cursive",
    C: {
        sky: '#0284c7', skyS: '#e0f2fe', lime: '#65a30d', limeS: '#ecfccb', indigo: '#4f46e5', indigoS: '#e0e7ff',
        rose: '#e11d48', roseS: '#ffe4e6', amber: '#d97706', amberS: '#fef3c7', green: '#059669', greenS: '#d1fae5',
        violet: '#7c3aed', violetS: '#ede9fe', slate: '#475569', slateS: '#f1f5f9', ink: '#1e293b', red: '#dc2626'
    },
    rng(seed) {
        let a = seed >>> 0;
        return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    },
    gauss(r) { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); },
    h(tag, attrs, ...kids) {
        const e = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs || {})) {
            if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
            else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
            else if (k === 'html') e.innerHTML = v;
            else e.setAttribute(k, v);
        }
        for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(String(c)));
        return e;
    },
    s(tag, attrs, ...kids) {
        const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (const [k, v] of Object.entries(attrs || {})) {
            if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
        }
        for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(String(c)));
        return e;
    },
    svg(w, h, cls) { return V.s('svg', { viewBox: `0 0 ${w} ${h}`, class: 'vis-svg ' + (cls || '') }); },
    // Hand-drawn line
    rough(x1, y1, x2, y2, r, amt = 1) {
        r = r || Math.random;
        const L = Math.hypot(x2 - x1, y2 - y1) || 1, off = Math.min(amt * 1.6, Math.max(0.5, L * 0.012 * amt));
        const j = () => (r() * 2 - 1) * off, dx = x2 - x1, dy = y2 - y1, px = -dy / L, py = dx / L;
        let d = '';
        for (let k = 0; k < 2; k++) {
            const bow = j() * 1.2;
            d += `M${x1 + j()} ${y1 + j()} C${x1 + dx * .35 + px * bow + j()} ${y1 + dy * .35 + py * bow + j()} ${x1 + dx * .7 + px * bow + j()} ${y1 + dy * .7 + py * bow + j()} ${x2 + j()} ${y2 + j()} `;
        }
        return d;
    },
    rline(g, x1, y1, x2, y2, color, width = 1.8, r, extra = {}) {
        const p = V.s('path', Object.assign({ d: extra['stroke-dasharray'] ? `M${x1} ${y1} L${x2} ${y2}` : V.rough(x1, y1, x2, y2, r), fill: 'none', stroke: color || V.C.ink, 'stroke-width': width, 'stroke-linecap': 'round' }, extra));
        g.append(p); return p;
    },
    rrect(g, x, y, w, h, color, fill, r, width = 1.6) {
        if (fill) g.append(V.s('rect', { x: x + 1, y: y + 1, width: Math.max(0, w - 2), height: Math.max(0, h - 2), rx: 4, fill }));
        V.rline(g, x, y, x + w, y, color, width, r); V.rline(g, x + w, y, x + w, y + h, color, width, r);
        V.rline(g, x + w, y + h, x, y + h, color, width, r); V.rline(g, x, y + h, x, y, color, width, r);
    },
    text(g, x, y, str, o = {}) {
        const t = V.s('text', { x, y, 'font-family': V.FONT, 'font-size': o.size || 15, fill: o.color || V.C.ink, 'text-anchor': o.anchor || 'middle', 'dominant-baseline': 'middle', 'font-weight': o.weight || 400 });
        t.textContent = str; g.append(t); return t;
    },
    arrow(g, x1, y1, x2, y2, color, width = 1.8, r) {
        V.rline(g, x1, y1, x2, y2, color, width, r);
        const a = Math.atan2(y2 - y1, x2 - x1);
        for (const s of [-1, 1]) { const b = a + Math.PI - s * 0.45; V.rline(g, x2, y2, x2 + 10 * Math.cos(b), y2 + 10 * Math.sin(b), color, width, r, {}); }
    },
    slider(label, min, max, step, value, onInput, fmt) {
        const out = V.h('output', {}, fmt ? fmt(value) : value);
        const inp = V.h('input', { type: 'range', min, max, step, value });
        inp.addEventListener('input', () => { const v = Number(inp.value); out.textContent = fmt ? fmt(v) : v; onInput(v); });
        const w = V.h('label', { class: 'vis-slider' }, V.h('span', {}, label), inp, out);
        w.input = inp; return w;
    },
    seg(options, value, onChange) {
        const w = V.h('div', { class: 'vis-seg', role: 'group' });
        const btns = options.map(([v, label]) => {
            const b = V.h('button', { type: 'button', class: v === value ? 'on' : '' }, label);
            b.addEventListener('click', () => { btns.forEach(x => x.classList.remove('on')); b.classList.add('on'); onChange(v); });
            return b;
        });
        w.append(...btns); return w;
    },
    btn(label, onClick, cls) { return V.h('button', { type: 'button', class: 'vis-btn ' + (cls || ''), onclick: onClick }, label); },
    fmt(n, d = 2) { return Number.isFinite(n) ? Number(n.toFixed(d)).toLocaleString('en-US', { maximumFractionDigits: d }) : '-'; },
    clear(e) { while (e.firstChild) e.removeChild(e.firstChild); },
    median(a) { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; },
    mean(a) { return a.reduce((x, y) => x + y, 0) / a.length; },
    std(a) { const m = V.mean(a); return Math.sqrt(V.mean(a.map(x => (x - m) ** 2))); },
    quantile(a, q) { const s = [...a].sort((x, y) => x - y), p = (s.length - 1) * q, i = Math.floor(p); return s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * (p - i); },
    svgPoint(svg, e) { const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; return pt.matrixTransform(svg.getScreenCTM().inverse()); },
    reduced() { return !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches); },
    // Eased animation: fn(k) for k in 0..1; returns a cancel function
    tween(ms, fn, done) {
        if (V.reduced()) { fn(1); if (done) done(); return () => { }; }
        let start = null, stop = false;
        const step = t => {
            if (stop) return;
            if (start === null) start = t;
            const k = Math.min(1, (t - start) / ms);
            fn(k < .5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2);
            if (k < 1) requestAnimationFrame(step); else if (done) done();
        };
        requestAnimationFrame(step);
        return () => { stop = true; };
    },
    // Step rail with Prev / Play / Next (same pattern as the gp results app walkthrough)
    stepper(host, steps, opt = {}) {
        let i = 0, timer = null;
        const rail = V.h('div', { class: 'vis-steps' }), title = V.h('b', { class: 'vis-step-title' }), text = V.h('p', { class: 'vis-readout' });
        const count = V.h('span', { class: 'vis-note' });
        const stop = () => { clearInterval(timer); timer = null; play.textContent = 'Play'; };
        const go = k => {
            i = Math.max(0, Math.min(steps.length - 1, k));
            [...rail.children].forEach((b, j) => { b.classList.toggle('on', j === i); b.classList.toggle('done', j < i); });
            title.textContent = steps[i].title; text.innerHTML = steps[i].text || '';
            count.textContent = 'Step ' + (i + 1) + ' of ' + steps.length;
            if (steps[i].draw) steps[i].draw(i);
            renderMath(text);
        };
        const play = V.btn('Play', () => {
            if (timer) { stop(); return; }
            if (i >= steps.length - 1) go(0);
            play.textContent = 'Pause';
            timer = setInterval(() => { if (i >= steps.length - 1) stop(); else go(i + 1); }, opt.ms || 1700);
        });
        steps.forEach((s, k) => rail.append(V.h('button', { type: 'button', class: 'vis-step', title: s.title, onclick: () => { stop(); go(k); } }, String(k + 1))));
        host.append(rail, V.h('div', { class: 'vis-row' }, V.btn('← Prev', () => { stop(); go(i - 1); }), play, V.btn('Next →', () => { stop(); go(i + 1); }), count), title, text);
        go(opt.start || 0);
        return { go, get index() { return i; } };
    },
    // Horizontal bars with hover tooltips (same idea as the gp results app bar chart)
    hbars(svg, items, o = {}) {
        const x0 = o.x0 ?? 150, y0 = o.y0 ?? 10, w = o.w ?? 320, rh = o.rowH ?? 22, max = o.max ?? Math.max(...items.map(d => d.value), 1e-9);
        items.forEach((d, k) => {
            const y = y0 + k * rh, len = Math.max(2, w * Math.max(0, d.value) / max);
            const g = V.s('g', { opacity: d.dim ? 0.35 : 1 });
            g.append(V.s('title', {}, d.tip || (d.label + ': ' + d.value)));
            g.append(V.s('rect', { x: x0, y: y + 3, width: w, height: rh - 8, rx: 4, fill: '#f1f5f9' }));
            g.append(V.s('rect', { x: x0, y: y + 3, width: len, height: rh - 8, rx: 4, fill: d.color || V.C.sky }));
            V.text(g, x0 - 8, y + rh / 2 + 4, d.label, { size: o.size || 12, anchor: 'end', color: d.dim ? '#94a3b8' : '#334155' });
            if (d.text !== '') V.text(g, x0 + len + 6, y + rh / 2 + 4, d.text ?? V.fmt(d.value, 3), { size: (o.size || 12) - 1, anchor: 'start', color: '#64748b' });
            svg.append(g);
        });
    },

    // 3D helpers
    _three: null,
    three() {
        if (window.THREE) return Promise.resolve(window.THREE);
        if (!V._three) V._three = new Promise((res, rej) => {
            const sc = document.createElement('script');
            sc.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
            sc.onload = () => res(window.THREE); sc.onerror = rej; document.head.append(sc);
        });
        return V._three;
    },
    async scene3d(host, opt = {}) {
        let THREE;
        try { THREE = await V.three(); } catch (e) { host.append(V.h('p', { class: 'vis-note' }, '3D view needs an internet connection to load three.js.')); return null; }
        const canvas = V.h('canvas', { class: 'vis-canvas3d', 'aria-label': opt.label || '3D view' });
        host.append(canvas);
        let renderer;
        try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true }); } catch (e) { canvas.remove(); host.append(V.h('p', { class: 'vis-note' }, 'WebGL is not available in this browser.')); return null; }
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(40, 1.6, 0.1, 100);
        scene.add(new THREE.AmbientLight(0xffffff, 0.75));
        const dl = new THREE.DirectionalLight(0xffffff, 0.55); dl.position.set(3, 5, 4); scene.add(dl);
        const target = new THREE.Vector3(...(opt.target || [0, 0, 0]));
        let theta = opt.theta ?? 0.8, phi = opt.phi ?? 1.05, radius = opt.radius ?? 6, needs = true, visible = false, auto = opt.autoRotate ?? true;
        const frameFns = [];
        const place = () => { camera.position.set(target.x + radius * Math.sin(phi) * Math.cos(theta), target.y + radius * Math.cos(phi), target.z + radius * Math.sin(phi) * Math.sin(theta)); camera.lookAt(target); };
        const resize = () => { const w = host.clientWidth || 600, h = Math.round(w / (opt.aspect || 1.7)); renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); needs = true; };
        let drag = null;
        canvas.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; auto = false; canvas.setPointerCapture(e.pointerId); });
        canvas.addEventListener('pointermove', e => { if (!drag) return; theta += (e.clientX - drag[0]) * 0.01; phi = Math.min(2.9, Math.max(0.2, phi - (e.clientY - drag[1]) * 0.01)); drag = [e.clientX, e.clientY]; needs = true; });
        canvas.addEventListener('pointerup', () => { drag = null; });
        const zoom = d => { radius = Math.min(18, Math.max(2.5, radius * d)); needs = true; };
        const bar = V.h('div', { class: 'vis-row vis-3dbar' }, V.btn('+ zoom', () => zoom(0.85)), V.btn('- zoom', () => zoom(1.18)), V.btn('⟳ auto-rotate', () => { auto = !auto; needs = true; }), V.h('span', { class: 'vis-note' }, 'Drag to rotate'));
        host.append(bar);
        new IntersectionObserver(es => es.forEach(x => { visible = x.isIntersecting; if (visible) { resize(); loop(); } })).observe(canvas);
        window.addEventListener('resize', resize);
        let last = 0;
        function loop(t = 0) {
            if (!visible) return;
            const dt = Math.min(0.05, (t - last) / 1000 || 0); last = t;
            if (auto) { theta += dt * 0.25; needs = true; }
            frameFns.forEach(f => { if (f(dt)) needs = true; });
            if (needs) { place(); renderer.render(scene, camera); needs = false; }
            requestAnimationFrame(loop);
        }
        resize();
        return { THREE, scene, camera, renderer, dirty: () => { needs = true; }, onFrame: f => frameFns.push(f) };
    },
    label3d(THREE, text, color = '#1e293b', size = 0.35) {
        const c = document.createElement('canvas'), ctx = c.getContext('2d'), fs = 64;
        ctx.font = `700 ${fs}px Kalam, 'Segoe Print', cursive`;
        const w = Math.ceil(ctx.measureText(text).width) + 24; c.width = w; c.height = fs + 24;
        ctx.font = `700 ${fs}px Kalam, 'Segoe Print', cursive`; ctx.fillStyle = color; ctx.textBaseline = 'middle'; ctx.fillText(text, 12, c.height / 2 + 4);
        const tex = new THREE.CanvasTexture(c); tex.anisotropy = 4;
        const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
        sp.scale.set(size * w / c.height, size, 1); sp.renderOrder = 10; return sp;
    },
    line3d(THREE, pts, color, dashed) {
        const g = new THREE.BufferGeometry().setFromPoints(pts.map(p => new THREE.Vector3(...p)));
        const m = dashed ? new THREE.LineDashedMaterial({ color, dashSize: 0.12, gapSize: 0.08 }) : new THREE.LineBasicMaterial({ color });
        const l = new THREE.Line(g, m); if (dashed) l.computeLineDistances(); return l;
    },
    axes3d(THREE, scene, names, size = 2) {
        const col = 0x94a3b8;
        for (let i = 0; i < 3; i++) {
            const a = [0, 0, 0], b = [0, 0, 0]; a[i] = -size; b[i] = size;
            scene.add(V.line3d(THREE, [a, b], col));
            if (names[i]) { const s = V.label3d(THREE, names[i], '#475569', 0.28); s.position.set(b[0] * 1.15, b[1] * 1.15, b[2] * 1.15); scene.add(s); }
        }
    }
};

function initVisuals(root) {
    (root || document).querySelectorAll('.vis-widget:not([data-observed])').forEach(fig => {
        fig.setAttribute('data-observed', '1');
        const io = new IntersectionObserver(es => es.forEach(x => {
            if (!x.isIntersecting || fig.dataset.ready) return;
            io.disconnect();
            const fn = ML_VIS[fig.dataset.vis], host = fig.querySelector('.vis-root');
            if (!fn) { host.textContent = 'Missing visual: ' + fig.dataset.vis; return; }
            try { const r = fn(host); Promise.resolve(r).then(() => { fig.dataset.ready = '1'; renderMath(host); }); }
            catch (e) { console.error('visual ' + fig.dataset.vis, e); host.textContent = 'This visual failed to load.'; }
        }), { rootMargin: '200px' });
        io.observe(fig);
    });
}
// Generated by tools/precompute_l5.py from sklearn load_breast_cancer (569 samples, 30 features).
const L5_DATA = {"names":["mean radius","mean texture","mean perimeter","mean area","mean smoothness","mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension","radius error","texture error","perimeter error","area error","smoothness error","compactness error","concavity error","concave points error","symmetry error","fractal dimension error","worst radius","worst texture","worst perimeter","worst area","worst smoothness","worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"],"chi2":[266.1,93.9,2011.1,53991.66,0.15,5.4,19.71,10.54,0.26,0.0,34.68,0.01,250.57,8758.5,0.0,0.61,1.04,0.31,0.0,0.01,491.69,174.45,3665.04,112598.43,0.4,19.31,39.52,13.49,1.3,0.23],"rfeOrder":[25,8,16,4,11,17,9,29,14,18,19,2,1,6,0,28,5,12,3,24,15,26,13,7,22,21,10,20,27,23],"chi15":[0,1,2,3,6,10,12,13,20,21,22,23,25,26,27],"rfe15":[3,5,7,10,12,13,15,20,21,22,23,24,26,27,28],"overlap":10,"importance":[0.0348,0.0152,0.068,0.0605,0.008,0.0116,0.0669,0.107,0.0034,0.0026,0.0143,0.0037,0.0101,0.0296,0.0047,0.0056,0.0058,0.0038,0.0035,0.0059,0.0828,0.0175,0.0808,0.1394,0.0122,0.0199,0.0373,0.1322,0.0082,0.0045],"accByK":[89.1,90.34,92.26,93.85,93.85,94.38,94.2,94.2,94.2,93.85,94.56,94.38,94.2,94.2,94.03,95.78,95.78,95.96,96.49,96.66,96.84,96.84,96.84,96.84,96.84,96.84,96.84,96.66,96.66,96.66]};
// Lecture 1 visuals
ML_VIS.l1_clusters3d = async function (root) {
    const r = V.rng(7), centres = [[-1.1, -0.7, 0.9], [1.0, 0.9, 0.2], [0.2, -0.9, -1.1]], pts = [];
    centres.forEach(c => { for (let i = 0; i < 30; i++) pts.push(c.map(v => v + V.gauss(r) * 0.33)); });
    const info = V.h('p', { class: 'vis-readout' }, 'Unlabelled customers: nobody told the algorithm who is a budget, regular or premium shopper.');
    const ctl = V.h('div', { class: 'vis-row' });
    root.append(ctl, info);
    const S = await V.scene3d(root, { radius: 7.5, label: 'Customers in 3D: spend, visits, basket size' });
    if (!S) return;
    const { THREE, scene } = S;
    V.axes3d(THREE, scene, ['spend', 'visits', 'basket'], 2);
    const geo = new THREE.SphereGeometry(0.07, 14, 10), grey = 0x94a3b8, cols = [0x0284c7, 0xe11d48, 0x059669];
    const meshes = pts.map(p => { const m = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: grey })); m.position.set(...p); scene.add(m); return m; });
    const cgeo = new THREE.OctahedronGeometry(0.17);
    // Starting centres
    const START = [[-0.3, 0.6, 1.6], [1.8, 0.0, -0.6], [-0.6, -1.8, -1.4]];
    let cents = [], cmesh = [], trails = [], iter = 0, timer = null, phase = 'assign', done = false;
    function reset() {
        clearInterval(timer); iter = 0; phase = 'assign'; done = false; meshes.forEach(m => m.material.color.setHex(grey));
        cmesh.forEach(m => scene.remove(m)); trails.forEach(t => scene.remove(t)); cmesh = []; trails = []; cents = [];
        info.textContent = 'Unlabelled customers: nobody told the algorithm who is a budget, regular or premium shopper.'; S.dirty();
    }
    function step() {
        if (done) return true;
        if (!cents.length) {
            cents = START.map(c => [...c]);
            cmesh = cents.map((c, k) => { const m = new THREE.Mesh(cgeo, new THREE.MeshLambertMaterial({ color: cols[k] })); m.position.set(...c); m.userData.goal = [...c]; scene.add(m); return m; });
            info.textContent = 'Start: three centres (◆) placed at random positions, away from the real groups.'; S.dirty();
            return false;
        }
        if (phase === 'assign') {
            const lab = pts.map(p => { let b = 0, bd = 1e9; cents.forEach((c, k) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2; if (d < bd) { bd = d; b = k; } }); return b; });
            lab.forEach((k, i) => meshes[i].material.color.setHex(cols[k]));
            cmesh.forEach(m => m.userData.lab = lab);
            iter++; phase = 'update'; S.dirty();
            info.textContent = `Iteration ${iter} · assign: every customer joins its nearest centre (colour = group).`;
            return false;
        }
        const lab = cmesh[0].userData.lab;
        let moved = 0;
        cents = cents.map((c, k) => {
            const mine = pts.filter((_, i) => lab[i] === k); if (!mine.length) return c;
            const n = [0, 1, 2].map(d => V.mean(mine.map(p => p[d]))), dist = Math.hypot(n[0] - c[0], n[1] - c[1], n[2] - c[2]);
            moved += dist;
            if (dist > 0.01) { const t = V.line3d(THREE, [c, n], cols[k], true); scene.add(t); trails.push(t); }
            cmesh[k].userData.goal = n; return n;
        });
        phase = 'assign'; S.dirty();
        done = moved < 0.01;
        info.textContent = done ? `Converged after ${iter} iterations - the centres stopped moving: three groups discovered without any labels.`
            : `Iteration ${iter} · update: each centre moves to the average of its group (dashed trail = how far it moved: ${V.fmt(moved, 2)}).`;
        return done;
    }
    S.onFrame(() => {
        let moving = false;
        cmesh.forEach(m => { const g = m.userData.goal, p = m.position; if (!g) return; const dx = g[0] - p.x, dy = g[1] - p.y, dz = g[2] - p.z; if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) > 0.002) { p.x += dx * 0.12; p.y += dy * 0.12; p.z += dz * 0.12; moving = true; } m.rotation.y += 0.03; });
        return moving || cmesh.length > 0;
    });
    ctl.append(V.btn('Run k-means (k = 3)', () => { reset(); step(); timer = setInterval(() => { if (step()) clearInterval(timer); }, 1100); }, 'primary'), V.btn('Step', () => step()), V.btn('Reset', reset));
};

ML_VIS.l1_knn = function (root) {
    const W = 520, H = 320, r = V.rng(3), pts = [];
    for (let i = 0; i < 18; i++) pts.push({ x: 0.55 + V.gauss(r) * 0.15, y: 0.6 + V.gauss(r) * 0.15, c: 1 });
    for (let i = 0; i < 18; i++) pts.push({ x: 0.3 + V.gauss(r) * 0.14, y: 0.3 + V.gauss(r) * 0.13, c: 0 });
    pts.forEach(p => { p.x = Math.min(.97, Math.max(.03, p.x)); p.y = Math.min(.97, Math.max(.03, p.y)); });
    // Linear boundary
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
    pts.forEach(p => { const f = [1, p.x, p.y], t = p.c ? 1 : -1; for (let i = 0; i < 3; i++) { b[i] += f[i] * t; for (let j = 0; j < 3; j++) A[i][j] += f[i] * f[j]; } });
    const w = solve(A, b);
    let q = { x: 0.48, y: 0.45 }, k = 5, showModel = false;
    const svg = V.svg(W, H), read = V.h('p', { class: 'vis-readout' });
    const sx = x => 40 + x * (W - 60), sy = y => H - 30 - y * (H - 50);
    svg.addEventListener('click', e => { const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; const p = pt.matrixTransform(svg.getScreenCTM().inverse()); q = { x: Math.min(1, Math.max(0, (p.x - 40) / (W - 60))), y: Math.min(1, Math.max(0, (H - 30 - p.y) / (H - 50))) }; draw(); });
    root.append(V.h('div', { class: 'vis-row' }, V.slider('k neighbours', 1, 9, 2, k, v => { k = v; draw(); }), V.seg([[false, 'Instance-based (KNN)'], [true, '+ Model-based line']], false, v => { showModel = v; draw(); })), svg, read,
        V.h('p', { class: 'vis-note' }, 'Click anywhere to place a new student. Blue = enjoyed ML, red = did not.'));
    function draw() {
        V.clear(svg); const rr = V.rng(11);
        V.arrow(svg, 40, H - 30, W - 10, H - 30, '#94a3b8', 1.4, rr); V.arrow(svg, 40, H - 30, 40, 10, '#94a3b8', 1.4, rr);
        V.text(svg, W - 80, H - 12, 'likes programming →', { size: 13, color: '#64748b' });
        V.text(svg, 50, 18, '↑ likes statistics', { size: 13, color: '#64748b', anchor: 'start' });
        if (showModel) { const y0 = -(w[0] + w[1] * 0) / w[2], y1 = -(w[0] + w[1] * 1) / w[2]; V.rline(svg, sx(0), sy(y0), sx(1), sy(y1), V.C.amber, 2.4, rr); }
        const near = pts.map(p => ({ p, d: Math.hypot(p.x - q.x, p.y - q.y) })).sort((a, c) => a.d - c.d).slice(0, k);
        near.forEach(n => V.rline(svg, sx(q.x), sy(q.y), sx(n.p.x), sy(n.p.y), '#94a3b8', 1.2, rr, { 'stroke-dasharray': '4 4' }));
        pts.forEach(p => svg.append(V.s('circle', { cx: sx(p.x), cy: sy(p.y), r: 6, fill: p.c ? V.C.sky : V.C.rose, stroke: '#fff', 'stroke-width': 1.5 })));
        const votes = near.filter(n => n.p.c).length, pred = votes * 2 > k;
        svg.append(V.s('path', { d: `M${sx(q.x)} ${sy(q.y) - 11} l11 11 l-11 11 l-11 -11 z`, fill: pred ? V.C.sky : V.C.rose, stroke: V.C.ink, 'stroke-width': 2 }));
        const mPred = w[0] + w[1] * q.x + w[2] * q.y > 0;
        read.innerHTML = `<b>Instance-based:</b> ${votes} of ${k} nearest students enjoyed ML → predict <b>${pred ? 'enjoys ML' : "doesn't"}</b>.` +
            (showModel ? ` &nbsp;|&nbsp; <b>Model-based:</b> the point is on the ${mPred ? 'blue' : 'red'} side of the learned line → <b>${mPred ? 'enjoys ML' : "doesn't"}</b> (no stored examples needed).` : '');
    }
    draw();
};

function solve(A, b) {
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
        let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
        [M[c], M[p]] = [M[p], M[c]];
        if (Math.abs(M[c][c]) < 1e-14) continue;
        for (let r = 0; r < n; r++) { if (r === c) continue; const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    return M.map((r, i) => r[n] / (r[i] || 1e-14));
}

ML_VIS.l1_overfit = function (root) {
    const r = V.rng(21), f = x => Math.sin(2.6 * x) * 0.8 + 0.3 * x;
    const mk = n => Array.from({ length: n }, () => { const x = r() * 2 - 1; return { x, y: f(x) + V.gauss(r) * 0.22 }; });
    const train = mk(12), test = mk(30);
    const fit = d => { const n = d + 1, A = Array.from({ length: n }, () => Array(n).fill(0)), b = Array(n).fill(0); train.forEach(p => { const v = Array.from({ length: n }, (_, i) => p.x ** i); for (let i = 0; i < n; i++) { b[i] += v[i] * p.y; for (let j = 0; j < n; j++) A[i][j] += v[i] * v[j]; } }); for (let i = 0; i < n; i++) A[i][i] += 1e-9; return solve(A, b); };
    const pred = (c, x) => c.reduce((s, ci, i) => s + ci * x ** i, 0);
    const mse = (c, set) => V.mean(set.map(p => (pred(c, p.x) - p.y) ** 2));
    const D = 11, models = [], errs = [];
    for (let d = 1; d <= D; d++) { const c = fit(d); models[d] = c; errs[d] = [mse(c, train), mse(c, test)]; }
    let deg = 3;
    const W = 560, H = 300, svg = V.svg(W, H), curve = V.svg(300, 300), read = V.h('p', { class: 'vis-readout' });
    root.append(V.h('div', { class: 'vis-row' }, V.slider('Model complexity (polynomial degree)', 1, D, 1, deg, v => { deg = v; draw(); })),
        V.h('div', { class: 'vis-split' }, svg, curve), read,
        V.h('p', { class: 'vis-note' }, '● training points (the model sees these) · ○ test points (unseen). Errors are mean squared error.'));
    // Fit the y-axis to all points so none are cut off
    const ys = train.concat(test).map(p => p.y), yLo = Math.min(...ys) - 0.15, yHi = Math.max(...ys) + 0.15, PAD = 14;
    const sx = x => 30 + (x + 1) / 2 * (W - 50), sy = y => PAD + (yHi - y) / (yHi - yLo) * (H - 2 * PAD);
    function draw() {
        V.clear(svg); V.clear(curve); const rr = V.rng(5);
        V.rline(svg, 20, sy(0), W - 10, sy(0), '#e2e8f0', 1, rr);
        let d = ''; for (let i = 0; i <= 200; i++) { const x = -1 + 2 * i / 200, y = Math.max(yLo - 1, Math.min(yHi + 1, pred(models[deg], x))); d += (i ? 'L' : 'M') + sx(x) + ' ' + sy(y); }
        svg.append(V.s('path', { d, fill: 'none', stroke: deg <= 2 ? V.C.violet : deg >= 8 ? V.C.rose : V.C.green, 'stroke-width': 3, 'stroke-linecap': 'round' }));
        test.forEach(p => svg.append(V.s('circle', { cx: sx(p.x), cy: sy(p.y), r: 4.5, fill: '#fff', stroke: V.C.amber, 'stroke-width': 2 })));
        train.forEach(p => svg.append(V.s('circle', { cx: sx(p.x), cy: sy(p.y), r: 5.5, fill: V.C.sky })));
        const lg = v => Math.log10(Math.max(v, 1e-4)), lo = -2.2, hi = 1.2, cx = d => 40 + (d - 1) / (D - 1) * 240, cy = v => 250 - (Math.max(lo, Math.min(hi, lg(v))) - lo) / (hi - lo) * 220;
        V.arrow(curve, 40, 252, 290, 252, '#94a3b8', 1.2, rr); V.arrow(curve, 40, 252, 40, 20, '#94a3b8', 1.2, rr);
        V.text(curve, 165, 278, 'model complexity →', { size: 13, color: '#64748b' });
        V.text(curve, 22, 130, 'error', { size: 13, color: '#64748b' });
        [[0, V.C.sky, 'training'], [1, V.C.amber, 'test']].forEach(([k, col, name]) => {
            let p = ''; for (let dd = 1; dd <= D; dd++) p += (dd > 1 ? 'L' : 'M') + cx(dd) + ' ' + cy(errs[dd][k]);
            curve.append(V.s('path', { d: p, fill: 'none', stroke: col, 'stroke-width': 2.5 }));
            V.text(curve, 292, cy(errs[D][k]) + (k ? -10 : 10), name, { size: 13, color: col, anchor: 'end' });
        });
        curve.append(V.s('line', { x1: cx(deg), x2: cx(deg), y1: 20, y2: 252, stroke: V.C.ink, 'stroke-dasharray': '4 4' }));
        const best = errs.reduce((b, e, i) => (e && (!b || e[1] < errs[b][1]) ? i : b), 0);
        V.text(curve, cx(best), 14, 'sweet spot', { size: 12, color: V.C.green });
        const [tr, te] = errs[deg];
        const verdict = deg <= 2 ? '<b style="color:#7c3aed">Underfitting</b> - too simple: both errors are high.' : te > 3 * Math.max(tr, 0.02) && deg >= 6 ? '<b style="color:#e11d48">Overfitting</b> - the curve chases the training points; test error grows.' : '<b style="color:#059669">Good generalization</b> - captures the trend, not the noise.';
        read.innerHTML = `Degree ${deg}: training error <b>${V.fmt(tr, 3)}</b>, test error <b>${V.fmt(te, 3)}</b>. ${verdict}`;
    }
    draw();
};
// Lecture 2 visuals
ML_VIS.l2_impute = function (root) {
    let outlier = 1000, strat = 'median';
    const W = 560, H = 260, svg = V.svg(W, H), table = V.h('div', { class: 'vis-table-wrap' }), read = V.h('p', { class: 'vis-readout' });
    root.append(V.h('div', { class: 'vis-row' }, V.slider('Product E price (RM)', 120, 3000, 10, outlier, v => { outlier = v; draw(); }),
        V.seg([['mean', 'Fill with mean'], ['median', 'Fill with median']], strat, v => { strat = v; draw(); })), svg, read, table);
    function draw() {
        const known = [100, 120, 110, outlier], mean = V.mean(known), med = V.median(known), fill = strat === 'mean' ? mean : med;
        V.clear(svg); const rr = V.rng(2), maxV = Math.max(outlier, mean) * 1.1, sy = v => H - 30 - v / maxV * (H - 60);
        const names = ['A', 'B', 'C (missing)', 'D', 'E'], vals = [100, 120, null, 110, outlier];
        V.rline(svg, 30, H - 30, W - 10, H - 30, '#94a3b8', 1.4, rr);
        vals.forEach((v, i) => {
            const x = 50 + i * 100, val = v ?? fill, y = sy(val);
            V.rrect(svg, x, y, 60, H - 30 - y, v == null ? V.C.amber : V.C.sky, v == null ? V.C.amberS : V.C.skyS, rr);
            V.text(svg, x + 30, H - 14, names[i], { size: 13 });
            V.text(svg, x + 30, y - 12, (v == null ? '≈' : '') + V.fmt(val, 1), { size: 13, weight: 700, color: v == null ? V.C.amber : V.C.ink });
        });
        [[mean, V.C.rose, 'mean ' + V.fmt(mean, 1)], [med, V.C.green, 'median ' + V.fmt(med, 1)]].forEach(([v, c, t], k) => {
            V.rline(svg, 30, sy(v), W - 10, sy(v), c, 1.6, rr, { 'stroke-dasharray': '6 5' });
            V.text(svg, W - 12, sy(v) + (k ? 12 : -10), t, { size: 13, color: c, anchor: 'end' });
        });
        read.innerHTML = `Mean of the known prices = (100 + 120 + 110 + ${outlier}) / 4 = <b>${V.fmt(mean, 1)}</b>; median = <b>${V.fmt(med, 1)}</b>. ` +
            (outlier > 300 ? `The outlier drags the mean ${V.fmt(mean - med, 1)} above the median, so <b>median</b> is the more representative fill.` : 'Without an outlier, mean and median are close - either works.');
        table.innerHTML = `<table class="vis-table"><tr><th>Product</th><th>A</th><th>B</th><th>C</th><th>D</th><th>E</th></tr><tr><td>price</td><td>100</td><td>120</td><td class="hl">${V.fmt(fill, 1)}</td><td>110</td><td>${outlier}</td></tr></table><code class="vis-code">df["price"].fillna(df["price"].${strat}())</code>`;
    }
    draw();
};

ML_VIS.l2_scaling = function (root) {
    const ages = [18, 25, 30, 41, 52, 70], incomes = [20000, 45000, 55000, 80000, 120000, 200000];
    let A = { age: 25, inc: 45000 }, B = { age: 30, inc: 55000 }, mode = 'raw';
    const read = V.h('p', { class: 'vis-readout' }), svg = V.svg(560, 190);
    const sl = (who, key, min, max, step, lab) => V.slider(lab, min, max, step, who[key], v => { who[key] = v; draw(); }, v => key === 'inc' ? 'RM' + v.toLocaleString() : v);
    root.append(V.h('div', { class: 'vis-row' }, sl(A, 'age', 18, 70, 1, 'A age'), sl(A, 'inc', 20000, 200000, 1000, 'A income')),
        V.h('div', { class: 'vis-row' }, sl(B, 'age', 18, 70, 1, 'B age'), sl(B, 'inc', 20000, 200000, 1000, 'B income')),
        V.seg([['raw', 'No scaling'], ['minmax', 'Min-max'], ['z', 'Z-score']], mode, v => { mode = v; draw(); }), svg, read);
    const tf = (v, arr) => mode === 'raw' ? v : mode === 'minmax' ? (v - Math.min(...arr)) / (Math.max(...arr) - Math.min(...arr)) : (v - V.mean(arr)) / V.std(arr);
    function draw() {
        const da = tf(A.age, ages) - tf(B.age, ages), di = tf(A.inc, incomes) - tf(B.inc, incomes);
        const ca = da * da, ci = di * di, tot = ca + ci || 1, dist = Math.sqrt(ca + ci);
        V.clear(svg); const rr = V.rng(4);
        V.text(svg, 10, 22, 'Share of the distance between A and B coming from each feature', { size: 15, anchor: 'start', weight: 700 });
        const wA = 520 * ca / tot;
        V.rrect(svg, 20, 50, Math.max(wA, 2), 56, V.C.sky, V.C.skyS, rr); V.rrect(svg, 20 + wA, 50, Math.max(520 - wA, 2), 56, V.C.amber, V.C.amberS, rr);
        V.text(svg, 20 + Math.max(wA, 60) / 2, 78, `age ${V.fmt(100 * ca / tot, 1)}%`, { size: 14, weight: 700, color: V.C.sky, anchor: wA < 60 ? 'start' : 'middle' });
        V.text(svg, 20 + wA + (520 - wA) / 2, 78, `income ${V.fmt(100 * ci / tot, 1)}%`, { size: 14, weight: 700, color: V.C.amber });
        V.text(svg, 20, 135, `Δage = ${V.fmt(da, 3)}     Δincome = ${V.fmt(di, 3)}     distance = √(Δage² + Δincome²) = ${V.fmt(dist, 3)}`, { size: 14, anchor: 'start' });
        V.text(svg, 20, 165, mode === 'raw' ? 'Raw units: income differences are thousands, age differences are single digits.' : mode === 'minmax' ? "x' = (x - min) / (max - min) puts both on 0-1." : 'z = (x - μ) / σ puts both on "standard deviations from the mean".', { size: 13, anchor: 'start', color: '#64748b' });
        read.innerHTML = mode === 'raw' ? 'Without scaling, income decides almost the whole distance - not because it matters more, only because its units are bigger. Distance-based models (KNN, SVM) would effectively ignore age.' : 'After scaling, both features contribute on comparable footing. The order of customers is unchanged - same pattern, different ruler.';
    }
    draw();
};

ML_VIS.l2_groupby = function (root) {
    const rows = [['New', 100], ['Premium', 500], ['New', 200], ['Premium', 700], ['Returning', 150], ['Returning', 250]];
    const col = { New: V.C.sky, Premium: V.C.violet, Returning: V.C.amber };
    let step = 0, agg = 'mean';
    const stage = V.h('div', { class: 'vis-groupby' }), code = V.h('code', { class: 'vis-code' });
    const aggf = { mean: V.mean, sum: a => a.reduce((x, y) => x + y, 0), max: a => Math.max(...a), count: a => a.length };
    const stepBtn = V.btn('Next step', () => { step = (step + 1) % 4; draw(); }, 'primary');
    root.append(V.h('div', { class: 'vis-row' }, stepBtn, V.seg(Object.keys(aggf).map(k => [k, k + '()']), agg, v => { agg = v; draw(); })), code, stage);
    function draw() {
        code.textContent = `df.groupby("user_type")["price"].${agg}()`;
        const groups = {}; rows.forEach(([g, p]) => (groups[g] = groups[g] || []).push(p));
        const chip = (g, txt) => V.h('span', { class: 'vis-chip', style: { borderColor: col[g], background: col[g] + '18' } }, txt);
        const panel = (title, on, body) => V.h('div', { class: 'vis-panel' + (on ? ' on' : '') }, V.h('b', {}, title), body);
        V.clear(stage);
        stage.append(
            panel('0 · Raw rows', step === 0, V.h('div', {}, rows.map(([g, p]) => chip(g, `${g} · ${p}`)))),
            panel('1 · Split', step === 1, step >= 1 ? V.h('div', {}, Object.entries(groups).map(([g, ps]) => V.h('div', { class: 'vis-group' }, V.h('small', {}, g), ps.map(p => chip(g, p))))) : V.h('i', {}, '...')),
            panel('2 · Apply ' + agg + '()', step === 2, step >= 2 ? V.h('div', {}, Object.entries(groups).map(([g, ps]) => V.h('div', {}, chip(g, `${g}: ${V.fmt(aggf[agg](ps), 1)}`)))) : V.h('i', {}, '...')),
            panel('3 · Combine', step === 3, step >= 3 ? V.h('table', { class: 'vis-table' }, V.h('tr', {}, V.h('th', {}, 'user_type'), V.h('th', {}, agg + ' price')), Object.entries(groups).map(([g, ps]) => V.h('tr', {}, V.h('td', {}, g), V.h('td', {}, V.fmt(aggf[agg](ps), 1))))) : V.h('i', {}, '...')));
        stepBtn.textContent = step === 3 ? '↺ Start again' : 'Next step';
    }
    draw();
};

ML_VIS.l2_chart_picker = function (root) {
    const Q = [['line', 'How did monthly sales change over the year?'], ['bar', 'Which branch sells the most?'], ['hist', 'What do product prices look like?'], ['box', 'Are there unusual ages?'], ['scatter', 'Does advertising relate to sales?']];
    const svg = V.svg(520, 250), read = V.h('p', { class: 'vis-readout' });
    root.append(V.seg(Q.map(([k, q]) => [k, q]), 'line', v => draw(v)), svg, read);
    const why = { line: 'Line chart - a trend over time.', bar: 'Bar chart - compare a quantity across discrete categories.', hist: 'Histogram - the distribution of one numeric variable (skew, long tail).', box: 'Box plot - centre, spread and outliers at a glance.', scatter: 'Scatter plot - the relationship between two numeric variables.' };
    function draw(k) {
        V.clear(svg); const rr = V.rng(8), X0 = 50, Y0 = 215;
        V.arrow(svg, X0, Y0, 500, Y0, '#94a3b8', 1.4, rr); V.arrow(svg, X0, Y0, X0, 15, '#94a3b8', 1.4, rr);
        if (k === 'line') { const ys = [60, 70, 65, 85, 90, 110, 120, 115, 135, 150, 145, 170]; let d = ''; ys.forEach((y, i) => { d += (i ? 'L' : 'M') + (X0 + 20 + i * 36) + ' ' + (Y0 - y); }); svg.append(V.s('path', { d, fill: 'none', stroke: V.C.sky, 'stroke-width': 3 })); ['Jan', 'Jun', 'Dec'].forEach((m, i) => V.text(svg, X0 + 20 + i * 198, Y0 + 15, m, { size: 12 })); }
        if (k === 'bar') [['KL', 160], ['Penang', 95], ['Ipoh', 120], ['Johor', 70]].forEach(([b, v], i) => { V.rrect(svg, X0 + 30 + i * 105, Y0 - v, 70, v, V.C.lime, V.C.limeS, rr); V.text(svg, X0 + 65 + i * 105, Y0 + 14, b, { size: 13 }); });
        if (k === 'hist') [20, 70, 140, 165, 120, 80, 45, 25, 12, 6, 3, 8].forEach((v, i) => { V.rrect(svg, X0 + 10 + i * 36, Y0 - v * 1.1, 36, v * 1.1, V.C.indigo, V.C.indigoS, rr); });
        if (k === 'box') { const y = 120; V.rline(svg, 90, y, 170, y, V.C.ink, 2, rr); V.rrect(svg, 170, y - 35, 160, 70, V.C.rose, V.C.roseS, rr); V.rline(svg, 240, y - 35, 240, y + 35, V.C.ink, 2.5, rr); V.rline(svg, 330, y, 400, y, V.C.ink, 2, rr); [455, 478].forEach(x => svg.append(V.s('circle', { cx: x, cy: y, r: 6, fill: 'none', stroke: V.C.red, 'stroke-width': 2 }))); V.text(svg, 466, y - 24, 'outliers', { size: 13, color: V.C.red }); }
        if (k === 'scatter') { const r = V.rng(9); for (let i = 0; i < 40; i++) { const x = r(); svg.append(V.s('circle', { cx: X0 + 20 + x * 420, cy: Y0 - 20 - (x * 150 + V.gauss(r) * 18), r: 5, fill: V.C.amber, opacity: .8 })); } V.text(svg, 300, Y0 + 14, 'advertising spend →', { size: 12 }); }
        read.innerHTML = '<b>' + why[k] + '</b>';
    }
    draw('line');
};

ML_VIS.l2_leakage = function (root) {
    const train = [10, 20, 30];
    let test = 100, mode = 'train';
    const svg = V.svg(560, 230), read = V.h('p', { class: 'vis-readout' }), code = V.h('pre', { class: 'lesson-code' });
    root.append(V.h('div', { class: 'vis-row' }, V.slider('Test value', 35, 200, 1, test, v => { test = v; draw(); }),
        V.seg([['train', 'Fit scaler on TRAIN only'], ['all', 'Fit scaler on ALL data (leak)']], mode, v => { mode = v; draw(); })), svg, read, code);
    function draw() {
        const all = [...train, test], lo = Math.min(...(mode === 'train' ? train : all)), hi = Math.max(...(mode === 'train' ? train : all));
        const sc = v => (v - lo) / (hi - lo);
        V.clear(svg); const rr = V.rng(3), X = v => 40 + (v + 0.2) / 5 * 480;
        V.text(svg, 20, 24, `learned: min = ${lo}, max = ${hi}`, { size: 15, anchor: 'start', weight: 700, color: mode === 'train' ? V.C.green : V.C.red });
        V.arrow(svg, 30, 120, 540, 120, '#94a3b8', 1.4, rr);
        [0, 1, 2, 3, 4].forEach(t => { V.rline(svg, X(t), 114, X(t), 126, '#94a3b8', 1.2, rr); V.text(svg, X(t), 142, t, { size: 12, color: '#64748b' }); });
        svg.append(V.s('rect', { x: X(0), y: 104, width: X(1) - X(0), height: 32, fill: V.C.greenS, opacity: .6 }));
        train.forEach(v => { svg.append(V.s('circle', { cx: X(sc(v)), cy: 120, r: 8, fill: V.C.sky })); });
        const tx = Math.min(X(sc(test)), 535);
        svg.append(V.s('path', { d: `M${tx} 108 l10 12 l-10 12 l-10 -12 z`, fill: V.C.amber, stroke: V.C.ink }));
        V.text(svg, tx, 92, `test ${test} → ${V.fmt(sc(test), 2)}`, { size: 14, weight: 700, color: V.C.amber });
        V.text(svg, X(0.5), 165, 'training range after scaling', { size: 12, color: V.C.green });
        V.text(svg, 280, 205, mode === 'train' ? 'The test point is scaled with parameters it never influenced.' : 'max came FROM the test point - the scaler already "saw" the exam.', { size: 15, color: mode === 'train' ? V.C.green : V.C.red });
        read.innerHTML = mode === 'train' ? `Train-only fit: (${test} - 10) / (30 - 10) = <b>${V.fmt(sc(test), 2)}</b>. Values outside 0-1 are fine and honest: the test set really is unlike the training data.` : `Leaky fit: the test value set the maximum, so it becomes exactly <b>1</b> and the training data is squashed to ${V.fmt(sc(10), 2)}-${V.fmt(sc(30), 2)}. Test information has entered preprocessing, so evaluation will look better than reality.`;
        code.textContent = mode === 'train' ? 'scaler.fit(X_train)\nX_train = scaler.transform(X_train)\nX_test  = scaler.transform(X_test)   # correct' : 'scaler.fit(X_all)                     # wrong: includes test rows\nX_train = scaler.transform(X_train)\nX_test  = scaler.transform(X_test)';
    }
    draw();
};
// Lecture 3 visuals
ML_VIS.l3_interaction3d = async function (root) {
    let mode = 'both';
    const read = V.h('p', { class: 'vis-readout' });
    root.append(V.seg([['add', 'Additive: a·Attack + b·Defense'], ['inter', 'Interaction: Attack x Defense'], ['both', 'Both']], mode, v => { mode = v; show(); }), read);
    const S = await V.scene3d(root, { radius: 7, phi: 1.0, theta: 0.9, target: [0, 0.6, 0], label: 'Additive plane versus interaction surface' });
    if (!S) return;
    const { THREE, scene } = S, N = 24, sz = 3;
    const surf = (fz, color, opacity) => {
        const g = new THREE.PlaneGeometry(sz, sz, N, N); g.rotateX(-Math.PI / 2);
        const p = g.attributes.position;
        for (let i = 0; i < p.count; i++) { const a = (p.getX(i) / sz + 0.5) * 100, d = (p.getZ(i) / sz + 0.5) * 100; p.setY(i, fz(a, d)); }
        g.computeVertexNormals();
        const m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity }));
        const w = new THREE.LineSegments(new THREE.WireframeGeometry(g), new THREE.LineBasicMaterial({ color, transparent: true, opacity: .25 }));
        const grp = new THREE.Group(); grp.add(m, w); scene.add(grp); return grp;
    };
    const add = surf((a, d) => (a + d) / 100 * 1.1, 0x0284c7, 0.45), inter = surf((a, d) => a * d / 10000 * 2.2, 0xe11d48, 0.55);
    const o = -sz / 2;
    scene.add(V.line3d(THREE, [[o, 0, o], [-o, 0, o]], 0x64748b), V.line3d(THREE, [[o, 0, o], [o, 0, -o]], 0x64748b), V.line3d(THREE, [[o, 0, o], [o, 2.4, o]], 0x64748b));
    [['Attack →', [0.2, 0, o - 0.35]], ['Defense →', [o - 0.45, 0, 0.2]], ['prediction', [o, 2.65, o]]].forEach(([t, p]) => { const s = V.label3d(THREE, t, '#475569', 0.28); s.position.set(...p); scene.add(s); });
    const px = 49 / 100 * sz + o, pz = 49 / 100 * sz + o, py = 49 * 49 / 10000 * 2.2;
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.08, 16, 12), new THREE.MeshLambertMaterial({ color: 0xd97706 }));
    dot.position.set(px, py, pz); scene.add(dot);
    const lab = V.label3d(THREE, '49 x 49 = 2401', '#b45309', 0.3); lab.position.set(px + 0.2, py + 0.7, pz); scene.add(lab);
    scene.add(V.line3d(THREE, [[px, 0, pz], [px, py, pz]], 0xd97706, true));
    function show() {
        add.visible = mode !== 'inter'; inter.visible = mode !== 'add'; dot.visible = lab.visible = mode !== 'add'; S.dirty();
        read.innerHTML = mode === 'add' ? 'Additive model: raising Attack always adds the same amount, whatever Defense is - a flat, tilted plane.' : mode === 'inter' ? 'Interaction feature: the surface bends upward only when <b>both</b> are high. The engineered column Attack x Defense lets even a simple model capture that.' : 'Compare: the plane (blue) cannot rise faster in the "both high" corner; the interaction surface (red) can.';
    }
    show();
};

ML_VIS.l3_poly_count = function (root) {
    let n = 2, d = 2;
    const read = V.h('div', { class: 'vis-readout' }), svg = V.svg(560, 120);
    root.append(V.h('div', { class: 'vis-row' }, V.slider('original features n', 1, 50, 1, n, v => { n = v; draw(); }), V.slider('degree', 1, 5, 1, d, v => { d = v; draw(); })), read, svg);
    const C = (a, b) => { let r = 1; for (let i = 1; i <= b; i++) r = r * (a - b + i) / i; return Math.round(r); };
    function terms() {
        const out = [], rec = (start, left, cur) => { if (cur.length) out.push(cur.slice()); if (!left) return; for (let i = start; i < n; i++) { cur.push(i); rec(i, left - 1, cur); cur.pop(); } };
        rec(0, d, []);
        return out.map(t => { const c = {}; t.forEach(i => c[i] = (c[i] || 0) + 1); return Object.entries(c).map(([i, p]) => `x<sub>${+i + 1}</sub>${p > 1 ? `<sup>${p}</sup>` : ''}`).join(''); });
    }
    function draw() {
        const count = C(n + d, d) - 1;
        read.innerHTML = `PolynomialFeatures(degree=${d}, include_bias=False) on ${n} feature${n > 1 ? 's' : ''} → <b style="font-size:1.25em">${count.toLocaleString()}</b> features` +
            (count <= 30 ? `<div class="vis-terms">${terms().join(' · ')}</div>` : `<div class="vis-terms">Too many to list - that is <b>feature explosion</b>.</div>`);
        V.clear(svg); const rr = V.rng(6), w = Math.min(540, 20 + Math.log10(count + 1) / 7 * 520);
        V.rrect(svg, 10, 20, w, 44, count > 1000 ? V.C.rose : count > 50 ? V.C.amber : V.C.green, count > 1000 ? V.C.roseS : count > 50 ? V.C.amberS : V.C.greenS, rr);
        V.text(svg, 10, 92, 'bar length on a log scale: 10 · 100 · 1,000 · 10,000 · 100,000 · 1,000,000 features', { size: 13, anchor: 'start', color: '#64748b' });
    }
    draw();
};

function skewedIncome() {
    const r = V.rng(42); return Array.from({ length: 300 }, () => Math.round(Math.exp(10.4 + 0.75 * V.gauss(r)) / 100) * 100);
}

ML_VIS.l3_binning = function (root) {
    const data = skewedIncome().sort((a, b) => a - b), lo = data[0], hi = data[data.length - 1];
    let mode = 'fixed', k = 4;
    const svg = V.svg(560, 300), read = V.h('p', { class: 'vis-readout' }), code = V.h('code', { class: 'vis-code' });
    root.append(V.h('div', { class: 'vis-row' }, V.seg([['fixed', 'Fixed-width (pd.cut)'], ['quantile', 'Quantile (pd.qcut)']], mode, v => { mode = v; draw(); }), V.slider('bins', 2, 10, 1, k, v => { k = v; draw(); })), code, svg, read);
    function draw() {
        const edges = mode === 'fixed' ? Array.from({ length: k + 1 }, (_, i) => lo + (hi - lo) * i / k) : Array.from({ length: k + 1 }, (_, i) => data[Math.min(data.length - 1, Math.round(i / k * (data.length - 1)))]);
        const counts = Array(k).fill(0); data.forEach(v => { let b = edges.findIndex((e, i) => i > 0 && v <= e) - 1; if (b < 0) b = 0; counts[Math.min(k - 1, b)]++; });
        code.textContent = mode === 'fixed' ? `pd.cut(df["income"], bins=${k})   # equal WIDTH` : `pd.qcut(df["income"], q=${k})   # equal COUNT`;
        V.clear(svg); const rr = V.rng(10), X = v => 30 + (v - lo) / (hi - lo) * 510;
        data.forEach((v, i) => svg.append(V.s('circle', { cx: X(v), cy: 40 + (i % 6) * 6, r: 2.2, fill: V.C.slate, opacity: .55 })));
        V.text(svg, 30, 14, 'each dot = one developer income (right-skewed)', { size: 13, anchor: 'start', color: '#64748b' });
        edges.forEach(e => V.rline(svg, X(e), 30, X(e), 260, V.C.indigo, 1.4, rr, { 'stroke-dasharray': '5 4' }));
        const maxC = Math.max(...counts);
        counts.forEach((c, i) => {
            const x0 = X(edges[i]), x1 = X(edges[i + 1]), h = c / maxC * 150;
            V.rrect(svg, x0 + 1, 255 - h, Math.max(x1 - x0 - 2, 3), h, c === 0 ? V.C.red : V.C.indigo, c === 0 ? '#fee2e2' : V.C.indigoS, rr);
            if (x1 - x0 > 26) V.text(svg, (x0 + x1) / 2, 245 - h, c, { size: 13, weight: 700 });
        });
        V.text(svg, 30, 285, 'RM' + Math.round(lo / 1000) + 'k', { size: 12, anchor: 'start', color: '#64748b' });
        V.text(svg, 540, 285, 'RM' + Math.round(hi / 1000) + 'k', { size: 12, anchor: 'end', color: '#64748b' });
        const empty = counts.filter(c => c < 6).length;
        read.innerHTML = mode === 'fixed' ? `Equal widths of about RM${Math.round((hi - lo) / k / 1000)}k. Counts: <b>${counts.join(' · ')}</b>. ${empty ? `${empty} bin(s) are nearly empty - the long tail wastes them.` : ''}` : `Every bin holds about ${Math.round(data.length / k)} people; the widths adapt (narrow where data is dense, wide in the tail). Counts: <b>${counts.join(' · ')}</b>.`;
    }
    draw();
};

ML_VIS.l3_log = function (root) {
    const data = skewedIncome();
    let mode = 'raw';
    const svg = V.svg(560, 250), read = V.h('p', { class: 'vis-readout' });
    root.append(V.seg([['raw', 'Raw income'], ['log', 'log10(income)']], mode, v => { mode = v; draw(); }), svg, read,
        V.h('div', { class: 'vis-terms' }, '10 → 1 · 100 → 2 · 1,000 → 3 · 1,000,000 → 6 - big values shrink far more than small ones'));
    const skew = a => { const m = V.mean(a), s = V.std(a); return V.mean(a.map(x => ((x - m) / s) ** 3)); };
    function draw() {
        const vals = mode === 'raw' ? data : data.map(v => Math.log10(v)), lo = Math.min(...vals), hi = Math.max(...vals), B = 20, counts = Array(B).fill(0);
        vals.forEach(v => counts[Math.min(B - 1, Math.floor((v - lo) / (hi - lo) * B))]++);
        V.clear(svg); const rr = V.rng(12), maxC = Math.max(...counts);
        V.rline(svg, 20, 210, 545, 210, '#94a3b8', 1.4, rr);
        counts.forEach((c, i) => { const h = c / maxC * 170; if (c) V.rrect(svg, 22 + i * 26, 210 - h, 24, h, mode === 'raw' ? V.C.amber : V.C.green, mode === 'raw' ? V.C.amberS : V.C.greenS, rr); });
        const f = v => mode === 'raw' ? 'RM' + Math.round(v / 1000) + 'k' : V.fmt(v, 2);
        V.text(svg, 20, 230, f(lo), { size: 12, anchor: 'start', color: '#64748b' }); V.text(svg, 545, 230, f(hi), { size: 12, anchor: 'end', color: '#64748b' });
        const sk = skew(vals);
        read.innerHTML = `Skewness = <b>${V.fmt(sk, 2)}</b>. ` + (mode === 'raw' ? 'A long right tail: a few very high incomes stretch the scale and dominate.' : 'After log10 the distribution is far more balanced, order is preserved, and the extreme values no longer dominate.');
    }
    draw();
};

ML_VIS.l3_encoder = function (root) {
    let cats = ['Action', 'Adventure', 'Sports'], mode = 'onehot', buckets = 4;
    const inp = V.h('input', { type: 'text', value: cats.join(', '), class: 'vis-input', 'aria-label': 'Categories, comma separated' });
    inp.addEventListener('input', () => { const c = inp.value.split(',').map(s => s.trim()).filter(Boolean); cats = [...new Set(c)].slice(0, 8); if (cats.length) draw(); });
    const out = V.h('div', { class: 'vis-table-wrap' }), read = V.h('p', { class: 'vis-readout' });
    const bucketSl = V.slider('hash buckets k', 2, 8, 1, buckets, v => { buckets = v; draw(); });
    root.append(V.h('div', { class: 'vis-row' }, V.h('label', { class: 'vis-slider' }, V.h('span', {}, 'categories'), inp)),
        V.seg([['label', 'Label'], ['onehot', 'One-hot'], ['dummy', 'Dummy'], ['effect', 'Effect'], ['hash', 'Hashing']], mode, v => { mode = v; draw(); }), bucketSl, out, read);
    const hash = s => { let h = 0; for (const ch of s) h = (h * 31 + ch.charCodeAt(0)) >>> 0; return h % buckets; };
    function draw() {
        bucketSl.style.display = mode === 'hash' ? '' : 'none';
        let head = [], rows = [];
        if (mode === 'label') { head = ['code']; rows = cats.map((c, i) => [i]); }
        if (mode === 'onehot') { head = cats; rows = cats.map((_, i) => cats.map((_, j) => +(i === j))); }
        if (mode === 'dummy' || mode === 'effect') { head = cats.slice(1); rows = cats.map((_, i) => i === 0 ? head.map(() => mode === 'effect' ? -1 : 0) : head.map((_, j) => +(i === j + 1))); }
        if (mode === 'hash') { head = Array.from({ length: buckets }, (_, i) => 'h' + i); rows = cats.map(c => head.map((_, j) => +(hash(c) === j))); }
        out.innerHTML = `<table class="vis-table"><tr><th>category</th>${head.map(h => `<th>${h}</th>`).join('')}</tr>${rows.map((r, i) => `<tr${(mode === 'dummy' || mode === 'effect') && i === 0 ? ' class="ref"' : ''}><td>${cats[i]}</td>${r.map(v => `<td class="${v === 1 ? 'one' : v === -1 ? 'neg' : ''}">${v}</td>`).join('')}</tr>`).join('')}</table>`;
        const m = cats.length, coll = mode === 'hash' ? m - new Set(cats.map(hash)).size : 0;
        read.innerHTML = {
            label: `1 column. Warning: implies ${cats.join(' &lt; ')} - fine only if the categories are genuinely <b>ordinal</b>.`,
            onehot: `m = ${m} categories → <b>${m}</b> columns, exactly one "hot" per row. No false order. <code>pd.get_dummies(df, columns=["genre"])</code>`,
            dummy: `<b>${m - 1}</b> columns: ${cats[0]} is the reference row (all 0). Avoids the dummy-variable trap. <code>drop_first=True</code>`,
            effect: `<b>${m - 1}</b> columns, but the reference ${cats[0]} is coded -1 everywhere - comparisons are against the overall mean.`,
            hash: `Fixed <b>${buckets}</b> columns whatever the number of categories. ${coll ? `<b style="color:#dc2626">${coll} collision(s)</b>: different categories share a bucket.` : 'No collisions with these categories.'}`
        }[mode];
    }
    draw();
};
// Lecture 4 visuals
const STOP = new Set('a an the is are was were be been am at and or but of to in on for with as by it its this that these those i you he she we they my your our their not no'.split(' '));
const LEMMA = { ran: 'run', running: 'run', runs: 'run', cats: 'cat', mice: 'mouse', better: 'good', best: 'good', studies: 'study', studying: 'study', went: 'go', geese: 'goose', children: 'child', quickly: 'quick', was: 'be', are: 'be', is: 'be' };
function stem(w) {
    for (const [suf, rep] of [['ies', 'i'], ['ing', ''], ['edly', ''], ['ed', ''], ['ly', ''], ['es', ''], ['s', '']]) {
        if (w.length > suf.length + 2 && w.endsWith(suf)) { w = w.slice(0, -suf.length) + rep; break; }
    }
    if (/(.)\1$/.test(w) && !/[lsz]$/.test(w)) w = w.slice(0, -1);
    return w;
}

ML_VIS.l4_text_pipeline = function (root) {
    const inp = V.h('input', { type: 'text', class: 'vis-input', value: 'The cats are running quickly! Machine, machine and MACHINE studies.', 'aria-label': 'Sentence to preprocess' });
    const out = V.h('div', { class: 'vis-pipeline' });
    inp.addEventListener('input', draw);
    root.append(V.h('label', { class: 'vis-slider', style: { width: '100%' } }, V.h('span', {}, 'type a sentence'), inp), out);
    function row(title, toks, note, cls) { return V.h('div', { class: 'vis-prow' }, V.h('b', {}, title), V.h('div', {}, toks.length ? toks.map(t => V.h('span', { class: 'vis-chip ' + (cls ? cls(t) : '') }, t)) : V.h('i', {}, '(empty)')), note ? V.h('small', {}, note) : null); }
    function draw() {
        const tokens = inp.value.match(/[A-Za-z0-9']+|[^\sA-Za-z0-9]/g) || [];
        const lower = tokens.map(t => t.toLowerCase()), clean = lower.filter(t => /[a-z0-9]/.test(t)), nostop = clean.filter(t => !STOP.has(t));
        V.clear(out);
        out.append(
            row('1 · Tokenize', tokens, `${tokens.length} tokens`),
            row('2 · Lowercase', lower, `${new Set(tokens).size} → ${new Set(lower).size} distinct`),
            row('3 · Remove special characters', clean, 'punctuation and symbols dropped'),
            row('4 · Remove stopwords', nostop, 'the, is, are, and ... carry little information', null),
            row('5a · Stemming (heuristic)', nostop.map(stem), 'chops suffixes - may give non-words like "studi"'),
            row('5b · Lemmatization (dictionary)', nostop.map(w => LEMMA[w] || w), 'returns real words: ran → run, mice → mouse'));
    }
    draw();
};

ML_VIS.l4_tfidf = function (root) {
    let docs = ['the machine is running normally', 'the machine has a bearing failure', 'the machine needs oil and the machine is noisy', 'the compressor failure after bearing noise in the machine'];
    let view = 'tfidf', bigrams = false;
    const areas = V.h('div', { class: 'vis-docs' }), out = V.h('div', { class: 'vis-table-wrap' }), heat = V.h('div', { class: 'vis-table-wrap' }), read = V.h('p', { class: 'vis-readout' });
    docs.forEach((d, i) => { const t = V.h('textarea', { rows: 2, class: 'vis-input', 'aria-label': 'Document ' + (i + 1) }); t.value = d; t.addEventListener('input', () => { docs[i] = t.value; draw(); }); areas.append(V.h('label', {}, V.h('small', {}, 'Doc ' + (i + 1)), t)); });
    root.append(areas, V.h('div', { class: 'vis-row' }, V.seg([['bow', 'Bag of Words counts'], ['tfidf', 'TF-IDF weights']], view, v => { view = v; draw(); }), V.seg([[false, 'unigrams'], [true, '+ bigrams']], false, v => { bigrams = v; draw(); })), out, read, heat);
    function draw() {
        const toks = docs.map(d => { const w = (d.toLowerCase().match(/[a-z0-9_]+/g) || []); return bigrams ? w.concat(w.slice(1).map((x, i) => w[i] + ' ' + x)) : w; });
        const vocab = [...new Set(toks.flat())].sort(), N = docs.length;
        const df = Object.fromEntries(vocab.map(t => [t, toks.filter(d => d.includes(t)).length]));
        const idf = t => Math.log(N / df[t]);
        const mat = toks.map(d => vocab.map(t => { const c = d.filter(x => x === t).length; return view === 'bow' ? c : (d.length ? c / d.length : 0) * idf(t); }));
        const max = Math.max(1e-9, ...mat.flat());
        const shown = vocab.filter(t => !bigrams || mat.some((r, i) => r[vocab.indexOf(t)] > 0)).slice(0, 24);
        out.innerHTML = `<table class="vis-table vis-heat"><tr><th>term</th>${docs.map((_, i) => `<th>Doc ${i + 1}</th>`).join('')}${view === 'tfidf' ? '<th>IDF</th>' : ''}</tr>` +
            shown.map(t => { const j = vocab.indexOf(t); return `<tr><td><b>${t}</b></td>${mat.map(r => `<td style="background:rgba(225,29,72,${(r[j] / max * 0.75).toFixed(2)})">${view === 'bow' ? r[j] : V.fmt(r[j], 2)}</td>`).join('')}${view === 'tfidf' ? `<td class="idf">${V.fmt(idf(t), 2)}</td>` : ''}</tr>`; }).join('') + '</table>' +
            (vocab.length > shown.length ? `<small>showing ${shown.length} of ${vocab.length} terms</small>` : '');
        const everywhere = vocab.filter(t => df[t] === N);
        read.innerHTML = view === 'bow' ? 'Raw counts: frequent generic words such as "the" and "machine" get the biggest numbers.' : `IDF(t) = log(N / df(t)) with N = ${N}. ${everywhere.length ? `<b>${everywhere.join(', ')}</b> appear in every document → IDF = log(1) = 0, so their weight vanishes.` : ''} Rare, distinctive terms like "failure" and "compressor" get the strongest weights.`;
        // Cosine similarity
        const vec = toks.map(d => vocab.map(t => d.filter(x => x === t).length * (view === 'tfidf' ? idf(t) : 1)));
        const cos = (a, b) => { const dot = a.reduce((s, x, i) => s + x * b[i], 0), na = Math.hypot(...a), nb = Math.hypot(...b); return na && nb ? dot / na / nb : 0; };
        heat.innerHTML = `<b style="display:block;margin:8px 0 4px">Cosine similarity between documents (${view === 'bow' ? 'counts' : 'TF-IDF'})</b><table class="vis-table vis-heat"><tr><th></th>${docs.map((_, i) => `<th>Doc ${i + 1}</th>`).join('')}</tr>` +
            vec.map((a, i) => `<tr><td><b>Doc ${i + 1}</b></td>${vec.map(b => { const c = cos(a, b); return `<td style="background:rgba(2,132,199,${(c * 0.7).toFixed(2)})">${V.fmt(c, 2)}</td>`; }).join('')}</tr>`).join('') + '</table>';
    }
    draw();
};

ML_VIS.l4_cosine = function (root) {
    const W = 420, H = 300, O = { x: 60, y: 250 }, svg = V.svg(W, H, 'vis-drag'), read = V.h('p', { class: 'vis-readout' });
    let A = { x: 260, y: 90 }, B = { x: 300, y: 190 }, drag = null;
    root.append(V.h('div', { class: 'vis-row' }, V.btn('Make B twice as long', () => { B = { x: O.x + (B.x - O.x) * 2, y: O.y + (B.y - O.y) * 2 }; clamp(B); draw(); }), V.btn('Point B the same way as A', () => { const s = Math.hypot(B.x - O.x, B.y - O.y) / Math.hypot(A.x - O.x, A.y - O.y); B = { x: O.x + (A.x - O.x) * s, y: O.y + (A.y - O.y) * s }; clamp(B); draw(); })), svg, read,
        V.h('p', { class: 'vis-note' }, 'Drag the arrow heads. Each arrow is a document vector (two terms shown).'));
    const clamp = p => { p.x = Math.min(W - 10, Math.max(O.x + 1, p.x)); p.y = Math.max(10, Math.min(O.y - 1, p.y)); };
    const pos = e => { const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; return pt.matrixTransform(svg.getScreenCTM().inverse()); };
    svg.addEventListener('pointerdown', e => { const p = pos(e); drag = Math.hypot(p.x - A.x, p.y - A.y) < Math.hypot(p.x - B.x, p.y - B.y) ? A : B; svg.setPointerCapture(e.pointerId); });
    svg.addEventListener('pointermove', e => { if (!drag) return; const p = pos(e); drag.x = p.x; drag.y = p.y; clamp(drag); draw(); });
    svg.addEventListener('pointerup', () => drag = null);
    function draw() {
        V.clear(svg); const rr = V.rng(2);
        V.arrow(svg, O.x, O.y, W - 5, O.y, '#cbd5e1', 1.2, rr); V.arrow(svg, O.x, O.y, O.x, 5, '#cbd5e1', 1.2, rr);
        V.text(svg, W - 40, O.y + 16, '"cat" count', { size: 12, color: '#64748b' }); V.text(svg, O.x + 8, 14, '"dog" count', { size: 12, color: '#64748b', anchor: 'start' });
        const a = [A.x - O.x, O.y - A.y], b = [B.x - O.x, O.y - B.y], c = (a[0] * b[0] + a[1] * b[1]) / Math.hypot(...a) / Math.hypot(...b), th = Math.acos(Math.min(1, Math.max(-1, c))) * 180 / Math.PI;
        const r = 50, aa = Math.atan2(-a[1], a[0]), ab = Math.atan2(-b[1], b[0]);
        svg.append(V.s('path', { d: `M${O.x + r * Math.cos(aa)} ${O.y + r * Math.sin(aa)} A${r} ${r} 0 0 ${ab > aa ? 1 : 0} ${O.x + r * Math.cos(ab)} ${O.y + r * Math.sin(ab)}`, fill: 'none', stroke: V.C.amber, 'stroke-width': 2 }));
        V.arrow(svg, O.x, O.y, A.x, A.y, V.C.sky, 3, rr); V.arrow(svg, O.x, O.y, B.x, B.y, V.C.rose, 3, rr);
        [[A, 'A', V.C.sky], [B, 'B', V.C.rose]].forEach(([p, n, col]) => { svg.append(V.s('circle', { cx: p.x, cy: p.y, r: 9, fill: col, opacity: .25 })); V.text(svg, p.x + 14, p.y - 10, n, { size: 16, weight: 700, color: col }); });
        read.innerHTML = `θ = <b>${V.fmt(th, 1)}°</b> → cos θ = <b>${V.fmt(c, 3)}</b>. ${c > 0.95 ? 'Very similar documents (same direction) - length does not matter.' : c < 0.2 ? 'Nearly orthogonal: the documents share almost no terms.' : 'Partly similar.'}`;
    }
    draw();
};

ML_VIS.l4_embed3d = async function (root) {
    const W = { king: [1.3, 1.1, 0.2], queen: [1.3, -1.1, 0.2], man: [-0.2, 1.1, 0.3], woman: [-0.2, -1.1, 0.3], prince: [0.7, 0.9, -0.7], princess: [0.7, -0.9, -0.7], cat: [-0.9, 0.1, 1.5], dog: [-0.5, 0.6, 1.8], kitten: [-1.3, -0.4, 1.2], car: [-1.4, 0.2, -1.4], truck: [-1.7, 0.7, -1.2], bus: [-0.9, -0.3, -1.8] };
    const read = V.h('p', { class: 'vis-readout' }, 'Toy 3-D embedding (real ones have 100-300 dimensions). Similar words sit close together.');
    const ctl = V.h('div', { class: 'vis-row' });
    root.append(ctl, read);
    const S = await V.scene3d(root, { radius: 7.5, label: 'Word embedding space' });
    if (!S) return;
    const { THREE, scene } = S, group = { king: 0xd97706, queen: 0xd97706, prince: 0xd97706, princess: 0xd97706, man: 0x0284c7, woman: 0x0284c7, cat: 0x059669, dog: 0x059669, kitten: 0x059669, car: 0x7c3aed, truck: 0x7c3aed, bus: 0x7c3aed };
    V.axes3d(THREE, scene, ['', '', ''], 2);
    const geo = new THREE.SphereGeometry(0.09, 16, 12);
    Object.entries(W).forEach(([w, p]) => { const m = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: group[w] })); m.position.set(...p); scene.add(m); const l = V.label3d(THREE, w, '#' + group[w].toString(16).padStart(6, '0'), 0.3); l.position.set(p[0], p[1] + 0.25, p[2]); scene.add(l); });
    let extras = [];
    const clear = () => { extras.forEach(o => scene.remove(o)); extras = []; S.dirty(); };
    const add = o => { scene.add(o); extras.push(o); };
    ctl.append(V.btn('king - man + woman = ?', () => {
        clear(); const k = W.king, d = W.woman.map((v, i) => v - W.man[i]), t = k.map((v, i) => v + d[i]);
        add(V.line3d(THREE, [W.man, W.woman], 0x0284c7, true)); add(V.line3d(THREE, [k, t], 0xe11d48));
        const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.15), new THREE.MeshLambertMaterial({ color: 0xe11d48 })); m.position.set(...t); add(m);
        read.innerHTML = 'Take the "man → woman" direction (dashed blue) and apply it to <b>king</b> (red arrow): you land on <b>queen</b>. Directions in the space capture relationships.'; S.dirty();
    }, 'primary'), V.btn('Nearest words to "cat"', () => {
        clear(); const near = Object.entries(W).filter(([w]) => w !== 'cat').map(([w, p]) => [w, Math.hypot(...p.map((v, i) => v - W.cat[i]))]).sort((a, b) => a[1] - b[1]).slice(0, 3);
        near.forEach(([w]) => add(V.line3d(THREE, [W.cat, W[w]], 0x059669, true)));
        read.innerHTML = `Closest to <b>cat</b>: ${near.map(([w, d]) => `${w} (${V.fmt(d, 2)})`).join(', ')}. With one-hot vectors, cat would be exactly as far from car as from dog.`; S.dirty();
    }), V.btn('Clear', () => { clear(); read.textContent = 'Toy 3-D embedding (real ones have 100-300 dimensions). Similar words sit close together.'; }));
};

ML_VIS.l4_timestamp = function (root) {
    const zones = [['8', 'Malaysia (UTC+8)'], ['0', 'London (UTC+0)'], ['-5', 'New York (UTC-5)'], ['9', 'Tokyo (UTC+9)']];
    const dt = V.h('input', { type: 'datetime-local', step: 1, value: '2026-09-23T14:35:22', class: 'vis-input', 'aria-label': 'Local date and time' });
    const tz = V.h('select', { class: 'vis-input', 'aria-label': 'Time zone' }, zones.map(([v, l]) => V.h('option', { value: v }, l)));
    const out = V.h('div', { class: 'vis-featgrid' }), other = V.h('p', { class: 'vis-readout' });
    dt.addEventListener('input', draw); tz.addEventListener('change', draw);
    root.append(V.h('div', { class: 'vis-row' }, V.h('label', { class: 'vis-slider' }, V.h('span', {}, 'local time'), dt), V.h('label', { class: 'vis-slider' }, V.h('span', {}, 'recorded in'), tz)), out, other);
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    function draw() {
        const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?/.exec(dt.value); if (!m) return;
        const [y, mo, d, h, mi, s] = m.slice(1).map(x => Number(x || 0)), off = Number(tz.value);
        const localMs = Date.UTC(y, mo - 1, d, h, mi, s), utcMs = localMs - off * 3600e3, L = new Date(localMs), U = new Date(utcMs);
        const dow = (L.getUTCDay() + 6) % 7, doy = Math.floor((localMs - Date.UTC(y, 0, 1)) / 864e5) + 1;
        const tod = h < 12 ? 'Morning' : h < 18 ? 'Afternoon' : 'Night', ordinal = Math.floor(localMs / 864e5) + 719163;
        const f = [['year', y], ['month', mo], ['day', d], ['hour', h], ['quarter', 'Q' + Math.ceil(mo / 3)], ['day_of_year', doy], ['dayofweek', `${dow} (${days[dow]})`], ['is_weekend', dow > 4], ['time_of_day', tod],
            ['UTC', U.toISOString().replace('T', ' ').slice(0, 19)], ['epoch seconds', Math.floor(utcMs / 1000).toLocaleString()], ['Gregorian ordinal', ordinal.toLocaleString()]];
        V.clear(out); f.forEach(([k, v], i) => out.append(V.h('div', { class: 'vis-feat' + (i >= 9 ? ' alt' : '') }, V.h('small', {}, k), V.h('b', {}, String(v)))));
        other.innerHTML = 'The <b>same instant</b> elsewhere: ' + zones.filter(([v]) => v !== tz.value).map(([v, l]) => { const t = new Date(utcMs + Number(v) * 3600e3); return `${l.split(' (')[0]} ${t.toISOString().slice(11, 16)}`; }).join(' · ') + '. Without converting to UTC first, a model would think these happened hours apart.';
    }
    draw();
};

ML_VIS.l4_rgb3d = async function (root) {
    const art = ['..yyyy..', '.yyyyyy.', 'yybyybyy', 'yyyyyyyy', 'yryyyyry', 'yyrrrryy', '.yyyyyy.', '..yyyy..'];
    const pal = { '.': [0.55, 0.78, 0.98], y: [0.99, 0.8, 0.15], b: [0.1, 0.1, 0.18], r: [0.85, 0.15, 0.25] };
    const px = art.map(row => [...row].map(c => pal[c]));
    let mode = 'rgb';
    const read = V.h('p', { class: 'vis-readout' });
    root.append(V.seg([['rgb', 'Colour image'], ['split', 'Split channels (8, 8, 3)'], ['gray', 'Grayscale (8, 8)'], ['flat', 'Flatten (64,)']], mode, v => { mode = v; layout(); }), read);
    const S = await V.scene3d(root, { radius: 9, phi: 1.15, theta: 1.2, autoRotate: false, label: 'Image as stacked channel matrices' });
    if (!S) return;
    const { THREE, scene } = S, g = new THREE.PlaneGeometry(0.36, 0.36), tiles = [];
    // Layers: 0 colour, 1-3 R/G/B, 4 gray
    for (let L = 0; L < 5; L++) for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
        const [r, gg, b] = px[i][j], Y = 0.2125 * r + 0.7154 * gg + 0.0721 * b;
        const col = L === 0 ? [r, gg, b] : L === 1 ? [r, 0, 0] : L === 2 ? [0, gg, 0] : L === 3 ? [0, 0, b] : [Y, Y, Y];
        const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: new THREE.Color(...col), side: THREE.DoubleSide, transparent: true }));
        m.userData = { L, i, j }; m.position.set(0, 0, 0); scene.add(m); tiles.push(m);
    }
    const labels = ['', 'R', 'G', 'B'].map((t, k) => { if (!t) return null; const s = V.label3d(THREE, t, ['#000', '#dc2626', '#16a34a', '#2563eb'][k], 0.5); scene.add(s); return s; });
    const goal = new Map();
    function layout() {
        tiles.forEach(m => {
            const { L, i, j } = m.userData, x = (j - 3.5) * 0.4, y = (3.5 - i) * 0.4; let p = [0, 0, 0], vis = false, rot = 0;
            if (mode === 'rgb') { vis = L === 0; p = [x, y, 0]; }
            if (mode === 'split') { vis = L >= 1 && L <= 3; p = [x, y, (L - 2) * 1.3]; }
            if (mode === 'gray') { vis = L === 4; p = [x, y, 0]; }
            if (mode === 'flat') { vis = L === 4; const k = i * 8 + j; p = [(k % 16 - 7.5) * 0.4, 1.4 - Math.floor(k / 16) * 0.95, 0]; }
            goal.set(m, { p, vis, rot });
        });
        labels.forEach((s, k) => { if (s) { s.visible = mode === 'split'; s.position.set(-1.9, 1.6, (k - 2) * 1.3); } });
        read.innerHTML = { rgb: 'One colour image = an <b>(m, n, 3)</b> array: every pixel has red, green and blue intensities.', split: 'The same image as <b>three stacked matrices</b> - the R, G and B channels (drag to look from the side).', gray: 'Grayscale: <b>Y = 0.2125R + 0.7154G + 0.0721B</b> collapses three channels into one <b>(m, n)</b> matrix.', flat: 'Flattening: the 8 x 8 matrix becomes one row of <b>64</b> numbers - a feature vector, but neighbours are no longer next to each other.' }[mode];
        S.dirty();
    }
    S.onFrame(() => {
        let moving = false;
        goal.forEach((gl, m) => {
            m.visible = gl.vis || m.material.opacity > 0.05;
            const t = m.position; const dx = gl.p[0] - t.x, dy = gl.p[1] - t.y, dz = gl.p[2] - t.z;
            if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) > 0.002) { t.x += dx * 0.15; t.y += dy * 0.15; t.z += dz * 0.15; moving = true; }
            const op = gl.vis ? 1 : 0; if (Math.abs(m.material.opacity - op) > 0.02) { m.material.opacity += (op - m.material.opacity) * 0.2; moving = true; }
        });
        return moving;
    });
    layout();
};

ML_VIS.l4_edges = function (root) {
    const N = 48, img = [];
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        let v = 0.15 + 0.03 * Math.sin(x * 0.7) * Math.cos(y * 0.5);
        if ((x - 15) ** 2 + (y - 16) ** 2 < 90) v = 0.9;
        if (x > 26 && x < 42 && y > 24 && y < 40) v = 0.65;
        img.push(v);
    }
    const at = (x, y) => img[Math.min(N - 1, Math.max(0, y)) * N + Math.min(N - 1, Math.max(0, x))];
    const gx = [], gy = [], mag = [];
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const a = -at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1) + at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1);
        const b = -at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1) + at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1);
        gx.push(a); gy.push(b); mag.push(Math.hypot(a, b));
    }
    let thr = 0.6;
    const S = 4, mk = () => { const c = V.h('canvas', { width: N * S, height: N * S, class: 'vis-pix' }); return c; };
    const c1 = mk(), c2 = mk(), c3 = mk();
    root.append(V.slider('edge threshold', 0.1, 2, 0.05, thr, v => { thr = v; draw(); }),
        V.h('div', { class: 'vis-trio' }, V.h('figure', {}, c1, V.h('figcaption', {}, 'pixels (grayscale)')), V.h('figure', {}, c2, V.h('figcaption', {}, 'edges: sharp intensity change')), V.h('figure', {}, c3, V.h('figcaption', {}, 'HOG idea: dominant edge direction per 8x8 cell'))),
        V.h('p', { class: 'vis-readout' }, 'Edges keep shapes and boundaries and ignore flat regions. HOG goes one step further and records which way the edges point in each cell - a compact description of shape.'));
    function draw() {
        const paint = (c, f) => { const ctx = c.getContext('2d'); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const v = Math.round(255 * f(y * N + x)); ctx.fillStyle = `rgb(${v},${v},${v})`; ctx.fillRect(x * S, y * S, S, S); } return ctx; };
        paint(c1, i => img[i]);
        paint(c2, i => mag[i] > thr ? 1 : 0);
        const ctx = paint(c3, () => 0.97);
        ctx.strokeStyle = '#e2e8f0'; for (let k = 0; k <= N; k += 8) { ctx.beginPath(); ctx.moveTo(k * S, 0); ctx.lineTo(k * S, N * S); ctx.moveTo(0, k * S); ctx.lineTo(N * S, k * S); ctx.stroke(); }
        for (let cy = 0; cy < N; cy += 8) for (let cx = 0; cx < N; cx += 8) {
            const bins = Array(8).fill(0);
            for (let y = cy; y < cy + 8; y++) for (let x = cx; x < cx + 8; x++) { const i = y * N + x; if (mag[i] < thr * 0.5) continue; let a = Math.atan2(gy[i], gx[i]) + Math.PI / 2; a = ((a % Math.PI) + Math.PI) % Math.PI; bins[Math.floor(a / Math.PI * 8) % 8] += mag[i]; }
            bins.forEach((w, b) => {
                if (w < 0.5) return; const a = (b + 0.5) / 8 * Math.PI, L = Math.min(16, 4 + w * 1.2), mx = (cx + 4) * S, my = (cy + 4) * S;
                ctx.strokeStyle = '#e11d48'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(mx - Math.cos(a) * L, my - Math.sin(a) * L); ctx.lineTo(mx + Math.cos(a) * L, my + Math.sin(a) * L); ctx.stroke();
            });
        }
    }
    draw();
};
// Lecture 5 visuals: feature scaling and feature selection

// Three scalers on the lecture's video-view counts
ML_VIS.l5_scalers = function (root) {
    const base = [1, 5, 25, 300, 1295];
    let viral = 19000, mode = 'all', prev = null;
    const svg = V.svg(620, 330), read = V.h('p', { class: 'vis-readout' }), stats = V.h('div', { class: 'vis-table-wrap' });
    const ticks = [1000, 2500, 5000, 10000, 19000, 40000, 100000, 250000];
    root.append(
        V.h('div', { class: 'vis-row' },
            V.slider('Views of the viral video', 0, ticks.length - 1, 1, 4, v => { viral = ticks[v]; draw(); }, v => ticks[v].toLocaleString('en-US')),
            V.seg([['all', 'All three'], ['std', 'StandardScaler'], ['mm', 'MinMaxScaler'], ['rob', 'RobustScaler']], mode, v => { mode = v; draw(); })),
        svg, read, stats,
        V.h('p', { class: 'vis-note' }, 'Statistics are computed live (population standard deviation and linear-interpolated quartiles, as scikit-learn does).'));
    const scalers = {
        std: { name: 'StandardScaler', color: V.C.sky, f: (x, a) => (x - V.mean(a)) / V.std(a), ref: a => `mean ${V.fmt(V.mean(a), 1)}, SD ${V.fmt(V.std(a), 1)}` },
        mm: { name: 'MinMaxScaler', color: V.C.amber, f: (x, a) => (x - Math.min(...a)) / (Math.max(...a) - Math.min(...a)), ref: a => `min ${V.fmt(Math.min(...a))}, max ${V.fmt(Math.max(...a))}` },
        rob: { name: 'RobustScaler', color: V.C.green, f: (x, a) => (x - V.median(a)) / (V.quantile(a, .75) - V.quantile(a, .25)), ref: a => `median ${V.fmt(V.median(a), 1)}, IQR ${V.fmt(V.quantile(a, .75) - V.quantile(a, .25), 1)}` },
    };
    function rows() { return mode === 'all' ? ['std', 'mm', 'rob'] : [mode]; }
    function draw() {
        const data = base.concat([viral]);
        const next = {};
        rows().forEach(k => { next[k] = data.map(x => scalers[k].f(x, data)); });
        const from = prev && prev.mode === mode ? prev.vals : null;
        prev = { mode, vals: next };
        V.tween(from ? 500 : 1, t => paint(data, from ? lerp(from, next, t) : next));
        read.innerHTML = rows().map(k => {
            const v = next[k], normal = v.slice(0, 5);
            return `<b>${scalers[k].name}</b>: the five ordinary videos spread over <b>${V.fmt(Math.max(...normal) - Math.min(...normal), 3)}</b> scaled units (from ${V.fmt(Math.min(...normal), 2)} to ${V.fmt(Math.max(...normal), 2)}); the viral video lands at <b>${V.fmt(v[5], 2)}</b>.`;
        }).join('<br>') + (mode === 'all' ? '<br>All three rows share one axis from -1 to 2. MinMax crushes the ordinary videos into a sliver; RobustScaler keeps them readable - and every scaler still keeps the outlier.' : '');
        stats.innerHTML = '<table class="vis-table"><tr><th>Views</th>' + rows().map(k => `<th>${scalers[k].name}</th>`).join('') + '</tr>' +
            data.map((x, i) => `<tr${i === 5 ? ' class="ref"' : ''}><td>${x.toLocaleString('en-US')}</td>` + rows().map(k => `<td>${V.fmt(next[k][i], 4)}</td>`).join('') + '</tr>').join('') +
            '<tr><td><i>learned from training data</i></td>' + rows().map(k => `<td><i>${scalers[k].ref(data)}</i></td>`).join('') + '</tr></table>';
    }
    function lerp(a, b, t) { const o = {}; for (const k in b) o[k] = b[k].map((v, i) => a[k] ? a[k][i] + (v - a[k][i]) * t : v); return o; }
    function paint(data, vals) {
        V.clear(svg); const rr = V.rng(7), ks = rows(), rowH = ks.length === 1 ? 220 : 100;
        const lo = -1, hi = 2, sx = x => 40 + (Math.max(lo, Math.min(hi, x)) - lo) / (hi - lo) * 520;
        ks.forEach((k, r) => {
            const v = vals[k], y = 52 + r * rowH + (ks.length === 1 ? 60 : 0);
            V.text(svg, 40, y - 30, scalers[k].name, { size: 15, weight: 700, color: scalers[k].color, anchor: 'start' });
            V.rline(svg, 40, y, 560, y, '#94a3b8', 1.4, rr);
            [-1, -0.5, 0, 0.5, 1, 1.5, 2].forEach(t => {
                svg.append(V.s('line', { x1: sx(t), x2: sx(t), y1: y - 4, y2: y + 4, stroke: '#94a3b8' }));
                V.text(svg, sx(t), y + 18, String(t), { size: 11, color: '#64748b' });
            });
            const nv = v.slice(0, 5), a = sx(Math.min(...nv)), b = sx(Math.max(...nv));
            svg.append(V.s('rect', { x: a - 9, y: y - 17, width: Math.max(18, b - a + 18), height: 30, rx: 9, fill: scalers[k].color, opacity: .1 }));
            v.forEach((val, i) => {
                if (i === 5) return;
                const c = V.s('circle', { cx: sx(val), cy: y - 3 + (i % 2 ? -5 : 5), r: 5.5, fill: scalers[k].color, stroke: '#fff', 'stroke-width': 1.3 });
                c.append(V.s('title', {}, `${data[i].toLocaleString('en-US')} views → ${V.fmt(val, 4)}`));
                svg.append(c);
            });
            const vx = v[5], off = vx > hi;
            const c = V.s('circle', { cx: sx(vx), cy: y - 3, r: 8, fill: V.C.rose, stroke: '#fff', 'stroke-width': 1.5 });
            c.append(V.s('title', {}, `${data[5].toLocaleString('en-US')} views → ${V.fmt(vx, 4)}`)); svg.append(c);
            if (off) { V.arrow(svg, 566, y - 3, 600, y - 3, V.C.rose, 2, rr); V.text(svg, 600, y - 20, 'viral → ' + V.fmt(vx, 2), { size: 12, color: V.C.rose, anchor: 'end', weight: 700 }); }
            V.text(svg, (a + b) / 2, y + 34, `ordinary videos: width ${V.fmt(Math.max(...nv) - Math.min(...nv), 2)}`, { size: 11, color: scalers[k].color });
        });
    }
    draw();
};

// Raw units versus standardized units in 3D: the nearest neighbour changes
ML_VIS.l5_space3d = async function (root) {
    const r = V.rng(17), people = [];
    for (let i = 0; i < 34; i++) {
        const age = 22 + r() * 43, exp = Math.max(0, Math.min(40, age - 22 - r() * 6)), sal = 30000 + exp * 2800 + V.gauss(r) * 12000;
        people.push({ age, exp, sal, name: 'customer ' + (i + 1) });
    }
    const Q = { age: 25, sal: 80000, exp: 3, name: 'new applicant' };
    people.push({ age: 60, sal: 80500, exp: 35, name: 'A: age 60, RM80,500' }, { age: 26, sal: 86000, exp: 4, name: 'B: age 26, RM86,000' });
    const all = people.concat([Q]);
    const stat = k => ({ m: V.mean(all.map(p => p[k])), s: V.std(all.map(p => p[k])) });
    const S = { age: stat('age'), exp: stat('exp'), sal: stat('sal') };
    let mode = 'raw';
    const read = V.h('p', { class: 'vis-readout' });
    root.append(V.seg([['raw', 'Raw units (RM and years)'], ['z', 'Standardized (z-scores)']], mode, v => { mode = v; morph(); }), read);
    const sc = await V.scene3d(root, { radius: 7.5, phi: 1.05, theta: 0.75, label: 'Customers in raw and standardized space' });
    if (!sc) return;
    const { THREE, scene } = sc;
    V.axes3d(THREE, scene, ['age', 'salary', 'experience'], 2.4);
    const pos = (p, m) => m === 'raw'
        ? [(p.age - S.age.m) / S.sal.s * 1.0, (p.sal - S.sal.m) / S.sal.s * 1.0, (p.exp - S.exp.m) / S.sal.s * 1.0]
        : [(p.age - S.age.m) / S.age.s * 0.9, (p.sal - S.sal.m) / S.sal.s * 0.9, (p.exp - S.exp.m) / S.exp.s * 0.9];
    const geo = new THREE.SphereGeometry(0.07, 14, 10);
    const meshes = all.map((p, i) => {
        const col = p === Q ? 0xd97706 : i === all.length - 3 ? 0xe11d48 : i === all.length - 2 ? 0x059669 : 0x0284c7;
        const m = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: col }));
        if (p === Q || i >= all.length - 3) m.scale.setScalar(1.7);
        m.position.set(...pos(p, 'raw')); scene.add(m); return m;
    });
    let link = null;
    const dist = (a, b, m) => m === 'raw' ? Math.hypot(a.age - b.age, a.sal - b.sal, a.exp - b.exp)
        : Math.hypot((a.age - b.age) / S.age.s, (a.sal - b.sal) / S.sal.s, (a.exp - b.exp) / S.exp.s);
    function nearest(m) { return people.reduce((b, p) => (!b || dist(p, Q, m) < dist(b, Q, m) ? p : b), null); }
    function morph() {
        const from = meshes.map(o => o.position.clone()), to = all.map(p => new THREE.Vector3(...pos(p, mode)));
        V.tween(900, t => { meshes.forEach((o, i) => o.position.lerpVectors(from[i], to[i], t)); if (link) scene.remove(link); link = null; sc.dirty(); }, () => {
            const nn = nearest(mode), i = all.indexOf(nn);
            link = V.line3d(THREE, [meshes[all.length - 1].position.toArray(), meshes[i].position.toArray()], 0xd97706, true); scene.add(link); sc.dirty();
            const dA = nn.age - Q.age, dS = nn.sal - Q.sal, dE = nn.exp - Q.exp;
            read.innerHTML = (mode === 'raw'
                ? 'In raw units the salary axis is thousands of times longer than age or experience, so the cloud collapses onto the salary axis. '
                : 'After standardizing, each axis is measured in its own standard deviations, so all three features shape the distance. ')
                + `Nearest neighbour of the new applicant (age 25, RM80,000, 3 years): <b>${nn.name}</b> (Δage ${V.fmt(dA, 0)}, Δsalary RM${V.fmt(dS, 0)}, Δexperience ${V.fmt(dE, 0)}).`;
        });
    }
    morph();
};

// Curse of dimensionality: edge length needed to capture 10% of uniformly spread data
ML_VIS.l5_curse3d = async function (root) {
    let dim = 3, frac = 0.1;
    const read = V.h('p', { class: 'vis-readout' }), chart = V.svg(560, 200);
    root.append(V.h('div', { class: 'vis-row' },
        V.seg([[1, '1 feature'], [2, '2 features'], [3, '3 features']], dim, v => { dim = v; update(); }),
        V.slider('Share of data to capture', 1, 50, 1, 10, v => { frac = v / 100; update(); }, v => v + '%')), read);
    const sc = await V.scene3d(root, { radius: 6, phi: 1.1, theta: 0.7, target: [0.5, 0.5, 0.5], label: 'Unit cube with a neighbourhood box' });
    root.append(chart, V.h('p', { class: 'vis-note' }, 'Edge length of a neighbourhood that holds the chosen share of uniformly spread data: share^(1/d). With 10 features you need 79% of every feature\'s range just to find the nearest 10% of observations.'));
    const r = V.rng(9), pts = Array.from({ length: 400 }, () => [r(), r(), r()]);
    let box = null, cloud = null, edge = 0, shown = 0;
    if (sc) {
        const { THREE, scene } = sc;
        const g = new THREE.BoxGeometry(1, 1, 1), e = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color: 0x94a3b8 }));
        e.position.set(.5, .5, .5); scene.add(e);
        box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshLambertMaterial({ color: 0x0d9488, transparent: true, opacity: .25 }));
        scene.add(box);
        const geo = new THREE.BufferGeometry(), colors = new Float32Array(pts.length * 3);
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts.length * 3), 3));
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        cloud = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.045, vertexColors: true }));
        scene.add(cloud);
    }
    function place(k) {
        if (!sc) return;
        const p = cloud.geometry.attributes.position, c = cloud.geometry.attributes.color;
        pts.forEach((q, i) => {
            const v = [q[0], dim > 1 ? q[1] : 0, dim > 2 ? q[2] : 0];
            p.setXYZ(i, v[0], v[1], v[2]);
            const inBox = isIn(q);
            c.setXYZ(i, inBox ? 0.05 : 0.6, inBox ? 0.58 : 0.65, inBox ? 0.53 : 0.72);
        });
        p.needsUpdate = c.needsUpdate = true;
        const s = shown + (edge - shown) * k;
        box.scale.set(s, dim > 1 ? s : 0.02, dim > 2 ? s : 0.02);
        box.position.set(s / 2, dim > 1 ? s / 2 : 0, dim > 2 ? s / 2 : 0);
        sc.dirty();
    }
    function isIn(q) { return q[0] < edge && (dim < 2 || q[1] < edge) && (dim < 3 || q[2] < edge); }
    function update() {
        edge = frac ** (1 / dim);
        const inside = pts.filter(isIn).length;
        V.tween(600, place, () => { shown = edge; });
        read.innerHTML = `With <b>${dim}</b> feature${dim > 1 ? 's' : ''}, a corner box must stretch <b>${V.fmt(100 * edge, 1)}%</b> of every feature's range to hold ${V.fmt(100 * frac, 0)}% of the data (${inside} of ${pts.length} points here). Neighbourhoods stop being local as dimensions grow.`;
        drawChart();
    }
    function drawChart() {
        V.clear(chart); const rr = V.rng(3), sx = d => 50 + (d - 1) / 19 * 480, sy = e => 170 - e * 150;
        V.arrow(chart, 50, 170, 540, 170, '#94a3b8', 1.2, rr); V.arrow(chart, 50, 170, 50, 12, '#94a3b8', 1.2, rr);
        V.text(chart, 300, 192, 'number of features d →', { size: 13, color: '#64748b' });
        V.text(chart, 30, 20, 'edge', { size: 12, color: '#64748b' });
        [0.5, 1].forEach(e => V.text(chart, 40, sy(e), String(e), { size: 11, color: '#64748b', anchor: 'end' }));
        let d = ''; for (let k = 1; k <= 20; k++) d += (k > 1 ? 'L' : 'M') + sx(k) + ' ' + sy(frac ** (1 / k));
        chart.append(V.s('path', { d, fill: 'none', stroke: '#0d9488', 'stroke-width': 3 }));
        for (let k = 1; k <= 20; k++) {
            const c = V.s('circle', { cx: sx(k), cy: sy(frac ** (1 / k)), r: k === dim ? 6 : 3, fill: k === dim ? '#d97706' : '#0d9488' });
            c.append(V.s('title', {}, `${k} features: edge ${V.fmt(frac ** (1 / k), 3)}`)); chart.append(c);
        }
    }
    update();
};

// VarianceThreshold: variance is not predictive relevance
ML_VIS.l5_variance = function (root) {
    const feats = [
        ['Gen 1', 0.1644, 0.10], ['Gen 2', 0.1149, 0.05], ['Gen 3', 0.16, 0.12], ['Gen 4', 0.1282, 0.08],
        ['Gen 5', 0.1637, 0.06], ['Gen 6', 0.0919, 0.04], ['Constant column', 0, 0], ['Rare fraud flag (1% of rows)', 0.0099, 0.62],
    ];
    let thr = 0.15, showRel = false;
    const svg = V.svg(600, 230), read = V.h('p', { class: 'vis-readout' });
    root.append(V.h('div', { class: 'vis-row' },
        V.slider('Variance threshold', 0, 25, 1, 15, v => { thr = v / 100; draw(); }, v => (v / 100).toFixed(2)),
        V.seg([[false, 'Variance only'], [true, '+ association with the target']], showRel, v => { showRel = v; draw(); })), svg, read,
        V.h('p', { class: 'vis-note' }, 'Gen 1-6 use the lecture\'s variances (threshold 0.15 keeps Gen 1, 3 and 5). The two extra rows and the association scores are illustrative.'));
    function draw() {
        V.clear(svg);
        V.hbars(svg, feats.map(([n, v, a]) => ({ label: n, value: v, color: v > thr ? '#0d9488' : '#cbd5e1', dim: v <= thr, text: 'var ' + v.toFixed(4) + (v <= thr ? '  removed' : ''), tip: `${n}: variance ${v}` })),
            { x0: 210, w: showRel ? 170 : 330, max: 0.2, rowH: 26 });
        const x = 210 + (showRel ? 170 : 330) * thr / 0.2;
        svg.append(V.s('line', { x1: x, x2: x, y1: 6, y2: 222, stroke: V.C.rose, 'stroke-width': 2, 'stroke-dasharray': '5 4' }));
        V.text(svg, x + 4, 222, 'threshold', { size: 11, color: V.C.rose, anchor: 'start' });
        if (showRel) {
            V.text(svg, 520, 6 + 4, 'link to target', { size: 11, color: '#64748b' });
            feats.forEach(([n, v, a], k) => {
                const y = 10 + k * 26;
                svg.append(V.s('rect', { x: 470, y: y + 3, width: 110, height: 18, rx: 4, fill: '#f1f5f9' }));
                svg.append(V.s('rect', { x: 470, y: y + 3, width: Math.max(2, 110 * a), height: 18, rx: 4, fill: a > 0.5 ? V.C.amber : '#fcd34d' }));
            });
        }
        const kept = feats.filter(f => f[1] > thr).map(f => f[0]);
        read.innerHTML = `Kept (variance &gt; ${thr.toFixed(2)}): <b>${kept.join(', ') || 'nothing'}</b>. ` + (showRel
            ? 'The rare fraud flag has the lowest variance but the strongest link to the target - VarianceThreshold never looks at the target, so it throws it away.'
            : 'VarianceThreshold is unsupervised: it measures how much a column varies, never whether that variation predicts y.');
    }
    draw();
};

// SelectKBest with chi-square on the real Wisconsin data, plus accuracy versus k
ML_VIS.l5_filter = function (root) {
    const D = L5_DATA, order = D.chi2.map((v, i) => i).sort((a, b) => D.chi2[b] - D.chi2[a]);
    let k = 15;
    const bars = V.svg(600, 30 * 18 + 16), chart = V.svg(600, 220), read = V.h('p', { class: 'vis-readout' });
    root.append(V.slider('k (features to keep)', 1, 30, 1, k, v => { k = v; draw(); }), read, chart, bars,
        V.h('p', { class: 'vis-note' }, 'Real scikit-learn numbers on the Wisconsin breast-cancer data (569 samples, 30 features). Bars use a log scale because chi-square scores range from almost 0 to about 112,600. Accuracy: 5-fold cross-validation with MinMaxScaler + SelectKBest inside each fold - the lecture reports 95.09% (30) and 95.26% (15) with a different setup.'));
    function draw() {
        V.clear(bars); V.clear(chart);
        const lg = v => Math.log10(v + 1e-4) + 4;
        V.hbars(bars, order.map((i, r) => ({ label: D.names[i], value: lg(D.chi2[i]), color: r < k ? '#0d9488' : '#cbd5e1', dim: r >= k, text: V.fmt(D.chi2[i], 2), tip: `${D.names[i]}: chi-square ${D.chi2[i]}` })),
            { x0: 190, w: 330, max: lg(Math.max(...D.chi2)), rowH: 18, size: 11 });
        const rr = V.rng(2), sx = j => 50 + (j - 1) / 29 * 520, lo = Math.min(...D.accByK) - 0.5, hi = Math.max(...D.accByK) + 0.5, sy = a => 180 - (a - lo) / (hi - lo) * 160;
        V.arrow(chart, 50, 180, 585, 180, '#94a3b8', 1.2, rr); V.arrow(chart, 50, 180, 50, 10, '#94a3b8', 1.2, rr);
        V.text(chart, 320, 205, 'k = number of selected features →', { size: 13, color: '#64748b' });
        V.text(chart, 56, 14, 'CV accuracy %', { size: 12, color: '#64748b', anchor: 'start' });
        let d = ''; D.accByK.forEach((a, j) => { d += (j ? 'L' : 'M') + sx(j + 1) + ' ' + sy(a); });
        chart.append(V.s('path', { d, fill: 'none', stroke: '#0d9488', 'stroke-width': 2.5 }));
        D.accByK.forEach((a, j) => { const c = V.s('circle', { cx: sx(j + 1), cy: sy(a), r: j + 1 === k ? 6 : 2.5, fill: j + 1 === k ? '#d97706' : '#0d9488' }); c.append(V.s('title', {}, `k = ${j + 1}: ${a}%`)); chart.append(c); });
        [Math.ceil(lo), Math.floor(hi)].forEach(a => V.text(chart, 44, sy(a), String(a), { size: 11, color: '#64748b', anchor: 'end' }));
        const best = D.accByK.indexOf(Math.max(...D.accByK)) + 1;
        read.innerHTML = `Keeping the top <b>${k}</b> chi-square features gives <b>${D.accByK[k - 1]}%</b> cross-validated accuracy (all 30: ${D.accByK[29]}%, best here: k = ${best}, ${D.accByK[best - 1]}%). ` +
            'Small differences like these are within normal evaluation noise, which is exactly why a 0.17-point gain is not proof that fewer features generalize better.';
    }
    draw();
};

// Recursive feature elimination, animated, then compared with chi-square
ML_VIS.l5_rfe = function (root) {
    const D = L5_DATA, n = 30;
    let remain = 30, timer = null;
    const grid = V.h('div', { class: 'vis-chipgrid' }), read = V.h('p', { class: 'vis-readout' });
    const chips = D.names.map((nm, i) => { const c = V.h('span', { class: 'vis-chip', title: nm }, nm); grid.append(c); return c; });
    const sl = V.slider('Features remaining', 1, 30, 1, 30, v => { stop(); remain = v; draw(); });
    const play = V.btn('Run RFE (30 → 15)', () => {
        if (timer) { stop(); return; }
        if (remain <= 15) remain = 30;
        play.textContent = 'Pause';
        timer = setInterval(() => { remain--; sl.input.value = remain; sl.querySelector('output').textContent = remain; draw(); if (remain <= 15) stop(); }, 450);
    });
    function stop() { clearInterval(timer); timer = null; play.textContent = 'Run RFE (30 → 15)'; }
    let cmp = false;
    root.append(V.h('div', { class: 'vis-row' }, sl, play, V.seg([[false, 'RFE only'], [true, 'Compare with chi-square top 15']], cmp, v => { cmp = v; draw(); })), grid, read,
        V.h('p', { class: 'vis-note' }, 'Real elimination order from RFE(LogisticRegression, step=1) on standardized Wisconsin data. Each step refits the model and drops the lowest-ranked feature.'));
    function draw() {
        const dropped = new Set(D.rfeOrder.slice(0, n - remain)), last = D.rfeOrder[n - remain - 1], chi = new Set(D.chi15);
        chips.forEach((c, i) => {
            c.className = 'vis-chip';
            if (dropped.has(i)) c.classList.add('off');
            if (i === last) c.classList.add('just');
            if (cmp && remain === 15) { if (!dropped.has(i) && chi.has(i)) c.classList.add('both'); else if (!dropped.has(i)) c.classList.add('rfe'); else if (chi.has(i)) c.classList.add('chi'); }
        });
        const both = D.rfe15.filter(i => chi.has(i)).length;
        read.innerHTML = remain === 30 ? 'Step 0: fit Logistic Regression on all 30 features and rank them by coefficient size.'
            : `<b>${remain}</b> features left. Last removed: <b>${D.names[last]}</b> (the weakest after refitting on ${remain + 1} features).` +
            (remain === 15 ? (cmp ? ` Green = chosen by both, violet = RFE only, outlined = chi-square only: they agree on <b>${both} of 15</b> (the lecture's run agreed on 8). They define "useful" differently.` : ' Switch on the comparison to see the overlap with chi-square.') : '');
    }
    draw();
};

// Embedded selection: random forest importance
ML_VIS.l5_importance = function (root) {
    const D = L5_DATA, order = D.importance.map((v, i) => i).sort((a, b) => D.importance[b] - D.importance[a]);
    let view = 'top';
    const svg = V.svg(600, 260), read = V.h('p', { class: 'vis-readout' });
    root.append(V.seg([['top', 'Top 10 features'], ['family', 'Grouped: mean / error / worst']], view, v => { view = v; draw(); }), svg, read);
    function draw() {
        V.clear(svg);
        if (view === 'top') {
            V.hbars(svg, order.slice(0, 10).map((i, r) => ({ label: D.names[i], value: D.importance[i], color: r < 3 ? '#0d9488' : '#5eead4', text: V.fmt(D.importance[i], 3) })), { x0: 190, w: 330, rowH: 24 });
            read.innerHTML = `RandomForestClassifier(random_state=42).feature_importances_: the top three (${order.slice(0, 3).map(i => D.names[i]).join(', ')}) carry ${V.fmt(100 * order.slice(0, 3).reduce((s, i) => s + D.importance[i], 0), 0)}% of the total importance. High importance means "the trees split on it a lot", not "it causes cancer".`;
        } else {
            const fam = { mean: 0, error: 0, worst: 0 };
            D.names.forEach((nm, i) => { fam[nm.startsWith('mean') ? 'mean' : nm.includes('error') ? 'error' : 'worst'] += D.importance[i]; });
            V.hbars(svg, Object.entries(fam).map(([k, v]) => ({ label: k + ' features', value: v, color: '#0d9488', text: V.fmt(100 * v, 1) + '%' })), { x0: 190, w: 330, rowH: 40 });
            read.innerHTML = 'Each nucleus measurement appears three times (mean, standard error, worst). Correlated copies share importance between them - one reason impurity importance is not a ranking of real-world causes.';
        }
    }
    draw();
};
// Lecture 6 visuals: regression, logistic regression, decision trees and random forests

// Shared toy classification data: age (18-70) and monthly income (RM thousands, 1-15) -> buys or not
function l6Data(seed = 4, n = 46) {
    const r = V.rng(seed), pts = [];
    for (let i = 0; i < n; i++) {
        const x = 18 + r() * 52, y = 1 + r() * 14;
        let c = (y > 6 && x > 28) || (y > 11) ? 1 : 0;
        if (r() < 0.1) c = 1 - c;
        pts.push({ x, y, c });
    }
    return pts;
}
const L6X = [18, 70], L6Y = [1, 15];
function l6Gini(pts) { if (!pts.length) return 0; const p = pts.filter(q => q.c).length / pts.length; return 1 - p * p - (1 - p) * (1 - p); }
function l6Candidates(pts, f) { const v = [...new Set(pts.map(p => p[f]))].sort((a, b) => a - b); return v.slice(1).map((x, i) => (x + v[i]) / 2); }
function l6Best(pts, feats) {
    let best = null;
    feats.forEach(f => l6Candidates(pts, f).forEach(t => {
        const L = pts.filter(p => p[f] <= t), R = pts.filter(p => p[f] > t);
        if (!L.length || !R.length) return;
        const g = (L.length * l6Gini(L) + R.length * l6Gini(R)) / pts.length;
        if (!best || g < best.g - 1e-12) best = { f, t, g };
    }));
    return best;
}
function l6Grow(pts, depth, pick, minLeaf = 1) {
    const ones = pts.filter(p => p.c).length, node = { n: pts.length, ones, gini: l6Gini(pts), leaf: true, cls: ones * 2 >= pts.length ? 1 : 0 };
    if (depth === 0 || node.gini === 0 || pts.length < 2 * minLeaf) return node;
    const b = l6Best(pts, pick ? pick() : ['x', 'y']);
    if (!b || b.g >= node.gini - 1e-12) return node;
    Object.assign(node, { leaf: false, f: b.f, t: b.t });
    node.L = l6Grow(pts.filter(p => p[b.f] <= b.t), depth - 1, pick, minLeaf);
    node.R = l6Grow(pts.filter(p => p[b.f] > b.t), depth - 1, pick, minLeaf);
    return node;
}
function l6Predict(node, p) { while (!node.leaf) node = p[node.f] <= node.t ? node.L : node.R; return node.cls; }
function l6Scatter(svg, W, H, pad) {
    const sx = x => pad + (x - L6X[0]) / (L6X[1] - L6X[0]) * (W - pad - 10), sy = y => H - pad + 10 - (y - L6Y[0]) / (L6Y[1] - L6Y[0]) * (H - pad - 10);
    return { sx, sy, ix: px => L6X[0] + (px - pad) / (W - pad - 10) * (L6X[1] - L6X[0]), iy: py => L6Y[0] + (H - pad + 10 - py) / (H - pad - 10) * (L6Y[1] - L6Y[0]) };
}
function l6Points(svg, pts, sx, sy, r = 5) {
    pts.forEach(p => {
        const c = V.s(p.c ? 'circle' : 'rect', p.c ? { cx: sx(p.x), cy: sy(p.y), r, fill: V.C.violet, stroke: '#fff', 'stroke-width': 1.2 }
            : { x: sx(p.x) - r, y: sy(p.y) - r, width: 2 * r, height: 2 * r, fill: V.C.amber, stroke: '#fff', 'stroke-width': 1.2 });
        c.append(V.s('title', {}, `age ${V.fmt(p.x, 0)}, income RM${V.fmt(p.y, 1)}k: ${p.c ? 'buys' : 'does not buy'}`));
        svg.append(c);
    });
}
function l6Axes(svg, W, H, pad, rr) {
    V.arrow(svg, pad, H - pad + 10, W - 4, H - pad + 10, '#94a3b8', 1.2, rr); V.arrow(svg, pad, H - pad + 10, pad, 4, '#94a3b8', 1.2, rr);
    V.text(svg, W - 60, H - 8, 'age →', { size: 12, color: '#64748b' });
    V.text(svg, pad + 6, 12, 'income (RM k)', { size: 12, color: '#64748b', anchor: 'start' });
}

// Least squares: residuals drawn as squares
ML_VIS.l6_residuals = function (root) {
    const pts = [[1, 44], [2, 52], [3, 53], [4, 61], [5, 64], [6, 72], [7, 73], [8, 81]].map(([x, y]) => ({ x, y }));
    let m = 3, c = 50, cancel = null, drag = null;
    const W = 600, H = 330, svg = V.svg(W, H), read = V.h('p', { class: 'vis-readout' });
    const sx = x => 50 + x / 9 * 520, sy = y => 300 - (y - 30) / 70 * 280, iy = py => 30 + (300 - py) / 280 * 70;
    const sm = V.slider('slope m', 0, 100, 1, m * 10, v => { m = v / 10; draw(); }, v => (v / 10).toFixed(1));
    const scl = V.slider('intercept c', 0, 80, 1, c, v => { c = v; draw(); });
    const ols = () => { const mx = V.mean(pts.map(p => p.x)), my = V.mean(pts.map(p => p.y)); const b = pts.reduce((s, p) => s + (p.x - mx) * (p.y - my), 0) / pts.reduce((s, p) => s + (p.x - mx) ** 2, 0); return [b, my - b * mx]; };
    const sse = (mm, cc) => pts.reduce((s, p) => s + (p.y - (mm * p.x + cc)) ** 2, 0);
    root.append(V.h('div', { class: 'vis-row' }, sm, scl, V.btn('Fit by least squares', () => {
        const [bm, bc] = ols(), m0 = m, c0 = c; if (cancel) cancel();
        cancel = V.tween(1100, t => { m = m0 + (bm - m0) * t; c = c0 + (bc - c0) * t; sync(); draw(); });
    }, 'primary'), V.btn('Lecture line: 5h + 40', () => { m = 5; c = 40; sync(); draw(); })), svg, read,
        V.h('p', { class: 'vis-note' }, 'Drag any student up or down. Each square\'s area is that student\'s squared error; OLS chooses the line with the smallest total area.'));
    function sync() { sm.input.value = Math.round(m * 10); sm.querySelector('output').textContent = m.toFixed(1); scl.input.value = Math.round(c); scl.querySelector('output').textContent = Math.round(c); }
    svg.addEventListener('pointermove', e => { if (drag === null) return; const p = V.svgPoint(svg, e); pts[drag].y = Math.max(32, Math.min(99, iy(p.y))); draw(); });
    svg.addEventListener('pointerup', () => { drag = null; });
    svg.addEventListener('pointerleave', () => { drag = null; });
    function draw() {
        V.clear(svg); const rr = V.rng(8);
        V.arrow(svg, 50, 300, 585, 300, '#94a3b8', 1.2, rr); V.arrow(svg, 50, 300, 50, 12, '#94a3b8', 1.2, rr);
        V.text(svg, 540, 318, 'study hours →', { size: 12, color: '#64748b' }); V.text(svg, 56, 16, 'score', { size: 12, color: '#64748b', anchor: 'start' });
        for (let h = 1; h <= 8; h++) V.text(svg, sx(h), 312, String(h), { size: 11, color: '#94a3b8' });
        pts.forEach(p => {
            const yh = m * p.x + c, e = p.y - yh, side = Math.abs(sy(p.y) - sy(yh));
            svg.append(V.s('rect', { x: sx(p.x), y: Math.min(sy(p.y), sy(yh)), width: side, height: side, fill: e >= 0 ? 'rgba(124,58,237,.16)' : 'rgba(225,29,72,.14)', stroke: e >= 0 ? V.C.violet : V.C.rose, 'stroke-width': 1 }));
            svg.append(V.s('line', { x1: sx(p.x), x2: sx(p.x), y1: sy(p.y), y2: sy(yh), stroke: e >= 0 ? V.C.violet : V.C.rose, 'stroke-width': 2 }));
        });
        svg.append(V.s('line', { x1: sx(0), y1: sy(c), x2: sx(9), y2: sy(m * 9 + c), stroke: V.C.ink, 'stroke-width': 2.5 }));
        pts.forEach((p, i) => {
            const dot = V.s('circle', { cx: sx(p.x), cy: sy(p.y), r: 7, fill: V.C.sky, stroke: '#fff', 'stroke-width': 2, style: 'cursor:ns-resize' });
            dot.append(V.s('title', {}, `${p.x} h, score ${V.fmt(p.y, 1)}, residual ${V.fmt(p.y - (m * p.x + c), 1)}`));
            dot.addEventListener('pointerdown', e => { drag = i; svg.setPointerCapture(e.pointerId); });
            svg.append(dot);
        });
        const [bm, bc] = ols(), cur = sse(m, c), best = sse(bm, bc);
        read.innerHTML = `ŷ = ${m.toFixed(2)}·hours + ${c.toFixed(1)}  ·  SSE = Σ(y - ŷ)² = <b>${V.fmt(cur, 1)}</b> (least-squares minimum: ${V.fmt(best, 1)} at ŷ = ${bm.toFixed(2)}·hours + ${bc.toFixed(1)}). ` +
            `Prediction for 6 hours: ${V.fmt(m * 6 + c, 1)}. Positive residuals (violet) mean the line underpredicts; negative (red) mean it overpredicts - squaring stops them cancelling.`;
    }
    draw();
};

// Multiple regression in 3D: a plane through the slide's car data
ML_VIS.l6_plane3d = async function (root) {
    // The 36-car dataset whose first six rows appear on the slide: [name, volume (cm3), weight (kg), CO2 (g/km)]
    const raw = '1000,790,99;1200,1160,95;1000,929,95;900,865,90;1500,1140,105;1000,929,105;1400,1109,90;1500,1365,92;1500,1112,98;1600,1150,99;1100,980,99;1300,990,101;1000,1112,99;1600,1252,94;1600,1326,97;1600,1330,97;1600,1365,99;2200,1280,104;1600,1119,104;2000,1328,105;1600,1584,94;2000,1428,99;2100,1365,99;1600,1415,99;2000,1415,99;1500,1465,102;2000,1490,104;2000,1725,114;1600,1523,109;2000,1705,114;2100,1605,115;2000,1746,117;1600,1235,104;1600,1390,108;1600,1405,109;2500,1395,120';
    const named = ['Toyota Aygo', 'Mitsubishi Space Star', 'Skoda Citigo', 'Fiat 500', 'Mini Cooper', 'VW Up!'];
    const cars = raw.split(';').map((r, i) => { const [v, w, c] = r.split(',').map(Number); return [named[i] || '', v, w, c]; });
    let mode = 'plane';
    const read = V.h('p', { class: 'vis-readout' });
    root.append(V.seg([['line', 'Volume only (simple regression)'], ['plane', 'Volume + Weight (multiple regression)']], mode, v => { mode = v; show(); }), read);
    const sc = await V.scene3d(root, { radius: 7, phi: 1.15, theta: 0.6, target: [0, 0.2, 0], label: 'Regression plane through car data' });
    if (!sc) return;
    const { THREE, scene } = sc;
    const X = cars.map(r => r[1]), Wt = cars.map(r => r[2]), Y = cars.map(r => r[3]);
    const nx = v => (v - 1700) / 800 * 1.6, nz = v => (v - 1268) / 478 * 1.6, ny = v => (v - 105) / 15 * 1.2;
    const fitPlane = () => {
        const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
        cars.forEach((r, i) => { const f = [1, X[i], Wt[i]]; for (let a = 0; a < 3; a++) { b[a] += f[a] * Y[i]; for (let d = 0; d < 3; d++) A[a][d] += f[a] * f[d]; } });
        return solve(A, b);
    };
    const fitLine = () => { const mx = V.mean(X), my = V.mean(Y); const s = X.reduce((t, x, i) => t + (x - mx) * (Y[i] - my), 0) / X.reduce((t, x) => t + (x - mx) ** 2, 0); return [my - s * mx, s]; };
    const P = fitPlane(), Ln = fitLine();
    scene.add(V.line3d(THREE, [[-2, -1.3, -2], [2, -1.3, -2]], 0x94a3b8), V.line3d(THREE, [[-2, -1.3, -2], [-2, -1.3, 2]], 0x94a3b8), V.line3d(THREE, [[-2, -1.3, -2], [-2, 1.6, -2]], 0x94a3b8));
    [['engine volume →', [0.6, -1.3, -2.35]], ['weight →', [-2.4, -1.3, 0.6]], ['CO₂', [-2, 1.85, -2]]].forEach(([t, p]) => { const s = V.label3d(THREE, t, '#475569', 0.26); s.position.set(...p); scene.add(s); });
    const g = new THREE.PlaneGeometry(3.6, 3.6, 8, 8); g.rotateX(-Math.PI / 2);
    const pos = g.attributes.position;
    for (let i = 0; i < pos.count; i++) { const vx = pos.getX(i) / 1.6 * 800 + 1700, vz = pos.getZ(i) / 1.6 * 478 + 1268; pos.setY(i, ny(P[0] + P[1] * vx + P[2] * vz)); }
    g.computeVertexNormals();
    const plane = new THREE.Group();
    plane.add(new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: 0x7c3aed, transparent: true, opacity: .28, side: THREE.DoubleSide })),
        new THREE.LineSegments(new THREE.WireframeGeometry(g), new THREE.LineBasicMaterial({ color: 0x7c3aed, transparent: true, opacity: .3 })));
    scene.add(plane);
    const line = V.line3d(THREE, [[nx(900), ny(Ln[0] + Ln[1] * 900), 0], [nx(2500), ny(Ln[0] + Ln[1] * 2500), 0]], 0x7c3aed);
    scene.add(line);
    const dots = [], sticks = [];
    cars.forEach((r, i) => {
        const m = new THREE.Mesh(new THREE.SphereGeometry(r[0] ? 0.075 : 0.055, 14, 10), new THREE.MeshLambertMaterial({ color: r[0] ? 0xd97706 : 0x0284c7 }));
        scene.add(m); dots.push(m);
        const lab = V.label3d(THREE, r[0] || ' ', '#334155', 0.16); scene.add(lab); dots.push(lab);
    });
    function show() {
        sticks.forEach(s => scene.remove(s)); sticks.length = 0;
        let sse = 0;
        cars.forEach((r, i) => {
            const z = mode === 'plane' ? nz(r[2]) : 0, yh = mode === 'plane' ? P[0] + P[1] * r[1] + P[2] * r[2] : Ln[0] + Ln[1] * r[1];
            dots[2 * i].position.set(nx(r[1]), ny(r[3]), z); dots[2 * i + 1].position.set(nx(r[1]), ny(r[3]) + 0.22, z);
            const s = V.line3d(THREE, [[nx(r[1]), ny(r[3]), z], [nx(r[1]), ny(yh), z]], 0xe11d48); scene.add(s); sticks.push(s);
            sse += (r[3] - yh) ** 2;
        });
        plane.visible = mode === 'plane'; line.visible = mode === 'line'; sc.dirty();
        read.innerHTML = mode === 'plane'
            ? `Fitted plane on 36 cars (amber = the six on the slide): CO₂ = ${V.fmt(P[0], 1)} + ${V.fmt(P[1], 5)}·Volume + ${V.fmt(P[2], 5)}·Weight. Red sticks are residuals (SSE ${V.fmt(sse, 1)}). Each coefficient is the change in predicted CO₂ per unit of its feature while the other is held constant - still a linear model. A 1,300 cm³, 2,300 kg car would be predicted ${V.fmt(P[0] + P[1] * 1300 + P[2] * 2300, 1)} g/km.`
            : `Fitted line: CO₂ = ${V.fmt(Ln[0], 1)} + ${V.fmt(Ln[1], 4)}·Volume (SSE ${V.fmt(sse, 1)}). With one predictor, cars with the same engine but different weights get the same prediction.`;
    }
    show();
};

// Sigmoid and threshold versus a straight regression line on 0/1 labels
ML_VIS.l6_sigmoid = function (root) {
    const r = V.rng(12), pts = [];
    for (let i = 0; i < 22; i++) { const x = r() * 10, c = x + V.gauss(r) * 1.4 > 5.5 ? 1 : 0; pts.push({ x, c }); }
    let w = 1.4, b = -7.7, thr = 0.5, probe = 6;
    const W = 600, H = 300, svg = V.svg(W, H), read = V.h('p', { class: 'vis-readout' });
    const sx = x => 50 + x / 10 * 520, sy = p => 250 - p * 200;
    const mx = V.mean(pts.map(p => p.x)), my = V.mean(pts.map(p => p.c));
    const lb = pts.reduce((s, p) => s + (p.x - mx) * (p.c - my), 0) / pts.reduce((s, p) => s + (p.x - mx) ** 2, 0), la = my - lb * mx;
    root.append(V.h('div', { class: 'vis-row' },
        V.slider('weight w', 2, 40, 1, 14, v => { w = v / 10; draw(); }, v => (v / 10).toFixed(1)),
        V.slider('bias b', -200, 0, 1, -77, v => { b = v / 10; draw(); }, v => (v / 10).toFixed(1)),
        V.slider('threshold', 10, 90, 5, 50, v => { thr = v / 100; draw(); }, v => (v / 100).toFixed(2)),
        V.slider('new transaction amount', 0, 100, 1, 60, v => { probe = v / 10; draw(); }, v => 'RM' + (v / 10).toFixed(1) + 'k')), svg, read);
    const sig = z => 1 / (1 + Math.exp(-z));
    function draw() {
        V.clear(svg); const rr = V.rng(4);
        svg.append(V.s('rect', { x: 50, y: 10, width: 520, height: 40, fill: 'rgba(225,29,72,.06)' }), V.s('rect', { x: 50, y: 250, width: 520, height: 40, fill: 'rgba(225,29,72,.06)' }));
        V.text(svg, 565, 24, 'impossible probability > 1', { size: 11, color: V.C.rose, anchor: 'end' });
        V.text(svg, 565, 282, 'impossible probability < 0', { size: 11, color: V.C.rose, anchor: 'end' });
        V.arrow(svg, 50, 250, 585, 250, '#94a3b8', 1.2, rr); V.rline(svg, 50, 290, 50, 10, '#94a3b8', 1.2, rr);
        [0, 0.5, 1].forEach(p => V.text(svg, 42, sy(p), String(p), { size: 11, color: '#64748b', anchor: 'end' }));
        V.text(svg, 520, 268, 'transaction amount →', { size: 12, color: '#64748b' });
        svg.append(V.s('line', { x1: 50, x2: 570, y1: sy(thr), y2: sy(thr), stroke: '#94a3b8', 'stroke-dasharray': '5 4' }));
        V.text(svg, 56, sy(thr) - 8, 'threshold ' + thr.toFixed(2), { size: 11, color: '#64748b', anchor: 'start' });
        svg.append(V.s('line', { x1: sx(0), y1: sy(la), x2: sx(10), y2: sy(la + lb * 10), stroke: V.C.slate, 'stroke-width': 2, 'stroke-dasharray': '7 5' }));
        let d = ''; for (let i = 0; i <= 200; i++) { const x = i / 20; d += (i ? 'L' : 'M') + sx(x) + ' ' + sy(sig(w * x + b)); }
        svg.append(V.s('path', { d, fill: 'none', stroke: V.C.violet, 'stroke-width': 3 }));
        let ok = 0;
        pts.forEach(p => {
            const pred = sig(w * p.x + b) >= thr ? 1 : 0; ok += pred === p.c;
            const c = V.s('circle', { cx: sx(p.x), cy: sy(p.c), r: 6, fill: p.c ? V.C.rose : V.C.green, stroke: pred === p.c ? '#fff' : V.C.ink, 'stroke-width': pred === p.c ? 1.5 : 2.5 });
            c.append(V.s('title', {}, `RM${p.x.toFixed(1)}k: actually ${p.c ? 'fraud' : 'legit'}, predicted ${pred ? 'fraud' : 'legit'}`)); svg.append(c);
        });
        const z = w * probe + b, pp = sig(z);
        svg.append(V.s('line', { x1: sx(probe), x2: sx(probe), y1: 250, y2: sy(pp), stroke: V.C.amber, 'stroke-width': 2 }));
        svg.append(V.s('circle', { cx: sx(probe), cy: sy(pp), r: 7, fill: V.C.amber, stroke: '#fff', 'stroke-width': 2 }));
        read.innerHTML = `New transaction: z = w·x + b = ${w.toFixed(1)}x${probe.toFixed(1)} + (${b.toFixed(1)}) = <b>${z.toFixed(2)}</b>, σ(z) = 1/(1+e<sup>-z</sup>) = <b>${pp.toFixed(3)}</b> → <b>${pp >= thr ? 'Fraud' : 'Legitimate'}</b> (threshold ${thr.toFixed(2)}). ` +
            `Training accuracy ${ok}/${pts.length}; outlined points are misclassified. The dashed straight line is linear regression on the same 0/1 labels - it leaves the 0-1 band, so it cannot be a probability.`;
    }
    draw();
};

// Decision tree: explore one split, or grow a tree
ML_VIS.l6_tree = function (root) {
    const pts = l6Data();
    let mode = 'split', feat = 'y', thr = 6.5, depth = 2, cancel = null;
    const W = 340, H = 300, pad = 34, svg = V.svg(W, H), side = V.svg(420, 300), read = V.h('p', { class: 'vis-readout' });
    const { sx, sy } = l6Scatter(svg, W, H, pad);
    const tLabel = t => feat === 'x' ? V.fmt(t, 0) + ' years' : 'RM' + V.fmt(t, 1) + 'k';
    const tSl = V.slider('threshold', 0, 100, 1, 39, v => { const R = feat === 'x' ? L6X : L6Y; thr = R[0] + v / 100 * (R[1] - R[0]); draw(); }, v => { const R = feat === 'x' ? L6X : L6Y; return tLabel(R[0] + v / 100 * (R[1] - R[0])); });
    const dSl = V.slider('max depth', 1, 4, 1, depth, v => { depth = v; draw(); });
    const fSeg = V.seg([['y', 'split on income'], ['x', 'split on age']], feat, v => { feat = v; thr = feat === 'x' ? 40 : 6.5; syncT(); draw(); });
    const best = V.btn('Find best split', () => {
        const cand = l6Candidates(pts, feat); if (cancel) cancel();
        cancel = V.tween(1600, t => { thr = cand[Math.min(cand.length - 1, Math.floor(t * cand.length))]; syncT(); draw(); }, () => { const b = l6Best(pts, [feat]); thr = b.t; syncT(); draw(); });
    }, 'primary');
    const ctrl = V.h('div', { class: 'vis-row' });
    root.append(V.seg([['split', 'Explore one split (Gini)'], ['grow', 'Grow a tree (CART)']], mode, v => { mode = v; draw(); }), ctrl, V.h('div', { class: 'vis-split' }, svg, side), read,
        V.h('p', { class: 'vis-note' }, 'Violet circles buy, amber squares do not. Gini = 1 - Σ pᵢ²; a split is scored by the size-weighted Gini of its two children.'));
    function syncT() { const R = feat === 'x' ? L6X : L6Y; tSl.input.value = Math.round((thr - R[0]) / (R[1] - R[0]) * 100); tSl.querySelector('output').textContent = tLabel(thr); }
    function base() { V.clear(svg); l6Axes(svg, W, H, pad, V.rng(6)); }
    function draw() {
        V.clear(ctrl); V.clear(side);
        if (mode === 'split') {
            ctrl.append(fSeg, tSl, best);
            base();
            const L = pts.filter(p => p[feat] <= thr), R = pts.filter(p => p[feat] > thr), wg = (L.length * l6Gini(L) + R.length * l6Gini(R)) / pts.length;
            if (feat === 'x') svg.append(V.s('line', { x1: sx(thr), x2: sx(thr), y1: 4, y2: H - pad + 10, stroke: V.C.ink, 'stroke-width': 2.5 }));
            else svg.append(V.s('line', { x1: pad, x2: W - 6, y1: sy(thr), y2: sy(thr), stroke: V.C.ink, 'stroke-width': 2.5 }));
            l6Points(svg, pts, sx, sy);
            const cand = l6Candidates(pts, feat), gs = cand.map(t => { const a = pts.filter(p => p[feat] <= t), b = pts.filter(p => p[feat] > t); return (a.length * l6Gini(a) + b.length * l6Gini(b)) / pts.length; });
            const R0 = feat === 'x' ? L6X : L6Y, cx = t => 40 + (t - R0[0]) / (R0[1] - R0[0]) * 360, cy = g => 250 - g / 0.5 * 220, rr = V.rng(2);
            V.arrow(side, 40, 250, 410, 250, '#94a3b8', 1.2, rr); V.arrow(side, 40, 250, 40, 20, '#94a3b8', 1.2, rr);
            V.text(side, 220, 275, (feat === 'x' ? 'age' : 'income') + ' threshold →', { size: 12, color: '#64748b' });
            V.text(side, 46, 18, 'weighted Gini', { size: 12, color: '#64748b', anchor: 'start' });
            [0, 0.25, 0.5].forEach(g => V.text(side, 34, cy(g), String(g), { size: 11, color: '#64748b', anchor: 'end' }));
            side.append(V.s('path', { d: cand.map((t, i) => (i ? 'L' : 'M') + cx(t) + ' ' + cy(gs[i])).join(''), fill: 'none', stroke: V.C.violet, 'stroke-width': 2.2 }));
            side.append(V.s('circle', { cx: cx(thr), cy: cy(wg), r: 6, fill: V.C.amber }));
            const f = feat === 'x' ? 'age' : 'income', fmt = v => feat === 'x' ? V.fmt(v, 1) : 'RM' + V.fmt(v, 1) + 'k';
            read.innerHTML = `Root: ${pts.length} customers, Gini ${V.fmt(l6Gini(pts), 3)}. Split "${f} ≤ ${fmt(thr)}": left ${L.length} (${L.filter(p => p.c).length} buy, Gini ${V.fmt(l6Gini(L), 3)}), right ${R.length} (${R.filter(p => p.c).length} buy, Gini ${V.fmt(l6Gini(R), 3)}). ` +
                `Weighted Gini = ${L.length}/${pts.length}·${V.fmt(l6Gini(L), 3)} + ${R.length}/${pts.length}·${V.fmt(l6Gini(R), 3)} = <b>${V.fmt(wg, 3)}</b>. The tree picks the threshold at the bottom of the curve.`;
        } else {
            ctrl.append(dSl);
            base();
            const tree = l6Grow(pts, depth, null);
            const nX = 34, nY = 28;
            for (let i = 0; i < nX; i++) for (let j = 0; j < nY; j++) {
                const x = L6X[0] + (i + .5) / nX * (L6X[1] - L6X[0]), y = L6Y[0] + (j + .5) / nY * (L6Y[1] - L6Y[0]), cls = l6Predict(tree, { x, y });
                svg.append(V.s('rect', { x: sx(L6X[0] + i / nX * (L6X[1] - L6X[0])), y: sy(L6Y[0] + (j + 1) / nY * (L6Y[1] - L6Y[0])), width: (W - pad - 10) / nX + .5, height: (H - pad - 10) / nY + .5, fill: cls ? 'rgba(124,58,237,.14)' : 'rgba(217,119,6,.12)' }));
            }
            l6Points(svg, pts, sx, sy);
            const acc = pts.filter(p => l6Predict(tree, p) === p.c).length;
            let leaves = 0;
            const lay = (nd, x0, x1, y, d) => {
                const x = (x0 + x1) / 2, span = x1 - x0, fs = Math.max(8, Math.min(11, span / 7));
                if (nd.leaf) {
                    leaves++;
                    const lw = Math.min(44, span - 4);
                    V.rrect(side, x - lw / 2, y - 12, lw, 24, nd.cls ? V.C.violet : V.C.amber, nd.cls ? V.C.violetS : V.C.amberS, V.rng(d + x));
                    V.text(side, x, y, `${nd.ones}/${nd.n - nd.ones}`, { size: fs }); return;
                }
                const yc = y + 62, nw = Math.min(80, span - 4);
                [[nd.L, x0, x], [nd.R, x, x1]].forEach(([c, a, b]) => side.append(V.s('line', { x1: x, y1: y + 12, x2: (a + b) / 2, y2: yc - 12, stroke: '#94a3b8' })));
                V.rrect(side, x - nw / 2, y - 13, nw, 26, V.C.slate, '#fff', V.rng(d * 7 + x));
                V.text(side, x, y, `${nd.f === 'x' ? 'age' : 'inc'} ≤ ${V.fmt(nd.t, 1)}`, { size: fs });
                lay(nd.L, x0, x, yc, d + 1); lay(nd.R, x, x1, yc, d + 1);
            };
            lay(tree, 0, 420, 22, 0);
            read.innerHTML = `Depth ${depth}: ${leaves} leaves, training accuracy ${acc}/${pts.length}. Leaves show buyers/non-buyers. ` +
                (depth >= 4 ? 'Deep trees carve tiny boxes around individual noisy points - perfect on training data, unreliable on new customers (overfitting).' : 'Each level asks one more yes/no question; the shaded regions are the IF-THEN rules.');
        }
    }
    draw();
};

// Random forest: bootstrap samples + random feature choice, then majority vote
ML_VIS.l6_forest = function (root) {
    const pts = l6Data(4), MAXT = 25;
    const r = V.rng(31), forest = [];
    for (let t = 0; t < MAXT; t++) {
        const sample = Array.from({ length: pts.length }, () => pts[Math.floor(r() * pts.length)]);
        const tree = l6Grow(sample, 3, () => [r() < 0.5 ? 'x' : 'y']);
        forest.push({ tree, uniq: new Set(sample).size });
    }
    let nT = 1, probe = { x: 35, y: 7 }, timer = null;
    const W = 340, H = 300, pad = 34, svg = V.svg(W, H), minis = V.h('div', { class: 'vis-minis' }), read = V.h('p', { class: 'vis-readout' });
    const { sx, sy, ix, iy } = l6Scatter(svg, W, H, pad);
    const sl = V.slider('trees in the forest', 1, MAXT, 1, nT, v => { nT = v; draw(); });
    const grow = V.btn('Grow forest', () => {
        if (timer) { clearInterval(timer); timer = null; grow.textContent = 'Grow forest'; return; }
        nT = 0; grow.textContent = 'Pause';
        timer = setInterval(() => { nT++; sl.input.value = nT; sl.querySelector('output').textContent = nT; draw(); if (nT >= MAXT) { clearInterval(timer); timer = null; grow.textContent = 'Grow forest'; } }, 260);
    }, 'primary');
    svg.addEventListener('click', e => { const p = V.svgPoint(svg, e); probe = { x: Math.max(L6X[0], Math.min(L6X[1], ix(p.x))), y: Math.max(L6Y[0], Math.min(L6Y[1], iy(p.y))) }; draw(); });
    root.append(V.h('div', { class: 'vis-row' }, sl, grow), V.h('div', { class: 'vis-split' }, svg, minis), read,
        V.h('p', { class: 'vis-note' }, 'Click the map to place a new customer. Each tree sees a bootstrap sample (drawn with replacement) and may only use one randomly chosen feature at each split. Shading = share of trees voting "buys".'));
    function vote(p, k) { return forest.slice(0, k).filter(f => l6Predict(f.tree, p)).length; }
    function draw() {
        V.clear(svg); l6Axes(svg, W, H, pad, V.rng(6));
        const nX = 34, nY = 28;
        for (let i = 0; i < nX; i++) for (let j = 0; j < nY; j++) {
            const x = L6X[0] + (i + .5) / nX * (L6X[1] - L6X[0]), y = L6Y[0] + (j + .5) / nY * (L6Y[1] - L6Y[0]), s = vote({ x, y }, nT) / nT;
            svg.append(V.s('rect', { x: sx(L6X[0] + i / nX * (L6X[1] - L6X[0])), y: sy(L6Y[0] + (j + 1) / nY * (L6Y[1] - L6Y[0])), width: (W - pad - 10) / nX + .5, height: (H - pad - 10) / nY + .5, fill: s >= .5 ? `rgba(124,58,237,${.06 + .22 * s})` : `rgba(217,119,6,${.06 + .2 * (1 - s)})` }));
        }
        l6Points(svg, pts, sx, sy, 4);
        const v = vote(probe, nT);
        svg.append(V.s('path', { d: `M${sx(probe.x)} ${sy(probe.y) - 10} l10 10 l-10 10 l-10 -10 z`, fill: v * 2 >= nT ? V.C.violet : V.C.amber, stroke: V.C.ink, 'stroke-width': 2 }));
        V.clear(minis);
        forest.slice(0, Math.min(nT, 6)).forEach((f, t) => {
            const m = V.svg(140, 110, 'vis-mini'), mx = x => (x - L6X[0]) / (L6X[1] - L6X[0]) * 140, my = y => 110 - (y - L6Y[0]) / (L6Y[1] - L6Y[0]) * 110;
            for (let i = 0; i < 20; i++) for (let j = 0; j < 16; j++) {
                const x = L6X[0] + (i + .5) / 20 * (L6X[1] - L6X[0]), y = L6Y[0] + (j + .5) / 16 * (L6Y[1] - L6Y[0]), c = l6Predict(f.tree, { x, y });
                m.append(V.s('rect', { x: mx(L6X[0] + i / 20 * (L6X[1] - L6X[0])), y: my(L6Y[0] + (j + 1) / 16 * (L6Y[1] - L6Y[0])), width: 7.5, height: 7.4, fill: c ? '#ddd6fe' : '#fde68a' }));
            }
            const says = l6Predict(f.tree, probe);
            m.append(V.s('circle', { cx: mx(probe.x), cy: my(probe.y), r: 4, fill: says ? V.C.violet : V.C.amber, stroke: V.C.ink }));
            minis.append(V.h('figure', { class: 'vis-minifig' }, m, V.h('figcaption', {}, `Tree ${t + 1}: ${f.uniq} unique of ${pts.length} rows · says ${says ? 'buys' : 'no'}`)));
        });
        const single = l6Predict(forest[0].tree, probe);
        read.innerHTML = `New customer (age ${V.fmt(probe.x, 0)}, RM${V.fmt(probe.y, 1)}k): <b>${v} of ${nT}</b> tree${nT > 1 ? 's' : ''} vote "buys" → forest predicts <b>${v * 2 >= nT ? 'buys' : 'does not buy'}</b> (majority vote). Tree 1 alone says ${single ? 'buys' : 'no'}. ` +
            'For regression the forest would average the trees instead, e.g. (400,000 + 430,000 + 420,000) / 3 = RM416,667. More trees smooth the boundary because their individual mistakes differ.';
    }
    draw();
};
// Lecture 7 visuals: SVM, kernels, Naive Bayes, KNN and neural networks

// Linear soft-margin SVM (dual coordinate descent), draggable points and a C slider
ML_VIS.l7_svm = function (root) {
    const pts = [
        [6.2, 7.1, 1], [7.4, 6.0, 1], [8.1, 8.2, 1], [6.8, 8.8, 1], [8.8, 6.6, 1], [5.6, 8.4, 1], [7.6, 4.9, 1], [4.6, 6.1, 1],
        [2.2, 3.1, -1], [3.4, 2.0, -1], [1.6, 4.6, -1], [3.9, 3.8, -1], [2.6, 1.4, -1], [4.4, 1.9, -1], [1.2, 2.2, -1], [5.6, 4.7, -1],
    ].map(([x, y, c]) => ({ x, y, c }));
    let C = 1, model = null, drag = null, cancel = null;
    const W = 560, H = 340, svg = V.svg(W, H), read = V.h('p', { class: 'vis-readout' });
    const sx = x => 40 + x / 10 * 500, sy = y => 320 - y / 10 * 300, ix = px => (px - 40) / 500 * 10, iy = py => (320 - py) / 300 * 10;
    const Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100];
    root.append(V.slider('C (penalty for margin violations)', 0, Cs.length - 1, 1, 4, v => { C = Cs[v]; retrain(); }, v => String(Cs[v])), svg, read,
        V.h('p', { class: 'vis-note' }, 'Drag any point. Solid line: w·x + b = 0. Dashed: the margins w·x + b = ±1. Ringed points are support vectors (α > 0); red outlines violate the margin.'));
    function train() {
        const B = 5, X = pts.map(p => [p.x, p.y, B]), a = new Array(pts.length).fill(0), w = [0, 0, 0];
        const Q = X.map(x => x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        for (let pass = 0; pass < 800; pass++) {
            let maxPG = 0;
            for (let i = 0; i < pts.length; i++) {
                const y = pts[i].c, G = y * (w[0] * X[i][0] + w[1] * X[i][1] + w[2] * X[i][2]) - 1;
                const PG = a[i] === 0 ? Math.min(G, 0) : a[i] === C ? Math.max(G, 0) : G;
                maxPG = Math.max(maxPG, Math.abs(PG));
                if (Math.abs(PG) > 1e-12) { const old = a[i]; a[i] = Math.min(Math.max(a[i] - G / Q[i], 0), C); const d = (a[i] - old) * y; w[0] += d * X[i][0]; w[1] += d * X[i][1]; w[2] += d * X[i][2]; }
            }
            if (maxPG < 1e-6) break;
        }
        return { w: [w[0], w[1]], b: w[2] * B, a };
    }
    function retrain() {
        const next = train(), prev = model;
        if (cancel) cancel();
        if (!prev) { model = next; draw(); return; }
        cancel = V.tween(450, t => { model = { w: [prev.w[0] + (next.w[0] - prev.w[0]) * t, prev.w[1] + (next.w[1] - prev.w[1]) * t], b: prev.b + (next.b - prev.b) * t, a: next.a }; draw(); });
    }
    svg.addEventListener('pointermove', e => { if (drag === null) return; const p = V.svgPoint(svg, e); pts[drag].x = Math.max(0.2, Math.min(9.8, ix(p.x))); pts[drag].y = Math.max(0.2, Math.min(9.8, iy(p.y))); model = train(); draw(); });
    svg.addEventListener('pointerup', () => { drag = null; }); svg.addEventListener('pointerleave', () => { drag = null; });
    const lineAt = (w, b, k) => {
        // points of w·x + b = k clipped to the 0..10 box
        const out = [];
        if (Math.abs(w[1]) > 1e-9) [0, 10].forEach(x => { const y = (k - b - w[0] * x) / w[1]; if (y >= 0 && y <= 10) out.push([x, y]); });
        if (Math.abs(w[0]) > 1e-9) [0, 10].forEach(y => { const x = (k - b - w[1] * y) / w[0]; if (x >= 0 && x <= 10) out.push([x, y]); });
        return out.slice(0, 2);
    };
    function draw() {
        V.clear(svg); const { w, b, a } = model, rr = V.rng(3), nw = Math.hypot(w[0], w[1]);
        for (let i = 0; i < 25; i++) for (let j = 0; j < 15; j++) {
            const x = (i + .5) / 25 * 10, y = (j + .5) / 15 * 10, s = w[0] * x + w[1] * y + b;
            svg.append(V.s('rect', { x: sx(i / 25 * 10), y: sy((j + 1) / 15 * 10), width: 20.5, height: 20.5, fill: s >= 0 ? 'rgba(2,132,199,.07)' : 'rgba(225,29,72,.06)' }));
        }
        V.rline(svg, 40, 320, 545, 320, '#94a3b8', 1.2, rr); V.rline(svg, 40, 320, 40, 15, '#94a3b8', 1.2, rr);
        V.text(svg, 490, 334, 'income →', { size: 12, color: '#64748b' }); V.text(svg, 46, 14, 'savings', { size: 12, color: '#64748b', anchor: 'start' });
        [[-1, '5 4'], [1, '5 4'], [0, null]].forEach(([k, dash]) => {
            const seg = lineAt(w, b, k); if (seg.length < 2) return;
            svg.append(V.s('line', Object.assign({ x1: sx(seg[0][0]), y1: sy(seg[0][1]), x2: sx(seg[1][0]), y2: sy(seg[1][1]), stroke: k ? '#64748b' : V.C.ink, 'stroke-width': k ? 1.6 : 3 }, dash ? { 'stroke-dasharray': dash } : {})));
        });
        let sv = 0, viol = 0, err = 0;
        pts.forEach((p, i) => {
            const m = p.c * (w[0] * p.x + w[1] * p.y + b), isSV = a[i] > 1e-6;
            sv += isSV; viol += m < 1 - 1e-6; err += m < 0;
            if (isSV) svg.append(V.s('circle', { cx: sx(p.x), cy: sy(p.y), r: 12, fill: 'none', stroke: V.C.amber, 'stroke-width': 2.5 }));
            const dot = V.s('circle', { cx: sx(p.x), cy: sy(p.y), r: 7, fill: p.c > 0 ? V.C.sky : V.C.rose, stroke: m < 1 - 1e-6 ? V.C.red : '#fff', 'stroke-width': m < 1 - 1e-6 ? 3 : 1.5, style: 'cursor:grab' });
            dot.append(V.s('title', {}, `${p.c > 0 ? 'repaid' : 'defaulted'} · y(w·x+b) = ${m.toFixed(2)}${isSV ? ' · support vector' : ''}`));
            dot.addEventListener('pointerdown', e => { drag = i; svg.setPointerCapture(e.pointerId); });
            svg.append(dot);
        });
        read.innerHTML = `C = ${C}: margin width 2/‖w‖ = <b>${V.fmt(2 / nw, 2)}</b> units, <b>${sv}</b> support vectors, ${viol} margin violation${viol === 1 ? '' : 's'}, ${err} training error${err === 1 ? '' : 's'}. ` +
            (C >= 10 ? 'Large C punishes every violation, so the margin shrinks to fit the training points - risk of fitting noise.' : C <= 0.1 ? 'Small C tolerates violations, so the margin widens and more points become support vectors - a more regularized boundary.' : 'Only the ringed support vectors decide where the boundary sits; move any other point and nothing changes.');
    }
    retrain();
};

// Kernel idea in 3D: lift concentric circles with z = x² + y²
ML_VIS.l7_kernel3d = async function (root) {
    const r = V.rng(5), pts = [];
    for (let i = 0; i < 70; i++) { const a = r() * 2 * Math.PI, d = Math.sqrt(r()) * 0.42; pts.push({ x: d * Math.cos(a), y: d * Math.sin(a), c: 0 }); }
    for (let i = 0; i < 110; i++) { const a = r() * 2 * Math.PI, d = 0.72 + r() * 0.28; pts.push({ x: d * Math.cos(a), y: d * Math.sin(a), c: 1 }); }
    let lifted = 0, target = 1;
    const read = V.h('p', { class: 'vis-readout' });
    const btn = V.btn('Lift with z = x² + y²', () => { target = target ? 0 : 1; btn.textContent = target ? 'Drop back to 2D' : 'Lift with z = x² + y²'; animate(); }, 'primary');
    root.append(V.h('div', { class: 'vis-row' }, btn), read);
    const sc = await V.scene3d(root, { radius: 6.4, phi: 1.12, theta: 0.6, target: [0, 0.7, 0], label: 'Circles lifted into 3D' });
    if (!sc) return;
    const { THREE, scene } = sc, S = 1.6, H = 2.0;
    const floor = []; for (let k = 0; k <= 64; k++) { const a = k / 64 * 2 * Math.PI; floor.push([1.75 * Math.cos(a), 0, 1.75 * Math.sin(a)]); }
    scene.add(V.line3d(THREE, floor, 0xcbd5e1));
    const meshes = pts.map(p => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.04, 10, 8), new THREE.MeshLambertMaterial({ color: p.c ? 0xe11d48 : 0x0284c7 })); scene.add(m); return m; });
    const level = 0.33;
    const plane = new THREE.Mesh(new THREE.CircleGeometry(1.75, 48), new THREE.MeshLambertMaterial({ color: 0xc026d3, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
    plane.rotation.x = -Math.PI / 2; plane.position.y = level * H; scene.add(plane);
    const ring = []; for (let k = 0; k <= 64; k++) { const a = k / 64 * 2 * Math.PI; ring.push([Math.sqrt(level) * S * Math.cos(a), 0.01, Math.sqrt(level) * S * Math.sin(a)]); }
    const circ = V.line3d(THREE, ring, 0xc026d3); scene.add(circ);
    function place() {
        pts.forEach((p, i) => meshes[i].position.set(p.x * S, lifted * (p.x * p.x + p.y * p.y) * H, p.y * S));
        plane.material.opacity = Math.max(0, lifted - 0.4) * 0.32; circ.visible = lifted > 0.6 || lifted < 0.05; sc.dirty();
        read.innerHTML = lifted > 0.5
            ? 'Lifted: every point gets height z = x² + y² (its squared distance from the centre). The blue inner disc sits low, the red ring sits high - one flat plane now separates them. Seen from above, that plane is the purple circle: a curved boundary in the original 2D space.'
            : 'Flat 2D view: the blue disc is surrounded by the red ring, so no straight line can separate them. Press Lift to add a third coordinate.';
    }
    function animate() { const from = lifted; V.tween(1400, t => { lifted = from + (target - from) * t; place(); }); }
    lifted = 0; place(); animate();
};

// Naive Bayes calculator on the lecture's customer-buying data (slide 74)
ML_VIS.l7_bayes = function (root) {
    const F = ['Weather', 'Temperature', 'Humidity', 'Wind', 'Day', 'Promotion'];
    const VALS = [['Sunny', 'Cloudy', 'Rainy'], ['Hot', 'Mild', 'Cool'], ['High', 'Normal'], ['Weak', 'Strong'], ['Weekday', 'Weekend'], ['Yes', 'No']];
    const rows = [
        ['Sunny', 'Hot', 'High', 'Weak', 'Weekday', 'Yes', 'No'], ['Sunny', 'Hot', 'High', 'Strong', 'Weekend', 'No', 'No'],
        ['Cloudy', 'Mild', 'High', 'Weak', 'Weekday', 'Yes', 'Yes'], ['Rainy', 'Cool', 'Normal', 'Weak', 'Weekend', 'Yes', 'Yes'],
        ['Rainy', 'Cool', 'Normal', 'Strong', 'Weekday', 'No', 'No'], ['Cloudy', 'Cool', 'Normal', 'Strong', 'Weekend', 'No', 'Yes'],
        ['Sunny', 'Mild', 'High', 'Weak', 'Weekend', 'Yes', 'No'], ['Sunny', 'Cool', 'Normal', 'Weak', 'Weekday', 'Yes', 'Yes'],
        ['Rainy', 'Mild', 'High', 'Weak', 'Weekend', 'No', 'Yes'], ['Cloudy', 'Hot', 'Normal', 'Strong', 'Weekday', 'Yes', 'Yes'],
    ];
    const x = ['Sunny', 'Cool', 'High', 'Weak', 'Weekend', 'Yes'];
    let laplace = false, shown = null;
    const pick = V.h('div', { class: 'vis-row' }), table = V.h('div', { class: 'vis-table-wrap' }), svg = V.svg(560, 110), read = V.h('p', { class: 'vis-readout' });
    F.forEach((f, j) => {
        const sel = V.h('select', { class: 'vis-input', 'aria-label': f });
        VALS[j].forEach(v => sel.append(V.h('option', v === x[j] ? { value: v, selected: 'selected' } : { value: v }, v)));
        sel.addEventListener('change', () => { x[j] = sel.value; draw(); });
        pick.append(V.h('label', { class: 'vis-slider' }, V.h('span', {}, f), sel));
    });
    root.append(pick, V.seg([[false, 'Plain counts'], [true, 'Laplace (add-one) smoothing']], laplace, v => { laplace = v; draw(); }), table, svg, read,
        V.h('p', { class: 'vis-note' }, 'Counts come from the ten rows on slide 74. Try Weather = Cloudy: no "No" customer was cloudy, so the plain No-score becomes exactly 0.'));
    function draw() {
        const res = ['Yes', 'No'].map(cls => {
            const sub = rows.filter(r => r[6] === cls), n = sub.length, prior = n / rows.length;
            const terms = x.map((v, j) => { const k = sub.filter(r => r[j] === v).length; return laplace ? [k + 1, n + VALS[j].length] : [k, n]; });
            const score = terms.reduce((s, [a, b]) => s * a / b, prior);
            return { cls, n, prior, terms, score };
        });
        const tot = res[0].score + res[1].score;
        table.innerHTML = '<table class="vis-table"><tr><th>Class</th><th>Prior</th>' + F.map((f, j) => `<th>P(${x[j]} | class)</th>`).join('') + '<th>Score</th></tr>' +
            res.map(r => `<tr><td><b>Buy = ${r.cls}</b></td><td>${r.n}/10</td>` + r.terms.map(([a, b]) => `<td${a === 0 ? ' class="neg"' : ''}>${a}/${b}</td>`).join('') + `<td><b>${r.score.toFixed(5)}</b></td></tr>`).join('') + '</table>';
        const post = res.map(r => tot ? r.score / tot : 0.5);
        const from = shown || post; shown = post;
        V.tween(500, t => {
            V.clear(svg);
            res.forEach((r, k) => {
                const v = from[k] + (post[k] - from[k]) * t, y = 14 + k * 46;
                V.text(svg, 110, y + 15, `P(Buy = ${r.cls} | X)`, { size: 13, anchor: 'end' });
                svg.append(V.s('rect', { x: 120, y, width: 380, height: 28, rx: 6, fill: '#f1f5f9' }), V.s('rect', { x: 120, y, width: Math.max(2, 380 * v), height: 28, rx: 6, fill: r.cls === 'Yes' ? V.C.green : V.C.rose }));
                V.text(svg, 126 + 380 * v, y + 15, (100 * v).toFixed(1) + '%', { size: 13, anchor: 'start', weight: 700 });
            });
        });
        const win = res[0].score >= res[1].score ? 'Yes' : 'No';
        read.innerHTML = `Score(C) = P(C) x Π P(xᵢ | C): Yes ${res[0].score.toFixed(5)}, No ${res[1].score.toFixed(5)} → predict <b>Buy = ${win}</b>. ` +
            `Normalizing (dividing by the sum, which plays the role of P(X)) gives the posteriors. ` +
            (res.some(r => r.score === 0) ? '<b>Zero-frequency problem:</b> one zero count wipes out the whole class - switch on Laplace smoothing.' : laplace ? 'Smoothing adds 1 to every count and K (the number of values) to every denominator, so nothing is exactly zero.' : 'For the default customer this reproduces the recount: Yes ≈ 0.00370, No ≈ 0.00703.');
    }
    draw();
};

// KNN decision regions: small K is jagged, large K is smooth; unscaled features distort the geometry
ML_VIS.l7_knn_k = function (root) {
    const r = V.rng(8), pts = [];
    for (let i = 0; i < 34; i++) pts.push({ x: 0.32 + V.gauss(r) * 0.15, y: 0.62 + V.gauss(r) * 0.16, c: 0 });
    for (let i = 0; i < 34; i++) pts.push({ x: 0.66 + V.gauss(r) * 0.15, y: 0.38 + V.gauss(r) * 0.16, c: 1 });
    pts.forEach(p => { p.x = Math.min(.98, Math.max(.02, p.x)); p.y = Math.min(.98, Math.max(.02, p.y)); });
    let K = 1, unscaled = false, probe = { x: 0.5, y: 0.5 };
    const W = 560, H = 330, svg = V.svg(W, H), read = V.h('p', { class: 'vis-readout' });
    const sx = x => 40 + x * 510, sy = y => 315 - y * 300;
    root.append(V.h('div', { class: 'vis-row' }, V.slider('K (neighbours that vote)', 0, 15, 1, 0, v => { K = 2 * v + 1; draw(); }, v => String(2 * v + 1)),
        V.seg([[false, 'Scaled features'], [true, 'Unscaled: x in RM, y in years']], unscaled, v => { unscaled = v; draw(); })), svg, read,
        V.h('p', { class: 'vis-note' }, 'Click to place a new customer. Shading shows what KNN would predict everywhere; dashed lines join the new customer to its K neighbours.'));
    svg.addEventListener('click', e => { const p = V.svgPoint(svg, e); probe = { x: Math.min(1, Math.max(0, (p.x - 40) / 510)), y: Math.min(1, Math.max(0, (315 - p.y) / 300)) }; draw(); });
    const d2 = (a, b) => unscaled ? ((a.x - b.x) * 25) ** 2 + (a.y - b.y) ** 2 : (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
    const knn = (q, k, skip) => pts.filter(p => p !== skip).map(p => [d2(p, q), p]).sort((a, b) => a[0] - b[0]).slice(0, k);
    const predict = (q, k, skip) => knn(q, k, skip).filter(n => n[1].c).length * 2 > k ? 1 : 0;
    function draw() {
        V.clear(svg);
        const nX = 56, nY = 30;
        for (let i = 0; i < nX; i++) for (let j = 0; j < nY; j++) {
            const q = { x: (i + .5) / nX, y: (j + .5) / nY };
            svg.append(V.s('rect', { x: sx(i / nX), y: sy((j + 1) / nY), width: 510 / nX + .4, height: 300 / nY + .4, fill: predict(q, K) ? 'rgba(192,38,211,.16)' : 'rgba(2,132,199,.13)' }));
        }
        const near = knn(probe, K);
        near.forEach(([, p]) => svg.append(V.s('line', { x1: sx(probe.x), y1: sy(probe.y), x2: sx(p.x), y2: sy(p.y), stroke: '#64748b', 'stroke-dasharray': '4 3' })));
        pts.forEach(p => svg.append(V.s('circle', { cx: sx(p.x), cy: sy(p.y), r: 5, fill: p.c ? '#c026d3' : V.C.sky, stroke: '#fff', 'stroke-width': 1.3 })));
        const pr = predict(probe, K);
        svg.append(V.s('path', { d: `M${sx(probe.x)} ${sy(probe.y) - 11} l11 11 l-11 11 l-11 -11 z`, fill: pr ? '#c026d3' : V.C.sky, stroke: V.C.ink, 'stroke-width': 2 }));
        const trainAcc = pts.filter(p => predict(p, K) === p.c).length, loo = pts.filter(p => predict(p, K, p) === p.c).length;
        read.innerHTML = `K = ${K}: ${near.filter(n => n[1].c).length} of ${K} neighbours are magenta → predict <b>${pr ? 'magenta' : 'blue'}</b>. Accuracy on the stored points: ${trainAcc}/${pts.length}; leave-one-out (each point predicted from the others): <b>${loo}/${pts.length}</b>. ` +
            (unscaled ? 'Without scaling, the x-axis numbers are 25x larger, so distance is almost all x: the regions become vertical stripes and y is ignored.' : K === 1 ? 'K = 1 memorizes: every stored point is "correct", but the boundary wraps around noise (overfitting).' : K >= 21 ? 'Large K smooths everything into a broad split and can ignore local structure (underfitting).' : 'Moderate K averages over a few neighbours, smoothing out individual noisy points.');
    }
    draw();
};

// One artificial neuron: weighted sum, bias and activation
ML_VIS.l7_neuron = function (root) {
    let x1 = 3, x2 = 4, w1 = 2, w2 = -1, b = 1, act = 'sigmoid', pulse = 1;
    const W = 600, H = 250, svg = V.svg(W, H), read = V.h('p', { class: 'vis-readout' });
    const f = { step: z => (z >= 0 ? 1 : 0), sigmoid: z => 1 / (1 + Math.exp(-z)), tanh: z => Math.tanh(z), relu: z => Math.max(0, z) };
    const sl = (lab, min, max, step, v, set) => V.slider(lab, min, max, step, v, nv => { set(nv); fire(); }, nv => Number(nv).toFixed(1));
    root.append(V.h('div', { class: 'vis-row' }, sl('x₁', -5, 5, 0.5, x1, v => { x1 = v; }), sl('x₂', -5, 5, 0.5, x2, v => { x2 = v; }), sl('bias b', -5, 5, 0.5, b, v => { b = v; })),
        V.h('div', { class: 'vis-row' }, sl('w₁', -3, 3, 0.5, w1, v => { w1 = v; }), sl('w₂', -3, 3, 0.5, w2, v => { w2 = v; }),
            V.seg([['step', 'step'], ['sigmoid', 'sigmoid'], ['tanh', 'tanh'], ['relu', 'ReLU']], act, v => { act = v; fire(); })), svg, read);
    function fire() { V.tween(700, t => { pulse = t; draw(); }); }
    function draw() {
        V.clear(svg); const rr = V.rng(2), z = w1 * x1 + w2 * x2 + b, a = f[act](z);
        const inputs = [[70, 60, 'x₁', x1, w1], [70, 140, 'x₂', x2, w2], [70, 215, '1', 1, b]];
        inputs.forEach(([x, y, lab, v, w]) => {
            const col = w >= 0 ? V.C.sky : V.C.rose;
            svg.append(V.s('line', { x1: x + 24, y1: y, x2: 296, y2: 130, stroke: col, 'stroke-width': 1.5 + Math.abs(w) * 1.6, opacity: .75 }));
            const px = x + 24 + (296 - x - 24) * pulse, py = y + (130 - y) * pulse;
            if (pulse < 1) svg.append(V.s('circle', { cx: px, cy: py, r: 5, fill: V.C.amber }));
            V.rrect(svg, x - 24, y - 18, 48, 36, V.C.slate, '#fff', rr);
            V.text(svg, x, y - 2, lab, { size: 15, weight: 700 }); V.text(svg, x, y + 12, lab === '1' ? 'bias' : V.fmt(v, 1), { size: 11, color: '#64748b' });
            V.text(svg, (x + 24 + 296) / 2 - 10, (y + 130) / 2 - 8, (lab === '1' ? 'b = ' : 'w = ') + V.fmt(w, 1), { size: 12, color: col, weight: 700 });
        });
        svg.append(V.s('circle', { cx: 320, cy: 130, r: 26, fill: V.C.violetS, stroke: V.C.violet, 'stroke-width': 2.5 }));
        V.text(svg, 320, 124, 'Σ', { size: 20, weight: 700, color: V.C.violet }); V.text(svg, 320, 144, 'z = ' + V.fmt(z, 1), { size: 11 });
        V.arrow(svg, 348, 130, 392, 130, V.C.ink, 2, rr);
        V.rrect(svg, 396, 70, 120, 120, V.C.slate, '#fff', rr);
        const gx = v => 404 + (v + 6) / 12 * 104, gy = v => 178 - (act === 'relu' ? v / 6 : act === 'tanh' ? (v + 1) / 2 : v) * 96;
        let d = ''; for (let i = 0; i <= 60; i++) { const zz = -6 + i / 5; d += (i ? 'L' : 'M') + gx(zz) + ' ' + gy(Math.min(6, f[act](zz))); }
        svg.append(V.s('path', { d, fill: 'none', stroke: V.C.violet, 'stroke-width': 2.2 }));
        const zc = Math.max(-6, Math.min(6, z));
        svg.append(V.s('circle', { cx: gx(zc), cy: gy(Math.min(6, f[act](zc))), r: 5, fill: V.C.amber, stroke: '#fff' }));
        V.text(svg, 456, 62, 'a = f(z)', { size: 13, weight: 700 });
        V.arrow(svg, 518, 130, 552, 130, V.C.ink, 2, rr);
        V.text(svg, 575, 124, 'a', { size: 15, weight: 700 }); V.text(svg, 575, 142, V.fmt(a, 3), { size: 13, color: V.C.violet, weight: 700 });
        read.innerHTML = `z = w₁x₁ + w₂x₂ + b = (${w1})(${x1}) + (${w2})(${x2}) + (${b}) = <b>${V.fmt(z, 2)}</b>; ${act}(z) = <b>${V.fmt(a, 3)}</b>. ` +
            (act === 'step' ? 'A step neuron is the original perceptron: it outputs 1 or 0, which gives no gradient to learn from.' : act === 'relu' ? 'ReLU passes positive z unchanged and turns every negative z into 0.' : act === 'tanh' ? 'tanh squashes z into (-1, 1) and is zero-centred.' : 'Sigmoid squashes z into (0, 1); with the defaults this is the lecture\'s example: z = 3, σ(3) ≈ 0.953.') +
            ' Thicker lines are larger |w|; blue weights push z up, red weights push it down.';
    }
    draw(); fire();
};

// XOR forward pass with the lecture's weights (slides 105-108)
ML_VIS.l7_xor = function (root) {
    const Wt = { a1: -4.8, b1: 5.1, c1: -2.6, a2: 4.6, b2: -5.2, c2: -3.2, o1: 5.9, o2: 5.2, co: -2.7 };
    const sig = z => 1 / (1 + Math.exp(-z));
    let inp = [0, 1], step = 0;
    const net = V.svg(380, 260), space = V.svg(220, 260), host = V.h('div');
    root.append(V.seg([['00', 'x = (0, 0)'], ['01', 'x = (0, 1)'], ['10', 'x = (1, 0)'], ['11', 'x = (1, 1)']], '01', v => { inp = [Number(v[0]), Number(v[1])]; st.go(st.index); }),
        V.h('div', { class: 'vis-split' }, net, space), host);
    const calc = () => {
        const [x1, x2] = inp, z1 = Wt.a1 * x1 + Wt.b1 * x2 + Wt.c1, z2 = Wt.a2 * x1 + Wt.b2 * x2 + Wt.c2, h1 = sig(z1), h2 = sig(z2);
        const zo = Wt.o1 * h1 + Wt.o2 * h2 + Wt.co, y = sig(zo);
        return { x1, x2, z1, z2, h1, h2, zo, y };
    };
    function drawNet(k) {
        V.clear(net); const rr = V.rng(9), v = calc();
        const N = { x1: [50, 70], x2: [50, 190], h1: [190, 70], h2: [190, 190], y: [320, 130] };
        const edges = [['x1', 'h1', Wt.a1, 1], ['x2', 'h1', Wt.b1, 1], ['x1', 'h2', Wt.a2, 2], ['x2', 'h2', Wt.b2, 2], ['h1', 'y', Wt.o1, 3], ['h2', 'y', Wt.o2, 3]];
        edges.forEach(([a, b, w, s]) => {
            const on = k === s || (k === 4 && s === 3) || k >= 5, A = N[a], B = N[b];
            net.append(V.s('line', { x1: A[0], y1: A[1], x2: B[0], y2: B[1], stroke: w >= 0 ? V.C.sky : V.C.rose, 'stroke-width': on ? 3.2 : 1.4, opacity: on ? 1 : .35, class: on ? 'vis-flow' : '' }));
            V.text(net, A[0] + (B[0] - A[0]) * .38, A[1] + (B[1] - A[1]) * .38 - 8, String(w), { size: 11, color: w >= 0 ? V.C.sky : V.C.rose, weight: 700 });
        });
        const node = (key, label, val, hot) => {
            const [x, y] = N[key];
            net.append(V.s('circle', { cx: x, cy: y, r: 25, fill: hot ? V.C.amberS : '#fff', stroke: hot ? V.C.amber : V.C.slate, 'stroke-width': hot ? 3 : 1.8 }));
            V.text(net, x, y - 6, label, { size: 14, weight: 700 }); if (val !== null) V.text(net, x, y + 10, val, { size: 11, color: '#475569' });
        };
        node('x1', 'x₁', String(v.x1), k === 0); node('x2', 'x₂', String(v.x2), k === 0);
        node('h1', 'H₁', k >= 1 ? v.h1.toFixed(4) : null, k === 1); node('h2', 'H₂', k >= 2 ? v.h2.toFixed(5) : null, k === 2);
        node('y', 'Y', k >= 4 ? v.y.toFixed(4) : k === 3 ? 'z=' + v.zo.toFixed(2) : null, k >= 3);
        V.text(net, 190, 30, 'bias -2.6', { size: 11, color: '#64748b' }); V.text(net, 190, 232, 'bias -3.2', { size: 11, color: '#64748b' }); V.text(net, 320, 92, 'bias -2.7', { size: 11, color: '#64748b' });
    }
    const clip = 'xorclip' + Math.floor(Math.random() * 1e6);
    function drawSpace() {
        V.clear(space); const rr = V.rng(4), s = v => 30 + v * 160, t = v => 210 - v * 160;
        space.append(V.s('defs', {}, V.s('clipPath', { id: clip }, V.s('rect', { x: 8, y: 14, width: 210, height: 214 }))));
        V.rline(space, 20, 210, 210, 210, '#94a3b8', 1.2, rr); V.rline(space, 30, 220, 30, 20, '#94a3b8', 1.2, rr);
        V.text(space, 200, 226, 'x₁', { size: 12, color: '#64748b' }); V.text(space, 16, 28, 'x₂', { size: 12, color: '#64748b' });
        const line = (a, b, c, col) => { const p = []; [-0.2, 1.2].forEach(x => p.push([x, -(a * x + c) / b])); space.append(V.s('line', { x1: s(p[0][0]), y1: t(p[0][1]), x2: s(p[1][0]), y2: t(p[1][1]), stroke: col, 'stroke-width': 2, 'stroke-dasharray': '6 4', 'clip-path': 'url(#' + clip + ')' })); };
        line(Wt.a1, Wt.b1, Wt.c1, V.C.violet); line(Wt.a2, Wt.b2, Wt.c2, V.C.green);
        [[0, 0], [0, 1], [1, 0], [1, 1]].forEach(([a, b]) => {
            const on = a === inp[0] && b === inp[1], xor = a ^ b;
            space.append(V.s('circle', { cx: s(a), cy: t(b), r: on ? 11 : 8, fill: xor ? V.C.amber : '#fff', stroke: V.C.ink, 'stroke-width': on ? 3 : 1.5 }));
        });
        V.text(space, 110, 246, 'H₁ line (violet), H₂ line (green)', { size: 11, color: '#64748b' });
    }
    const steps = [
        { title: 'Inputs', text: () => `x₁ = ${inp[0]}, x₂ = ${inp[1]}. XOR should output ${inp[0] ^ inp[1]}.` },
        { title: 'Hidden neuron H₁', text: () => { const v = calc(); return `net = (${inp[0]})(-4.8) + (${inp[1]})(5.1) + (-2.6) = <b>${v.z1.toFixed(1)}</b>; H₁ = σ(${v.z1.toFixed(1)}) = <b>${v.h1.toFixed(4)}</b>.`; } },
        { title: 'Hidden neuron H₂', text: () => { const v = calc(); return `net = (${inp[0]})(4.6) + (${inp[1]})(-5.2) + (-3.2) = <b>${v.z2.toFixed(1)}</b>; H₂ = σ(${v.z2.toFixed(1)}) = <b>${v.h2.toFixed(5)}</b>.`; } },
        { title: 'Output weighted sum', text: () => { const v = calc(); return `net_Y = ${v.h1.toFixed(4)}(5.9) + ${v.h2.toFixed(5)}(5.2) + (-2.7) = <b>${v.zo.toFixed(4)}</b>.`; } },
        { title: 'Output activation and class', text: () => { const v = calc(); return `Y = σ(${v.zo.toFixed(4)}) = <b>${v.y.toFixed(4)}</b> → class <b>${v.y >= 0.5 ? 1 : 0}</b> (threshold 0.5). Expected XOR(${inp[0]}, ${inp[1]}) = ${inp[0] ^ inp[1]}, so the network is right.`; } },
        { title: 'Why the hidden layer matters', text: () => 'Each hidden neuron draws one straight line in input space (right). H₁ fires only above the violet line, H₂ only below the green one, and the output fires when either does - two lines together isolate the opposite corners (0, 1) and (1, 0), which no single line can do.' },
    ];
    const st = V.stepper(host, steps.map((s, k) => ({ title: s.title, text: '', draw: () => { drawNet(k); drawSpace(); host.querySelector('.vis-readout').innerHTML = s.text(); } })), { ms: 2200 });
};

// Learning rate on a one-dimensional loss curve
ML_VIS.l7_lr = function (root) {
    const L = w => (w - 3) ** 2 + 1, dL = w => 2 * (w - 3);
    let eta = 0.1, path = [-2], timer = null;
    const W = 560, H = 300, svg = V.svg(W, H), read = V.h('p', { class: 'vis-readout' });
    const sx = w => 40 + (w + 4) / 14 * 500, sy = l => 280 - Math.min(l, 70) / 70 * 260;
    const sl = V.slider('learning rate η', 2, 110, 1, 10, v => { eta = v / 100; reset(); }, v => (v / 100).toFixed(2));
    const run = V.btn('Take 12 steps', () => {
        reset(); let k = 0;
        timer = setInterval(() => { const w = path[path.length - 1]; path.push(w - eta * dL(w)); draw(); if (++k >= 12 || Math.abs(path[path.length - 1]) > 1e4) { clearInterval(timer); timer = null; } }, 280);
    }, 'primary');
    root.append(V.h('div', { class: 'vis-row' }, sl, run, V.btn('Reset', reset)), svg, read);
    function reset() { if (timer) { clearInterval(timer); timer = null; } path = [-2]; draw(); }
    function draw() {
        V.clear(svg); const rr = V.rng(6);
        V.arrow(svg, 40, 280, 548, 280, '#94a3b8', 1.2, rr); V.rline(svg, 40, 280, 40, 14, '#94a3b8', 1.2, rr);
        V.text(svg, 520, 294, 'weight w →', { size: 12, color: '#64748b' }); V.text(svg, 46, 14, 'loss L(w)', { size: 12, color: '#64748b', anchor: 'start' });
        let d = ''; for (let i = 0; i <= 140; i++) { const w = -4 + i / 10; d += (i ? 'L' : 'M') + sx(w) + ' ' + sy(L(w)); }
        svg.append(V.s('path', { d, fill: 'none', stroke: V.C.violet, 'stroke-width': 3 }));
        svg.append(V.s('circle', { cx: sx(3), cy: sy(1), r: 5, fill: V.C.green })); V.text(svg, sx(3), sy(1) + 16, 'minimum', { size: 11, color: V.C.green });
        path.forEach((w, i) => {
            const cw = Math.max(-4, Math.min(10, w));
            if (i) { const p = Math.max(-4, Math.min(10, path[i - 1])); svg.append(V.s('line', { x1: sx(p), y1: sy(L(p)), x2: sx(cw), y2: sy(L(cw)), stroke: V.C.amber, 'stroke-width': 2 })); }
            svg.append(V.s('circle', { cx: sx(cw), cy: sy(L(cw)), r: i === path.length - 1 ? 7 : 4, fill: V.C.amber, stroke: '#fff' }));
        });
        const w = path[path.length - 1], kind = eta < 0.5 ? (eta < 0.06 ? 'too small: each step barely moves, so training is slow' : 'steps shrink smoothly towards the minimum')
            : eta < 1 ? 'each step overshoots to the other side, but the bounces shrink (oscillating convergence)' : eta === 1 ? 'every step jumps exactly across the valley and never settles' : 'too large: every overshoot is bigger than the last, so the loss explodes (divergence)';
        read.innerHTML = `Update: w ← w - η·dL/dw with L(w) = (w - 3)² + 1. Step ${path.length - 1}: w = ${V.fmt(w, 3)}, loss = ${V.fmt(L(w), 3)}. With η = ${eta.toFixed(2)}, ${kind}.`;
    }
    draw();
};

// Gradient descent on a 3D loss surface
ML_VIS.l7_loss3d = async function (root) {
    const A = 0.5, Bc = 2.5, L = (u, v) => A * u * u + Bc * v * v, start = [-2.6, 1.3];
    let eta = 0.15;
    const read = V.h('p', { class: 'vis-readout' });
    root.append(V.h('div', { class: 'vis-row' }, V.slider('learning rate η', 2, 42, 1, 15, v => { eta = v / 100; run(); }, v => (v / 100).toFixed(2))), read);
    const sc = await V.scene3d(root, { radius: 9.5, phi: 0.9, theta: 0.85, target: [0, 1.2, 0], label: 'Loss surface with a descent path' });
    if (!sc) return;
    const { THREE, scene } = sc, Hs = 0.22;
    const g = new THREE.PlaneGeometry(6, 4, 40, 28); g.rotateX(-Math.PI / 2);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) p.setY(i, L(p.getX(i), p.getZ(i)) * Hs);
    g.computeVertexNormals();
    scene.add(new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: 0x7c3aed, transparent: true, opacity: .33, side: THREE.DoubleSide })),
        new THREE.LineSegments(new THREE.WireframeGeometry(g), new THREE.LineBasicMaterial({ color: 0x7c3aed, transparent: true, opacity: .18 })));
    const s1 = V.label3d(THREE, 'w₁', '#475569', .3); s1.position.set(3.3, 0, 0); scene.add(s1);
    const s2 = V.label3d(THREE, 'w₂', '#475569', .3); s2.position.set(0, 0, 2.3); scene.add(s2);
    const min = new THREE.Mesh(new THREE.SphereGeometry(.07, 12, 10), new THREE.MeshLambertMaterial({ color: 0x059669 })); scene.add(min);
    let group = null, cancel = null;
    function run() {
        if (cancel) cancel();
        if (group) scene.remove(group);
        group = new THREE.Group(); scene.add(group);
        const path = [start.slice()];
        for (let k = 0; k < 25; k++) { const [u, v] = path[path.length - 1]; path.push([u - eta * 2 * A * u, v - eta * 2 * Bc * v]); }
        const P = path.map(([u, v]) => { const cu = Math.max(-3, Math.min(3, u)), cv = Math.max(-2, Math.min(2, v)); return [cu, L(cu, cv) * Hs + 0.04, cv]; });
        cancel = V.tween(2200, t => {
            while (group.children.length) group.remove(group.children[0]);
            const n = Math.max(1, Math.round(t * (P.length - 1)));
            group.add(V.line3d(THREE, P.slice(0, n + 1), 0xd97706));
            P.slice(0, n + 1).forEach(q => { const m = new THREE.Mesh(new THREE.SphereGeometry(.05, 10, 8), new THREE.MeshLambertMaterial({ color: 0xd97706 })); m.position.set(...q); group.add(m); });
            sc.dirty();
        });
        const end = path[path.length - 1], diverge = Math.abs(end[1]) > 2 || !Number.isFinite(end[1]);
        read.innerHTML = `L(w₁, w₂) = 0.5·w₁² + 2.5·w₂² is a valley that is steep across w₂ and shallow along w₁. With η = ${eta.toFixed(2)}: ` +
            (diverge ? '<b>diverges</b> - the steep direction overshoots more on every step.' : eta > 0.2 ? 'it <b>zig-zags</b> across the steep sides while creeping along the valley floor.' : eta < 0.06 ? 'it moves very <b>slowly</b>: 25 steps are not enough to reach the bottom.' : 'it descends steadily towards the green minimum.') +
            ` After 25 steps the loss is ${Number.isFinite(L(...end)) ? V.fmt(L(...end), 4) : '∞'}.`;
    }
    run();
};
