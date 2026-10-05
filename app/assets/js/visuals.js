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
    fmt(n, d = 2) { return Number.isFinite(n) ? Number(n.toFixed(d)).toLocaleString('en-US', { maximumFractionDigits: d }) : '—'; },
    clear(e) { while (e.firstChild) e.removeChild(e.firstChild); },
    median(a) { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; },
    mean(a) { return a.reduce((x, y) => x + y, 0) / a.length; },
    std(a) { const m = V.mean(a); return Math.sqrt(V.mean(a.map(x => (x - m) ** 2))); },

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
        const bar = V.h('div', { class: 'vis-row vis-3dbar' }, V.btn('＋ zoom', () => zoom(0.85)), V.btn('－ zoom', () => zoom(1.18)), V.btn('⟳ auto-rotate', () => { auto = !auto; needs = true; }), V.h('span', { class: 'vis-note' }, 'Drag to rotate'));
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
        info.textContent = done ? `Converged after ${iter} iterations — the centres stopped moving: three groups discovered without any labels.`
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
        const verdict = deg <= 2 ? '<b style="color:#7c3aed">Underfitting</b> — too simple: both errors are high.' : te > 3 * Math.max(tr, 0.02) && deg >= 6 ? '<b style="color:#e11d48">Overfitting</b> — the curve chases the training points; test error grows.' : '<b style="color:#059669">Good generalization</b> — captures the trend, not the noise.';
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
            (outlier > 300 ? `The outlier drags the mean ${V.fmt(mean - med, 1)} above the median, so <b>median</b> is the more representative fill.` : 'Without an outlier, mean and median are close — either works.');
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
        V.text(svg, 20, 165, mode === 'raw' ? 'Raw units: income differences are thousands, age differences are single digits.' : mode === 'minmax' ? "x' = (x − min) / (max − min) puts both on 0–1." : 'z = (x − μ) / σ puts both on "standard deviations from the mean".', { size: 13, anchor: 'start', color: '#64748b' });
        read.innerHTML = mode === 'raw' ? 'Without scaling, income decides almost the whole distance — not because it matters more, only because its units are bigger. Distance-based models (KNN, SVM) would effectively ignore age.' : 'After scaling, both features contribute on comparable footing. The order of customers is unchanged — same pattern, different ruler.';
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
            panel('1 · Split', step === 1, step >= 1 ? V.h('div', {}, Object.entries(groups).map(([g, ps]) => V.h('div', { class: 'vis-group' }, V.h('small', {}, g), ps.map(p => chip(g, p))))) : V.h('i', {}, '…')),
            panel('2 · Apply ' + agg + '()', step === 2, step >= 2 ? V.h('div', {}, Object.entries(groups).map(([g, ps]) => V.h('div', {}, chip(g, `${g}: ${V.fmt(aggf[agg](ps), 1)}`)))) : V.h('i', {}, '…')),
            panel('3 · Combine', step === 3, step >= 3 ? V.h('table', { class: 'vis-table' }, V.h('tr', {}, V.h('th', {}, 'user_type'), V.h('th', {}, agg + ' price')), Object.entries(groups).map(([g, ps]) => V.h('tr', {}, V.h('td', {}, g), V.h('td', {}, V.fmt(aggf[agg](ps), 1))))) : V.h('i', {}, '…')));
        stepBtn.textContent = step === 3 ? '↺ Start again' : 'Next step';
    }
    draw();
};

ML_VIS.l2_chart_picker = function (root) {
    const Q = [['line', 'How did monthly sales change over the year?'], ['bar', 'Which branch sells the most?'], ['hist', 'What do product prices look like?'], ['box', 'Are there unusual ages?'], ['scatter', 'Does advertising relate to sales?']];
    const svg = V.svg(520, 250), read = V.h('p', { class: 'vis-readout' });
    root.append(V.seg(Q.map(([k, q]) => [k, q]), 'line', v => draw(v)), svg, read);
    const why = { line: 'Line chart — a trend over time.', bar: 'Bar chart — compare a quantity across discrete categories.', hist: 'Histogram — the distribution of one numeric variable (skew, long tail).', box: 'Box plot — centre, spread and outliers at a glance.', scatter: 'Scatter plot — the relationship between two numeric variables.' };
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
        V.text(svg, 280, 205, mode === 'train' ? 'The test point is scaled with parameters it never influenced.' : 'max came FROM the test point — the scaler already "saw" the exam.', { size: 15, color: mode === 'train' ? V.C.green : V.C.red });
        read.innerHTML = mode === 'train' ? `Train-only fit: (${test} − 10) / (30 − 10) = <b>${V.fmt(sc(test), 2)}</b>. Values outside 0–1 are fine and honest: the test set really is unlike the training data.` : `Leaky fit: the test value set the maximum, so it becomes exactly <b>1</b> and the training data is squashed to ${V.fmt(sc(10), 2)}–${V.fmt(sc(30), 2)}. Test information has entered preprocessing, so evaluation will look better than reality.`;
        code.textContent = mode === 'train' ? 'scaler.fit(X_train)\nX_train = scaler.transform(X_train)\nX_test  = scaler.transform(X_test)   # correct' : 'scaler.fit(X_all)                     # wrong: includes test rows\nX_train = scaler.transform(X_train)\nX_test  = scaler.transform(X_test)';
    }
    draw();
};
// Lecture 3 visuals
ML_VIS.l3_interaction3d = async function (root) {
    let mode = 'both';
    const read = V.h('p', { class: 'vis-readout' });
    root.append(V.seg([['add', 'Additive: a·Attack + b·Defense'], ['inter', 'Interaction: Attack × Defense'], ['both', 'Both']], mode, v => { mode = v; show(); }), read);
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
    const lab = V.label3d(THREE, '49 × 49 = 2401', '#b45309', 0.3); lab.position.set(px + 0.2, py + 0.7, pz); scene.add(lab);
    scene.add(V.line3d(THREE, [[px, 0, pz], [px, py, pz]], 0xd97706, true));
    function show() {
        add.visible = mode !== 'inter'; inter.visible = mode !== 'add'; dot.visible = lab.visible = mode !== 'add'; S.dirty();
        read.innerHTML = mode === 'add' ? 'Additive model: raising Attack always adds the same amount, whatever Defense is — a flat, tilted plane.' : mode === 'inter' ? 'Interaction feature: the surface bends upward only when <b>both</b> are high. The engineered column Attack × Defense lets even a simple model capture that.' : 'Compare: the plane (blue) cannot rise faster in the "both high" corner; the interaction surface (red) can.';
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
            (count <= 30 ? `<div class="vis-terms">${terms().join(' · ')}</div>` : `<div class="vis-terms">Too many to list — that is <b>feature explosion</b>.</div>`);
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
        read.innerHTML = mode === 'fixed' ? `Equal widths of about RM${Math.round((hi - lo) / k / 1000)}k. Counts: <b>${counts.join(' · ')}</b>. ${empty ? `${empty} bin(s) are nearly empty — the long tail wastes them.` : ''}` : `Every bin holds about ${Math.round(data.length / k)} people; the widths adapt (narrow where data is dense, wide in the tail). Counts: <b>${counts.join(' · ')}</b>.`;
    }
    draw();
};

ML_VIS.l3_log = function (root) {
    const data = skewedIncome();
    let mode = 'raw';
    const svg = V.svg(560, 250), read = V.h('p', { class: 'vis-readout' });
    root.append(V.seg([['raw', 'Raw income'], ['log', 'log10(income)']], mode, v => { mode = v; draw(); }), svg, read,
        V.h('div', { class: 'vis-terms' }, '10 → 1 · 100 → 2 · 1,000 → 3 · 1,000,000 → 6 — big values shrink far more than small ones'));
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
            label: `1 column. Warning: implies ${cats.join(' &lt; ')} — fine only if the categories are genuinely <b>ordinal</b>.`,
            onehot: `m = ${m} categories → <b>${m}</b> columns, exactly one "hot" per row. No false order. <code>pd.get_dummies(df, columns=["genre"])</code>`,
            dummy: `<b>${m - 1}</b> columns: ${cats[0]} is the reference row (all 0). Avoids the dummy-variable trap. <code>drop_first=True</code>`,
            effect: `<b>${m - 1}</b> columns, but the reference ${cats[0]} is coded −1 everywhere — comparisons are against the overall mean.`,
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
            row('4 · Remove stopwords', nostop, 'the, is, are, and … carry little information', null),
            row('5a · Stemming (heuristic)', nostop.map(stem), 'chops suffixes — may give non-words like "studi"'),
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
        read.innerHTML = `θ = <b>${V.fmt(th, 1)}°</b> → cos θ = <b>${V.fmt(c, 3)}</b>. ${c > 0.95 ? 'Very similar documents (same direction) — length does not matter.' : c < 0.2 ? 'Nearly orthogonal: the documents share almost no terms.' : 'Partly similar.'}`;
    }
    draw();
};

ML_VIS.l4_embed3d = async function (root) {
    const W = { king: [1.3, 1.1, 0.2], queen: [1.3, -1.1, 0.2], man: [-0.2, 1.1, 0.3], woman: [-0.2, -1.1, 0.3], prince: [0.7, 0.9, -0.7], princess: [0.7, -0.9, -0.7], cat: [-0.9, 0.1, 1.5], dog: [-0.5, 0.6, 1.8], kitten: [-1.3, -0.4, 1.2], car: [-1.4, 0.2, -1.4], truck: [-1.7, 0.7, -1.2], bus: [-0.9, -0.3, -1.8] };
    const read = V.h('p', { class: 'vis-readout' }, 'Toy 3-D embedding (real ones have 100–300 dimensions). Similar words sit close together.');
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
    ctl.append(V.btn('king − man + woman = ?', () => {
        clear(); const k = W.king, d = W.woman.map((v, i) => v - W.man[i]), t = k.map((v, i) => v + d[i]);
        add(V.line3d(THREE, [W.man, W.woman], 0x0284c7, true)); add(V.line3d(THREE, [k, t], 0xe11d48));
        const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.15), new THREE.MeshLambertMaterial({ color: 0xe11d48 })); m.position.set(...t); add(m);
        read.innerHTML = 'Take the "man → woman" direction (dashed blue) and apply it to <b>king</b> (red arrow): you land on <b>queen</b>. Directions in the space capture relationships.'; S.dirty();
    }, 'primary'), V.btn('Nearest words to "cat"', () => {
        clear(); const near = Object.entries(W).filter(([w]) => w !== 'cat').map(([w, p]) => [w, Math.hypot(...p.map((v, i) => v - W.cat[i]))]).sort((a, b) => a[1] - b[1]).slice(0, 3);
        near.forEach(([w]) => add(V.line3d(THREE, [W.cat, W[w]], 0x059669, true)));
        read.innerHTML = `Closest to <b>cat</b>: ${near.map(([w, d]) => `${w} (${V.fmt(d, 2)})`).join(', ')}. With one-hot vectors, cat would be exactly as far from car as from dog.`; S.dirty();
    }), V.btn('Clear', () => { clear(); read.textContent = 'Toy 3-D embedding (real ones have 100–300 dimensions). Similar words sit close together.'; }));
};

ML_VIS.l4_timestamp = function (root) {
    const zones = [['8', 'Malaysia (UTC+8)'], ['0', 'London (UTC+0)'], ['-5', 'New York (UTC−5)'], ['9', 'Tokyo (UTC+9)']];
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
        read.innerHTML = { rgb: 'One colour image = an <b>(m, n, 3)</b> array: every pixel has red, green and blue intensities.', split: 'The same image as <b>three stacked matrices</b> — the R, G and B channels (drag to look from the side).', gray: 'Grayscale: <b>Y = 0.2125R + 0.7154G + 0.0721B</b> collapses three channels into one <b>(m, n)</b> matrix.', flat: 'Flattening: the 8 × 8 matrix becomes one row of <b>64</b> numbers — a feature vector, but neighbours are no longer next to each other.' }[mode];
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
        V.h('div', { class: 'vis-trio' }, V.h('figure', {}, c1, V.h('figcaption', {}, 'pixels (grayscale)')), V.h('figure', {}, c2, V.h('figcaption', {}, 'edges: sharp intensity change')), V.h('figure', {}, c3, V.h('figcaption', {}, 'HOG idea: dominant edge direction per 8×8 cell'))),
        V.h('p', { class: 'vis-readout' }, 'Edges keep shapes and boundaries and ignore flat regions. HOG goes one step further and records which way the edges point in each cell — a compact description of shape.'));
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
