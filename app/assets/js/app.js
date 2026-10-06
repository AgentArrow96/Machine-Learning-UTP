
        function switchConcept(group, panel, button) {
            document.querySelectorAll('[id^="' + group + '-"]').forEach(x => x.classList.remove('active'));
            const target = document.getElementById(group + '-' + panel);
            if (target) target.classList.add('active');
            const parent = button.closest('.teach-panel');
            if (parent) parent.querySelectorAll('.concept-tab').forEach(x => x.classList.remove('active'));
            button.classList.add('active');
        }
        function revealTeach(id) {
            const target = document.getElementById(id);
            if (target) target.classList.toggle('show');
        }

        // Maths rendering
        function renderMath(el) {
            if (!el || typeof renderMathInElement !== 'function') return;
            renderMathInElement(el, {
                delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
                ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
                throwOnError: false
            });
            fitMath(el);
        }

        // Section navigation
        function jumpTo(id) {
            const el = document.getElementById(id); if (!el) return;
            for (let p = el; p; p = p.parentElement) if (p.tagName === 'DETAILS') p.open = true;
            const y = el.getBoundingClientRect().top + window.scrollY - 16;
            window.scrollTo({ top: y, behavior: 'smooth' });
            el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
        }
        // Shrink wide display maths to fit the screen instead of scrolling sideways
        function fitMath(root) {
            (root || document).querySelectorAll('.katex-display').forEach(d => {
                const k = d.querySelector('.katex'); if (!k || !d.clientWidth) return;
                k.style.fontSize = '';
                const ratio = d.clientWidth / k.scrollWidth;
                if (ratio < 1) k.style.fontSize = Math.max(0.5, ratio * 0.98).toFixed(3) + 'em';
            });
        }
        document.addEventListener('toggle', e => { if (e.target.open) fitMath(e.target); }, true);
        window.addEventListener('resize', () => fitMath(document.querySelector('.page-section:not(.hidden)')));

        // Tap a hand-drawn diagram on a phone to view it full screen (turned sideways in portrait)
        function openDiagram(fig) {
            if (window.innerWidth > 780) return;
            const svg = fig.querySelector('svg.sketch'); if (!svg) return;
            const box = document.createElement('div'); box.className = 'diagram-viewer';
            const rotate = window.innerHeight > window.innerWidth;
            box.innerHTML = '<button type="button" class="diagram-close" aria-label="Close">Close</button>';
            const c = svg.cloneNode(true); if (rotate) c.classList.add('rotated'); box.append(c);
            box.addEventListener('click', () => box.remove());
            document.body.append(box);
        }
        document.addEventListener('click', e => { const f = e.target.closest('.vis-sketch'); if (f) openDiagram(f); });

        // Event wiring
        const ACTIONS = {
            print: () => window.print(), 'reset-progress': () => resetProgress(), 'start-quiz': () => startQuiz(), retry: () => retryMistakes(),
            'toggle-sidebar': () => toggleSidebar(), contents: () => jumpToContents(), submit: () => submitAnswer(), next: () => nextQuestion()
        };
        document.addEventListener('click', e => {
            const el = e.target.closest('[data-nav],[data-reveal],[data-tab],[data-jump],[data-toc],[data-sections],[data-quiz],[data-learned],[data-option],[data-action]');
            if (!el) return;
            const d = el.dataset;
            if (d.nav) navTo(d.nav);
            else if (d.reveal) revealTeach(d.reveal);
            else if (d.tab) switchConcept(d.tab, d.panel, el);
            else if (d.toc) tocGo(el, d.toc);
            else if (d.jump) jumpTo(d.jump);
            else if (d.sections) toggleSections(el, d.sections === 'open');
            else if (d.quiz) launchModuleQuiz(Number(d.quiz), d.quizMode);
            else if (d.learned) markLearned(Number(d.learned));
            else if (d.option) selectOption(d.option);
            else if (ACTIONS[d.action]) ACTIONS[d.action]();
        });
        document.addEventListener('input', e => { if (e.target.matches('.toc-search')) tocFilter(e.target); });
        document.addEventListener('change', e => { if (e.target.id === 'quiz-mode') updateModeNote(); });

        // Diagrams
        function fillDiagrams(root) {
            const src = window.ML_DIAGRAMS || {};
            (root || document).querySelectorAll('[data-diagram]').forEach(f => {
                const slot = f.querySelector('.sketch-slot');
                if (slot && !slot.firstChild && src[f.dataset.diagram]) slot.innerHTML = src[f.dataset.diagram];
            });
        }

        // Section list
        function tocGo(btn, id) { const p = btn.closest('details.toc-panel'); if (p) p.open = false; jumpTo(id); }
        function tocFilter(inp) {
            const q = inp.value.trim().toLowerCase(), list = inp.parentElement.querySelector('.toc-list');
            let group = null, shown = 0;
            list.querySelectorAll('li').forEach(li => {
                if (li.classList.contains('toc-group')) { group = li; li.hidden = !!q; return; }
                const hit = !q || (li.textContent + ' ' + (li.dataset.full || '')).toLowerCase().includes(q); li.hidden = !hit; if (hit) shown++;
            });
            list.dataset.empty = shown ? '' : 'No matching sections';
        }
        // Contents button
        function activeToc() { const s = document.querySelector('.page-section:not(.hidden)'); return s ? s.querySelector('.topic-index') : null; }
        function jumpToContents() { const t = activeToc(); if (!t) return; t.open = true; jumpTo(t.id); const i = t.querySelector('.toc-search'); if (i) i.focus({ preventScroll: true }); }
        function updateContentsFab() {
            const fab = document.getElementById('contents-fab'), t = activeToc();
            if (fab) fab.hidden = !(t && t.getBoundingClientRect().bottom < 0);
        }
        window.addEventListener('scroll', updateContentsFab, { passive: true });
        function toggleSections(btn, open) {
            const box = btn.closest('.markdown-body');
            if (box) box.querySelectorAll('details.note-section').forEach(d => d.open = open);
        }

        const QUESTION_BANK = window.QUESTION_BANK || [], DIFFICULT_BANK = window.DIFFICULT_BANK || [], HARD_BANK = window.HARD_BANK || [];
        const BLOOM = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'];
        let quizQuestions = [], quizIndex = 0, quizCorrect = 0, quizAnswered = 0, selectedAnswers = new Set(), lastMistakes = [], currentMistakes = [], quizLog = [], quizMode = 'standard';

        // Page loading: each page's content lives in assets/pages/<id>.js and calls ML_PAGE when loaded
        const pageLoads = {}, pageReady = {};
        function setupPage(sec) {
            fillDiagrams(sec);
            renderMath(sec);
            initVisuals(sec);
            if (sec.id === 'sec-home') updateProgressDashboard();
        }
        function ML_PAGE(id, html) {
            const sec = document.getElementById('sec-' + id); if (!sec) return;
            sec.innerHTML = html;
            sec.dataset.loaded = '1';
            setupPage(sec);
            if (pageReady[id]) pageReady[id]();
        }
        function loadPage(id) {
            const sec = document.getElementById('sec-' + id);
            if (!sec) return Promise.resolve();
            if (sec.dataset.loaded) return Promise.resolve(sec);
            if (!pageLoads[id]) pageLoads[id] = new Promise(resolve => {
                pageReady[id] = () => resolve(sec);
                sec.innerHTML = '<p class="page-loading">Loading...</p>';
                const s = document.createElement('script');
                s.src = 'assets/pages/' + id + '.js';
                s.onerror = () => { sec.innerHTML = '<p class="page-loading">Could not load this page.</p>'; resolve(sec); };
                document.head.append(s);
            });
            return pageLoads[id];
        }
        function preloadPages() {
            const ids = [...document.querySelectorAll('.page-section')].map(s => s.dataset.page).filter(id => !document.getElementById('sec-' + id).dataset.loaded);
            const idle = window.requestIdleCallback || (fn => setTimeout(fn, 50));
            const next = () => { const id = ids.shift(); if (id) loadPage(id).then(() => idle(next)); };
            idle(next);
        }

        function navTo(page) {
            document.querySelectorAll('.page-section').forEach(s => s.classList.add('hidden'));
            const target = document.getElementById('sec-' + page);
            if (target) { target.classList.remove('hidden'); fitMath(target); }
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            const btn = document.getElementById('nav-' + page);
            if (btn) { btn.classList.add('active'); const group = btn.closest('details.nav-module'); if (group) group.open = true; }
            window.scrollTo({ top: 0, behavior: 'smooth' });
            history.replaceState(null, '', '#' + page);
            updateContentsFab();
            if (window.innerWidth <= 1024) closeSidebar();
            return loadPage(page).then(sec => { if (sec && !sec.classList.contains('hidden')) { fitMath(sec); updateContentsFab(); } return sec; });
        }
        function closeSidebar() {
            document.getElementById('sidebar').classList.remove('open');
            const menu = document.getElementById('mobile-menu'); if (menu) { menu.setAttribute('aria-expanded', 'false'); menu.setAttribute('aria-label', 'Open navigation'); }
        }
        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar'), menu = document.getElementById('mobile-menu');
            const isOpen = sidebar.classList.toggle('open');
            if (menu) { menu.setAttribute('aria-expanded', String(isOpen)); menu.setAttribute('aria-label', isOpen ? 'Close navigation' : 'Open navigation'); }
        }

        function getProgress() {
            try { return JSON.parse(localStorage.getItem('mlStudyProgress') || '{}'); } catch (e) { return {}; }
        }
        function saveProgress(p) { try { localStorage.setItem('mlStudyProgress', JSON.stringify(p)); } catch (e) { } updateProgressDashboard(); }
        function markLearned(n) { const p = getProgress(); p[n] = p[n] || { learned: false, best: 0 }; p[n].learned = true; saveProgress(p); alert('Lecture ' + n + ' marked as learned.'); }
        function resetProgress() { if (!confirm('Reset learned status and best quiz scores?')) return; try { localStorage.removeItem('mlStudyProgress'); } catch (e) { } updateProgressDashboard(); }
        function updateProgressDashboard() {
            const p = getProgress();
            for (let n = 1; n <= 4; n++) {
                const x = p[n] || { learned: false, best: 0 }; const pct = Math.round(((x.learned ? 1 : 0) + (x.best >= 80 ? 1 : 0)) / 2 * 100);
                const status = document.getElementById('status-m' + n), best = document.getElementById('best-m' + n), fill = document.getElementById('progress-m' + n);
                if (status) status.textContent = x.learned ? 'Learned' : 'Not learned'; if (best) best.textContent = 'Best: ' + (x.best || 0) + '%'; if (fill) fill.style.width = pct + '%';
                const hard = document.getElementById('hard-m' + n);
                if (hard) hard.textContent = 'Hard: ' + (x.hardTwin != null ? x.hardTwin + '%' : 'not tried') + ' · Bloom: ' + (x.hard != null ? x.hard + '%' : 'not tried');
            }
        }
        function shuffle(arr) { const a = [...arr]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1));[a[i], a[j]] = [a[j], a[i]]; } return a; }
        function launchModuleQuiz(n, mode) {
            return navTo('simulator').then(() => {
                mode = mode || 'standard';
                document.getElementById('quiz-mode').value = mode; document.getElementById('quiz-module').value = String(n);
                document.getElementById('quiz-count').value = mode === 'standard' ? '20' : 'all';
                document.getElementById('quiz-order').value = mode === 'standard' || mode === 'hard' ? 'shuffle' : 'sequence';
                updateModeNote(); startQuiz();
            });
        }
        function updateModeNote() { const note = document.getElementById('quiz-mode-note'); if (note) note.hidden = !isBloomMode(document.getElementById('quiz-mode').value); }
        function isBloomMode(mode) { return mode === 'difficult' || mode === 'extensions'; }
        // Bloom challenge: lowest Bloom level first; Bloom-graded before extensions within a level
        function bloomOrder(pool, mix) {
            const key = q => q.bloom_rank * 2 + (q.ext ? 1 : 0);
            if (mix) return shuffle(pool).sort((a, b) => a.bloom_rank - b.bloom_rank);
            return [...pool].sort((a, b) => key(a) - key(b) || a.module - b.module || Number(a.number.slice(1)) - Number(b.number.slice(1)));
        }
        function startQuiz(custom = null) {
            const mod = document.getElementById('quiz-module').value, count = document.getElementById('quiz-count').value, order = document.getElementById('quiz-order').value;
            if (!custom) quizMode = document.getElementById('quiz-mode').value;
            const bloom = isBloomMode(quizMode), source = bloom ? DIFFICULT_BANK : quizMode === 'hard' ? HARD_BANK : QUESTION_BANK;
            let pool = custom ? [...custom] : source.filter(q => (mod === 'all' || String(q.module) === mod) && (quizMode !== 'extensions' || q.ext));
            if (bloom) {
                if (count !== 'all' && !custom) pool = shuffle(pool).slice(0, Number(count));
                pool = bloomOrder(pool, order === 'shuffle' || !!custom);
            } else {
                const no = q => q.twin || q.number;
                if (order === 'shuffle' || custom) pool = shuffle(pool); else pool.sort((a, b) => a.module - b.module || no(a) - no(b));
                if (count !== 'all' && !custom) pool = pool.slice(0, Number(count));
            }
            quizQuestions = pool; quizIndex = 0; quizCorrect = 0; quizAnswered = 0; currentMistakes = []; quizLog = []; selectedAnswers = new Set();
            document.getElementById('retry-btn').style.display = 'none'; updateQuizStats(); renderQuestion();
        }
        function retryMistakes() { if (!lastMistakes.length) return; document.getElementById('quiz-count').value = 'all'; startQuiz(lastMistakes); }
        // Quiz
        function renderQuestion() {
            const card = document.getElementById('quiz-card');
            if (!quizQuestions.length) { card.innerHTML = '<div class="empty-state"><div><h3>No matching questions</h3><p>Choose a different lecture or question count.</p></div></div>'; return; }
            if (quizIndex >= quizQuestions.length) { renderResults(); return; }
            const q = quizQuestions[quizIndex]; selectedAnswers = new Set();
            const options = q.options.map(([letter, text]) => `<button class="quiz-option" data-letter="${letter}" data-option="${letter}"><span class="letter">${letter}</span><span>${text}</span></button>`).join('');
            const badges = q.bloom ? `<div class="bloom-badges"><span class="bloom-chip bloom-${q.bloom_rank}">${q.bloom}</span>${q.ext ? `<span class="ext-badge">Extension · ${q.ext}</span>` : ''}</div>`
                : q.twin ? `<div class="bloom-badges"><span class="twin-badge">Hard · harder version of Q${q.twin}</span></div>` : '';
            card.innerHTML = `<div class="quiz-meta"><span>Lecture ${q.module} · ${q.bloom || q.twin ? q.number : 'Question ' + q.number}</span><span>${quizIndex + 1} of ${quizQuestions.length}</span></div>${badges}<div class="quiz-question">${q.question}</div><div class="quiz-options">${options}</div><div class="quiz-footer"><span class="quiz-hint">${q.multiple ? 'Select all required responses, then submit.' : 'Select one response.'}</span>${q.multiple ? '<button class="primary-btn" id="submit-multi" data-action="submit">Submit selections</button>' : ''}</div><div id="quiz-feedback"></div>`;
            renderMath(card);
            document.getElementById('quiz-progress').style.width = ((quizIndex) / quizQuestions.length * 100) + '%';
        }
        function selectOption(letter) {
            const q = quizQuestions[quizIndex]; const btn = document.querySelector(`.quiz-option[data-letter="${letter}"]`);
            if (!q.multiple) { selectedAnswers = new Set([letter]); document.querySelectorAll('.quiz-option').forEach(x => x.classList.remove('selected')); btn.classList.add('selected'); submitAnswer(); return; }
            if (selectedAnswers.has(letter)) { selectedAnswers.delete(letter); btn.classList.remove('selected'); } else { selectedAnswers.add(letter); btn.classList.add('selected'); }
        }
        function sameSet(a, b) { return a.size === b.size && [...a].every(x => b.has(x)); }
        function submitAnswer() {
            const q = quizQuestions[quizIndex]; if (!selectedAnswers.size) return;
            const correctSet = new Set(q.answers), isCorrect = sameSet(selectedAnswers, correctSet); quizAnswered++; quizLog.push([q, isCorrect]); if (isCorrect) quizCorrect++; else currentMistakes.push(q);
            document.querySelectorAll('.quiz-option').forEach(btn => { const l = btn.dataset.letter; btn.disabled = true; btn.classList.remove('selected'); if (correctSet.has(l)) btn.classList.add('correct'); else if (selectedAnswers.has(l)) btn.classList.add('incorrect'); });
            const submit = document.getElementById('submit-multi'); if (submit) submit.style.display = 'none';
            const fb = document.getElementById('quiz-feedback'); fb.className = 'quiz-feedback ' + (isCorrect ? 'good' : 'bad'); fb.innerHTML = `<strong>${isCorrect ? 'Correct.' : 'Not quite.'} Answer: ${q.answers.join(', ')}</strong><div style="margin-top:5px">${q.explanation || ''}</div>${twinCompare(q)}<div style="margin-top:12px"><button class="primary-btn" data-action="next">${quizIndex + 1 === quizQuestions.length ? 'View result' : 'Next question →'}</button></div>`;
            renderMath(fb);
            updateQuizStats(); document.getElementById('quiz-progress').style.width = ((quizIndex + 1) / quizQuestions.length * 100) + '%';
        }
        // Hard mode: show the standard question the twin was built from
        function twinCompare(q) {
            const o = q.twin && QUESTION_BANK.find(x => x.module === q.module && x.number === q.twin);
            if (!o) return '';
            const opts = o.options.map(([l, t]) => `<div class="bank-option"><b>${l}.</b> ${t}</div>`).join('');
            return `<details class="twin-compare"><summary>Compare with the standard question (Q${o.number})</summary><div class="twin-compare-body"><div class="harder-question">${o.question}</div><div class="bank-options">${opts}</div><p><b>Answer: ${o.answers.join(', ')}</b></p></div></details>`;
        }
        function nextQuestion() { quizIndex++; renderQuestion(); }
        function updateQuizStats() { document.getElementById('quiz-score').textContent = quizCorrect + ' / ' + quizAnswered; document.getElementById('quiz-accuracy').textContent = (quizAnswered ? Math.round(quizCorrect / quizAnswered * 100) : 0) + '%'; }
        function renderResults() {
            const pct = quizAnswered ? Math.round(quizCorrect / quizAnswered * 100) : 0; lastMistakes = [...currentMistakes];
            document.getElementById('retry-btn').style.display = lastMistakes.length ? 'block' : 'none';
            const lectures = [...new Set(quizQuestions.map(q => q.module))], hard = quizQuestions.some(q => q.bloom);
            const key = quizMode === 'difficult' ? 'hard' : quizMode === 'hard' ? 'hardTwin' : quizMode === 'standard' ? 'best' : null;
            if (lectures.length === 1 && key) {
                const n = lectures[0], p = getProgress(); p[n] = p[n] || { learned: false, best: 0 };
                p[n][key] = Math.max(p[n][key] || 0, pct);
                saveProgress(p);
            }
            let levels = '';
            if (hard) {
                const rows = BLOOM.map((b, i) => [b, 'bloom-chip bloom-' + (i + 1), quizLog.filter(([q]) => q.bloom === b)]);
                rows.push(['Extensions', 'ext-badge', quizLog.filter(([q]) => q.ext)]);
                levels = '<div class="bloom-results">' + rows.filter(r => r[2].length).map(([b, cls, xs]) => { const ok = xs.filter(x => x[1]).length; return `<div class="bloom-result"><span class="${cls}">${b}</span><strong>${ok} / ${xs.length}</strong><div class="bloom-bar"><i data-w="${Math.round(ok / xs.length * 100)}"></i></div></div>`; }).join('') + '<p class="bloom-note">Each Bloom row counts every question at that level, extensions included.</p></div>';
            }
            const label = pct >= 90 ? 'Strongly prepared' : pct >= 80 ? 'Nearly ready' : pct >= 65 ? 'Core knowledge is present' : 'Review the weak concepts before retesting';
            document.getElementById('quiz-card').innerHTML = `<div style="text-align:center"><div class="result-circle"><div><strong>${pct}%</strong><div style="font-size:11px;color:#64748b">${quizCorrect} / ${quizAnswered}</div></div></div><h3 style="font-size:27px;color:#0f172a;margin:0 0 6px">${label}</h3><p style="color:#64748b">Missed questions: ${lastMistakes.length}. ${lastMistakes.length ? 'Use Retry missed questions to repair only the gaps.' : 'No mistakes. Try a larger mixed test.'}</p>${levels}<div class="action-row" style="justify-content:center"><button class="primary-btn" data-action="start-quiz">Retake / reshuffle</button>${lastMistakes.length ? '<button class="secondary-btn" data-action="retry">Retry mistakes</button>' : ''}</div></div>`;
            document.querySelectorAll('.bloom-bar i').forEach(i => { i.style.width = i.dataset.w + '%'; });
            document.getElementById('quiz-progress').style.width = '100%'; updateQuizStats();
        }

        const LAST_UPDATED_REPO = 'AgentArrow96/Machine-Learning-UTP', LAST_UPDATED_BRANCH = 'main', LAST_UPDATED_CACHE_KEY = 'mlLastPushISO';
        let lastPushDate = null;
        function relativeTimeFrom(date) {
            const sec = Math.floor((Date.now() - date.getTime()) / 1000);
            if (sec < 45) return 'just now';
            const min = Math.floor(sec / 60); if (min < 60) return min + (min === 1 ? ' minute ago' : ' minutes ago');
            const hr = Math.floor(min / 60); if (hr < 24) return hr + (hr === 1 ? ' hour ago' : ' hours ago');
            const day = Math.floor(hr / 24); if (day < 30) return day + (day === 1 ? ' day ago' : ' days ago');
            const mo = Math.floor(day / 30); if (mo < 12) return mo + (mo === 1 ? ' month ago' : ' months ago');
            const yr = Math.floor(mo / 12); return yr + (yr === 1 ? ' year ago' : ' years ago');
        }
        function renderLastUpdated() {
            const wrap = document.getElementById('last-updated'), text = document.getElementById('last-updated-text');
            if (!wrap || !text || !lastPushDate) return;
            text.textContent = 'Last updated ' + relativeTimeFrom(lastPushDate);
            wrap.title = 'Last GitHub push: ' + lastPushDate.toLocaleString();
        }
        async function fetchLastPush() {
            try {
                const res = await fetch(`https://api.github.com/repos/${LAST_UPDATED_REPO}/commits/${LAST_UPDATED_BRANCH}`, { headers: { Accept: 'application/vnd.github+json' } });
                if (!res.ok) throw new Error('bad response');
                const data = await res.json();
                const iso = data && data.commit && data.commit.committer && data.commit.committer.date;
                if (!iso) throw new Error('no date in response');
                lastPushDate = new Date(iso);
                try { localStorage.setItem(LAST_UPDATED_CACHE_KEY, iso); } catch (e) { }
                renderLastUpdated();
            } catch (e) {
                if (!lastPushDate) {
                    let cached = null; try { cached = localStorage.getItem(LAST_UPDATED_CACHE_KEY); } catch (e2) { }
                    if (cached) { lastPushDate = new Date(cached); renderLastUpdated(); }
                    else { const text = document.getElementById('last-updated-text'); if (text) text.textContent = 'Last updated: unavailable'; }
                }
            }
        }
        function initLastUpdated() {
            fetchLastPush();
            setInterval(renderLastUpdated, 30000);
            setInterval(fetchLastPush, 5 * 60 * 1000);
        }
        document.addEventListener('DOMContentLoaded', () => {
            initLastUpdated();
            const hash = location.hash.replace('#', '');
            navTo(hash && document.getElementById('sec-' + hash) ? hash : 'home').then(preloadPages);
        });
