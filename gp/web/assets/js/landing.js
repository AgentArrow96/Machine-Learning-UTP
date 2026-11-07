'use strict';

(() => {
    const ROUTES = ['overview', 'dataset', 'preprocessing', 'methodology', 'walkthrough', 'results', 'explorer', 'training', 'interpretation', 'reproducibility',
        'summary', 'problem', 'data', 'prepare', 'features', 'method', 'evaluate', 'sustain', 'limits', 'sources',
        'changes', 'improve', 'undersample', 'imbalance'];
    const SECTIONS = ['why', 'data', 'how', 'found', 'impact', 'limits'];
    const hash = location.hash.slice(1);
    if (ROUTES.includes(hash) && !SECTIONS.includes(hash)) {
        location.replace(`app.html${location.search}#${hash}`);
        return;
    }

    const KEY = 'floodnet-landing-theme';
    const read = () => {
        try {
            return localStorage.getItem(KEY);
        } catch {
            return null;
        }
    };
    const stored = read();
    document.documentElement.dataset.theme = stored === 'light' || stored === 'dark' ? stored : 'dark';
    const still = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

    const setTheme = theme => {
        document.documentElement.dataset.theme = theme;
        try {
            localStorage.setItem(KEY, theme);
        } catch {
        }
        const button = document.getElementById('lp-theme');
        if (button) {
            button.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
            button.setAttribute('aria-pressed', String(theme === 'light'));
        }
    };

    const rail = (id, direction) => {
        const element = document.getElementById(id);
        if (!element) return;
        const card = element.querySelector(':scope > *');
        const step = card ? card.getBoundingClientRect().width + 24 : element.clientWidth * 0.8;
        element.scrollBy({ left: direction * step, behavior: still() ? 'auto' : 'smooth' });
    };

    const updateRail = element => {
        const max = element.scrollWidth - element.clientWidth - 2;
        document.querySelectorAll(`[data-rail-prev="${element.id}"]`).forEach(b => { b.disabled = element.scrollLeft <= 2; });
        document.querySelectorAll(`[data-rail-next="${element.id}"]`).forEach(b => { b.disabled = element.scrollLeft >= max; });
    };

    const stack = () => {
        const figure = document.getElementById('lp-stack');
        if (!figure) return;
        if (still()) {
            figure.style.setProperty('--lift', '0.8');
            return;
        }
        let frame = 0;
        const lift = () => {
            frame = 0;
            const progress = Math.min(1, Math.max(0, scrollY / 300));
            figure.style.setProperty('--lift', (0.35 + 0.65 * progress).toFixed(3));
        };
        addEventListener('scroll', () => { if (!frame) frame = requestAnimationFrame(lift); }, { passive: true });
        lift();
        const hero = figure.closest('.lp-hero');
        hero?.addEventListener('pointermove', event => {
            if (event.pointerType !== 'mouse') return;
            const box = figure.getBoundingClientRect();
            const x = (event.clientX - (box.left + box.width / 2)) / box.width;
            const y = (event.clientY - (box.top + box.height / 2)) / box.height;
            figure.style.setProperty('--tx', `${(Math.max(-1, Math.min(1, x)) * 4).toFixed(2)}deg`);
            figure.style.setProperty('--ty', `${(Math.max(-1, Math.min(1, y)) * -4).toFixed(2)}deg`);
        });
        hero?.addEventListener('pointerleave', () => {
            figure.style.setProperty('--tx', '0deg');
            figure.style.setProperty('--ty', '0deg');
        });
    };

    const tiles = () => {
        const host = document.getElementById('lp-tiles');
        if (!host) return;
        const cells = [];
        for (let row = 0; row < 6; row++) {
            for (let col = 0; col < 8; col++) cells.push(`<i style="background-position:${(col / 7 * 100).toFixed(2)}% ${(row / 5 * 100).toFixed(2)}%"></i>`);
        }
        host.innerHTML = cells.join('');
    };

    const countUp = element => {
        const target = element.textContent.trim();
        const match = target.match(/^(-?\d+(?:\.\d+)?)(.*)$/);
        if (!match) return;
        const value = Number(match[1]);
        const decimals = (match[1].split('.')[1] || '').length;
        const suffix = match[2];
        const start = performance.now();
        const step = now => {
            const t = Math.min(1, (now - start) / 900);
            const eased = 1 - Math.pow(2, -10 * t);
            element.textContent = t < 1 ? `${(value * eased).toFixed(decimals)}${suffix}` : target;
            if (t < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    };

    const loupe = () => {
        const plate = document.querySelector('#lp-scenes .lp-figure-plate');
        const image = plate?.querySelector('img');
        const lens = plate?.querySelector('.lp-loupe');
        if (!plate || !image || !lens) return;
        const zoom = 2.5;
        plate.addEventListener('pointermove', event => {
            if (event.pointerType !== 'mouse') return;
            const box = image.getBoundingClientRect();
            const x = event.clientX - box.left;
            const y = event.clientY - box.top;
            if (x < 0 || y < 0 || x > box.width || y > box.height) {
                plate.classList.remove('lp-loupe-on');
                return;
            }
            const plateBox = plate.getBoundingClientRect();
            lens.style.left = `${event.clientX - plateBox.left}px`;
            lens.style.top = `${event.clientY - plateBox.top}px`;
            lens.style.backgroundImage = `url("${image.currentSrc || image.src}")`;
            lens.style.backgroundSize = `${box.width * zoom}px ${box.height * zoom}px`;
            lens.style.backgroundPosition = `${-(x * zoom - 88)}px ${-(y * zoom - 88)}px`;
            plate.classList.add('lp-loupe-on');
            if (!still()) {
                plate.style.setProperty('--ry', `${((x / box.width - 0.5) * 4).toFixed(2)}deg`);
                plate.style.setProperty('--rx', `${((0.5 - y / box.height) * 4).toFixed(2)}deg`);
            }
        });
        plate.addEventListener('pointerleave', () => {
            plate.classList.remove('lp-loupe-on');
            plate.style.setProperty('--rx', '0deg');
            plate.style.setProperty('--ry', '0deg');
        });
    };

    document.addEventListener('DOMContentLoaded', () => {
        setTheme(document.documentElement.dataset.theme);
        document.getElementById('lp-theme')?.addEventListener('click', () => {
            setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
        });

        const back = document.getElementById('lp-back');
        if (back && !location.hostname.endsWith('github.io')) {
            const fromRepo = location.protocol === 'file:' || location.pathname.includes('/gp/web/');
            back.href = fromRepo ? '../../app/index.html' : `${location.protocol}//${location.hostname}:8766/app/index.html`;
        }

        document.addEventListener('click', event => {
            const prev = event.target.closest('[data-rail-prev]');
            const next = event.target.closest('[data-rail-next]');
            if (prev) rail(prev.dataset.railPrev, -1);
            if (next) rail(next.dataset.railNext, 1);
        });

        document.querySelectorAll('.lp-rail').forEach(element => {
            updateRail(element);
            element.addEventListener('scroll', () => updateRail(element), { passive: true });
            element.addEventListener('keydown', event => {
                if (event.target !== element || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
                event.preventDefault();
                rail(element.id, event.key === 'ArrowRight' ? 1 : -1);
            });
        });
        addEventListener('resize', () => document.querySelectorAll('.lp-rail').forEach(updateRail));

        tiles();
        stack();
        loupe();

        const needle = document.getElementById('lp-needle');
        const cards = [...document.querySelectorAll('.lp-versus-card')];
        const links = [...document.querySelectorAll('.lp-nav a')];
        const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
        if (still() || !('IntersectionObserver' in window)) {
            needle?.classList.add('lp-lifted');
            document.body.classList.add('lp-static');
            cards.forEach(card => card.classList.add('lp-in-view'));
        }
        if (!('IntersectionObserver' in window)) return;

        const spy = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (!entry.isIntersecting) return;
                links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === `#${entry.target.id}`));
            });
        }, { rootMargin: '-45% 0px -50% 0px' });
        sections.forEach(s => spy.observe(s));
        if (still()) return;

        const once = (elements, action, margin = '0px 0px -15% 0px') => {
            const observer = new IntersectionObserver(entries => {
                entries.forEach(entry => {
                    if (!entry.isIntersecting) return;
                    action(entry.target);
                    observer.unobserve(entry.target);
                });
            }, { rootMargin: margin });
            elements.forEach(element => observer.observe(element));
        };
        const reveal = [...document.querySelectorAll('.lp-section, .lp-cta')];
        reveal.forEach(element => element.classList.add('lp-reveal'));
        once(reveal, element => {
            element.classList.add('lp-in');
            if (element.id === 'found') element.querySelectorAll('[data-count]').forEach(countUp);
        }, '0px 0px -8% 0px');
        if (needle) once([needle], element => element.classList.add('lp-lifted'), '0px 0px -25% 0px');
        once(cards, element => element.classList.add('lp-in-view'), '0px 0px -20% 0px');
    });
})();
