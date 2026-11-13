// Browser-side checks, installed as window.__probe.
(() => {
  const parse = (c) => {
    const m = /rgba?\(([^)]+)\)/.exec(c || '');
    if (!m) return null;
    const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number);
    return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 };
  };
  const lum = (c) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }; return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
  const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  const blend = (fg, bg, a) => ({ r: fg.r * a + bg.r * (1 - a), g: fg.g * a + bg.g * (1 - a), b: fg.b * a + bg.b * (1 - a), a: 1 });
  const desc = (el) => {
    let s = el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (el.getAttribute('class') ? '.' + el.getAttribute('class').trim().replace(/\s+/g, '.') : '');
    const sec = el.closest('section');
    return (sec ? '[' + sec.id + '] ' : '') + s;
  };
  const effOpacity = (el) => { let o = 1; for (let p = el; p && p.nodeType === 1; p = p.parentElement) o *= +getComputedStyle(p).opacity; return o; };
  const htmlBg = (el) => {
    const stack = [];
    for (let p = el; p; p = p.parentElement) { const c = parse(getComputedStyle(p).backgroundColor); if (c && c.a > 0) { stack.push(c); if (c.a >= 1) break; } }
    let bg = parse(getComputedStyle(document.body).backgroundColor);
    for (let i = stack.length - 1; i >= 0; i--) bg = blend(stack[i], bg, stack[i].a);
    return bg;
  };
  const visible = (el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== 'hidden'; };
  const inter = (a, b) => Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) * Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));

  function layout() {
    const W = document.documentElement.clientWidth;
    const off = [], clipped = [];
    for (const el of document.querySelectorAll('body *')) {
      const r = el.getBoundingClientRect();
      if (r.width === 0) continue;
      if (r.right > W + 0.5 || r.left < -0.5) {
        let p = el.parentElement, how = null;
        while (p && p !== document.body) { const ox = getComputedStyle(p).overflowX; if (ox !== 'visible') { how = ox; break; } p = p.parentElement; }
        if (!how) off.push(desc(el) + ' left=' + Math.round(r.left) + ' right=' + Math.round(r.right));
        else if (how === 'hidden' || how === 'clip') clipped.push(desc(el) + ' clipped by ' + desc(p));
      }
      const cs = getComputedStyle(el);
      if ((cs.overflowX === 'hidden' || cs.overflowX === 'clip') && el.scrollWidth > el.clientWidth + 1 && !el.classList.contains('bar')) clipped.push(desc(el) + ' scrollW ' + el.scrollWidth + ' > ' + el.clientWidth);
      if ((cs.overflowY === 'hidden' || cs.overflowY === 'clip') && el.scrollHeight > el.clientHeight + 1 && !el.classList.contains('bar')) clipped.push(desc(el) + ' scrollH ' + el.scrollHeight + ' > ' + el.clientHeight);
      if (cs.textOverflow === 'ellipsis') clipped.push(desc(el) + ' ellipsis');
    }
    return { W, scrollW: document.documentElement.scrollWidth, bodyScrollW: document.body.scrollWidth, H: document.documentElement.scrollHeight, off: off.slice(0, 40), nOff: off.length, clipped: clipped.slice(0, 40) };
  }

  function svgFillBehind(svg, t) {
    const tr = t.getBoundingClientRect();
    const cx = (tr.left + tr.right) / 2, cy = (tr.top + tr.bottom) / 2;
    const all = [...svg.querySelectorAll('rect,circle,ellipse,polygon,path')].filter((s) => !s.closest('defs,marker'));
    let found = null;
    for (const s of all) {
      if (s.compareDocumentPosition(t) & Node.DOCUMENT_POSITION_PRECEDING) continue; // s after t
      const cs = getComputedStyle(s);
      if (cs.fill === 'none' || cs.display === 'none') continue;
      const c = parse(cs.fill); if (!c) continue;
      const r = s.getBoundingClientRect();
      if (cx < r.left || cx > r.right || cy < r.top || cy > r.bottom) continue;
      if (s.tagName === 'circle') { const rr = r.width / 2, dx = cx - (r.left + rr), dy = cy - (r.top + rr); if (dx * dx + dy * dy > rr * rr) continue; }
      if (s.tagName === 'path' && s.isPointInFill) { const m = s.getScreenCTM().inverse(); const pt = new DOMPoint(cx, cy).matrixTransform(m); if (!s.isPointInFill(pt)) continue; }
      found = { el: s, c: { ...c, a: c.a * (+cs.fillOpacity) * effOpacity(s) } };
    }
    let bg = htmlBg(svg);
    if (found) bg = blend(found.c, bg, found.c.a);
    return { bg, under: found ? desc(found.el) : 'html' };
  }

  // sample screen points along visible stroked shapes
  function strokePoints(svg) {
    const out = [];
    for (const s of svg.querySelectorAll('line,path,polyline')) {
      if (s.closest('defs,marker')) continue;
      const cs = getComputedStyle(s);
      if (cs.stroke === 'none' || cs.display === 'none' || +cs.strokeOpacity === 0 || effOpacity(s) < 0.2) continue;
      const cls = s.getAttribute('class') || '';
      if (/\bblob\b/.test(cls)) continue; // area highlights
      const sw = parseFloat(cs.strokeWidth) || 1;
      let L = 0; try { L = s.getTotalLength(); } catch { continue; }
      if (!L) continue;
      const m = s.getScreenCTM();
      const n = Math.min(400, Math.max(4, Math.ceil(L / 3)));
      const pts = [];
      for (let i = 0; i <= n; i++) { const p = s.getPointAtLength((L * i) / n); pts.push(new DOMPoint(p.x, p.y).matrixTransform(m)); }
      out.push({ el: s, cls, pts, sw });
    }
    return out;
  }

  function svgCheck(root) {
    const res = { overlapText: [], outside: [], lineOverText: [], textOverNode: [], blackFill: [], lowContrast: [] };
    for (const svg of root.querySelectorAll('svg')) {
      if (!visible(svg)) continue;
      const sr = svg.getBoundingClientRect();
      const texts = [...svg.querySelectorAll('text')].filter((t) => visible(t) && t.textContent.trim() && effOpacity(t) > 0.05);
      const rects = texts.map((t) => t.getBoundingClientRect());
      const label = (t) => JSON.stringify(t.textContent.trim().slice(0, 24)) + '.' + (t.getAttribute('class') || '');
      for (let i = 0; i < texts.length; i++) {
        const a = rects[i];
        if (a.left < sr.left - 0.5 || a.right > sr.right + 0.5 || a.top < sr.top - 0.5 || a.bottom > sr.bottom + 0.5) res.outside.push(desc(svg) + ' ' + label(texts[i]) + ` text [${Math.round(a.left - sr.left)},${Math.round(a.right - sr.left)}] svg w ${Math.round(sr.width)} h ${Math.round(sr.height)} top ${Math.round(a.top - sr.top)} bottom ${Math.round(a.bottom - sr.top)}`);
        for (let j = i + 1; j < texts.length; j++) {
          const b = rects[j];
          const ar = inter(a, b);
          // shrink: glyph boxes include line gap; count only a solid overlap
          if (ar > 0.18 * Math.min(a.width * a.height, b.width * b.height)) res.overlapText.push(desc(svg) + ' ' + label(texts[i]) + ' x ' + label(texts[j]) + ' ' + Math.round(100 * ar / Math.min(a.width * a.height, b.width * b.height)) + '%');
        }
      }
      // node circles
      const circles = [...svg.querySelectorAll('circle')].filter((c) => !c.closest('defs') && visible(c));
      for (let i = 0; i < texts.length; i++) {
        const a = rects[i]; const cx = (a.left + a.right) / 2, cy = (a.top + a.bottom) / 2;
        for (const c of circles) {
          const r = c.getBoundingClientRect(); const rr = r.width / 2; const ox = r.left + rr, oy = r.top + rr;
          const own = (cx - ox) ** 2 + (cy - oy) ** 2 <= rr * rr;
          if (own || rr < 6) continue;
          // nearest point of text rect to circle centre
          const nx = Math.max(a.left + 1, Math.min(ox, a.right - 1)), ny = Math.max(a.top + 2, Math.min(oy, a.bottom - 2));
          if ((nx - ox) ** 2 + (ny - oy) ** 2 < (rr - 1.5) ** 2) res.textOverNode.push(desc(svg) + ' ' + label(texts[i]) + ' overlaps node circle ' + (c.getAttribute('class') || ''));
        }
      }
      // lines through text
      const strokes = strokePoints(svg);
      for (let i = 0; i < texts.length; i++) {
        const t = texts[i]; const a = rects[i];
        // glyph ink box is smaller than the layout box; shrink vertically
        const box = { left: a.left + 1, right: a.right - 1, top: a.top + a.height * 0.22, bottom: a.bottom - a.height * 0.2 };
        // background plate right before the text?
        const plates = [];
        for (const s of svg.querySelectorAll('rect,circle')) {
          if (s.closest('defs')) continue;
          const cs = getComputedStyle(s); if (cs.fill === 'none' || effOpacity(s) < 0.9) continue;
          const c = parse(cs.fill); if (!c || c.a < 0.9) continue;
          const r = s.getBoundingClientRect();
          const inside = (x, y) => { if (s.tagName === 'circle') { const rr = r.width / 2; return (x - r.left - rr) ** 2 + (y - r.top - rr) ** 2 <= rr * rr; } return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom; };
          if (inside(box.left, box.top) && inside(box.right, box.top) && inside(box.left, box.bottom) && inside(box.right, box.bottom)) plates.push(s);
        }
        for (const st of strokes) {
          const hit = st.pts.some((p) => p.x > box.left && p.x < box.right && p.y > box.top && p.y < box.bottom);
          if (!hit) continue;
          const after = !!(t.compareDocumentPosition(st.el) & Node.DOCUMENT_POSITION_FOLLOWING);
          const platedOk = plates.some((pl) => !!(st.el.compareDocumentPosition(pl) & Node.DOCUMENT_POSITION_FOLLOWING) && !!(pl.compareDocumentPosition(t) & Node.DOCUMENT_POSITION_FOLLOWING));
          if (after || !platedOk) res.lineOverText.push(desc(svg) + ' ' + label(t) + ' crossed by ' + st.el.tagName + '.' + st.cls + (after ? ' (DRAWN ON TOP)' : ' (under, no plate)'));
        }
      }
      // default black / literal fills
      for (const s of svg.querySelectorAll('text,rect,circle,ellipse,polygon,path,line,tspan')) {
        if (s.closest('defs') && !s.closest('marker')) continue;
        const cs = getComputedStyle(s);
        if (cs.fill === 'rgb(0, 0, 0)' && s.tagName !== 'line') res.blackFill.push(desc(svg) + ' ' + s.tagName + '.' + (s.getAttribute('class') || '') + ' ' + (s.textContent || '').slice(0, 20));
      }
      // contrast of svg text
      for (let i = 0; i < texts.length; i++) {
        const t = texts[i];
        const fc = parse(getComputedStyle(t).fill); if (!fc) continue;
        const { bg, under } = svgFillBehind(svg, t);
        const op = effOpacity(t);
        const fg = blend(fc, bg, fc.a * op);
        const r = ratio(fg, bg);
        const fs = parseFloat(getComputedStyle(t).fontSize) * (svg.getBoundingClientRect().width / (svg.viewBox.baseVal && svg.viewBox.baseVal.width || svg.getBoundingClientRect().width));
        if (r < 3 || (r < 4.5 && op >= 0.99)) res.lowContrast.push(desc(svg) + ' ' + label(t) + ' ratio ' + r.toFixed(2) + ' fill ' + getComputedStyle(t).fill + ' on ' + under + ' op ' + op.toFixed(2) + ' px ' + fs.toFixed(1));
      }
    }
    return res;
  }

  function htmlContrast(root) {
    const bad = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const seen = new Set();
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const el = n.parentElement;
      if (!el || seen.has(el) || !n.textContent.trim() || el.closest('svg,script,style')) continue;
      seen.add(el);
      if (!visible(el)) continue;
      const cs = getComputedStyle(el);
      const fc = parse(cs.color); const bg = htmlBg(el);
      const op = effOpacity(el);
      const fg = blend(fc, bg, fc.a * op);
      const r = ratio(fg, bg);
      const px = parseFloat(cs.fontSize); const bold = +cs.fontWeight >= 700;
      const need = px >= 24 || (bold && px >= 18.66) ? 3 : 4.5;
      if (r < need) bad.push(desc(el) + ' ' + JSON.stringify(n.textContent.trim().slice(0, 30)) + ' ratio ' + r.toFixed(2) + ' (' + px + 'px, op ' + op.toFixed(2) + ')');
    }
    return bad;
  }

  function steppers() { return [...document.querySelectorAll('.stepper')].map((s) => ({ id: s.id, n: +s.querySelector('input[type=range]').max + 1 })); }
  function goto(id, i) {
    const s = document.getElementById(id); const rng = s.querySelector('input[type=range]');
    rng.value = i; rng.dispatchEvent(new Event('input', { bubbles: true }));
    const pics = s.querySelector('.pics'), panel = s.querySelector('.panel');
    const txt = panel.innerText;
    return { now: s.querySelector('.now').textContent, picsKids: pics.children.length, svgs: pics.querySelectorAll('svg').length, panelLen: txt.trim().length, bad: /\b(null|undefined|NaN)\b|\[object/.test(s.innerText) ? (s.innerText.match(/.{0,30}\b(null|undefined|NaN)\b.{0,30}|.{0,30}\[object.{0,20}/) || [''])[0] : '' , h: Math.round(s.getBoundingClientRect().height) };
  }
  window.__probe = { layout, svgCheck, htmlContrast, steppers, goto, desc };
})();
