'use strict';
/* ---------- data ---------- */
const D = JSON.parse(document.getElementById('data').textContent);
const ST = {};
for (const s of D.steps) ST[s.id] = s;
const CO = D.example.coords;
const NS = 'http://www.w3.org/2000/svg';
let UID = 0;

/* ---------- DOM helpers ---------- */
function add(e, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false || k === '') continue;
    e.append(k instanceof Node ? k : String(k));
  }
  return e;
}
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  if (attrs) for (const k in attrs) {
    const v = attrs[k];
    if (v == null || v === false) continue;
    e.setAttribute(k, v === true ? '' : v);
  }
  return add(e, kids);
}
function sv(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  if (attrs) for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  return e;
}
const ap = (e, ...k) => add(e, k);
const C = (s) => h('code', null, String(s));
const B = (s) => h('b', null, String(s));
const P = (...k) => h('p', null, k);
const blk = (title, ...kids) => h('div', { class: 'blk' }, h('span', { class: 'bh' }, title), kids);
const note = (...k) => h('p', { class: 'note' }, k);

/* ---------- formatting ---------- */
const INF = '∞';
const fnum = (v) => (typeof v === 'number' && !Number.isInteger(v) ? String(Math.round(v * 100) / 100) : String(v));
const fw = (v) => (v == null ? INF : fnum(v));
const fm = (v) => (v == null ? '—' : String(v));
const flist = (a) => (a && a.length ? '[' + a.join(', ') + ']' : '∅');
const fpair = (p) => p[0] + '–' + p[1];
const fpairs = (ps) => (ps && ps.length ? ps.map(fpair).join(', ') : 'none');
const farc = (a) => '(' + a[0] + ', ' + a[1] + ')';
const yesno = (b) => h('span', { class: b ? 'yes' : 'no' }, b ? 'yes' : 'no');
const pct = (x) => fnum(Math.round(x * 10000) / 100) + ' %';
const toMap = (pairs) => { const m = new Map(); for (const [k, v] of pairs) m.set(k, v); return m; };
const key2 = (a, b) => (a < b ? a + '-' + b : b + '-' + a);
const sid = (id) => 's-' + String(id).replace(/[^A-Za-z0-9]+/g, '-');
const thr = (name) => D.thresholds.find((t) => t.name === name);

/* ---------- geometry ---------- */
function place(ids, coords, o) {
  const sx = o.sx || 84, sy = o.sy || 74, px = o.px || 34, py = o.py || 32;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const id of ids) {
    const c = coords[id];
    if (!c) continue;
    x0 = Math.min(x0, c[0]); x1 = Math.max(x1, c[0]); y0 = Math.min(y0, c[1]); y1 = Math.max(y1, c[1]);
  }
  const pos = {};
  for (const id of ids) { const c = coords[id]; if (c) pos[id] = [px + (c[0] - x0) * sx, py + (c[1] - y0) * sy]; }
  return { pos, w: Math.round(px * 2 + (x1 - x0) * sx), h: Math.round(py * 2 + (y1 - y0) * sy) };
}
const quad = (P0, Cp, Q, t) => [
  (1 - t) * (1 - t) * P0[0] + 2 * (1 - t) * t * Cp[0] + t * t * Q[0],
  (1 - t) * (1 - t) * P0[1] + 2 * (1 - t) * t * Cp[1] + t * t * Q[1],
];
const RCACHE = new Map();
function route(P0, Q, obs, r, ck, prefer) {
  const k = ck + '|' + P0 + '|' + Q;
  if (RCACHE.has(k)) return RCACHE.get(k);
  const dx = Q[0] - P0[0], dy = Q[1] - P0[1], L = Math.hypot(dx, dy) || 1;
  const samples = (Cp) => {
    const pts = [];
    for (let t = 0.08; t < 0.93; t += 0.04) pts.push(Cp ? quad(P0, Cp, Q, t) : [P0[0] + dx * t, P0[1] + dy * t]);
    return pts;
  };
  const minD = (pts) => { let m = Infinity; for (const p of pts) for (const o of obs) m = Math.min(m, Math.hypot(o[0] - p[0], o[1] - p[1])); return m; };
  let res = null;
  if (minD(samples(null)) > r + 6) res = null;
  else {
    const nx = -dy / L, ny = dx / L;
    let best = null, bestD = -1;
    outer: for (const kk of [0.2, 0.3, 0.42, 0.56, 0.72]) {
      for (const sg of prefer ? [1, -1] : [-1, 1]) {
        const Cp = [(P0[0] + Q[0]) / 2 + sg * nx * L * kk, (P0[1] + Q[1]) / 2 + sg * ny * L * kk];
        const d = minD(samples(Cp));
        if (d > r + 6) { best = Cp; bestD = d; break outer; }
        if (d > bestD) { bestD = d; best = Cp; }
      }
    }
    res = best;
  }
  RCACHE.set(k, res);
  return res;
}
const f1 = (p) => p[0].toFixed(1) + ',' + p[1].toFixed(1);
function arrowColor(cls) {
  if (/\bfy\b|\bhop\b/.test(cls)) return 'bwd';
  if (/\b(hot|add|mt|apex|apexwin)\b/.test(cls)) return 'acc';
  if (/\b(new|push|rep|late|fill|queued)\b/.test(cls)) return 'fil';
  if (/\b(del|drop|capped)\b/.test(cls)) return 'bad';
  if (/\bprop\b/.test(cls)) return 'mut';
  return '';
}

/* ---------- graph drawing ----------
   spec.nodes: [{id, cls, big, sub}]; spec.edges: [{a, b, cls, arrow, tag, tagcls, z, bow}]
   spec.groups: [{members, links, tone}]; spec.coords; spec.layoutIds; spec.aria */
function graph(spec) {
  const coords = spec.coords || CO;
  const r = spec.r || 19;
  const ids = spec.layoutIds || spec.nodes.map((n) => n.id);
  const L = place(ids, coords, spec);
  const subs = spec.nodes.some((n) => n.sub != null && n.sub !== '');
  const svg = sv('svg', { class: L.w >= 440 ? 'gr' + (subs ? ' grs' : '') : null, viewBox: '0 0 ' + L.w + ' ' + L.h, width: L.w, height: L.h, role: 'img', 'aria-label': spec.aria || 'Graph' });
  const uid = ++UID;
  const defs = sv('defs');
  svg.append(defs);
  const mks = {};
  const marker = (c) => {
    if (!mks[c]) {
      const id = 'mk' + uid + (c || 'n');
      const m = sv('marker', { id, viewBox: '0 0 10 10', refX: 8.5, refY: 5, markerWidth: 9, markerHeight: 9, markerUnits: 'userSpaceOnUse', orient: 'auto-start-reverse' });
      m.append(sv('path', { d: 'M0,0 L10,5 L0,10 z', class: 'ah ' + c }));
      defs.append(m);
      mks[c] = id;
    }
    return mks[c];
  };
  const ck = (spec.cacheKey || '') + ids.join(',') + '/' + (spec.sx || 84) + '/' + (spec.sy || 74) + '/' + (spec.coords ? 'c' : 'n');
  for (const g of spec.groups || []) {
    const width = g.width || 2 * r + 16;
    for (const [a, b] of g.links || []) {
      const P0 = L.pos[a], Q = L.pos[b];
      if (P0 && Q) svg.append(sv('path', { class: 'blob ' + g.tone, d: 'M' + f1(P0) + 'L' + f1(Q), 'stroke-width': width }));
    }
    for (const m of g.members) { const p = L.pos[m]; if (p) svg.append(sv('circle', { class: 'blobc ' + g.tone, cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: width / 2 })); }
  }
  const all = (spec.obstacleIds || ids).filter((id) => L.pos[id]);
  const edges = (spec.edges || []).slice().sort((x, y) => (x.z || 0) - (y.z || 0));
  const tags = [];
  for (const e of edges) {
    const P0 = L.pos[e.a], Q = L.pos[e.b];
    if (!P0 || !Q) continue;
    const obs = all.filter((id) => String(id) !== String(e.a) && String(id) !== String(e.b)).map((id) => L.pos[id]);
    let Cp = null;
    if (e.bow) {
      const dx = Q[0] - P0[0], dy = Q[1] - P0[1], LL = Math.hypot(dx, dy) || 1;
      Cp = [(P0[0] + Q[0]) / 2 - (dy / LL) * LL * e.bow, (P0[1] + Q[1]) / 2 + (dx / LL) * LL * e.bow];
    } else if (!e.straight) Cp = route(P0, Q, obs, r, ck, (+e.a + +e.b) % 2 === 0);
    let a = P0, b = Q;
    if (e.arrow || e.trim) {
      const t1 = Cp || P0, t0 = Cp || Q;
      const dx = b[0] - t1[0], dy = b[1] - t1[1], l1 = Math.hypot(dx, dy) || 1;
      const ex = a[0] - t0[0], ey = a[1] - t0[1], l0 = Math.hypot(ex, ey) || 1;
      const rb = e.trim ? (e.trim[1] != null ? e.trim[1] : r + 2) : r + 3;
      const ra = e.trim ? (e.trim[0] != null ? e.trim[0] : r) : r;
      b = [b[0] - (dx / l1) * rb, b[1] - (dy / l1) * rb];
      a = [a[0] - (ex / l0) * ra, a[1] - (ey / l0) * ra];
    }
    const d = Cp ? 'M' + f1(a) + 'Q' + f1(Cp) + ' ' + f1(b) : 'M' + f1(a) + 'L' + f1(b);
    const el = sv('path', { class: e.cls || 'e', d });
    if (e.arrow) el.setAttribute('marker-end', 'url(#' + marker(arrowColor(e.cls || '')) + ')');
    svg.append(el);
    if (e.tag != null) tags.push([Cp ? quad(P0, Cp, Q, 0.5) : [(P0[0] + Q[0]) / 2, (P0[1] + Q[1]) / 2], String(e.tag), e.tagcls || '']);
  }
  for (const [p, t, c] of tags) {
    const w = t.length * 6.6 + 6;
    svg.append(sv('rect', { class: 'wlbg', x: (p[0] - w / 2).toFixed(1), y: (p[1] - 8).toFixed(1), width: w.toFixed(1), height: 15, rx: 3 }));
    svg.append(sv('text', { class: 'wl ' + c, x: p[0].toFixed(1), y: (p[1] + 3.6).toFixed(1), 'text-anchor': 'middle' }, t));
  }
  for (const n of spec.nodes) {
    const p = L.pos[n.id];
    if (!p) continue;
    const cls = n.cls || '';
    const isGone = /\bgone\b/.test(cls);
    if (isGone) svg.append(sv('circle', { class: 'ndbg', cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: n.r || r }));
    svg.append(sv('circle', { class: 'nd ' + cls, cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: n.r || r }));
    const inv = !isGone && /\b(cur|sep)\b/.test(cls) ? ' inv' : '';
    const gone = isGone ? ' gone' : '';
    if (n.nolabel) continue;
    const big = n.big != null ? String(n.big) : String(n.id);
    const sm = big.length > 3 ? ' sm' : '';
    if (n.sub != null && n.sub !== '') {
      svg.append(sv('text', { class: 'nl' + sm + inv + gone, x: p[0].toFixed(1), y: (p[1] + 1.5).toFixed(1), 'text-anchor': 'middle' }, big));
      svg.append(sv('text', { class: 'sub' + inv + gone, x: p[0].toFixed(1), y: (p[1] + 13.5).toFixed(1), 'text-anchor': 'middle' }, String(n.sub)));
    } else {
      svg.append(sv('text', { class: 'nl' + sm + inv + gone, x: p[0].toFixed(1), y: (p[1] + 5).toFixed(1), 'text-anchor': 'middle' }, big));
    }
  }
  if (spec.extra) spec.extra(svg, L.pos);
  return svg;
}
function wrapFig(svg, caption, o) {
  const pane = h('div', { class: 'pane' + (o && o.scroll ? ' scroll' : '') }, svg);
  if (o && o.maxw) pane.setAttribute('style', 'max-width:' + o.maxw + 'px');
  return h('figure', null, pane, caption ? h('figcaption', null, caption) : null);
}

/* ---------- elimination tree ----------
   spec.parent: Map rank -> parent rank or null; spec.ranks; spec.cls(r); spec.ecls(c, p); spec.sub(r) */
function tree(spec) {
  const ranks = spec.ranks.slice().sort((a, b) => a - b);
  const kids = new Map(ranks.map((r) => [r, []]));
  const roots = [];
  for (const r of ranks) {
    const p = spec.parent.get(r);
    if (p == null || !kids.has(p)) roots.push(r); else kids.get(p).push(r);
  }
  const pos = new Map();
  let leaf = 0, maxd = 0;
  const visit = (r, d) => {
    maxd = Math.max(maxd, d);
    const ch = kids.get(r);
    if (!ch.length) { pos.set(r, [leaf++, d]); return; }
    for (const c of ch) visit(c, d + 1);
    const xs = ch.map((c) => pos.get(c)[0]);
    pos.set(r, [(Math.min(...xs) + Math.max(...xs)) / 2, d]);
  };
  for (const r of roots.sort((a, b) => b - a)) visit(r, 0);
  const sx = spec.sx || 50, sy = spec.sy || 56, pad = 28, R = 20;
  const W = Math.round(pad * 2 + Math.max(0, leaf - 1) * sx), Hh = Math.round(pad * 2 + maxd * sy);
  const svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + Hh, width: W, height: Hh, role: 'img', 'aria-label': spec.aria || 'Elimination tree' });
  const XY = (r) => { const [x, d] = pos.get(r); return [pad + x * sx, pad + d * sy]; };
  for (const r of ranks) {
    const p = spec.parent.get(r);
    if (p == null || !kids.has(p)) continue;
    const [x1, y1] = XY(r), [x2, y2] = XY(p);
    svg.append(sv('line', { class: 'e ' + (spec.ecls ? spec.ecls(r, p) : 'te'), x1, y1, x2, y2 }));
  }
  for (const r of ranks) {
    const [x, y] = XY(r);
    const cls = spec.cls ? spec.cls(r) : '';
    const isGone = /\bgone\b/.test(cls);
    if (isGone) svg.append(sv('circle', { class: 'ndbg', cx: x, cy: y, r: R }));
    svg.append(sv('circle', { class: 'nd ' + cls, cx: x, cy: y, r: R }));
    const inv = !isGone && /\b(cur|sep)\b/.test(cls) ? ' inv' : '';
    const gone = isGone ? ' gone' : '';
    const sub = spec.sub ? spec.sub(r) : null;
    if (sub != null) {
      svg.append(sv('text', { class: 'nl sm' + inv + gone, x, y: y + 1, 'text-anchor': 'middle' }, String(r)));
      svg.append(sv('text', { class: 'sub' + inv + gone, x, y: y + 13, 'text-anchor': 'middle' }, String(sub)));
    } else svg.append(sv('text', { class: 'nl sm' + inv + gone, x, y: y + 4.5, 'text-anchor': 'middle' }, String(r)));
  }
  return svg;
}

/* ---------- matrix picture ----------
   spec.rows / spec.cols: labels; spec.cells: [{r, c, cls, text, x, out}] (indices into rows/cols) */
function matrix(spec) {
  const cs = spec.cs || 16, lw = spec.lw || 24, top = spec.top || 22;
  const nr = spec.rows.length, nc = spec.cols.length;
  const W = lw + nc * cs + 4, Hh = top + nr * cs + 4;
  const svg = sv('svg', { class: 'nat', viewBox: '0 0 ' + W + ' ' + Hh, width: W, height: Hh, role: 'img', 'aria-label': spec.aria || 'Matrix' });
  spec.cols.forEach((c, j) => svg.append(sv('text', { class: 'ml', x: lw + j * cs + cs / 2, y: top - 6, 'text-anchor': 'middle' }, String(c))));
  spec.rows.forEach((c, i) => svg.append(sv('text', { class: 'ml', x: lw - 4, y: top + i * cs + cs / 2 + 3.4, 'text-anchor': 'end' }, String(c))));
  const at = new Map();
  for (const c of spec.cells) at.set(c.r + ',' + c.c, c);
  for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) {
    const c = at.get(i + ',' + j);
    const x = lw + j * cs, y = top + i * cs;
    svg.append(sv('rect', { class: 'mc ' + (c && c.cls ? c.cls : '') + (c && c.out ? ' out' : ''), x, y, width: cs, height: cs }));
    if (c && c.x) {
      svg.append(sv('line', { class: 'mx', x1: x + 3, y1: y + 3, x2: x + cs - 3, y2: y + cs - 3 }));
      svg.append(sv('line', { class: 'mx', x1: x + cs - 3, y1: y + 3, x2: x + 3, y2: y + cs - 3 }));
    }
    if (c && c.text != null) svg.append(sv('text', { class: 'mt' + (/\bon\b/.test(c.cls || '') ? ' onx' : /\b(ink|a|f|b|pl)\b/.test(c.cls || '') ? ' inv' : ''), x: x + cs / 2, y: y + cs / 2 + 3.3, 'text-anchor': 'middle' }, String(c.text)));
  }
  for (const k of spec.rowRules || []) svg.append(sv('line', { class: 'mrule', x1: lw, x2: lw + nc * cs, y1: top + k * cs, y2: top + k * cs }));
  for (const k of spec.colRules || []) svg.append(sv('line', { class: 'mrule', y1: top, y2: top + nr * cs, x1: lw + k * cs, x2: lw + k * cs }));
  /* spec.boxes: outlines around cell blocks, [{r0, c0, r1, c1, cls}] with inclusive row/column indices */
  for (const b of spec.boxes || []) svg.append(sv('rect', { class: 'mbox ' + (b.cls || ''), x: lw + b.c0 * cs, y: top + b.r0 * cs, width: (b.c1 - b.c0 + 1) * cs, height: (b.r1 - b.r0 + 1) * cs }));
  return svg;
}
const range = (a, b) => { const o = []; for (let i = a; i <= b; i++) o.push(i); return o; };

/* ---------- strips, tables, legends ---------- */
function cellEl(c) {
  return h('span', { class: 'cell ' + (c.cls || '') }, c.ix != null ? h('span', { class: 'ix' }, String(c.ix)) : null, h('span', { class: 'v' }, c.v == null ? '' : String(c.v)));
}
const strip = (label, cells) => h('div', { class: 'strip' }, h('span', { class: 'lab' }, label), h('span', { class: 'cells' }, (cells.length ? cells : [{ v: 'none', cls: 'empty' }]).map(cellEl)));
const gstrip = (label, groups) => h('div', { class: 'strip' }, h('span', { class: 'lab' }, label),
  h('span', { class: 'grps' }, groups.map((g) => h('span', { class: 'grp' }, h('span', { class: 'gl' }, g.gl), h('span', { class: 'cells' }, g.cells.map(cellEl))))));
const arrays = (...s) => h('div', { class: 'arrays' }, s);
function table(head, rows, o) {
  const td = (c, j) => {
    const th = head[j];
    if (c instanceof Node) return h('td', { 'data-th': th }, c);
    if (c && typeof c === 'object' && !Array.isArray(c)) return h('td', { class: c.cls || null, 'data-th': th }, c.v instanceof Node ? c.v : String(c.v));
    return h('td', { 'data-th': th }, c == null ? '' : String(c));
  };
  return h('div', { class: 'tw' }, h('table', { class: 't' + (o && o.stack ? ' stack' : '') }, o && o.caption ? h('caption', null, o.caption) : null,
    h('thead', null, h('tr', null, head.map((x) => h('th', null, x)))),
    h('tbody', null, rows.map((r) => (Array.isArray(r) ? h('tr', null, r.map(td)) : h('tr', { class: r.cls || null }, r.cells.map(td)))))));
}
const prose = (v) => ({ v, cls: 'prose' });
function legend(items) {
  return h('div', { class: 'key' }, items.filter(Boolean).map(([k, cls, lab]) => h('span', null, h('i', { class: (k === 'l' ? 'ln ' : k === 'q' ? 'sw sq ' : 'sw ') + cls }), lab)));
}
function stats(items) {
  return h('div', { class: 'stats' }, items.map(([k, n, small]) => h('div', { class: 'stat' }, h('span', { class: 'k' }, k), h('span', { class: 'n' }, String(n), small ? h('small', null, ' ' + small) : null))));
}
function mini(head, rows) {
  return h('table', { class: 'mini' }, h('thead', null, h('tr', null, head.map((x) => h('th', null, x)))),
    h('tbody', null, rows.map((r) => h('tr', null, r.map((c) => (c instanceof Node ? h('td', null, c) : c && typeof c === 'object' ? h('td', { class: c.cls }, String(c.v)) : h('td', null, String(c))))))));
}
const chips = (items) => h('div', { class: 'chips' }, items.map(([t, c]) => h('span', { class: 'chip ' + (c || '') }, t)));
function bar(value, budget, max) {
  const top = Math.max(max || 0, value, budget || 0) || 1;
  const fill = h('i', { class: budget != null && value > budget ? 'over' : null, style: 'width:' + Math.min(100, (100 * value) / top).toFixed(1) + '%' });
  const mark = budget != null ? h('b', { style: 'left:' + ((100 * budget) / top).toFixed(1) + '%' }) : null;
  return h('div', { class: 'bar', role: 'img', 'aria-label': 'work so far ' + value + (budget != null ? ' of a budget of ' + budget : '') }, fill, mark);
}

/* ---------- keep the highlight in view on narrow screens ---------- */
const HL_FIRST = ['.nd.cur', '.e.hot', '.nd.ring', '.e.mt', '.e.new', '.e.push', '.nd.ring2', '.nd.top', '.nd.pool', '.nd.dead', '.e.add', '.e.hop', '.nd.ringbad', '.e.apexwin'];
function focusPanes(root) {
  requestAnimationFrame(() => {
    for (const pane of root.querySelectorAll('.pane')) {
      if (pane.scrollWidth <= pane.clientWidth + 1 || pane.querySelector('.lane')) continue;
      if (pane.dataset.cx != null) { pane.scrollLeft = Math.max(0, +pane.dataset.cx - pane.clientWidth / 2); continue; }
      const pr = pane.getBoundingClientRect();
      const box = (els) => {
        let a = Infinity, b = -Infinity;
        for (const el of els) { const r = el.getBoundingClientRect(); if (r.width || r.height) { a = Math.min(a, r.left); b = Math.max(b, r.right); } }
        return a === Infinity ? null : [a - pr.left + pane.scrollLeft, b - pr.left + pane.scrollLeft];
      };
      const all = box(pane.querySelectorAll(HL_FIRST.join(', ')));
      if (!all) continue;
      const W = pane.clientWidth, pad = 18;
      let x0 = all[0] - pad, x1 = all[1] + pad;
      if (x1 - x0 > W) {
        const first = box(pane.querySelectorAll(HL_FIRST.find((q) => pane.querySelector(q))));
        x0 = first[0] - pad; x1 = first[1] + pad;
        if (x1 - x0 > W) { const c = (first[0] + first[1]) / 2; x0 = c - W / 2; x1 = c + W / 2; }
      }
      const at = pane.scrollLeft;
      pane.scrollLeft = Math.max(0, x0 < at ? x0 : x1 > at + W ? x1 - W : at);
    }
  });
}

/* ---------- stepper ---------- */
function stepper(key, o) {
  const prev = h('button', { type: 'button', id: key + '-prev', 'aria-controls': key + '-panel' }, '◀ Back');
  const next = h('button', { type: 'button', id: key + '-next', 'aria-controls': key + '-panel' }, 'Next ▶');
  const rng = h('input', { type: 'range', id: key + '-step', min: 0, max: Math.max(0, o.n - 1), value: 0, 'aria-label': o.title + ': choose a step' });
  const now = h('span', { class: 'now', id: key + '-now', 'aria-live': 'polite' });
  const pics = h('div', { class: 'pics', id: key + '-pics' });
  const panel = h('div', { class: 'panel', id: key + '-panel' });
  const root = h('div', { class: 'stepper', id: key },
    h('div', { class: 'sth' }, o.title),
    h('div', { class: 'ctl' }, prev, rng, next, now),
    h('div', { class: o.wide ? 'stage one' : 'stage' }, pics, panel),
    o.legend ? legend(o.legend) : null);
  const show = (i) => {
    i = Math.max(0, Math.min(o.n - 1, i));
    rng.value = i;
    now.textContent = o.label(i);
    pics.textContent = '';
    panel.textContent = '';
    o.render(i, pics, panel);
    focusPanes(pics);
  };
  rng.addEventListener('input', () => show(+rng.value));
  prev.addEventListener('click', () => show(+rng.value - 1));
  next.addEventListener('click', () => show(+rng.value + 1));
  show(0);
  return root;
}

/* ---------- section scaffold ---------- */
function section(s) {
  const sec = h('section', { id: sid(s.id), 'aria-labelledby': sid(s.id) + '-h' });
  ap(sec, h('div', { class: 'fnhead' },
    h('span', { class: 'where' }, h('b', null, 'step ' + s.id), ' · ', s.design),
    h('h2', { id: sid(s.id) + '-h' }, s.title)));
  ap(sec, h('p', { class: 'what' }, s.what));
  if (s.calls && s.calls.length) ap(sec, callsBlock(s.calls, callsTitle(s)));
  return sec;
}
/* who runs each step's calls (03 §5, §6): the builder thread off the lock, or a writer under the lock */
const CALLS_WHO = {
  F1: 'the writer, under the lock', F2: 'the writer, under the lock', F3: 'the writer, under the lock', F4: 'the writer, under the lock',
  F7: 'the writer, under the lock', F8: 'the writer, under the lock',
  F5: 'the writer under the lock, then the Repair job on the builder thread, off the lock',
  F6: 'the fold job on the builder thread, off the lock; its publish under the lock',
};
const callsTitle = (s) => 'Calls, as the design writes them (' + (CALLS_WHO[s.id] || 'on the builder thread, off the lock') + ')';
function callsBlock(calls, title) {
  const host = calls.some((c) => /^host/.test(c.call));
  return h('div', { class: 'calls' }, h('span', { class: 'bh' }, title || 'Calls, as the design writes them'),
    host ? h('span', { class: 'io' }, '“host” marks a plain loop on that thread, not a GraphBLAS call.') : null,
    calls.map((c) => h('div', { class: 'call' }, h('code', null, c.call),
      (c.in || c.out) ? h('span', { class: 'io' }, (c.in ? 'reads ' + c.in.join(', ') : '') + (c.in && c.out ? ' · ' : '') + (c.out ? 'gives ' + c.out : '')) : null,
      c.note ? h('span', { class: 'io' }, c.note) : null)));
}
