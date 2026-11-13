// node run.mjs <light|dark|forcelight> <width> <scheme light|dark> [reduced]
import fs from 'node:fs';
import path from 'node:path';
import { open, here } from './cdp.mjs';

const [theme, wS, scheme, reduced] = process.argv.slice(2);
const width = +wS;
const pg = await open({ file: theme + '.html', width, scheme, reducedMotion: reduced ? 'reduce' : 'no-preference' });
await pg.ev(fs.readFileSync(path.join(here, 'probe.js'), 'utf8'));
// OPEN_DETAILS=1 opens every <details> (the collapsed appendix) before probing
if (process.env.OPEN_DETAILS === '1') await pg.ev(`document.querySelectorAll('details').forEach((d) => { d.open = true; }); new Promise((r) => setTimeout(r, 400))`);
const out = { theme, width, scheme };
out.layout = await pg.ev('__probe.layout()');
out.atRest = await pg.ev('__probe.svgCheck(document)');
out.htmlContrast = await pg.ev('__probe.htmlContrast(document.body)');
out.rootBg = await pg.ev('getComputedStyle(document.body).backgroundColor');
out.colorScheme = await pg.ev('getComputedStyle(document.documentElement).colorScheme');
const st = await pg.ev('__probe.steppers()');
out.steppers = [];
const agg = {};
const add = (k, msg, where) => { (agg[k] ||= {}); (agg[k][msg] ||= []).push(where); };
for (const s of st) {
  const frames = [];
  for (let i = 0; i < s.n; i++) {
    const g = await pg.ev(`__probe.goto(${JSON.stringify(s.id)}, ${i})`);
    frames.push(g);
    if (!g.svgs && !g.picsKids) add('emptyPics', s.id, i);
    if (!g.panelLen) add('emptyPanel', s.id, i);
    if (g.bad) add('badText', s.id + ' ' + g.bad, i);
    const c = await pg.ev(`__probe.svgCheck(document.getElementById(${JSON.stringify(s.id)}))`);
    for (const k of Object.keys(c)) for (const m of c[k]) add(k, m, s.id + '@' + i);
    for (const m of await pg.ev(`__probe.htmlContrast(document.getElementById(${JSON.stringify(s.id)}))`)) add('htmlContrast', m, s.id + '@' + i);
    const L = await pg.ev('({W: document.documentElement.clientWidth, sw: document.documentElement.scrollWidth})');
    if (L.sw > L.W) add('pageHScroll', 'scrollWidth ' + L.sw + ' > ' + L.W, s.id + '@' + i);
  }
  const hs = frames.map((f) => f.h);
  out.steppers.push({ id: s.id, n: s.n, hMin: Math.min(...hs), hMax: Math.max(...hs) });
  await pg.ev(`__probe.goto(${JSON.stringify(s.id)}, 0)`);
}
out.frames = agg;
out.logs = pg.logs;
const f = path.join(here, `report-${theme}-${width}-${scheme}${reduced ? '-rm' : ''}${process.env.OPEN_DETAILS === '1' ? '-open' : ''}.json`);
fs.writeFileSync(f, JSON.stringify(out, null, 1));
const summ = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, Array.isArray(v) ? v.length : typeof v === 'object' ? Object.keys(v).length : v]));
console.log(f);
console.log('layout', JSON.stringify({ W: out.layout.W, scrollW: out.layout.scrollW, H: out.layout.H, nOff: out.layout.nOff, clipped: out.layout.clipped.length }));
console.log('atRest', JSON.stringify(summ(out.atRest)), 'htmlContrast', out.htmlContrast.length, 'bg', out.rootBg, out.colorScheme);
console.log('frames (distinct msgs)', JSON.stringify(summ(agg)), 'logs', out.logs.length);
pg.close();
