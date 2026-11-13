// node frames.mjs <light|dark> <width> <outdir> <job>...   job = name=<css selector>[@stepperId:frame][~scale]
import fs from 'node:fs';
import path from 'node:path';
import { open, here } from './cdp.mjs';
const [theme, wS, outdir, ...jobs] = process.argv.slice(2);
const pg = await open({ file: theme + '.html', width: +wS, scheme: 'light' });
await pg.ev(fs.readFileSync(path.join(here, 'probe.js'), 'utf8'));
const dir = path.join(here, outdir); fs.mkdirSync(dir, { recursive: true });
for (const j of jobs) {
  const [name, rest0] = j.split('=');
  let [rest, sc] = rest0.split('~');
  const [sel, st] = rest.split('@');
  if (st) { const [id, f] = st.split(':'); await pg.ev(`__probe.goto(${JSON.stringify(id)}, ${+f})`); }
  await pg.ev('new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, 120))))');
  const clip = await pg.ev(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (!e) return null; const r = e.getBoundingClientRect(); return { x: Math.max(0, r.left + scrollX - 4), y: r.top + scrollY - 4, width: Math.min(r.width + 8, document.documentElement.clientWidth), height: r.height + 8 }; })()`);
  if (!clip) { console.log('MISSING', sel); continue; }
  await pg.shot(path.join(dir, name + '.png'), { ...clip, scale: +(sc || 1) });
  console.log(name, JSON.stringify(clip));
}
if (pg.logs.length) console.log('LOGS', pg.logs);
pg.close();
