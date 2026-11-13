// node shoot.mjs <light|dark> <width> <scheme> [scale] [full]
import fs from 'node:fs';
import path from 'node:path';
import { open, here } from './cdp.mjs';

const [theme, wS, scheme, scS, full] = process.argv.slice(2);
const width = +wS, scale = +(scS || 1);
const pg = await open({ file: theme + '.html', width, scheme });
// OPEN_DETAILS=1 opens every <details> (the collapsed appendix) and shoots into a separate folder
const openAll = process.env.OPEN_DETAILS === '1';
if (openAll) await pg.ev(`document.querySelectorAll('details').forEach((d) => { d.open = true; }); new Promise((r) => setTimeout(r, 400))`);
const dir = path.join(here, 'shots', `${theme}-${width}` + (openAll ? '-open' : ''));
fs.mkdirSync(dir, { recursive: true });
const H = await pg.ev('document.documentElement.scrollHeight');
const blocks = await pg.ev(`[...document.querySelectorAll('header.top, .part, section, footer, details > summary')].filter((e) => !e.parentElement.closest('details:not([open])') || e.matches('details > summary')).map((e) => { const r = e.getBoundingClientRect(); return { id: e.id || e.tagName.toLowerCase() + (e.className ? '.' + e.className : ''), y: r.top + scrollY, h: r.height }; })`);
let k = 0;
const list = [];
for (const b of blocks) {
  // split tall blocks into pieces of at most 2200 px
  const piece = 2200;
  for (let y = b.y, j = 0; y < b.y + b.h; y += piece, j++) {
    const h = Math.min(piece, b.y + b.h - y);
    const name = String(++k).padStart(3, '0') + '-' + b.id.replace(/[^\w.-]+/g, '_') + (j ? '-' + j : '') + '.png';
    await pg.shot(path.join(dir, name), { x: 0, y, width, height: Math.ceil(h), scale });
    list.push(name + ' y=' + Math.round(y) + ' h=' + Math.round(h));
  }
}
if (full) await pg.shot(path.join(here, 'shots', `full-${theme}-${width}.png`), { x: 0, y: 0, width, height: H, scale: +full });
fs.writeFileSync(path.join(dir, 'index.txt'), list.join('\n'));
console.log(dir, 'H', H, 'shots', list.length);
pg.close();
