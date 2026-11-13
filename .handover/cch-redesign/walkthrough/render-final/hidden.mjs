import fs from 'node:fs';
import path from 'node:path';
import { open, here } from './cdp.mjs';
const pg = await open({ file: 'light.html', width: 400 });
await pg.ev(fs.readFileSync(path.join(here, 'probe.js'), 'utf8'));
const check = `((root) => {
  const HL = '.nd.cur, .nd.ring, .nd.ring2, .nd.ringbad, .nd.top, .nd.pool, .e.hot, .e.mt, .e.new, .e.add, .e.del, .e.hop, .box.cur';
  const out = [];
  for (const el of root.querySelectorAll(HL)) {
    let p = el.parentElement; while (p && !(['auto','scroll'].includes(getComputedStyle(p).overflowX))) p = p.parentElement;
    if (!p) continue;
    const pr = p.getBoundingClientRect(), r = el.getBoundingClientRect();
    const visW = Math.max(0, Math.min(r.right, pr.right) - Math.max(r.left, pr.left));
    if (r.width > 0 && visW < r.width * 0.5) {
      const t = el.nextElementSibling && el.nextElementSibling.tagName === 'text' ? el.nextElementSibling.textContent : '';
      out.push((el.getAttribute('class') || '') + (t ? ' "' + t + '"' : '') + ' x=' + Math.round(r.left - pr.left) + '..' + Math.round(r.right - pr.left) + ' paneW=' + Math.round(pr.width));
    }
  }
  return out;
})`;
const atRest = await pg.ev(`[...document.querySelectorAll('section')].filter(s => !s.querySelector('.stepper')).map(s => [s.id, ${check}(s)]).filter(x => x[1].length)`);
console.log('AT REST (sections without steppers):'); for (const [id, l] of atRest) console.log(' ', id, l.length, l.slice(0, 4).join(' | '));
// also nodes 16,17,18 (any) hidden at rest in non-stepper figures
const st = await pg.ev('__probe.steppers()');
let total = 0, hit = 0;
for (const s of st) {
  const frames = [];
  for (let i = 0; i < s.n; i++) {
    await pg.ev(`__probe.goto(${JSON.stringify(s.id)}, ${i})`);
    await pg.ev('new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, 50))))');
    const l = await pg.ev(`${check}(document.getElementById(${JSON.stringify(s.id)}))`);
    total++;
    if (l.length) { hit++; frames.push(i + ': ' + l.slice(0, 2).join(' | ')); }
  }
  if (frames.length) console.log(s.id, frames.length + '/' + s.n, '\n   ' + frames.slice(0, 3).join('\n   '));
  await pg.ev(`__probe.goto(${JSON.stringify(s.id)}, 0)`);
}
console.log('frames with a hidden highlight:', hit, 'of', total);
pg.close();
