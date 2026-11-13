import { open } from './cdp.mjs';
const [theme, w] = process.argv.slice(2);
const pg = await open({ file: theme + '.html', width: +w });
const res = await pg.ev(`(() => {
  const out = {};
  for (const svg of document.querySelectorAll('svg')) {
    const sec = svg.closest('section'); const id = sec ? sec.id : '?';
    const vb = svg.viewBox.baseVal; const r = svg.getBoundingClientRect();
    if (!vb || !vb.width || !r.width) continue;
    const k = r.width / vb.width;
    for (const t of svg.querySelectorAll('text')) {
      const px = parseFloat(getComputedStyle(t).fontSize) * k;
      const o = out[id] ||= { min: 99, cls: '', n7: 0, n8: 0, scale: 9 };
      if (px < 7) o.n7++; if (px < 8) o.n8++;
      if (px < o.min) { o.min = +px.toFixed(2); o.cls = (t.getAttribute('class')||'') + ' "' + t.textContent.slice(0,8) + '"'; }
      o.scale = Math.min(o.scale, +k.toFixed(2));
    }
  }
  return out;
})()`);
const rows = Object.entries(res).filter(([k, v]) => v.min < 9).map(([k, v]) => k + ' min ' + v.min + 'px (' + v.cls + ') svgScale ' + v.scale + ' <7px:' + v.n7 + ' <8px:' + v.n8);
console.log(theme, w, '\n' + rows.join('\n'));
pg.close();
