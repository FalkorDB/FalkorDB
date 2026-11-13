// Minimal DevTools-protocol driver for headless Chrome.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const CH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
export const here = path.dirname(new URL(import.meta.url).pathname);
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function open({ file, width, height = 1000, scheme = 'light', reducedMotion = 'no-preference', dsf = 1 }) {
  const prof = fs.mkdtempSync(path.join(process.env.SCRATCH || os.tmpdir(), 'cdp-'));
  const port = 9400 + Math.floor(Math.random() * 500);
  const proc = spawn(CH, ['--headless=new', '--disable-gpu', '--no-first-run', '--hide-scrollbars', `--user-data-dir=${prof}`, `--remote-debugging-port=${port}`, 'about:blank'], { stdio: 'ignore' });
  let targets = [];
  for (let i = 0; i < 100; i++) {
    try { targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); if (targets.some((t) => t.type === 'page')) break; } catch {}
    await sleep(150);
  }
  const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener('open', r));
  let mid = 0;
  const pending = new Map(), logs = [], listeners = [];
  ws.addEventListener('message', (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
    if (m.method === 'Runtime.exceptionThrown') logs.push('EXC ' + m.params.exceptionDetails.text + ' ' + ((m.params.exceptionDetails.exception || {}).description || ''));
    if (m.method === 'Runtime.consoleAPICalled' && (m.params.type === 'error' || m.params.type === 'warning')) logs.push('CONSOLE.' + m.params.type + ' ' + m.params.args.map((a) => a.value || a.description).join(' '));
    if (m.method === 'Log.entryAdded') logs.push('LOG.' + m.params.entry.level + ' ' + m.params.entry.text + ' ' + (m.params.entry.url || ''));
    for (const f of listeners) f(m);
  });
  const send = (method, params = {}) => new Promise((r) => { const i = ++mid; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  await send('Page.enable'); await send('Runtime.enable'); await send('Log.enable'); await send('DOM.enable'); await send('CSS.enable');
  await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: scheme }, { name: 'prefers-reduced-motion', value: reducedMotion }] });
  await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: dsf, mobile: width < 500 });
  const loaded = new Promise((r) => listeners.push((m) => { if (m.method === 'Page.loadEventFired') r(); }));
  await send('Page.navigate', { url: 'file://' + path.join(here, file) });
  await loaded;
  await sleep(1500);
  const ev = async (expr) => {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (r.result.exceptionDetails) throw new Error('eval: ' + JSON.stringify(r.result.exceptionDetails).slice(0, 800));
    return r.result.result.value;
  };
  await ev('document.fonts.ready.then(() => 1)');
  const shot = async (out, clip) => {
    const r = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { ...clip, scale: clip.scale || 1 } });
    fs.writeFileSync(out, Buffer.from(r.result.data, 'base64'));
  };
  const close = () => { try { ws.close(); } catch {} proc.kill('SIGKILL'); };
  return { send, ev, shot, close, logs };
}
