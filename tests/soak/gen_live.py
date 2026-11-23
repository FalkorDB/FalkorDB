"""Regenerate the live section of report.html from the full-soak runner state.
usage: python3 gen_live.py   (run from the soak dir)"""
import html
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

S = Path(__file__).resolve().parent
sys.path.insert(0, str(S / "lib"))
FULL = S / "full"
ORDER = ["s07_mvcc_consistency", "s05_multitenant_churn", "s12_iot_retention", "s04_recommendation",
         "s01_social_feed", "s02_fraud_realtime", "s10_schema_under_load", "s09_replication",
         "s03_graphrag_vector", "s06_large_analytics", "s08_durability_recovery", "s11_client_chaos"]
_B = math.log(1.02)


def bval(b):
    return math.exp((b + 0.5) * _B)


def pct(h, p):
    n = sum(h.values())
    if not n:
        return None
    c = 0
    for b in sorted(h):
        c += h[b]
        if c >= p * n:
            return bval(b) / 1e3
    return None


def monitor(d):
    m = d / "monitor.csv"
    if not m.exists():
        return []
    out = []
    for l in m.read_text().splitlines()[1:]:
        p = l.split(",")
        try:
            out.append((float(p[0]), float(p[1]), float(p[2]), float(p[3]), p[7]))
        except (ValueError, IndexError):
            pass
    return out


def live_lat(d, window=60):
    """Last `window` seconds of per-op throughput and p99 from the workers' jsonl files."""
    rows = []
    for f in d.glob("lat-*.jsonl"):
        try:
            for line in f.read_text().splitlines()[-4000:]:
                x = json.loads(line)
                if "op" in x:
                    rows.append(x)
        except (OSError, json.JSONDecodeError):
            continue
    if not rows:
        return {}
    smax = max(r["s"] for r in rows)
    agg = {}
    for r in rows:
        if r["s"] <= smax - window or r["s"] == smax:
            continue
        a = agg.setdefault(r["op"], {"n": 0, "err": 0, "h": {}})
        a["n"] += r["n"]
        a["err"] += r["err"]
        for b, c in r["h"].items():
            a["h"][int(b)] = a["h"].get(int(b), 0) + c
    return {op: {"ops": (a["n"] - a["err"]) / (window - 1), "err": a["err"], "p50": pct(a["h"], .5),
                 "p99": pct(a["h"], .99)} for op, a in agg.items()}


def spark(vals, w=260, h=44, color="var(--accent)"):
    vals = [v for v in vals if v == v]
    if len(vals) < 2:
        return ""
    vals = vals[-600:]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    pts = " ".join(f"{i * (w - 4) / (len(vals) - 1) + 2:.1f},{h - 4 - (v - lo) / rng * (h - 8):.1f}"
                   for i, v in enumerate(vals))
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" style="max-width:100%">'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{pts}"/></svg>')


def fmt(v, nd=1):
    return "—" if v is None or v != v else f"{v:.{nd}f}"


def main():
    prog = (FULL / "progress.log").read_text().splitlines() if (FULL / "progress.log").exists() else []
    started, ended = {}, {}
    for l in prog:
        m = re.match(r"(\S+) (START|END)\s+(\S+) (\S+)(.*)", l)
        if m:
            k = (m.group(3), m.group(4))
            if m.group(2) == "START":
                started[k] = (m.group(1), m.group(5))
                ended.pop(k, None)  # a re-run supersedes the earlier END
            else:
                ended[k] = (m.group(1), m.group(5))
    now = time.strftime("%H:%M")
    # status table
    trs = []
    current = None
    for sc in ORDER:
        for eng in ("rust", "c") + (("c-samewindow",) if sc == "s12_iot_retention" else ()):
            k = (sc, eng)
            d = FULL / f"{sc}-{eng}"
            if k in ended:
                rc = ended[k][1].strip()
                ok = "rc=0" in rc
                st = f'<span class="pill {"ok" if ok else "crit"}">{"DONE" if ok else "EXIT " + rc.split()[0]}</span>'
                summ = d / "summary.json"
                extra = ""
                if summ.exists():
                    s = json.loads(summ.read_text())
                    fails = [c for c in s.get("checks_failed", [])]
                    crit = [c for c in fails if c.get("severity") == "critical"]
                    mon = s.get("monitor", {})
                    extra = (f'{len(s.get("checks", [])) - len(fails)}/{len(s.get("checks", []))} checks'
                             f'{" · <b>" + str(len(crit)) + " critical</b>" if crit else ""}'
                             f' · CPU {fmt(mon.get("cpu_avg_pct"), 0)}% · RSS max {fmt(mon.get("rss_max_mb"), 0)} MB'
                             f' · RSS slope {fmt(mon.get("rss_slope_mb_per_h"), 0)} MB/h')
                if eng.startswith("c") and not (d / ".commit").exists():
                    st = "<span class='pill inf'>QUEUED</span> <span class='note'>(Jul build done; v4.22.4 pending)</span>"
                trs.append(f"<tr><td>{sc}</td><td>{eng}</td><td>{st}</td><td class='num'>{started[k][0]}–{ended[k][0]}</td><td>{extra}</td></tr>")
            elif k in started:
                current = (sc, eng, d, started[k][0])
                trs.append(f"<tr><td>{sc}</td><td>{eng}</td><td><span class='pill maj'>RUNNING</span></td><td class='num'>since {started[k][0]}</td><td>see live panel</td></tr>")
            else:
                ready = (S / "scenarios" / f"{sc}.py").exists()
                trs.append(f"<tr><td>{sc}</td><td>{eng}</td><td><span class='pill inf'>{'QUEUED' if ready else 'NOT READY'}</span></td><td></td><td></td></tr>")
    # live panel
    livep = "<p class='note'>No run in progress.</p>"
    if current:
        sc, eng, d, t0 = current
        mon = monitor(d)
        phase = mon[-1][4] if mon else "starting"
        cpu = [x[1] for x in mon]
        rss = [x[2] for x in mon]
        um = [x[3] for x in mon]
        lat = live_lat(d)
        oprows = "".join(
            f"<tr><td>{html.escape(op)}</td><td class='num'>{fmt(v['ops'])}</td><td class='num'>{fmt(v['p50'], 2)}</td>"
            f"<td class='num'>{fmt(v['p99'], 2)}</td><td class='num'>{v['err']}</td></tr>"
            for op, v in sorted(lat.items()) if not op.startswith("_"))
        late = sum(v["ops"] for op, v in lat.items() if op.startswith("_"))
        livep = f"""
      <p><b>{sc}</b> on <b>{eng}</b>, phase <code>{html.escape(phase)}</code>, started {t0}, {len(mon)} s monitored.</p>
      <div class="two">
        <div class="card"><h3>Server CPU % · now {fmt(cpu[-1] if cpu else None, 0)}</h3>{spark(cpu)}<span class="note">avg {fmt(sum(c for c in cpu if c == c) / max(1, len([c for c in cpu if c == c])) if cpu else None, 0)}%, 100 = one core</span></div>
        <div class="card"><h3>RSS MB · now {fmt(rss[-1] if rss else None, 0)}</h3>{spark(rss, color="var(--warn)")}<span class="note">used_memory now {fmt(um[-1] if um else None, 0)} MB, peak RSS {fmt(max(rss) if rss else None, 0)} MB</span></div>
      </div>
      <div class="tbl"><table><thead><tr><th>Op (last 60 s)</th><th>ops/s</th><th>p50 ms</th><th>p99 ms</th><th>errors</th></tr></thead>
      <tbody>{oprows or "<tr><td colspan=5 class='note'>No load phase yet (ingest or verify).</td></tr>"}</tbody></table></div>
      {"<p class='note'>Harness signals (late or missed sends): " + fmt(late) + "/s in the last minute.</p>" if late else ""}"""
    # scores so far
    scores = ""
    try:
        subprocess.run([sys.executable, "-I", str(S / "score.py"), str(FULL)], capture_output=True, timeout=120)
        sj = json.loads((FULL / "scores.json").read_text())
        if sj["scenarios"]:
            r = "".join(
                f"<tr><td>{k}</td><td class='num'><b>{v['total']}</b></td><td class='num'>{v['correctness']}</td><td class='num'>{v['stability']}</td>"
                f"<td class='num'>{v['resources']}</td><td class='num'>{v['performance']}</td><td class='num'>{fmt(v['p99_vs_c_gmean'], 2)}</td>"
                f"<td class='num'>{fmt(v['thr_vs_c_gmean'], 2)}</td><td class='num'>{fmt(v['rss_vs_c'], 2)}</td></tr>"
                for k, v in sj["scenarios"].items())
            scores = f"""<h3>Scores so far (full-scale)</h3>
      <div class="tbl"><table><thead><tr><th>Scenario</th><th>Total /100</th><th>Correct /40</th><th>Stable /25</th><th>Resources /15</th><th>Perf /20</th><th>p99 Rust/C</th><th>Thru Rust/C</th><th>RSS Rust/C</th></tr></thead><tbody>{r}</tbody></table></div>
      <p class="note">Running release score: <b>{sj['release_score']}</b> over {len(sj['scenarios'])} scenarios. Blocked by: {', '.join(sj['blocked_by']) or 'none in full runs so far'}. A scenario is scored only after its C baseline finishes, so a C run still in progress shows as no C data.</p>"""
    except Exception as e:  # never break the page over scoring
        scores = f"<p class='note'>Scoring unavailable: {html.escape(str(e))}</p>"
    done = sum(1 for r in trs if "pill ok" in r or "pill crit" in r)
    sec = f"""<section id="live">
    <h2>Full-scale soak · live</h2>
    <p class="meta">Refreshed {now} (about every 25 minutes) · {done} of {len(trs)} runs finished (Rust and C v4.22.4) · Rust 20 min, C 5 min per scenario, one at a time</p>
    {livep}
    {scores}
    <div class="tbl"><table><thead><tr><th>Scenario</th><th>Engine</th><th>Status</th><th>Time</th><th>Result</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div>
  </section>"""
    rp = S / "report.html"
    t = rp.read_text()
    t = re.sub(r"<!--LIVE-START-->.*?<!--LIVE-END-->", lambda _: "<!--LIVE-START-->" + sec + "<!--LIVE-END-->", t, flags=re.S)
    t = re.sub(r"(<p class=\"meta\">Updated )[^·]*·", lambda m: m.group(1) + time.strftime("%Y-%m-%d %H:%M") + " ·", t, count=1)
    rp.write_text(t)
    print("ok", done, current[:2] if current else None)


if __name__ == "__main__":
    main()
