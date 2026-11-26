"""Build the final soak report (report.html) from full/ results, findings.json and scores.json.
usage: python -I gen_final.py

The gate line, the tiles and each scenario's labels come from scores.json and the runs it
scored. What a person wrote about one campaign (its label, the campaign it is compared
with, the machine, notes and caveats) lives in campaign.json beside this script, and the
report shows the notes as written for that campaign."""
import csv
import html
import json
import math
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

S = Path(__file__).resolve().parent
FULL = S / "full"
PY = sys.executable

SCN = [  # id, title, what it models, full-scale size
    ("s01_social_feed", "Social feed", "LDBC-like social network: feed, friend-of-friend with 8k-degree hubs, shortest path; 95/5 read/write", "1M persons, ~7M edges"),
    ("s02_fraud_realtime", "Fraud detection", "3,200 MERGE tx/s streaming ingest with ring, velocity, fan-in and cycle detection", "1.2M base nodes, +3.85M tx"),
    ("s03_graphrag_vector", "GraphRAG", "Vector KNN + 1-2 hop expansion, full-text, live ingest/update/delete of chunks", "250k chunks (384-d), 100k entities"),
    ("s04_recommendation", "Recommendations", "Co-purchase top-k, collaborative filtering, aggregations, 100-id batches", "600k nodes, 6.2M edges"),
    ("s05_multitenant_churn", "Multi-tenant", "3,000 graphs churned: create, copy, rename, delete; LIST/MEMORY/EXPLAIN", "3,000 graphs, 4.4M nodes"),
    ("s06_large_analytics", "Large analytics", "Var-length paths, shortest paths, algo.*, full-graph aggregates beside point lookups", "3.4M nodes, 32.8M edges"),
    ("s07_mvcc_consistency", "MVCC ledger", "Concurrent transfers with snapshot-invariant readers, failing writes, peak closed loop", "200k accounts, +1.1M transfers"),
    ("s08_durability_recovery", "Durability", "AOF always/everysec and RDB, BGSAVE under load, 8 kill -9 / restart segments", "1.5M nodes, 7.5M edges"),
    ("s09_replication", "Replication", "Primary + replica: 870 writes/s, replica reads, full resync, failover drill", "1M nodes, 5M edges"),
    ("s10_schema_under_load", "Online schema", "Create/drop range, full-text indexes and constraints under live reads and writes", "1.95M nodes, 5.2M edges"),
    ("s11_client_chaos", "Client chaos", "Connection churn, slow readers, CLIENT KILL, timeouts, error floods, 620 clients", "500k-node victim graph"),
    ("s12_iot_retention", "IoT retention", "4,000 inserts/s with equal retention deletes; window aggregations", "1.9M live readings"),
]
ENG = [("rust", "Rust", "var(--s1)"), ("c", "C", "var(--s2)")]
ISSUE = "https://github.com/FalkorDB/FalkorDB/issues/"


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def cdir(scn):
    sw = FULL / f"{scn}-c-samewindow"
    return sw if (sw / ".done").exists() else FULL / f"{scn}-c"


def run_dirs(scn, sc):
    """The Rust and C run directories score.py scored for this scenario (its `run` and
    `c_run`; c_run None: no C baseline), or the current ones when it has not scored it."""
    rust = FULL / sc["run"] if sc.get("run") else FULL / f"{scn}-rust"
    if "c_run" in sc:
        return {"rust": rust, "c": FULL / sc["c_run"] if sc["c_run"] else None}
    return {"rust": rust, "c": cdir(scn)}


def campaign():
    """campaign.json: label (run_rust_main.sh's <commit-label>), previous (compared through
    full/scores-<previous>.json), c_baseline, environment, notes {written_for, html},
    caveats. Without a label: the one most current Rust runs carry."""
    c = jl(S / "campaign.json") or {}
    if not c.get("label"):
        labels = [(FULL / f"{scn}-rust" / ".commit").read_text().strip() for scn, *_ in SCN
                  if (FULL / f"{scn}-rust" / ".commit").exists()]
        c["label"] = max(set(labels), key=labels.count) if labels else None
    return c


def label_of(d):
    return (d / ".commit").read_text().strip() if (d / ".commit").exists() else None


def fmt(v, nd=1, unit=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.{nd}f}{unit}"


def esc(s):
    return html.escape(str(s))


def linkify(s):
    import re
    return re.sub(r"#(\d{3,5})", lambda m: f'<a href="{ISSUE}{m.group(1)}">#{m.group(1)}</a>', esc(s))


# ------------------------------------------------------------- series for charts
def mon_series(d):
    m = d / "monitor.csv"
    if not m.exists():
        return None
    rows = []
    for l in m.read_text().splitlines()[1:]:
        p = l.split(",")
        try:
            rows.append((float(p[0]), float(p[1]), float(p[2])))
        except (ValueError, IndexError):
            pass
    if not rows:
        return None
    bins = {}
    for t, cpu, rss in rows:
        b = int(t // 15)
        bins.setdefault(b, []).append((cpu, rss))
    cpu, rss = [], []
    for b in sorted(bins):
        c = [x for x, _ in bins[b] if not math.isnan(x)]
        r = [y for _, y in bins[b] if not math.isnan(y)]
        x = round(b * 15 / 60, 2)
        if c:
            cpu.append([x, round(st.mean(c), 1)])
        if r:
            rss.append([x, round(max(r))])
    return cpu, rss


def lat_series(d, op):
    p = d / "latency_ts.csv"
    if not p.exists():
        return None, None
    per = {}
    thr = {}
    for r in csv.DictReader(p.open()):
        s = int(r["sec"])
        if not r["op"].startswith("_") and not r["op"][:2] in ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"):
            thr[s] = thr.get(s, 0) + int(r["ok"])
        if r["op"] == op and r["p99_ms"] not in ("nan", ""):
            per.setdefault(s, []).append(float(r["p99_ms"]))
    if not thr:
        return None, None
    s0 = min(thr)
    p99, tput = {}, {}
    for s, v in per.items():
        p99.setdefault((s - s0) // 15, []).extend(v)
    for s, v in thr.items():
        tput.setdefault((s - s0) // 15, []).append(v)
    return ([[round(b * 15 / 60, 2), round(st.median(v), 2)] for b, v in sorted(p99.items())],
            [[round(b * 15 / 60, 2), round(st.mean(v))] for b, v in sorted(tput.items())])


def main_op(summ):
    ops = {k: v for k, v in summ.get("ops", {}).items()
           if not k.startswith("_") and not k[:2] in ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8")}
    return max(ops, key=lambda k: ops[k]["count"]) if ops else None


# ------------------------------------------------------------- build
def build():
    scoring = subprocess.run([PY, "-I", str(S / "score.py"), str(FULL)], capture_output=True, text=True,
                             timeout=300)
    # a score.py that failed left the previous scores.json (if any): none of it is this run's
    scores = (jl(FULL / "scores.json") if scoring.returncode == 0 else None) or {"scenarios": {}}
    camp = campaign()
    label, previous = camp.get("label"), camp.get("previous")
    prev = (jl(FULL / f"scores-{previous}.json") if previous else None) or {"scenarios": {}}
    crashes = {"rust": 0, "c": 0}  # scenarios whose server log has crash markers
    findings = jl(S / "findings.json") or []
    charts = []
    rows_score, cards = [], []
    for scn, title, what, size in SCN:
        sc = scores["scenarios"].get(scn, {})
        dirs = run_dirs(scn, sc)
        sm = {e: jl(dirs[e] / "summary.json") if dirs[e] else None for e, _, _ in ENG}
        for e in crashes:
            crashes[e] += bool((sm[e] or {}).get("server_log_crash_markers"))
        r = sm["rust"] or {}
        mon_r, mon_c = r.get("monitor", {}), (sm["c"] or {}).get("monitor", {})
        crit = len(sc.get("critical_failures", []))
        maj = len(sc.get("major_failures", []))
        pill = ('<span class="pill crit">BLOCKER</span>' if crit or scn in scores.get("failed_runs", {}) else
                '<span class="pill maj">PASS w/ issues</span>' if maj else '<span class="pill ok">PASS</span>')
        on = label_of(dirs["rust"])
        was = prev["scenarios"].get(scn, {}).get("total")
        if on and on == label:
            sub = f"on {on}" + (f" · was {fmt(was, 1)} on {previous}" if previous and was is not None else "")
        else:
            sub = f"on {on or '?'}" + (f" · {label} re-run pending" if label else "")
        rows_score.append(
            f"<tr><td><b>{title}</b><div class='sub'>{esc(size)}</div></td><td class='num big'>{fmt(sc.get('total'), 1)}"
            f"<div class='sub'>{esc(sub)}</div></td>"
            f"<td>{pill}</td><td class='num'>{fmt(sc.get('correctness'), 0)}/40</td><td class='num'>{fmt(sc.get('stability'), 0)}/25</td>"
            f"<td class='num'>{fmt(sc.get('resources'), 1)}/15</td><td class='num'>{fmt(sc.get('performance'), 1)}/20</td>"
            f"<td class='num'>{fmt(sc.get('p99_vs_c_gmean'), 2, '×')}</td><td class='num'>{fmt(sc.get('rss_vs_c'), 2, '×')}</td>"
            f"<td class='num'>{fmt(mon_r.get('cpu_avg_pct'), 0, '%')} / {fmt(mon_c.get('cpu_avg_pct'), 0, '%')}</td></tr>")
        op = main_op(r)
        ser = {"cpu": [], "rss": [], "p99": [], "tput": []}
        for e, name, color in ENG:
            d = dirs[e]
            if d is None or not (d / ".done").exists():
                continue
            ms = mon_series(d)
            if ms:
                ser["cpu"].append({"name": name, "color": color, "pts": ms[0]})
                ser["rss"].append({"name": name, "color": color, "pts": ms[1]})
            p99, tput = lat_series(d, op) if op else (None, None)
            if p99:
                ser["p99"].append({"name": name, "color": color, "pts": p99})
            if tput:
                ser["tput"].append({"name": name, "color": color, "pts": tput})
        cid = scn[:3]
        charts.append({"id": cid, "charts": [
            {"key": "cpu", "title": "Server CPU", "unit": "%", "x": "min since start", "series": ser["cpu"]},
            {"key": "rss", "title": "Server RSS", "unit": " MB", "x": "min since start", "series": ser["rss"]},
            {"key": "p99", "title": f"p99 · {op or '—'}", "unit": " ms", "x": "min into load", "series": ser["p99"], "log": True},
            {"key": "tput", "title": "Throughput, all ops", "unit": " ops/s", "x": "min into load", "series": ser["tput"]},
        ]})
        notes = "".join(f"<li>{esc(n)}</li>" for n in sc.get("notes", []))
        fails = [c for c in (r.get("checks", []) or []) if not c["ok"]]
        fl = "".join(f"<li><span class='pill {'crit' if c['severity'] == 'critical' else 'maj' if c['severity'] == 'major' else 'inf'}'>"
                     f"{c['severity'].upper()}</span> {esc(c['name'])}</li>" for c in fails[:8])
        cards.append(f"""
      <article class="scn" id="{cid}">
        <header><div><div class="eyebrow">{cid}</div><h3>{title}</h3></div><div class="score">{fmt(sc.get('total'), 1)}<span>/100</span></div></header>
        <p class="note">{esc(what)} · <b>{esc(size)}</b></p>
        <div class="grid4">{''.join(f'<figure class="chart" data-c="{cid}" data-k="{k}"></figure>' for k in ("cpu", "rss", "p99", "tput"))}</div>
        <details><summary>Rust checks failed ({len(fails)}) and scoring notes</summary><ul>{fl or '<li>None</li>'}</ul><ul class="notes">{notes}</ul></details>
      </article>""")


    # findings
    order = {"critical": 0, "major": 1, "minor": 2, "info": 3}
    rust = [f for f in findings if f.get("engine") in ("rust", "both")]
    rust.sort(key=lambda f: (order.get(f["severity"], 9), f["id"]))
    cside = [f for f in findings if f.get("engine") == "c"]
    def frow(f):
        sev = f["severity"]
        cls = {"critical": "crit", "major": "maj"}.get(sev, "inf")
        status = f.get("status") or f.get("fix_pr") or f.get("issue") or "not filed"
        num = f.get("numbers", "")
        return (f"<tr><td class='num'>{f['id']}</td><td><span class='pill {cls}'>{sev.upper()}</span>"
                f"{'<div class=sub>both engines</div>' if f.get('engine') == 'both' else ''}</td>"
                f"<td><b>{linkify(f['title'])}</b>{'<div class=sub>' + linkify(num) + '</div>' if num else ''}</td>"
                f"<td class='note'>{esc(f.get('scenario', ''))}</td><td class='note'>{linkify(status)}</td></tr>")
    ncrit = sum(1 for f in rust if f["severity"] == "critical" and f.get("engine") == "rust" and "FIXED" not in str(f.get("status", "")))
    nmaj = sum(1 for f in rust if f["severity"] == "major" and "FIXED" not in str(f.get("status", "")))
    release = scores.get("release_score")
    blocked = scores.get("blocked_by", [])
    now = time.strftime("%Y-%m-%d %H:%M")
    crit_ids = ", ".join(f["id"] for f in rust if f["severity"] == "critical" and f.get("engine") == "rust"
                         and "FIXED" not in str(f.get("status", "")))
    fixed_ids = ", ".join(f["id"] for f in rust if "FIXED" in str(f.get("status", "")))
    nscored = len(scores["scenarios"])
    # not scored and not blocked: a run still going, a C baseline not finished, or never run
    unscored = [scn for scn, *_ in SCN if scn not in scores["scenarios"] and scn not in blocked]
    reasons = []
    if ncrit:
        reasons.append(f"{ncrit} open critical Rust bug{'' if ncrit == 1 else 's'} ({crit_ids})")
    if blocked:
        reasons.append(f"a critical check failed or the run failed in {', '.join(blocked)}")
    if scoring.returncode != 0:
        why = (scoring.stderr or "").strip().splitlines()[-1:] or ["no output"]
        gate = f"Not decided: score.py failed (exit {scoring.returncode}): {why[0][:200]}."
    elif reasons:
        gate = f"Blocked: {'; '.join(reasons)}."
    elif unscored:
        gate = f"Not decided: {len(unscored)} of {len(SCN)} scenarios are not scored yet."
    else:
        gate = "Not blocked: no open critical Rust bug, and no critical check or Rust run failed."
    gate += f" Scenario score {fmt(release, 1)}/100 over {nscored} of {len(SCN)} scenarios."
    if unscored and scoring.returncode == 0:
        gate += f" Not scored yet: {', '.join(unscored)}."
    passed = gate.startswith("Not blocked")
    c_name = camp.get("c_baseline") or "C"
    notes = camp.get("notes") or {}
    notes_html = (f"<p><b>Notes written for the {esc(notes.get('written_for') or '?')} campaign</b> "
                  f"(they describe those runs, not necessarily the scores above): {notes['html']}</p>"
                  if notes.get("html") else "")
    caveats = (f'<div class="card"><h3>Caveats</h3><p class="note">{esc(camp["caveats"])}</p></div>'
               if camp.get("caveats") else "")
    eyebrow = "Release readiness · FalkorDB-rs main" + (f" · {label}" if label else "") \
        + (f" (compared with {previous})" if previous else "")

    page = f"""<title>FalkorDB Release Soak</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;700&display=swap">
<style>
/* Layout: one reading column — verdict, scores, findings, per-scenario chart cards, method. */
:root{{
  --bg:#f6f7f9; --panel:#ffffff; --fg:#17202b; --muted:#5a6573; --line:#dde2e8; --grid:#e9edf1;
  --accent:#2357c6; --bad:#b4232f; --badbg:#fbe9eb; --warn:#9a5b00; --warnbg:#fdf2dc;
  --good:#1d7a46; --goodbg:#e4f4ea; --info:#4b5563; --infobg:#eceff3;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --display:"Space Grotesk",system-ui,sans-serif; --body:"IBM Plex Sans",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,monospace;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#0f141a; --panel:#161d26; --fg:#e4e9ef; --muted:#9aa6b4; --line:#273140; --grid:#1f2833;
  --accent:#7aa2ff; --bad:#ff7b86; --badbg:#3a1a1f; --warn:#f0b35a; --warnbg:#3a2a12;
  --good:#5fd394; --goodbg:#13301f; --info:#b6c0cc; --infobg:#202a36;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; color-scheme:dark}}}}
:root[data-theme="dark"]{{
  --bg:#0f141a; --panel:#161d26; --fg:#e4e9ef; --muted:#9aa6b4; --line:#273140; --grid:#1f2833;
  --accent:#7aa2ff; --bad:#ff7b86; --badbg:#3a1a1f; --warn:#f0b35a; --warnbg:#3a2a12;
  --good:#5fd394; --goodbg:#13301f; --info:#b6c0cc; --infobg:#202a36;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 var(--body);}}
.wrap{{max-width:1120px;margin:0 auto;padding-block:32px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:36px}}
h1,h2,h3{{font-family:var(--display);text-wrap:balance;margin:0;line-height:1.2}}
h1{{font-size:30px}} h2{{font-size:21px}} h3{{font-size:17px}}
p{{margin:0;max-width:80ch}}
.eyebrow{{font:500 12px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
.meta{{font:13px/1.5 var(--mono);color:var(--muted)}}
.count{{font:500 13px var(--mono);color:var(--muted);margin-left:8px}}
.verdict{{background:var(--badbg);border:1px solid var(--bad);border-radius:10px;padding:20px 22px;display:grid;gap:14px}}
.verdict .big{{font:700 22px/1.25 var(--display);color:var(--bad)}}
.verdict.pass{{background:var(--goodbg);border-color:var(--good)}} .verdict.pass .big{{color:var(--good)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
.tile{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;display:grid;gap:4px}}
.tile b{{font:600 24px/1.1 var(--display);font-variant-numeric:tabular-nums}}
.tile span{{font-size:13px;color:var(--muted)}}
section{{display:grid;gap:14px}}
.tbl{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel)}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th,td{{padding:8px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}}
th{{font:500 11.5px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.05em;color:var(--muted);background:var(--bg)}}
tr:last-child td{{border-bottom:0}}
td.num{{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}}
td.big{{font:600 16px var(--display)}}
.sub{{font-size:12.5px;color:var(--muted);margin-top:3px}}
code{{font:12.5px var(--mono);background:var(--infobg);padding:1px 5px;border-radius:4px;word-break:break-word}}
.pill{{display:inline-block;font:600 11px/1 var(--mono);padding:4px 7px;border-radius:99px;white-space:nowrap;letter-spacing:.03em}}
.crit{{background:var(--badbg);color:var(--bad)}} .maj{{background:var(--warnbg);color:var(--warn)}}
.ok{{background:var(--goodbg);color:var(--good)}} .inf{{background:var(--infobg);color:var(--info)}}
.note{{font-size:13.5px;color:var(--muted)}}
ul{{margin:0;padding-left:20px;display:grid;gap:6px}}
a{{color:var(--accent)}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;font:500 13px var(--mono);color:var(--fg)}}
.legend i{{display:inline-block;width:18px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}}
.scn{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;display:grid;gap:10px;min-width:0}}
.scn header{{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}}
.score{{font:700 26px var(--display);font-variant-numeric:tabular-nums}} .score span{{font-size:13px;color:var(--muted);font-weight:500}}
.grid4{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}}
.chart{{margin:0;position:relative;min-width:0}}
.chart .t{{font:500 12px var(--mono);color:var(--muted);margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.chart svg{{display:block;width:100%;height:auto;overflow:visible}}
.chart .ax{{font:10px var(--mono);fill:var(--muted)}}
.chart .lab{{font:600 10px var(--mono);fill:var(--fg)}}
.tip{{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:12px/1.4 var(--mono);color:var(--fg);box-shadow:0 2px 8px rgba(0,0,0,.12);white-space:nowrap;z-index:5}}
.tip i{{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}}
details summary{{cursor:pointer;font-size:13.5px;color:var(--muted)}}
details ul{{margin-top:8px;font-size:13px}}
.notes li{{color:var(--muted)}}
.two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;display:grid;gap:8px;min-width:0}}
@media (prefers-reduced-motion:reduce){{*{{transition:none!important}}}}
</style>

<div class="wrap">
  <header style="display:grid;gap:10px">
    <div class="eyebrow">{esc(eyebrow)}</div>
    <h1>Soak test of clean main</h1>
    <p class="meta">Updated {now} · {nscored} of {len(SCN)} scenarios scored{(' · ' + esc(camp['environment'])) if camp.get('environment') else ''}</p>
  </header>

  <div class="verdict{' pass' if passed else ''}">
    <div class="eyebrow">Release gate</div>
    <div class="big">{esc(gate)}</div>
    {notes_html}
    <div class="tiles">
      <div class="tile"><b>{fmt(release, 1)}</b><span>release score (mean of {nscored} scenarios)</span></div>
      <div class="tile"><b>{ncrit}</b><span>open critical Rust bugs (findings.json)</span></div>
      <div class="tile"><b>{nmaj}</b><span>open major Rust bugs</span></div>
      <div class="tile"><b>{crashes['rust']} / {crashes['c']}</b><span>scenarios that crashed: Rust / {esc(c_name)}</span></div>
      <div class="tile"><b>{sum(1 for s in scores['scenarios'].values() if s.get('total', 0) >= 90)}/{nscored}</b><span>scenarios scoring ≥ 90</span></div>
    </div>
  </div>

  <section>
    <h2>Scenario scores</h2>
    <p class="note">Per scenario: correctness 40 (any critical check failing gives 0), stability 25, resources 15 (memory at equal data vs C, growth per op), performance 20 (p99 and throughput vs C, drift).
    Ratios are Rust ÷ C, so below 1 is better. Where C crashed or could not finish, C ratios are scored as neutral. Scoring overrides with reasons are under each scenario.</p>
    <div class="tbl"><table><thead><tr><th>Scenario</th><th>Score</th><th>Gate</th><th>Correct</th><th>Stable</th><th>Resources</th><th>Perf</th><th>p99 vs C</th><th>RSS vs C</th><th>CPU Rust / C</th></tr></thead>
    <tbody>{''.join(rows_score)}</tbody></table></div>
  </section>

  <section>
    <h2>Rust findings</h2>
    <div class="tbl"><table><thead><tr><th>ID</th><th>Severity</th><th>What happens</th><th>Found in</th><th>Status</th></tr></thead>
    <tbody>{''.join(frow(f) for f in rust if "FIXED" not in str(f.get("status", "")))}</tbody></table></div>
    <details><summary>Problems in C seen while using it as a baseline ({len(cside)}; status column says whether each still happens on {esc(c_name)})</summary>
    <div class="tbl" style="margin-top:8px"><table><tbody>{''.join(frow(f) for f in cside)}</tbody></table></div></details>
  </section>

<!--LIVE-START--><!--LIVE-END-->

  <section>
    <h2>CPU, memory, latency and throughput</h2>
    <p class="note">Each card shows the full-scale runs: server CPU (100% = one core) and RSS in 15-second bins from server start, p99 of the busiest operation (median of per-second p99 in 15-second bins, log scale), and total successful ops/s. Hover a chart for exact values.</p>
    <div class="legend"><span><i style="background:var(--s1)"></i>Rust (main)</span><span><i style="background:var(--s2)"></i>{esc(c_name)}</span></div>
    {''.join(cards)}
  </section>
  <section>
    <h2>Method</h2>
    <div class="two">
      <div class="card"><h3>Scenarios</h3><p class="note">12 agents each built one production scenario with Python ground truth (seeded data plus a ledger of every acknowledged write) and smoke-tested it on Rust and C. The coordinator then ran each one alone at full scale: Rust 20 min, {esc(c_name)} 5 min at the same rates (s12 also 20 min).</p></div>
      <div class="card"><h3>Monitoring</h3><p class="note">1 Hz: server CPU, RSS, INFO used_memory, threads, clients, liveness. Per op per second: 2%-resolution latency histograms and throughput. Latency is timed from the actual send; sends more than 50 ms behind schedule count as backpressure.</p></div>
      {caveats}
    </div>
  </section>
</div>

<script>
const DATA = {json.dumps(charts, separators=(",", ":"))};
(function(){{
  const NS='http://www.w3.org/2000/svg';
  const W=300,H=150,L=40,R=46,T=8,B=22;
  function el(n,a,p){{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}}
  function nice(v){{if(v<=0)return 1;const p=Math.pow(10,Math.floor(Math.log10(v)));const f=v/p;return (f<=1?1:f<=2?2:f<=5?5:10)*p}}
  function fmtv(v,u){{return (Math.abs(v)>=100?Math.round(v).toLocaleString():(+v.toFixed(2)).toString())+u}}
  for(const g of DATA){{for(const c of g.charts){{
    const fig=document.querySelector(`figure[data-c="${{g.id}}"][data-k="${{c.key}}"]`); if(!fig)continue;
    const t=document.createElement('div');t.className='t';t.textContent=c.title;fig.appendChild(t);
    const ser=c.series.filter(s=>s.pts&&s.pts.length);
    if(!ser.length){{const p=document.createElement('div');p.className='note';p.textContent='No data';fig.appendChild(p);continue}}
    const xs=ser.flatMap(s=>s.pts.map(p=>p[0])), ys=ser.flatMap(s=>s.pts.map(p=>p[1])).filter(v=>v>0||!c.log);
    const xmax=Math.max(...xs)||1, log=!!c.log;
    let ymin=log?Math.max(0.01,Math.min(...ys)):0, ymax=Math.max(...ys)||1;
    if(log){{ymin=Math.pow(10,Math.floor(Math.log10(ymin)));ymax=Math.pow(10,Math.ceil(Math.log10(ymax)));}} else ymax=nice(ymax*1.05);
    const sx=x=>L+(x/xmax)*(W-L-R);
    const sy=y=>log?(T+(H-T-B)*(1-(Math.log10(Math.max(y,ymin))-Math.log10(ymin))/(Math.log10(ymax)-Math.log10(ymin)||1))):(T+(H-T-B)*(1-y/ymax));
    const svg=el('svg',{{viewBox:`0 0 ${{W}} ${{H}}`,role:'img','aria-label':c.title}},fig);
    const ticks=log?(()=>{{const a=[];for(let v=ymin;v<=ymax*1.0001;v*=10)a.push(v);return a}})():[0,ymax/2,ymax];
    for(const v of ticks){{el('line',{{x1:L,x2:W-R,y1:sy(v),y2:sy(v),stroke:'var(--grid)','stroke-width':1}},svg);
      const tx=el('text',{{x:L-4,y:sy(v)+3,'text-anchor':'end',class:'ax'}},svg);tx.textContent=fmtv(v,'')}}
    for(const v of [0,xmax/2,xmax]){{const tx=el('text',{{x:sx(v),y:H-6,'text-anchor':v===0?'start':v===xmax?'end':'middle',class:'ax'}},svg);tx.textContent=(+v.toFixed(0))+(v===xmax?' min':'')}}
    const ends=[];
    for(const s of ser){{const d=s.pts.map((p,i)=>(i?'L':'M')+sx(p[0]).toFixed(1)+' '+sy(p[1]).toFixed(1)).join(' ');
      el('path',{{d,fill:'none',stroke:s.color,'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round'}},svg);
      const lp=s.pts[s.pts.length-1];ends.push({{y:sy(lp[1]),x:sx(lp[0]),name:s.name}})}}
    ends.sort((a,b)=>a.y-b.y);for(let i=1;i<ends.length;i++)if(ends[i].y-ends[i-1].y<11)ends[i].y=ends[i-1].y+11;
    for(const e of ends){{const tx=el('text',{{x:Math.min(e.x+4,W-R+4),y:e.y+3,class:'lab'}},svg);tx.textContent=e.name}}
    const cross=el('line',{{y1:T,y2:H-B,stroke:'var(--muted)','stroke-width':1,'stroke-dasharray':'3 3',visibility:'hidden'}},svg);
    const dots=ser.map(s=>el('circle',{{r:4,fill:s.color,stroke:'var(--panel)','stroke-width':2,visibility:'hidden'}},svg));
    const tip=document.createElement('div');tip.className='tip';tip.hidden=true;fig.appendChild(tip);
    const hit=el('rect',{{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent'}},svg);
    function move(ev){{const r=svg.getBoundingClientRect();const px=(ev.clientX-r.left)*W/r.width;const x=Math.max(0,Math.min(xmax,(px-L)/(W-L-R)*xmax));
      cross.setAttribute('x1',sx(x));cross.setAttribute('x2',sx(x));cross.setAttribute('visibility','visible');
      let rows=[];ser.forEach((s,i)=>{{let best=s.pts[0];for(const p of s.pts)if(Math.abs(p[0]-x)<Math.abs(best[0]-x))best=p;
        if(Math.abs(best[0]-x)<=xmax*0.06){{dots[i].setAttribute('cx',sx(best[0]));dots[i].setAttribute('cy',sy(best[1]));dots[i].setAttribute('visibility','visible');
          rows.push(`<div><i style="background:${{s.color}}"></i>${{s.name}} <b>${{fmtv(best[1],c.unit)}}</b></div>`)}} else dots[i].setAttribute('visibility','hidden')}});
      tip.innerHTML=`<div>${{x.toFixed(1)}} ${{c.x}}</div>`+rows.join('');tip.hidden=!rows.length;
      const fw=fig.clientWidth;let left=(ev.clientX-r.left)+12;if(left+tip.offsetWidth>fw)left=(ev.clientX-r.left)-tip.offsetWidth-12;
      tip.style.left=Math.max(0,left)+'px';tip.style.top=(ev.clientY-fig.getBoundingClientRect().top+10)+'px'}}
    function leave(){{cross.setAttribute('visibility','hidden');dots.forEach(d=>d.setAttribute('visibility','hidden'));tip.hidden=true}}
    hit.addEventListener('pointermove',move);hit.addEventListener('pointerleave',leave);
  }}}}
}})();
</script>
"""
    # carry the live-progress section forward
    old = (S / "report.html").read_text() if (S / "report.html").exists() else ""
    import re
    m = re.search(r"<!--LIVE-START-->.*?<!--LIVE-END-->", old, flags=re.S)
    if m:
        page = page.replace("<!--LIVE-START--><!--LIVE-END-->", m.group(0))
    (S / "report.html").write_text(page)
    print("ok", len(rows_score), "scenarios,", len(page) // 1024, "KB")


if __name__ == "__main__":
    build()
