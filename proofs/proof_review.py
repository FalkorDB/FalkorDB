#!/usr/bin/env python3
"""The counterexample review of .github/workflows/proof-impact.yml's report job.

Usage: proof_review.py <verdicts.json> <impact.json> <added_theorems.json> <files.json> <posted.json>
Env: HEAD, RUN_URL, BOT_LOGIN.

Prints a pull-request review (JSON, for POST /pulls/{pr}/reviews) with one comment
per new counterexample, or {} when there is none to post.
"""
import hashlib, json, os, re, sys
verdicts_path, impact_path, added_path, files_path, posted_path = sys.argv[1:]
head, run_url, bot = os.environ['HEAD'], os.environ['RUN_URL'], os.environ['BOT_LOGIN']

def pages(path):
    # gh --paginate prints one JSON array per page.
    text, out, i, dec = open(path).read(), [], 0, json.JSONDecoder()
    while i < len(text):
        if text[i].isspace():
            i += 1; continue
        page, i = dec.raw_decode(text, i)
        out += page
    return out

def text(v, limit):
    return v.strip()[:limit] if isinstance(v, str) else ''

def fence(body, lang=''):
    ticks = '`' * max(3, max((len(m) for m in re.findall(r'`+', body)), default=0) + 1)
    return f'{ticks}{lang}\n{body}\n{ticks}'

def commentable(patch):
    # Lines of the PR head a review comment may sit on: added and context lines.
    lines, new = set(), 0
    for l in (patch or '').split('\n'):
        m = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', l)
        if m:
            new = int(m.group(1)); continue
        if l.startswith('-') or l.startswith('\\'):
            continue
        lines.add(new); new += 1
    return lines

def load(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default

verdicts = [v for v in load(verdicts_path, []) if isinstance(v, dict)]
# Only functions the diff touches (the gate's keys), and only theorems the gate
# saw the patch add as compiled, fully qualified declarations (not a comment
# or a Markdown line).
changed = {(c['file'], c['fn'], int(c['line'])) for c in load(impact_path, {}).get('changed', [])}
added = set(load(added_path, []))
by_path = {f['filename']: commentable(f.get('patch')) for f in pages(files_path)}
# Only this workflow's own earlier comments and reviews count as posted.
posted = '\n'.join(c.get('body') or '' for c in pages(posted_path)
                   if (c.get('user') or {}).get('login') == bot)
comments, loose = [], []
for v in verdicts:
    if v.get('verdict') != 'counterexample':
        continue
    path, fn, thm = text(v.get('file'), 300), text(v.get('fn'), 200), text(v.get('theorem'), 300)
    repro = text(v.get('repro'), 3000)
    why, before, after = text(v.get('why'), 600), text(v.get('before'), 1000), text(v.get('after'), 1000)
    try:
        line = int(v.get('line'))
    except (TypeError, ValueError):
        continue
    # All five fields (as the gate requires), so no review says "(not given)".
    if not (path and fn and thm and why and repro and before and after):
        continue
    if (path, fn, line) not in changed or thm not in added:
        continue
    marker = f'<!-- proof-cex {path}:{fn}:{hashlib.sha1(repro.encode()).hexdigest()[:12]} -->'
    if marker in posted:
        continue
    posted += '\n' + marker  # a duplicate verdict in this run is not posted twice
    body = '\n'.join([
        marker,
        f'**The Lean proofs found a probable bug in `{fn}`.**',
        '',
        why,
        '',
        'Reproduce:',
        fence(repro, 'cypher'),
        'Before this PR:', fence(before),
        'With this PR:', fence(after),
        '',
        f'<sub>Lean proves the counterexample on the model (`{thm}`, in the [run]({run_url})\'s `proof-gate` patch). '
        f'Confirm it on a server before acting on it. At {head[:9]}.</sub>',
    ])
    anchor = next((n for n in sorted(by_path.get(path, ())) if n >= line), None)
    if anchor is None:
        loose.append(f'`{path}` `{fn}`\n\n{body}')
    else:
        comments.append({'path': path, 'line': anchor, 'side': 'RIGHT', 'body': body})
if not comments and not loose:
    print('{}'); sys.exit(0)
summary = f'The Lean proofs found {len(comments) + len(loose)} probable bug(s) in this PR; details below.'
print(json.dumps({'commit_id': head, 'event': 'COMMENT',
                  'body': '\n\n---\n\n'.join([summary] + loose), 'comments': comments}))
