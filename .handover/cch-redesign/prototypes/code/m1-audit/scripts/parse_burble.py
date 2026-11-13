"""Attribute GraphBLAS burble entries to (threads, depth, path, call). Usage: parse_burble.py LOG"""
import re, sys, collections as co
log = sys.argv[1]
th = None; depth = None; path = None; call = None; last_call = None
in_audit = False
cur = None
rows = []   # (th, depth, path, call, text, sec)
timed = co.defaultdict(float)
np_rows = []
for line in open(log, errors='replace'):
    if line.startswith('LOAD threads='):
        th = int(re.search(r'threads=(\d+)', line).group(1))
        continue
    if line.startswith('@@DEPTH'):
        depth = int(line.split()[1]); path = 'R' if depth == 0 else 'split'; call = None; last_call = None
        continue
    if line.startswith('@@AUDIT begin'):
        in_audit = True; continue
    if line.startswith('@@AUDIT end'):
        in_audit = False; continue
    if line.startswith('@@NP'):
        np_rows.append((th, depth, path, line.split()[1:]))
        continue
    if line.startswith('@@END'):
        w = line.split(); timed[(th, depth, path, w[1])] += float(w[2]); call = 'untimed'
        continue
    if line.startswith('@@CALL'):
        c = line.split()[1]
        if depth is not None and depth > 0:
            if path == 'split' and c not in ('split_buildM', 'split_mxm'):
                path = 'R'
            if c == 'book' and path == 'R':
                path = 'B1'
            elif c == 'book' and last_call == 'bnd' and path == 'B1':
                path = 'B2'
        if depth == 0 and c == 'keep':
            path = 'keep'
        call = c; last_call = c
        continue
    if line.startswith(' [ '):
        cur = line.strip()[2:]
        if 'sec ]' in line:
            m = re.search(r'([0-9.eE+-]+) sec \]', line)
            rows.append((th, depth, path, call if not in_audit else 'AUDIT:' + str(call), cur, float(m.group(1)))); cur = None
        continue
    if cur is not None:
        if 'sec ]' in line:
            m = re.search(r'([0-9.eE+-]+) sec \]', line)
            rows.append((th, depth, path, call if not in_audit else 'AUDIT:' + str(call), cur, float(m.group(1)))); cur = None
        else:
            cur += ' ' + line.strip()

def kind(t):
    k = []
    if 'unjumble' in t: k.append('unjumble')
    if 'generic' in t or 'jit: undefined' in t: k.append('generic')
    if re.search(r'\((sparse|bitmap|full|hyper) to (sparse|bitmap|full|hyper)\)', t): k.append('convert')
    if 'transpose' in t: k.append('transpose')
    if '1 threads' in t or '1-thread' in t or 'single-threaded' in t: k.append('1thr')
    return k

want_th = int(sys.argv[2]) if len(sys.argv) > 2 else 1
agg = co.defaultdict(float); kagg = co.defaultdict(float); cnt = co.defaultdict(int)
desc = co.defaultdict(lambda: co.defaultdict(float))
for (t, d, p, c, text, s) in rows:
    if t != want_th: continue
    key = (d, p, c)
    agg[key] += s; cnt[key] += 1
    for k in kind(text):
        kagg[key + (k,)] += s
    # normalise descriptor
    dn = re.sub(r'[0-9.]+e[+-][0-9]+|\b\d+\b', '#', text)
    dn = re.sub(r'\s+', ' ', dn)[:150]
    desc[key][dn] += s
print(f'# threads={want_th}: burble seconds by (depth, path, call); flags = share of that time')
for key in sorted(agg, key=lambda k: (k[0] if k[0] is not None else -1, str(k[1]), -agg[k])):
    flags = ' '.join(f'{k}={kagg[key + (k,)]*1e3:.1f}' for k in ('unjumble', 'generic', 'convert', 'transpose', '1thr') if kagg.get(key + (k,)))
    tm = timed.get((want_th,) + key, 0.0)
    print(f'd={key[0]} {key[1]:5s} {str(key[2]):18s} n={cnt[key]:5d} burble_ms={agg[key]*1e3:9.2f} timer_ms={tm*1e3:9.2f}  {flags}')
if len(sys.argv) > 3:
    print('\n# descriptors')
    for key in sorted(desc, key=lambda k: -agg[k])[:int(sys.argv[3])]:
        print(f'== d={key[0]} {key[1]} {key[2]}  total {agg[key]*1e3:.2f} ms')
        for dn, s in sorted(desc[key].items(), key=lambda x: -x[1])[:6]:
            print(f'   {s*1e3:8.2f}  {dn}')
