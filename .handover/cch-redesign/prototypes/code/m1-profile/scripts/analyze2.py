#!/usr/bin/env python3
"""Cut a samply profile of the m1 bench by PHASE windows and attribute samples to the bench's
timed calls (Timer::time call sites, via inline frames from the dSYM).

usage: analyze2.py PROFILE.json.gz LOG DWARF SRC_DIR [--top N] [--out FILE] [--json FILE]

- Sample time = meta.startTime + cumsum(timeDeltas) (ms, epoch); PHASE t0/t1 are epoch ns.
- 'wall' = samples on the compute thread (1 ms each at 1 kHz); 'cpu' = threadCPUDelta (us), all threads.
- Worker-thread samples are attributed to the compute thread's timed call at the same tick.
"""
import bisect, collections, gzip, json, os, re, subprocess, sys

a = sys.argv[1:]
prof_path, log_path, dwarf, src_dir = a[:4]
top, out, jout = 25, None, None
k = 4
while k < len(a):
    if a[k] == '--top': top = int(a[k + 1]); k += 2
    elif a[k] == '--out': out = a[k + 1]; k += 2
    elif a[k] == '--json': jout = a[k + 1]; k += 2
    else: raise SystemExit('bad arg ' + a[k])

P = json.load(gzip.open(prof_path))
base = prof_path[:-3] if prof_path.endswith('.gz') else prof_path
S = json.load(open(base[:-5] + '.json.syms.json'))
st = S['string_table']
symtab = {}
for d in S['data']:
    rv = sorted((e['rva'], e['size'], st[e['symbol']]) for e in d['symbol_table'])
    symtab[d['debug_name']] = ([r[0] for r in rv], rv)
libs = P['libs']
start = P['meta']['startTime']
main_lib = [l['debugName'] for l in libs if l['path'].endswith('/graph-test')][0]


def lookup(lib, addr):
    t = symtab.get(lib)
    if not t:
        return '0x%x' % addr
    keys, rv = t
    j = bisect.bisect_right(keys, addr) - 1
    if j >= 0:
        r, sz, name = rv[j]
        if addr < r + sz or sz == 0:
            return name.lstrip('_') if name.startswith('_') and not name.startswith('__Z') else name
    return '0x%x' % addr


# call-site line -> timed call name, from the bench sources
callsite = {}
for fn in os.listdir(src_dir):
    if fn.startswith('m1_') and fn.endswith('.rs'):
        for i, line in enumerate(open(os.path.join(src_dir, fn)), 1):
            m = re.search(r'\.time\("([^"]+)"', line)
            if m:
                callsite[(fn, i)] = m.group(1)

phases = []
for line in open(log_path, errors='replace'):
    if line.startswith('PHASE '):
        kv = dict(x.split('=', 1) for x in line.split()[1:])
        phases.append((int(kv['t0']) / 1e6, int(kv['t1']) / 1e6, kv['name'], int(kv['threads']), int(kv['run']), int(kv['depth']), kv))
phases.sort()
pstarts = [p[0] for p in phases]


def phase_of(tms):
    j = bisect.bisect_right(pstarts, tms) - 1
    if j >= 0 and tms <= phases[j][1]:
        return phases[j]
    return None


# decode threads, gather main-binary addresses (leaf vs non-leaf)
threads = P['threads']
dec = []
need = set()
for t in threads:
    ft, fu, rt = t['frameTable'], t['funcTable'], t['resourceTable']
    fl = []
    for f in range(ft['length']):
        fn = ft['func'][f]
        res = fu['resource'][fn]
        lib = libs[rt['lib'][res]]['debugName'] if res is not None and res >= 0 else '?'
        fl.append((lib, ft['address'][f]))
    dec.append(fl)
    stk = t['stackTable']
    s = t['samples']
    seen = set()
    for si in s['stack']:
        x = si
        first = True
        while x is not None:
            key = (x, first)
            if key in seen:
                break
            seen.add(key)
            lib, addr = fl[stk['frame'][x]]
            if lib == main_lib and addr is not None:
                need.add(addr if first else addr - 1)
            first = False
            x = stk['prefix'][x]

inl = {}
todo = sorted(need)
for c in range(0, len(todo), 20000):
    chunk = todo[c:c + 20000]
    r = subprocess.run(['/opt/homebrew/opt/llvm/bin/llvm-symbolizer', '--obj=' + dwarf, '--inlining', '--functions=short', '--demangle'],
                       input='\n'.join('0x%x' % (x + 0x100000000) for x in chunk), capture_output=True, text=True)
    blocks = r.stdout.rstrip('\n').split('\n\n')
    assert len(blocks) == len(chunk), (len(blocks), len(chunk))
    for addr, b in zip(chunk, blocks):
        ls = b.split('\n')
        fr = []
        for j in range(0, len(ls) - 1, 2):
            name, loc = ls[j], ls[j + 1]
            fpath, _, rest = loc.rpartition(':')
            fpath, _, ln = fpath.rpartition(':')
            fr.append((name, os.path.basename(fpath), int(ln) if ln.isdigit() else 0))
        inl[addr] = fr


def expand(lib, addr, leaf):
    if lib == main_lib:
        a2 = addr if leaf else addr - 1
        fr = inl.get(a2)
        if fr and fr[0][0] != '??':
            return [(lib, n, f, l) for n, f, l in fr]
        return [(lib, lookup(lib, a2), '', 0)]
    return [(lib, lookup(lib, addr if leaf else addr - 1), '', 0)]


MEM = re.compile(r'^(_?platform_mem(move|set|cpy)|_?platform_bzero|bzero|__bzero|memcpy|memmove|memset|_platform_memset_pattern16|platform_memset_pattern16|GB_memcpy|GB_memset|copy_nonoverlapping|copy_from_slice)')
ALLOC = re.compile(r'^(malloc|free|calloc|realloc|_malloc|_free|_realloc|_calloc|malloc_|free_|xzm|_xzm|nanov2|_nanov2|szone|_szone|tiny_|small_|medium_|mvm_|_mvm|malloc_zone|_malloc_zone|GB_(malloc|calloc|realloc|free|xalloc|dealloc)_memory|madvise|mmap|munmap|__mmap|__munmap|mach_vm|_kernelrpc|vm_|__rust_alloc|__rust_dealloc|__rust_realloc|__rdl_|alloc_impl|grow_amortized|finish_grow|do_reserve_and_handle|reserve_for_push|RawVec|dealloc|drop_in_place)')
SORT = re.compile(r'(msort|qsort|quicksort|_sort|sort_|unjumble|heapsort|insertion_sort|merge_sort|small_sort|sort4|sort8|bidirectional_merge|ipnsort|GB_sort)')


def classify(frames):
    lib, name = frames[0][0], frames[0][1]
    libs_on = {f[0] for f in frames}
    if lib == 'libomp.dylib':
        return 'omp'
    if lib == 'libsystem_kernel.dylib' and 'libomp.dylib' in libs_on and not any(ALLOC.match(f[1]) for f in frames[:4]):
        return 'omp'
    if MEM.match(name):
        return 'mem'
    if lib == 'libsystem_malloc.dylib' or ALLOC.match(name):
        return 'alloc'
    if lib == 'libsystem_kernel.dylib' and len(frames) > 1 and (frames[1][0] == 'libsystem_malloc.dylib' or ALLOC.match(frames[1][1])):
        return 'alloc'
    # inline-aware: innermost few frames of a sort routine count as sort
    if any(SORT.search(f[1]) for f in frames[:3]):
        return 'sort'
    if lib == main_lib:
        # a GraphBLAS frame anywhere in the inline chain of the leaf address
        if name.startswith(('GB', 'GrB', 'GxB')) or frames[0][2].startswith('GB'):
            return 'gb'
        return 'host'
    return 'other'


def ctx_of(frames):
    names = [f[1] for f in frames]
    for key, pat in (('wait', 'GB_wait'), ('unjumble', 'GB_unjumble'), ('transpose', 'GB_transpose'), ('convert', 'GB_convert'),
                     ('builder', 'GB_builder'), ('hyper_hash', 'GB_hyper_hash_build')):
        if any(n.startswith(pat) for n in names):
            return key
    return '-'


def part_of(cat, frames):
    # exclusive partition: omp > GrB_wait work > format conversion > builder > mem > alloc > sort > GB kernel > host
    names = [f[1] for f in frames]
    if cat == 'omp':
        return 'omp'
    if any(n.startswith('GB_wait') for n in names):
        return 'gb_wait'
    if any(n.startswith('GB_convert') for n in names):
        return 'convert'
    if any(n.startswith('GB_builder') for n in names):
        return 'builder'
    if cat in ('mem', 'alloc', 'sort'):
        return cat
    if cat == 'gb':
        return 'kernel'
    return 'host' if cat == 'host' else 'other'


def call_of(frames):
    # innermost Timer::time<...> frame; its caller's file:line is the call site
    for j, f in enumerate(frames):
        if f[1].startswith('time<') and j + 1 < len(frames):
            c = frames[j + 1]
            return callsite.get((c[2], c[3]), f'?{c[2]}:{c[3]}')
    return None


def api_of(frames):
    for f in reversed(frames):
        if f[1].startswith(('GrB_', 'GxB_')):
            return f[1]
    return None


comp = None
for i, t in enumerate(threads):
    if 'm1_reuse' in (t.get('name') or ''):
        comp = i
assert comp is not None


def gkeys(ph):
    name, depth = ph[2], ph[5]
    if name in ('r', 'b2', 'b1', 'split'):
        return (f'{name}_d>=1', f'{name}_d{depth}')
    return (name,)


D = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
tick_call = []  # (time, call) on the compute thread
cache_by_thread = []
for ti, t in enumerate(threads):
    fl = dec[ti]
    stk = t['stackTable']
    s = t['samples']
    cpu_d = s.get('threadCPUDelta') or [0] * s['length']
    cache = {}
    tm = start
    for j in range(s['length']):
        tm += s['timeDeltas'][j]
        si = s['stack'][j]
        ph = phase_of(tm)
        if ph is None or si is None:
            continue
        if si not in cache:
            frames = []
            x = si
            first = True
            while x is not None:
                lib, addr = fl[stk['frame'][x]]
                frames.extend(expand(lib, addr, first))
                first = False
                x = stk['prefix'][x]
            names = set(f[1] for f in frames)
            cat0 = classify(frames)
            part0 = part_of(cat0, frames)
            chain = ' < '.join(f[1] for f in frames[:4]) if cat0 in ('omp', 'alloc', 'mem', 'sort') else ''
            cache[si] = (cat0, ctx_of(frames), call_of(frames), api_of(frames), frames[0][1] + '  [' + frames[0][0] + ']', names, chain, part0)
        cat, ctx, call, api, leaf, names, chain, part = cache[si]
        cpu = cpu_d[j] or 0
        main = ti == comp
        if main:
            tick_call.append((tm, call or '(untimed)', api))
        for g in gkeys(ph):
            A = D[g]
            A['cpu_cat'][cat] += cpu
            A['cpu_part'][part] += cpu
            A['cpu']['all'] += cpu
            A['self'][leaf] += cpu
            for n in names:
                A['incl'][n] += cpu
            A['ctx_cpu'][ctx] += cpu
            if chain:
                A['chain_' + cat][('main' if main else 'worker', chain)] += cpu if not main else 1
            if main:
                A['wall']['all'] += 1
                A['wall_cat'][cat] += 1
                A['wall_part'][part] += 1
                A['wall_call_part'][(call or '(untimed)', part)] += 1
                A['wall_call'][call or '(untimed)'] += 1
                A['wall_ctx'][ctx] += 1
                A['wall_call_cat'][(call or '(untimed)', cat)] += 1
                A['wall_call_ctx'][(call or '(untimed)', ctx)] += 1
                A['wall_call_leaf'][(call or '(untimed)', leaf)] += 1
                A['cpu_call'][call or '(untimed)'] += cpu
                A['cpu_call_cat'][(call or '(untimed)', cat)] += cpu
                A['cpu_call_part'][(call or '(untimed)', part)] += cpu
                A['cpu_call_leaf'][(call or '(untimed)', leaf)] += cpu
            else:
                A['wcpu']['all'] += cpu
                A['wcpu_cat'][cat] += cpu
                A['wsamp']['all'] += 1
                if cpu == 0:
                    A['wsamp']['idle'] += 1
    cache_by_thread.append(cache)

# second pass: worker samples -> compute thread's call at the nearest tick
tick_call.sort()
tt = [x[0] for x in tick_call]
for ti, t in enumerate(threads):
    if ti == comp:
        continue
    fl = dec[ti]
    s = t['samples']
    cpu_d = s.get('threadCPUDelta') or [0] * s['length']
    cache = cache_by_thread[ti]
    tm = start
    for j in range(s['length']):
        tm += s['timeDeltas'][j]
        si = s['stack'][j]
        ph = phase_of(tm)
        if ph is None or si is None or si not in cache:
            continue
        cpu = cpu_d[j] or 0
        q = bisect.bisect_left(tt, tm)
        cand = [x for x in (q - 1, q) if 0 <= x < len(tt)]
        if not cand:
            continue
        q = min(cand, key=lambda x: abs(tt[x] - tm))
        call = tick_call[q][1]
        cat = cache[si][0]
        part = cache[si][7]
        leaf = cache[si][4]
        for g in gkeys(ph):
            D[g]['cpu_call_leaf'][(call, leaf)] += cpu
            D[g]['cpu_call_part'][(call, part)] += cpu
            D[g]['cpu_call'][call] += cpu
            D[g]['cpu_call_cat'][(call, cat)] += cpu
            D[g]['wcpu_call_cat'][(call, cat)] += cpu

ru = collections.defaultdict(lambda: collections.defaultdict(float))
for p in phases:
    for g in gkeys(p):
        for f in ('wall', 'instr', 'cycles', 'user', 'sys', 'runnable'):
            ru[g][f] += float(p[6][f])
        ru[g]['n'] += 1

rep = []
E = rep.append
E(f'profile {os.path.basename(prof_path)}  log {os.path.basename(log_path)}  compute thread #{comp} {threads[comp].get("name")}  threads={len(threads)}')
CATS = ['gb', 'host', 'sort', 'mem', 'alloc', 'omp', 'other']


def pct(x, y):
    return 100.0 * x / y if y else 0.0


def catline(d, tot):
    return '  '.join(f'{c}={pct(d.get(c, 0), tot):5.1f}%' for c in CATS)


summary = {}
order = ['warm0', 'r0', 'keep0', 'labels', 'feral', 'split_d>=1', 'r_d>=1', 'b2_d>=1', 'b1_d>=1']
for g in order + sorted(x for x in D if x not in order):
    if g not in D:
        continue
    A = D[g]
    tot, wt = A['cpu']['all'], A['wall']['all']
    r = ru[g]
    summary[g] = {'cpu_cat': {c: pct(A['cpu_cat'].get(c, 0), tot) for c in CATS}, 'wall_cat': {c: pct(A['wall_cat'].get(c, 0), wt) for c in CATS},
                  'wall_ms': wt, 'cpu_s': tot / 1e6, 'worker_cpu_s': A['wcpu']['all'] / 1e6, 'ctx': {c: pct(v, tot) for c, v in A['ctx_cpu'].items()},
                  'worker_cat': {c: pct(A['wcpu_cat'].get(c, 0), A['wcpu']['all']) for c in CATS},
                  'wall_part': dict(A['wall_part']), 'cpu_part': dict(A['cpu_part']),
                  'wall_call_part': {f'{c}|{p}': v for (c, p), v in A['wall_call_part'].items()},
                  'cpu_call_part': {f'{c}|{p}': v for (c, p), v in A['cpu_call_part'].items()},
                  'wall_call': dict(A['wall_call']), 'cpu_call': dict(A['cpu_call'])}
    detail = g in ('r0', 'r_d>=1', 'b2_d>=1', 'split_d>=1', 'warm0')
    if not detail and not re.match(r'^(r|b2)_d\d+$', g):
        E(f'\n=== {g}: wall {r["wall"]:.3f}s instr {r["instr"]/1e9:.2f}G cycles {r["cycles"]/1e9:.2f}G | samples: compute {wt:.0f} ms, CPU {tot/1e6:.3f}s')
        E(f'    CPU  by leaf: {catline(A["cpu_cat"], tot)}')
        continue
    if not detail:
        # per depth: one line
        wl = A['wcpu']['all']
        E(f'  {g:8} wall {r["wall"]:.3f}s compute-ms {wt:5.0f}  main-wall: {catline(A["wall_cat"], wt)} | worker CPU {wl/1e6:.3f}s omp {pct(A["wcpu_cat"].get("omp", 0), wl):5.1f}% | all-CPU omp {pct(A["cpu_cat"].get("omp", 0), tot):5.1f}%')
        continue
    E(f'\n=== {g}: wall {r["wall"]:.3f}s instr {r["instr"]/1e9:.2f}G cycles {r["cycles"]/1e9:.2f}G cpu {r["user"]+r["sys"]:.3f}s (sys {r["sys"]:.3f}) windows {int(r["n"])}')
    E(f'    samples: compute thread {wt:.0f} ms, CPU all threads {tot/1e6:.3f}s, workers {A["wcpu"]["all"]/1e6:.3f}s ({A["wsamp"]["all"]:.0f} samples, {A["wsamp"]["idle"]:.0f} with zero CPU)')
    E(f'    compute-thread wall by leaf: {catline(A["wall_cat"], wt)}')
    E(f'    CPU (all threads) by leaf:   {catline(A["cpu_cat"], tot)}')
    PARTS = ['kernel', 'gb_wait', 'convert', 'builder', 'mem', 'alloc', 'sort', 'host', 'omp', 'other']
    E('    partition, compute-thread wall: ' + '  '.join(f'{x}={pct(A["wall_part"].get(x, 0), wt):.1f}%' for x in PARTS))
    E('    partition, CPU all threads:     ' + '  '.join(f'{x}={pct(A["cpu_part"].get(x, 0), tot):.1f}%' for x in PARTS))
    if A['wcpu']['all']:
        E(f'    worker CPU by leaf:          {catline(A["wcpu_cat"], A["wcpu"]["all"])}')
    E('    compute-thread wall inside: ' + '  '.join(f'{c}={pct(v, wt):.1f}%' for c, v in sorted(A['wall_ctx'].items(), key=lambda x: -x[1]) if c != '-'))
    E(f'    by timed call (compute-thread wall ms; CPU all threads s; leaf split of compute-thread wall):')
    for c, v in sorted(A['wall_call'].items(), key=lambda x: -x[1]):
        if v < 0.003 * wt:
            continue
        cats = '  '.join(f'{x}={pct(A["wall_call_cat"].get((c, x), 0), v):.0f}%' for x in CATS if A['wall_call_cat'].get((c, x), 0))
        ctxs = '  '.join(f'{x}={pct(A["wall_call_ctx"].get((c, x), 0), v):.0f}%' for x in ('wait', 'unjumble', 'builder', 'convert', 'transpose', 'hyper_hash') if A['wall_call_ctx'].get((c, x), 0))
        E(f'      {c:14} {v:6.0f} ms {pct(v, wt):5.1f}%  cpu {A["cpu_call"][c]/1e6:6.3f}s | {cats}' + (f' | in: {ctxs}' if ctxs else ''))
        leaves = sorted(((l, w) for (cc, l), w in A['wall_call_leaf'].items() if cc == c), key=lambda x: -x[1])[:4]
        E('          leaves: ' + '; '.join(f'{l} {pct(w, v):.0f}%' for l, w in leaves))
    for cc in ('omp', 'mem', 'alloc', 'sort'):
        items = sorted(A['chain_' + cc].items(), key=lambda x: -x[1])
        if not items:
            continue
        E(f'    {cc} stacks (main: compute-thread samples; worker: CPU us):')
        for who0 in ('main', 'worker'):
            for (who, ch), v in [x for x in items if x[0][0] == who0][:6]:
                E(f'      {who:6} {v:10.0f}  {ch}')
    E(f'    top self (CPU, all threads) -> timed calls it ran under:')
    for kk, v in sorted(A['self'].items(), key=lambda x: -x[1])[:top]:
        calls = sorted(((c, w) for (c, l), w in A['cpu_call_leaf'].items() if l == kk), key=lambda x: -x[1])[:3]
        E(f'      {pct(v, tot):5.1f}%  {kk}  <- ' + ', '.join(f'{c} {pct(w, v):.0f}%' for c, w in calls))
    E(f'    top total (CPU, inclusive, GraphBLAS and bench frames):')
    shown = 0
    for kk, v in sorted(A['incl'].items(), key=lambda x: -x[1]):
        if kk.startswith(('GB', 'GrB', 'GxB', 'book', 'build_b', 'contract_b2', 'coarsen_level', 'hierarchy', 'split_live', 'mat_tuples', 'kmp', '__kmp')) or 'time<' in kk:
            E(f'      {pct(v, tot):5.1f}%  {kk}')
            shown += 1
            if shown >= top:
                break

txt = '\n'.join(rep)
if out:
    open(out, 'w').write(txt + '\n')
if jout:
    json.dump(summary, open(jout, 'w'), indent=1)
print(txt)
