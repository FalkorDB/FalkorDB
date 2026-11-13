#!/usr/bin/env python3
"""Cut a samply profile of the m1 bench by the PHASE windows in its log and attribute samples.

usage: analyze_prof.py PROFILE.json.gz LOG [--inline BIN] [--top N] [--out FILE]

Sample time = meta.startTime + cumsum(timeDeltas) (ms, epoch). PHASE lines carry t0/t1 in epoch ns.
Weights: 'cpu' = threadCPUDelta (us), 'wall' = sample count on the compute thread.
"""
import bisect
import collections
import gzip
import json
import re
import subprocess
import sys

args = sys.argv[1:]
prof_path, log_path = args[0], args[1]
inline_bin = None
top = 25
out = None
i = 2
while i < len(args):
    if args[i] == '--inline':
        inline_bin = args[i + 1]; i += 2
    elif args[i] == '--top':
        top = int(args[i + 1]); i += 2
    elif args[i] == '--out':
        out = args[i + 1]; i += 2
    else:
        raise SystemExit('bad arg ' + args[i])

P = json.load(gzip.open(prof_path))
syms_path = prof_path[:-3] if prof_path.endswith('.gz') else prof_path
syms_path = syms_path[:-5] + '.json.syms.json' if syms_path.endswith('.json') else syms_path + '.syms.json'
S = json.load(open(syms_path))
st = S['string_table']
symtab = {}
for d in S['data']:
    rv = sorted((e['rva'], e['size'], st[e['symbol']]) for e in d['symbol_table'])
    symtab[d['debug_name']] = ([r[0] for r in rv], rv)

libs = P['libs']
start = P['meta']['startTime']


def lookup(lib, addr):
    t = symtab.get(lib)
    if not t:
        return '0x%x' % addr
    keys, rv = t
    k = bisect.bisect_right(keys, addr) - 1
    if k >= 0:
        r, sz, name = rv[k]
        if addr < r + sz or sz == 0:
            return name
    return '0x%x' % addr


# phases from the log
phases = []
for line in open(log_path, errors='replace'):
    if line.startswith('PHASE '):
        kv = dict(x.split('=', 1) for x in line.split()[1:])
        phases.append((int(kv['t0']) / 1e6, int(kv['t1']) / 1e6, kv['name'], int(kv['threads']), int(kv['run']), int(kv['depth']), kv))
phases.sort()
pstarts = [p[0] for p in phases]


def phase_of(tms):
    k = bisect.bisect_right(pstarts, tms) - 1
    if k >= 0 and tms <= phases[k][1]:
        return phases[k]
    return None


# optional inline expansion of main-binary frames via llvm-symbolizer
inline_cache = {}


def inline_names(addrs):
    if not inline_bin or not addrs:
        return
    todo = sorted(a for a in addrs if a not in inline_cache)
    if not todo:
        return
    inp = '\n'.join('0x%x' % (a + 0x100000000) for a in todo)
    r = subprocess.run(['/opt/homebrew/opt/llvm/bin/llvm-symbolizer', '--obj=' + inline_bin, '--inlining', '--functions=short', '--demangle', '--no-untag-mte'],
                       input=inp, capture_output=True, text=True)
    blocks = r.stdout.strip('\n').split('\n\n')
    for a, b in zip(todo, blocks):
        ls = b.split('\n')
        names = [ls[k] for k in range(0, len(ls), 2)]
        inline_cache[a] = names  # innermost first


def demangle_short(n):
    # rust legacy mangling _ZN...E -> path
    if n.startswith('__ZN') or n.startswith('_ZN'):
        s = n.lstrip('_')[2:]
        parts = []
        while s and s[0].isdigit():
            m = re.match(r'(\d+)', s)
            ln = int(m.group(1)); s = s[len(m.group(1)):]
            parts.append(s[:ln]); s = s[ln:]
        parts = [p for p in parts if not re.fullmatch(r'h[0-9a-f]{16}', p)]
        return '::'.join(parts).replace('$LT$', '<').replace('$GT$', '>').replace('$u20$', ' ').replace('..', '::')
    return n.lstrip('_') if n.startswith('_') else n


MEM = re.compile(r'^(_?platform_mem(move|set|cpy)|_?platform_bzero|memcpy|memmove|memset|bzero|__bzero|GB_memcpy|GB_memset|_platform_memset_pattern16)')
ALLOC = re.compile(r'^(malloc|free|calloc|realloc|_malloc|_free|_realloc|_calloc|malloc_|free_|_xzm|xzm_|nanov2|szone|_szone|_nanov2|tiny_|small_|medium_|large_|mvm_|_malloc_zone|malloc_zone|_mvm|GB_(malloc|calloc|realloc|free|xalloc|dealloc)_memory|madvise|mmap|munmap|__mmap|__munmap|mach_vm|_kernelrpc_mach_vm|vm_)')
SORT = re.compile(r'(msort|qsort|_sort|sort_|unjumble|GB_sort|heapsort|ikvsort|insertion_sort|quicksort|merge)')


def classify(frames):
    """frames: list of (lib, name) innermost first. Returns leaf category."""
    lib, name = frames[0]
    has_omp = any(l == 'libomp.dylib' for l, _ in frames)
    if lib == 'libomp.dylib':
        return 'omp'
    if lib == 'libsystem_kernel.dylib' and has_omp and not any(ALLOC.match(n) for _, n in frames[:4]):
        return 'omp'
    if MEM.match(name):
        return 'mem'
    if ALLOC.match(name) or lib == 'libsystem_malloc.dylib':
        return 'alloc'
    if lib in ('libsystem_kernel.dylib',) and len(frames) > 1 and (ALLOC.match(frames[1][1]) or frames[1][0] == 'libsystem_malloc.dylib'):
        return 'alloc'
    if SORT.search(name):
        return 'sort'
    if lib.startswith('graph-') or lib == 'graph-test':
        if name.startswith('GB') or name.startswith('GrB') or name.startswith('GxB') or name.startswith('LAGraph'):
            return 'gb'
        return 'host'
    return 'other:' + lib


def context(frames):
    """GraphBLAS context of a sample: inside a wait / transpose / conversion / unjumble / builder."""
    names = [n for _, n in frames]
    for key, pat in (('ctx_wait', 'GB_wait'), ('ctx_unjumble', 'GB_unjumble'), ('ctx_transpose', 'GB_transpose'),
                     ('ctx_convert', 'GB_convert'), ('ctx_build', 'GB_builder'), ('ctx_dup', 'GB_dup')):
        if any(n.startswith(pat) for n in names):
            return key
    return ''


API = re.compile(r'^(GrB_|GxB_)')


def api_of(frames):
    # outermost GrB_/GxB_ entry (closest to the root)
    for _, n in reversed(frames):
        if API.match(n):
            return n
    return None


def method_of(frames):
    # outermost GB_ internal method below the API frame (e.g. GB_AxB_dot2, GB_AxB_saxpy, GB_ewise...)
    seen_api = False
    for _, n in reversed(frames):
        if API.match(n):
            seen_api = True
            continue
        if seen_api and n.startswith('GB_'):
            return n
    return None


def kernel_of(frames):
    # innermost GB_ frame that looks like a worker kernel (outlined or factory/jit/generic)
    for _, n in frames:
        if n.startswith('GB') and ('omp_outlined' in n or n.startswith('GB__') or n.startswith('GB_jit') or 'generic' in n):
            return n
    for _, n in frames:
        if n.startswith('GB'):
            return n
    return None


threads = P['threads']
report = []


def emit(s=''):
    report.append(s)


# per-thread decoding
decoded = []
all_main_addrs = set()
for t in threads:
    ft, fu, rt, sa = t['frameTable'], t['funcTable'], t['resourceTable'], t['stringArray']
    stk = t['stackTable']
    fl = []
    for f in range(ft['length']):
        fn = ft['func'][f]
        res = fu['resource'][fn]
        lib = libs[rt['lib'][res]]['debugName'] if res is not None and res >= 0 else '?'
        addr = ft['address'][f]
        fl.append((lib, addr))
        if lib.startswith('graph-') and addr is not None and addr >= 0:
            all_main_addrs.add(addr)
    decoded.append((t, fl))

main_lib = None
for l in libs:
    if l['debugName'].startswith('graph-'):
        main_lib = l['debugName']
if inline_bin:
    inline_names(all_main_addrs)

for ti, (t, fl) in enumerate(decoded):
    pass

frame_name_cache = {}


def frame_names(lib, addr, leaf):
    key = (lib, addr, leaf)
    if key in frame_name_cache:
        return frame_name_cache[key]
    a = addr if leaf else addr - 1
    if inline_bin and lib == main_lib and addr in inline_cache:
        names = [demangle_short(n) for n in inline_cache[addr]]
        r = [(lib, n) for n in names]
    else:
        r = [(lib, demangle_short(lookup(lib, a)))]
    frame_name_cache[key] = r
    return r


# accumulate
Acc = lambda: collections.defaultdict(float)
by = collections.defaultdict(lambda: collections.defaultdict(Acc))  # by[group][table][key] += w


def group_keys(ph):
    name, th, run, depth = ph[2], ph[3], ph[4], ph[5]
    if name in ('r', 'b2', 'b1', 'split'):
        yield f'{name}_d>=1'
        yield f'{name}_d{depth}'
    else:
        yield f'{name}'


# compute thread: the test thread (name contains m1_reuse) else the busiest
comp_tid = None
for ti, (t, fl) in enumerate(decoded):
    if 'm1_reuse' in (t.get('name') or ''):
        comp_tid = ti
if comp_tid is None:
    comp_tid = max(range(len(decoded)), key=lambda k: decoded[k][0]['samples']['length'])

# main-thread API timeline for worker attribution
main_api_t = []
main_api_v = []

for ti, (t, fl) in enumerate(decoded):
    s = t['samples']
    stk = t['stackTable']
    tm = start
    cache = {}
    for k in range(s['length']):
        tm += s['timeDeltas'][k]
        cpu = (s.get('threadCPUDelta') or [0] * s['length'])[k] or 0
        si = s['stack'][k]
        ph = phase_of(tm)
        if ph is None or si is None:
            continue
        if si not in cache:
            frames = []
            x = si
            first = True
            while x is not None:
                lib, addr = fl[stk['frame'][x]]
                frames.extend(frame_names(lib, addr, first))
                first = False
                x = stk['prefix'][x]
            cache[si] = (classify(frames), context(frames), api_of(frames), method_of(frames), kernel_of(frames), frames[0], tuple(sorted(set(n for _, n in frames))))
        cat, ctx, api, meth, kern, leaf, allnames = cache[si]
        is_main = ti == comp_tid
        if is_main:
            main_api_t.append(tm)
            main_api_v.append((api, meth))
        for g in group_keys(ph):
            A = by[g]
            A['cpu_cat'][cat] += cpu
            A['cpu_total']['all'] += cpu
            if ctx:
                A['cpu_ctx'][ctx] += cpu
            A['self_cpu'][f'{leaf[1]}  [{leaf[0]}]'] += cpu
            for n in allnames:
                A['total_cpu'][n] += cpu
            if is_main:
                A['wall_cat'][cat] += 1
                A['wall_total']['all'] += 1
                if ctx:
                    A['wall_ctx'][ctx] += 1
                A['main_api'][f'{api} / {meth}'] += 1
            else:
                A['worker_cpu_cat'][cat] += cpu
                A['worker_cpu_total']['all'] += cpu
            if kern:
                A['kernel_cpu'][kern] += cpu

# worker attribution by main API at nearest tick (second pass)
order = sorted(range(len(main_api_t)), key=lambda k: main_api_t[k])
mt = [main_api_t[k] for k in order]
mv = [main_api_v[k] for k in order]
for ti, (t, fl) in enumerate(decoded):
    if ti == comp_tid:
        continue
    s = t['samples']
    tm = start
    for k in range(s['length']):
        tm += s['timeDeltas'][k]
        cpu = (s.get('threadCPUDelta') or [0] * s['length'])[k] or 0
        if not cpu:
            continue
        ph = phase_of(tm)
        if ph is None:
            continue
        j = bisect.bisect_left(mt, tm)
        cand = [x for x in (j - 1, j) if 0 <= x < len(mt)]
        if not cand:
            continue
        j = min(cand, key=lambda x: abs(mt[x] - tm))
        api, meth = mv[j]
        for g in group_keys(ph):
            by[g]['worker_by_main_api'][f'{api} / {meth}'] += cpu

# phase rusage sums from the log
ru = collections.defaultdict(lambda: collections.defaultdict(float))
for p in phases:
    kv = p[6]
    for g in group_keys(p):
        for f in ('wall', 'instr', 'cycles', 'user', 'sys', 'runnable'):
            ru[g][f] += float(kv[f])
        ru[g]['n'] += 1

emit(f'profile {prof_path}  log {log_path}  compute thread = {decoded[comp_tid][0].get("name")}  threads = {len(decoded)}')
emit('threads: ' + ', '.join(f'{k}:{t.get("name")}({t["samples"]["length"]})' for k, (t, _) in enumerate(decoded)))


def pct(a, b):
    return 100.0 * a / b if b else 0.0


groups = sorted(by.keys(), key=lambda g: (not g.startswith(('warm0', 'r0', 'keep0', 'labels', 'feral')), g))
for g in groups:
    A = by[g]
    tot = A['cpu_total']['all']
    wt = A['wall_total']['all']
    r = ru.get(g, {})
    emit('')
    emit(f'=== {g}: rusage wall={r.get("wall", 0):.3f}s instr={r.get("instr", 0) / 1e9:.3f}G cycles={r.get("cycles", 0) / 1e9:.3f}G user={r.get("user", 0):.3f}s sys={r.get("sys", 0):.3f}s runnable={r.get("runnable", 0):.3f}s windows={int(r.get("n", 0))}')
    emit(f'    samples: compute-thread {wt:.0f} (wall ms), all-thread CPU {tot / 1e6:.3f}s, workers CPU {A["worker_cpu_total"]["all"] / 1e6:.3f}s')
    emit('    CPU by leaf category: ' + ', '.join(f'{c}={pct(v, tot):.1f}%' for c, v in sorted(A['cpu_cat'].items(), key=lambda x: -x[1])))
    emit('    compute-thread wall by leaf category: ' + ', '.join(f'{c}={pct(v, wt):.1f}%' for c, v in sorted(A['wall_cat'].items(), key=lambda x: -x[1])))
    if A['worker_cpu_total']['all']:
        wtot = A['worker_cpu_total']['all']
        emit('    worker CPU by leaf category: ' + ', '.join(f'{c}={pct(v, wtot):.1f}%' for c, v in sorted(A['worker_cpu_cat'].items(), key=lambda x: -x[1])))
    if A['cpu_ctx']:
        emit('    CPU inside GB context: ' + ', '.join(f'{c}={pct(v, tot):.1f}%' for c, v in sorted(A['cpu_ctx'].items(), key=lambda x: -x[1])))
    if g.endswith('_d>=1') or g in ('r0', 'warm0', 'keep0'):
        emit(f'    top self (CPU):')
        for k, v in sorted(A['self_cpu'].items(), key=lambda x: -x[1])[:top]:
            emit(f'      {pct(v, tot):5.1f}%  {k}')
        emit(f'    top total (CPU, inclusive):')
        for k, v in sorted(A['total_cpu'].items(), key=lambda x: -x[1])[:top]:
            emit(f'      {pct(v, tot):5.1f}%  {k}')
        emit(f'    compute thread by API / method (wall):')
        for k, v in sorted(A['main_api'].items(), key=lambda x: -x[1])[:top]:
            emit(f'      {pct(v, wt):5.1f}%  {k}')
        if A['worker_by_main_api']:
            wtot = A['worker_cpu_total']['all']
            emit(f'    worker CPU by the compute thread\'s API at that tick:')
            for k, v in sorted(A['worker_by_main_api'].items(), key=lambda x: -x[1])[:15]:
                emit(f'      {pct(v, wtot):5.1f}%  {k}')
        emit(f'    kernels (CPU, innermost GB kernel frame):')
        for k, v in sorted(A['kernel_cpu'].items(), key=lambda x: -x[1])[:top]:
            emit(f'      {pct(v, tot):5.1f}%  {k}')

txt = '\n'.join(report)
if out:
    open(out, 'w').write(txt + '\n')
print(txt)
