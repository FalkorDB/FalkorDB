/* ===================== shared: scoped recompute ===================== */
const akey = (a) => a[0] + '-' + a[1];
function recomputeStepper(key, o) {
  const pops = o.pops;
  const heaps = [];
  let heap = new Map(o.front.map((a) => [akey(a), a]));
  heaps.push(new Map(heap));
  for (const p of pops) {
    heap.delete(akey(p.arc));
    for (const q of p.pushed) heap.set(akey(q), q);
    heaps.push(new Map(heap));
  }
  const n = pops.length + 1;
  const upOf = new Map();
  for (const a of o.arcs) { if (!upOf.has(a.lo)) upOf.set(a.lo, []); upOf.get(a.lo).push(a.hi); }
  const readersOf = (lo, hi) => (upOf.get(lo) || []).filter((w) => w !== hi).map((w) => [Math.min(hi, w), Math.max(hi, w)]);
  const title = o.title || 'Scoped recompute, one pop at a time';
  return stepper(key, {
    title, n,
    label: (i) => (i === 0 ? 'the front' : 'pop ' + i + ' of ' + pops.length),
    legend: [['l', 'hot', 'arc being recomputed'], ['l', 'apex', 'lower detour through an apex x'], ['l', 'push', 'arc it pushes (changed)'], ['l', 'queued', 'waiting in the heap'], ['l', 'done', 'recomputed earlier and changed'], ['n', 'ring', 'ends of the arc'], ['n', 't0', 'apex x']],
    render(i, pics, panel) {
      const cur = i > 0 ? pops[i - 1] : null;
      const before = heaps[i > 0 ? i - 1 : 0];
      const waiting = heaps[i];
      const doneChanged = new Set(pops.slice(0, Math.max(0, i - 1)).filter((p) => p.changed).map((p) => akey(p.arc)));
      const pushed = new Set(cur ? cur.pushed.map(akey) : []);
      const apexArcs = new Map();
      if (cur) for (const ax of cur.apexes) {
        const won = ax.up_won || ax.down_won;
        apexArcs.set(key2(ax.x, cur.arc[0]), won); apexArcs.set(key2(ax.x, cur.arc[1]), won);
      }
      const arcs = o.arcs.map((a) => {
        const k = a.lo + '-' + a.hi;
        let cls = 'e faint', z = 0;
        if (cur && k === akey(cur.arc)) { cls = 'e hot'; z = 5; }
        else if (pushed.has(k)) { cls = 'e push'; z = 4; }
        else if (apexArcs.has(k)) { cls = apexArcs.get(k) ? 'e apexwin' : 'e apex'; z = 3; }
        else if (waiting.has(k) || (i === 0 && before.has(k))) { cls = 'e queued'; z = 2; }
        else if (doneChanged.has(k)) { cls = 'e done'; z = 1; }
        return { lo: a.lo, hi: a.hi, cls, z };
      });
      const ends = new Set(cur ? [cur.arc[0], cur.arc[1]] : []);
      const apx = new Set(cur ? cur.apexes.map((x) => x.x) : []);
      const front0 = new Set(i === 0 ? o.front.flatMap((a) => [a[0], a[1]]) : []);
      ap(pics, wrapFig(rankGraph({ nodeOf: o.nodeOf, arcs, rcls: (r) => (ends.has(r) ? 'ring' : apx.has(r) ? 't0' : front0.has(r) ? 'ring2' : ''), extraNodes: o.extraNodes, aria: title + (cur ? ': arc (' + cur.arc[0] + ', ' + cur.arc[1] + ')' : ': the front') })));
      if (i === 0) {
        ap(panel, h('h3', null, 'The front'),
          o.frontWhy ? blk('why these arcs', P(o.frontWhy)) : null,
          blk('heap, keyed (lo, hi, id)', chips(o.front.map((a) => ['(' + a[0] + ', ' + a[1] + ') id ' + a[2], 'fil']))),
          o.start != null ? blk('work so far', bar(o.start, o.budget, o.max), h('span', { class: 'barlab' }, o.start + ' units before the first pop' + (o.budget != null ? ', budget ' + o.budget : ''))) : null);
        return;
      }
      const p = cur;
      const nm = ['up', 'down', 'middle up', 'middle down'];
      const lst = (as) => C(as.map((q) => '(' + q[0] + ', ' + q[1] + ')').join(', '));
      const readers = readersOf(p.arc[0], p.arc[1]);
      const pushedK = new Set(p.pushed.map(akey));
      const waitingR = readers.filter((q) => !pushedK.has(akey(q)));
      const inHeap = waitingR.every((q) => before.has(akey(q)));
      const already = waitingR.length ? [lst(waitingR), waitingR.length === 1 ? ' also reads it and ' : ' also read it and ', waitingR.length === 1 ? (inHeap ? 'is already in the heap' : 'was queued earlier') : (inHeap ? 'are already in the heap' : 'were queued earlier')] : null;
      const result = !p.changed ? 'Unchanged bit for bit, so it pushes nothing.'
        : !readers.length ? 'Changed, but no arc above it reads it, so it pushes nothing.'
          : !p.pushed.length ? ['Changed. The arcs above it that read it, ', lst(waitingR), ', ', inHeap ? 'are already in the heap' : 'were queued earlier', ', so nothing new is pushed.']
            : ['Changed, so it pushes the arcs above it that read it: ', lst(p.pushed), '.', already ? [' ', already, '.'] : ''];
      const ticks = p.apexes.some((a) => a.up_won || a.down_won);
      const fmt = (v, k) => (k < 2 ? fw(v) : fm(v));
      ap(panel, h('h3', null, 'Pop ' + i + ': arc (' + p.arc[0] + ', ' + p.arc[1] + '), id ' + p.arc[2]),
        blk('the arc', P('Upward it runs node ', B(p.nodes[0]), ' → node ', B(p.nodes[1]), '; downward the reverse. Its seed (smallest live weight per direction) is ', C(fw(p.seed[0]) + ' up, ' + fw(p.seed[1]) + ' down'), '.')),
        p.apexes.length ? blk('lower detours: x in down(lo) ∩ down(hi)', mini(['x', 'up via x', 'down via x'], p.apexes.map((a) => ['r' + a.x, { v: fw(a.try_up) + (a.up_won ? ' ✓' : ''), cls: a.up_won ? 'w' : '' }, { v: fw(a.try_down) + (a.down_won ? ' ✓' : ''), cls: a.down_won ? 'w' : '' }])),
          h('span', { class: 'barlab' }, ticks ? '✓ marks each x whose detour is strictly smaller than the value so far. The last ✓ in a column is the middle (the first x to reach the minimum).' : 'No detour is strictly smaller than the seed, so the seed is the value and there is no middle.')) : blk('lower detours', P('None: no rank below both ends joins them, so the seed is the value.')),
        blk('old → new', mini(['', ...nm], [['old', ...p.old.map((v, k) => fmt(v, k))], ['new', ...p.new.map((v, k) => ({ v: fmt(v, k), cls: v !== p.old[k] ? 'w' : '' }))]])),
        blk('result', P(result)),
        blk('heap after the pop', waiting.size ? chips([...waiting.values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]).map((a) => ['(' + a[0] + ', ' + a[1] + ')', 'fil'])) : P('Empty: the recompute is done.')),
        p.running_total != null ? blk('work so far', bar(p.running_total, o.budget, o.max), h('span', { class: 'barlab' }, p.running_total + ' units' + (o.budget != null ? ' of ' + o.budget + (p.running_total > o.budget ? ': over budget, stop' : '') : ''))) : null);
    },
  });
}

/* ===================== shared: S2 fill repair ===================== */
function s2Stepper(key, s2, o) {
  const tr = s2.overlay_trace;
  const n = tr.length + 2;
  const newSet = new Set(s2.new_arcs.map(akey));
  return stepper(key, {
    title: o.title || 'Fill repair (S2), one row at a time', n,
    label: (i) => (i === 0 ? 'insert' : i === n - 1 ? 'ids' : 'row ' + tr[i - 1].row),
    legend: [['l', 'new', 'arc added so far'], ['l', 'hot', 'added at this row'], ['n', 'cur', 'row being read'], ['n', 'ring', 'its up-row'], ['n', 't3', 'its parent']],
    render(i, pics, panel) {
      const made = new Map(s2.inserted_pairs.map((p) => [akey(p), 0]));
      for (let k = 0; k < Math.min(i, tr.length); k++) for (const e of tr[k].pushed_into_parent) made.set(tr[k].parent + '-' + e, k + 1);
      const row = i > 0 && i <= tr.length ? tr[i - 1] : null;
      const arcs = o.oldArcs.map((a) => ({ lo: a.lo, hi: a.hi, cls: 'e faint' }));
      for (const [k, at] of made) {
        const [lo, hi] = k.split('-').map(Number);
        arcs.push({ lo, hi, cls: (row && at === i) || (i === 0 && at === 0) ? 'e hot' : 'e new', z: 3 });
      }
      const up = new Set(row ? row.up : []);
      ap(pics, wrapFig(rankGraph({ nodeOf: o.nodeOf, arcs, rcls: (r) => (row && r === row.row ? 'cur' : row && r === row.parent ? 't3' : up.has(r) ? 'ring' : ''), extraNodes: o.extraNodes, aria: 'Fill repair: new arcs found so far' })));
      if (i === 0) {
        ap(panel, h('h3', null, 'Insert into the overlay'), blk('inserted pairs', P('Each pair (lo, hi) goes into up(lo) in a host overlay: ', C(s2.inserted_pairs.map(farc).join(', ')), '.')),
          blk('then', P('Rows are popped by rank. A row is read as the merge of base, delta and overlay. Its parent is its lowest entry, and everything else in the row is pushed into the parent’s row. A row that gains nothing pushes nothing.')));
      } else if (row) {
        ap(panel, h('h3', null, 'Row ' + row.row),
          blk('merged up-row', P(C(flist(row.up)), '. Its parent is the first entry, ', B(row.parent), '.')),
          blk('push U(r) ∖ {p} into up(p)', P(row.pushed_into_parent.length ? ['New in up(' + row.parent + '): ', C(flist(row.pushed_into_parent)), '. Each is a new arc (' + row.parent + ', x).'] : 'Everything is already in the parent’s row: nothing new.')));
      } else {
        ap(panel, h('h3', null, 'Fixpoint: sort and number'),
          blk('new arcs, sorted by (lo, hi)', arrays(strip('arc', s2.new_arcs.map((a) => ({ ix: 'id ' + a[2], v: a[0] + '–' + a[1], cls: 'fill' }))))),
          blk('then', P('Ids are appended from arc_capacity up. Each arc goes into both deltas and added_since_fold, and the metric is preset to (∞, ∞, none, none).')),
          blk('work', P(C(s2.work + ' units'), ' for S2: one per row entry the overlay reads and one per delta insert.')));
      }
    },
  });
}

/* ===================== shared: build timeline ===================== */
function boxLabels(e) {
  const n = e.name;
  let m;
  if ((m = n.match(/^Relax apex ranks (.+)$/))) return ['Relax', 'ranks ' + m[1]];
  if ((m = n.match(/^Seed \(.*= (\d+)\)$/))) return ['Seed', 'at v' + m[1]];
  if ((m = n.match(/^(.+?) \(ticket (\d+)\)$/))) return [m[1], 'ticket ' + m[2]];
  if ((m = n.match(/^Seal \(next = None: (.+)\)$/))) return ['Seal', 'then ' + m[1]];
  if (/^Order: leaves/.test(n)) return ['Order', 'leaves, π'];
  const w = n.split(' ');
  return [w[0], w.slice(1).join(' ')];
}
const FIT_WORD = { Same: 'Same', CatchUp: 'CatchUp', Gone: 'Gone' };
const fitWord = (f) => FIT_WORD[f] || 'fits';
function timeline(events, cur) {
  const n = events.length, sx = 96, bw = 88, x0 = 70;
  const W = x0 + n * sx + 6, Hh = 176;
  const svg = sv('svg', { class: 'nat', viewBox: '0 0 ' + W + ' ' + Hh, width: W, height: Hh, role: 'img', 'aria-label': 'Timeline of writer commits and batch or job publishes' });
  const lanes = [['writers', 14], ['batches, jobs', 84]];
  for (const [t, y] of lanes) { svg.append(sv('rect', { class: 'lane', x: 2, y, width: W - 4, height: 60, rx: 6 })); svg.append(sv('text', { class: 'lanet', x: 8, y: y + 34 }, t.split(',')[0])); }
  events.forEach((e, i) => {
    const x = x0 + i * sx;
    const w = e.actor === 'writer';
    const y = w ? 22 : 92;
    const cls = 'box ' + (w ? 'w' : e.actor === 'batch' ? 'b' : 'j') + (i === cur ? ' cur' : i > cur ? ' later' : '') + (e.fit === 'CatchUp' ? ' catch' : e.fit === 'Gone' ? ' gone' : '');
    svg.append(sv('rect', { class: cls, x, y, width: bw, height: 44, rx: 5 }));
    const [b1, b2] = w ? ['v' + e.version, (e.what || '').split(/[ ;:]/)[0]] : boxLabels(e);
    const l1 = b1, l2 = b2;
    svg.append(sv('text', { class: 'bt', x: x + bw / 2, y: y + 18, 'text-anchor': 'middle' }, l1.length > 12 ? l1.slice(0, 11) + '…' : l1));
    svg.append(sv('text', { class: 'bs', x: x + bw / 2, y: y + 33, 'text-anchor': 'middle' }, l2.length > 14 ? l2.slice(0, 13) + '…' : l2));
    if (!w && e.fork != null) svg.append(sv('text', { class: 'bs', x: x + bw / 2, y: 152, 'text-anchor': 'middle' }, 'fork v' + e.fork));
    if (!w && e.fit) svg.append(sv('text', { class: 'bs' + (e.fit === 'CatchUp' ? ' fil' : e.fit === 'Gone' ? ' bad' : ''), x: x + bw / 2, y: 165, 'text-anchor': 'middle' }, fitWord(e.fit)));
  });
  return svg;
}
function logChips(log) {
  return log && log.length ? chips(log.map((l) => ['seq ' + l.seq + ': ' + (l.kind === 'Pairs' ? fpairs(l.pairs) : l.kind), 'acc'])) : P(h('span', { class: 'no' }, 'empty'));
}
function entryText(e) {
  if (!e) return '—';
  const parts = [e.state];
  if (e.phase) parts.push(e.phase);
  if (e.generation != null) parts.push('generation ' + e.generation);
  if (e.upkeep) parts.push('upkeep ' + e.upkeep.kind + ', ticket ' + e.upkeep.ticket + ', since ' + e.upkeep.since);
  else if (e.state === 'Built') parts.push('no upkeep');
  return parts.join(' · ');
}
function timelineStepper(key, events, title) {
  return stepper(key, {
    title, n: events.length, wide: true,
    label: (i) => (events[i].actor === 'writer' ? 'writer v' + events[i].version : events[i].name),
    legend: [['q', 'as', 'writer commit (under the lock)'], ['q', 'bs2', 'batch publish'], ['q', 'p', 'built-entry or repair job'], ['q', 'out', 'CatchUp: writers committed since the fork']],
    render(i, pics, panel) {
      const pane = h('div', { class: 'pane scroll' }, timeline(events, i));
      ap(pics, h('figure', null, pane));
      requestAnimationFrame(() => { pane.scrollLeft = Math.max(0, 70 + i * 96 - pane.clientWidth / 2); });
      ap(panel, eventDetail(events[i]));
    },
  });
}
/* the details of one timeline event: a writer's commit, or a batch's or job's publish */
function eventDetail(e) {
  const panel = h('div', { style: 'display:contents' });
  if (e.actor === 'writer') {
    ap(panel, h('h3', null, 'Writer commit v' + e.version),
      blk('what it does', P(e.what)),
      e.log_node ? blk('log node it appends', chips([['seq ' + e.log_node.seq + ': ' + fpairs(e.log_node.pairs), 'acc']])) : null,
      e.work != null ? blk('work under the lock', bar(e.work, e.budget), h('span', { class: 'barlab' }, e.work + ' of ' + e.budget + ' units' + (e.stopped_at_pop ? ', stopped at pop ' + e.stopped_at_pop + '; finishing would take ' + e.full_cost.work : ''))) : null,
      e.outcome ? blk('outcome', P(e.outcome)) : null,
      e.note ? blk('note', P(e.note)) : null,
      e.decay_met && e.decay_met.length ? blk('decay signals met after the flush', chips(e.decay_met.map((x) => [x, 'fil']))) : null);
  } else {
    ap(panel, h('h3', null, e.name),
      e.fork != null ? blk('fork', P('Forked at v' + e.fork + (e.reads ? '; reads ' + e.reads : '') + '.')) : null,
      e.fit ? blk('publish', P(B(e.fit), e.why ? ': ' + e.why + '.' : '')) : null,
      e.relinked && e.relinked.length ? blk('relinked on top', P('Published nodes with seq ', C(e.relinked.join(', ')), ' are relinked onto the batch’s copy.')) : null,
      e.consumed && e.consumed.length ? blk('consumed', P('Resolves and frees the nodes with seq ', C(e.consumed.join(', ')), '.')) : null,
      e.absorbed ? blk('absorbed at adoption', bar(e.absorbed.work, e.absorbed.budget), h('span', { class: 'barlab' }, 'seq ' + e.absorbed.seqs.join(', ') + ': ' + e.absorbed.work + ' of ' + e.absorbed.budget + ' units')) : null,
      e.path ? blk('path', P(e.path)) : null,
      e.outcome ? blk('outcome', P(e.outcome)) : null,
      e.note ? blk('note', P(e.note)) : null);
  }
  ap(panel, blk('published log after', logChips(e.published_log)), blk('entry', P(entryText(e.entry))));
  return panel;
}

/* ===================== Part C ===================== */
function stepC1(s) {
  const sec = section(s), d = s.data;
  const r2n = toMap(ST.B11.data.rank_to_node);
  const nr = ST.B11.data.n_r;
  const rk = range(1, nr);
  const g = rankGraph({ nodeOf: r2n, arcs: s.draw.edges.B.map(([a, b]) => ({ lo: a, hi: b, cls: 'e' })), aria: 'B in rank space: each node labelled with its rank.' });
  const mx = matrix({ rows: rk, cols: rk, cells: d.B.map(([a, b]) => ({ r: a - 1, c: b - 1, cls: a < b ? 'a' : 'on' })), aria: 'B = S(π,π) over ranks' });
  const par = toMap(d.etree);
  const t = tree({ parent: par, ranks: rk, cls: (r) => (r2n.get(r) == null ? 'iso' : ''), sub: (r) => (r2n.get(r) == null ? 'pool' : 'n' + r2n.get(r)), aria: 'Elimination tree over B' });
  ap(sec, h('div', { class: 'two' },
    wrapFig(g, 'Big number: rank. Small: the node that holds it.'),
    h('figure', null, h('div', { class: 'pane scroll' }, mx), legend([['q', 'a', 'upper triangle: becomes an arc'], ['q', 'on', 'lower triangle (mirror)']]), h('figcaption', null, 'B is symmetric; the game reads its upper triangle.'))));
  const cc = toMap(d.col_counts);
  ap(sec, h('div', { class: 'two' },
    h('figure', null, h('div', { class: 'pane scroll' }, t), h('figcaption', null, 'Liu’s elimination tree over B: a rank’s parent is its lowest higher neighbour once lower ranks are eliminated.')),
    h('div', { style: 'display:grid;gap:12px;align-content:start' },
      arrays(strip('arcs per rank', rk.map((r) => ({ ix: r, v: cc.get(r) })))),
      stats([['arcs (from counts)', d.arcs], ['guard (b)', d.guard, 'arcs allowed'], ['entries in B', d.B.length]]))));
  return sec;
}

function stepC2(s) {
  const sec = section(s), d = s.data, dr = s.draw;
  const vsg = Math.max(...d.log.map((l) => l.seq));
  ap(sec, h('div', { class: 'two' },
    h('div', { style: 'display:grid;gap:12px;align-content:start' },
      blk('log nodes resolved at the fork', logChips(d.log)),
      table(['pair', 'live edge at v' + vsg, 'goes to'], d.resolved.map((r) => [fpair(r.pair), yesno(r.live_edge), r.live_edge ? 'A (added)' : 'D (dropped)'])),
      table(['node', 'degree', 'rank', 'kind'], d.allocation.map((a) => [a.node, a.degree, a.rank, a.kind]), { caption: 'Rank allocation, in id order, before any pair is classified' }),
      stats([['free ranks after', flist(d.free_ranks_after)], ['n_r′', d.n_r_prime], ['n_r by 03 §2', d.n_r.if_03_s2_counted_the_log, 'not used']])),
    (() => {
      const bg = bgEdges(relsAt(vsg), { faint: true });
      const rk = toMap(ST.C5.data.rank_of);
      const nodes = D.versions[String(vsg)].graph.nodes.map((n) => ({ id: n, cls: dr.nodes.new_pool_rank.includes(n) ? 'pool' : dr.nodes.new_top_rank.includes(n) ? 'top' : rk.has(n) ? '' : 'iso', sub: rk.has(n) ? 'r' + rk.get(n) : null }));
      const edges = [...bg, ...dr.edges.added.map(([a, b]) => ({ a, b, cls: 'e hot', z: 2 })), ...dr.edges.dropped.map(([a, b]) => ({ a, b, cls: 'e del', z: 2 }))];
      return h('div', { style: 'display:grid;gap:10px' }, wrapFig(graph({ nodes, edges, aria: 'The graph at the Supergraph fork: logged pairs added or dropped, and the two new nodes with their ranks.' }), 'The graph at v' + vsg + ', each node with its rank.'),
        legend([['l', 'hot', 'logged pair with a live edge: in A'], ['l', 'del', 'logged pair with no live edge: in D'], ['n', 'pool', 'takes a pool rank (degree ≤ 1)'], ['n', 'top', 'takes a new top rank'], ['n', 'iso', 'no edge, no rank']]));
    })()));
  const rk = range(1, d.n_r_prime);
  const inA = new Set(d.A.map(([a, b]) => a + ',' + b));
  const cells = d.B_prime.map(([a, b]) => ({ r: a - 1, c: b - 1, cls: inA.has(a + ',' + b) ? 'a' : 'on' }));
  for (const [a, b] of d.D) cells.push({ r: a - 1, c: b - 1, x: true });
  ap(sec, h('figure', null, h('div', { class: 'pane scroll' }, matrix({ rows: rk, cols: rk, cells, aria: 'B′ = (B ∖ D) ∪ A over ranks' })),
    legend([['q', 'on', 'kept from B'], ['q', 'a', 'A: added'], ['q', 'x', 'D: dropped']]),
    h('figcaption', null, 'B′ = (B ∖ D) ∪ A, the exact live pattern at the fork (revision 3.1).')));
  return sec;
}

function stepC3(s) {
  const sec = section(s), d = s.data;
  const nodeOf = BUILD.nodeOf;
  const w = d.why_noncontiguous;
  const nonc = new Set(d.noncontiguous_subtrees);
  const pool = new Set(w.new_pool_ranks.map((x) => x[0]));
  const moved = new Set(w.moved_by_the_drop.map((x) => x[0]));
  const topr = new Set(w.top_ranks_not_counted);
  const par = toMap(d.etree);
  const rk = d.etree.map(([c]) => c);
  const t = tree({ parent: par, ranks: rk,
    cls: (r) => (nodeOf.get(r) == null ? 'iso' : pool.has(r) ? 'pool' : topr.has(r) ? 'top' : nonc.has(r) ? 't3' : ''),
    ecls: (c) => (moved.has(c) || pool.has(c) ? 'te-moved' : 'te'),
    sub: (r) => (nodeOf.get(r) == null ? 'pool' : 'n' + nodeOf.get(r)), aria: 'Elimination tree over B′' });
  const cc = toMap(d.col_counts);
  ap(sec, h('div', { class: 'two' },
    h('figure', null, h('div', { class: 'pane scroll' }, t),
      legend([['n', 't3', 'root of a subtree whose ranks are no longer consecutive'], ['n', 'pool', 'new pool rank'], ['n', 'top', 'new top rank (not counted)'], ['l', 'te-moved', 'link that is new or moved'], ['n', 'iso', 'pool rank with no holder']]),
      h('figcaption', null, 'The elimination tree recomputed over B′. The postorder relabel is not redone.')),
    h('div', { style: 'display:grid;gap:12px;align-content:start' },
      arrays(strip('arcs per rank', d.col_counts.map(([r, c]) => ({ ix: r, v: c, cls: r > ST.B11.data.n_r ? 'fill' : '' })))),
      stats([['arcs (from counts)', d.arcs], ['guard (b)', d.guard], ['non-consecutive subtrees', d.noncontiguous_subtrees.length]]),
      h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'why subtrees broke'), h('ul', null,
        w.new_pool_ranks.map(([r, n, under]) => h('li', null, 'Pool rank ' + r + ' (node ' + n + ') hangs under rank ' + under + ', so every subtree above it contains rank ' + r + ': ' + flist(w.contain_a_new_pool_rank) + '.')),
        w.moved_by_the_drop.map(([r, a, b]) => h('li', null, 'Dropping ' + fpairs(w.dropped_roads) + ' moves rank ' + r + ' from under ' + a + ' to under ' + b + '.')),
        h('li', null, 'Broken by the drop: ' + flist(w.broken_by_the_drop) + '.'),
        h('li', null, 'Top rank ' + flist(w.top_ranks_not_counted) + ' is outside the ranks Order assigned, so it is not counted.'))))));
  return sec;
}

function stepC4(s) {
  const sec = section(s), d = s.data, dr = s.draw;
  const steps = d.steps;
  const nodeOf = BUILD.nodeOf;
  const fillAt = new Map();
  for (const st of steps) for (const f of st.fill) fillAt.set(st.rank + '-' + f, st.rank);
  const orig = dr.edges.original;
  const par = new Map(steps.map((st) => [st.rank, st.parent]));
  const upOf = new Map(steps.map((st) => [st.rank, st.up]));
  ap(sec, stepper('st-c4', {
    title: 'The elimination game, rank by rank', n: steps.length,
    label: (i) => 'rank ' + steps[i].rank,
    legend: [['n', 'cur', 'row being built'], ['n', 'ring', 'its up-row'], ['l', 'hot', 'its own edges from B′'], ['l', 'new', 'its fill (from children)'], ['l', 'fill', 'fill from earlier rows'], ['n', 'gone', 'done']],
    render(i, pics, panel) {
      const st = steps[i], r = st.rank;
      const up = new Set(st.up);
      const arcs = [];
      for (const [a, b] of orig) arcs.push({ lo: a, hi: b, cls: a === r ? 'e hot' : a < r ? 'e faint' : 'e', z: a === r ? 3 : 0 });
      for (const [k, at] of fillAt) if (at <= r) { const [a, b] = k.split('-').map(Number); arcs.push({ lo: a, hi: b, cls: at === r ? 'e new' : a < r ? 'e fill faint' : 'e fill', z: at === r ? 4 : 1 }); }
      ap(pics, wrapFig(rankGraph({ nodeOf, arcs, rcls: (x) => (x === r ? 'cur' : up.has(x) ? 'ring' : x < r ? 'gone' : ''), aria: 'The game at rank ' + r })));
      const tp = new Map([...par].filter(([c]) => c <= r));
      ap(pics, h('figure', null, h('div', { class: 'pane scroll' }, tree({ parent: par, ranks: steps.map((x) => x.rank), cls: (x) => (x === r ? 'cur' : x < r ? 'gone' : nodeOf.get(x) == null ? 'iso' : ''), ecls: (c) => (c === r ? 'te-new' : c > r ? 'te te-later' : 'te'), sub: (x) => (nodeOf.get(x) == null ? 'pool' : 'n' + nodeOf.get(x)), aria: 'Parent links found so far' })),
        h('figcaption', null, 'Parent links found so far (later ones faded).')));
      void tp;
      ap(panel, h('h3', null, 'Rank ' + r + (st.node == null ? ': pool rank, no holder' : ': node ' + st.node)));
      if (st.node == null) { ap(panel, P('A holderless rank has no row in B′, no arcs and no parent (P).')); return; }
      ap(panel, 
        blk('own neighbours above, from U0 = triu(B′, 1)', P(C(flist(st.own)))),
        blk('children', st.kids.length ? h('ul', null, st.kids.map((c) => h('li', null, 'row(' + c + ') = ' + flist(upOf.get(c)) + ' minus ' + r))) : P('none')),
        blk('row(' + r + ')', P(C(flist(st.up)), st.fill.length ? [' — fill: ', h('b', { class: 'fl' }, flist(st.fill))] : ' — no fill')),
        blk('tree link', P(st.parent == null ? 'Empty row: no parent, a root.' : ['Parent = first entry = ', B(st.parent), '.'])));
    },
  }));
  ap(sec, stats([['arcs', d.arcs], ['fill', d.fill], ['original', d.arcs - d.fill]]));
  return sec;
}

function stepC5(s) {
  const sec = section(s), d = s.data, b = d.base;
  const groupsUp = [], groupsUx = [];
  for (let i = 0; i + 1 < b.Up.Ap.length; i++) {
    const cells = [], cx = [];
    for (let k = b.Up.Ap[i]; k < b.Up.Ap[i + 1]; k++) { const f = FILL0.has(i + 1 + '-' + b.Up.Aj[k]); cells.push({ ix: k, v: b.Up.Aj[k], cls: f ? 'fill' : '' }); cx.push({ ix: k, v: b.Up.Ax[k], cls: f ? 'fill' : '' }); }
    if (cells.length) { groupsUp.push({ gl: 'row ' + i + ' · r' + (i + 1), cells }); groupsUx.push({ gl: 'row ' + i, cells: cx }); }
  }
  const groupsDn = [];
  for (let i = 0; i + 1 < b.Down.Ap.length; i++) {
    const cells = [];
    for (let k = b.Down.Ap[i]; k < b.Down.Ap[i + 1]; k++) cells.push({ ix: 'id ' + b.Down.Ax[k], v: b.Down.Aj[k], cls: FILL0.has(b.Down.Aj[k] + '-' + (i + 1)) ? 'fill' : '' });
    if (cells.length) groupsDn.push({ gl: 'row ' + i + ' · r' + (i + 1), cells });
  }
  ap(sec, h('figure', null, h('div', { class: 'pane arrays' },
    strip('Up.Ap', b.Up.Ap.map((v, i) => ({ ix: i, v }))),
    gstrip('Up.Aj (column rank)', groupsUp),
    gstrip('Up.Ax (arc id)', groupsUx)),
  h('figcaption', null, 'Up.base as stored. ' + b.row_of_rank + '. Amber entries are fill. Small numbers are array positions; id = Ap[row] + position in the row, so on a fresh build the id is the position.')));
  ap(sec, h('figure', null, h('div', { class: 'pane arrays' },
    strip('Down.Ap', b.Down.Ap.map((v, i) => ({ ix: i, v }))),
    gstrip('Down.Aj, with Ax', groupsDn)),
  h('figcaption', null, 'Down.base is the transpose and keeps the same ids: each cell shows the lower rank, with its arc id above.')));
  const rk = range(1, d.n);
  const cells = BUILD.arcs.map((a) => ({ r: a.lo - 1, c: a.hi - 1, cls: FILL0.has(a.lo + '-' + a.hi) ? 'f' : 'ink', text: a.id }));
  ap(sec, h('div', { class: 'two' },
    h('figure', null, h('div', { class: 'pane scroll' }, matrix({ rows: rk, cols: rk, cells, cs: 19, lw: 26, aria: 'Up.base as a matrix whose values are arc ids' })),
      legend([['q', 'ink', 'original arc'], ['q', 'f', 'fill arc']]), h('figcaption', null, 'Up as a GraphBLAS matrix: the value at (lo, hi) is the arc id.')),
    h('div', { style: 'display:grid;gap:12px;align-content:start' },
      arrays(strip('node_of[rank]', d.node_of.map(([r, n]) => ({ ix: r, v: n == null ? 'none' : n, cls: n == null ? 'iso' : '' }))),
        strip('rank_of[node]', d.rank_of.map(([n, r]) => ({ ix: n, v: r })))),
      stats([['arcs', d.arc_capacity, 'arc_capacity'], ['rows', d.n], ['free ranks', flist(d.free_ranks)], ['deltas', d.delta.Up == null ? 'empty' : 'set']]))));
  return sec;
}

/* ===================== Part D ===================== */
function stepD1(s) {
  const sec = section(s), d = s.data;
  const rows = d.per_slab_and_type.map((x) => [fpair(x.rows), x.type, x.m.length, fpairs(x.masked_by_dm.filter(([a, b]) => a < b)) + (x.masked_by_dm.length ? ' (' + x.masked_by_dm.length + ')' : ''), x.masked_by_dp.map(([a, b]) => a + '→' + b).join(', ') || 'none', x.dp.length, x.X.length]);
  ap(sec, table(['rows', 'type', 'm', 'masked by dm', 'masked by dp', 'dp', 'X (live)'], rows, { stack: true }));
  const vc = Math.max(...ST.C2.data.log.map((l) => l.seq));
  const Wm = new Map(d.W.map(([a, b, w]) => [a + ',' + b, w]));
  const tagFor = (a, b) => { const x = Wm.get(a + ',' + b), y = Wm.get(b + ',' + a); return x === y ? x : fw(x) + '/' + fw(y); };
  const edges = s.draw.edges.seeded.map(([a, b]) => ({ a, b, cls: 'e', tag: tagFor(a, b) }));
  const rk = BUILD.rankOf;
  const nodes = D.versions[String(vc)].graph.nodes.map((n) => ({ id: n, cls: rk.has(n) ? '' : 'iso', sub: rk.has(n) ? 'r' + rk.get(n) : null }));
  const stale = ST.A.data.stale_m_under_dp;
  ap(sec, h('div', { class: 'two' },
    wrapFig(graph({ nodes, edges, aria: 'The live weights W at the seed version, on the node graph.' }), 'W: the minimum live weight across types per ordered pair. Where the two directions differ the tag reads lo→hi / hi→lo.'),
    h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'the replaced road'),
      Object.entries(stale).map(([k, v]) => P(k + ': m still holds id ' + v.m_id + ' at weight ' + v.m_weight + '; dp holds id ' + v.dp_id + ' at weight ' + v.dp_weight + '. The ¬struct(dp) mask removes the m entry, so the seed reads ' + v.dp_weight + '. Reading both and taking MIN would seed the dead road.')),
      P('Seeds match the per-pair seeds: ', yesno(d.equals_per_pair_seeds), '.'))));
  const n = BUILD.nodeOf.size;
  const rks = range(1, n);
  const cells = d.W_rank.map(([a, b, w]) => ({ r: a - 1, c: b - 1, cls: a < b ? 'as' : 'bs2', text: w }));
  ap(sec, h('figure', null, h('div', { class: 'pane scroll' }, matrix({ rows: rks, cols: rks, cells, cs: 18, aria: 'W(π,π) over ranks' })),
    legend([['q', 'as', 'triu(Wπ): upward seeds'], ['q', 'bs2', 'below the diagonal: triu(Wπᵀ) gives downward seeds']]),
    h('figcaption', null, 'Wπ = W(π,π). Arc (lo, hi) is node_of(lo) → node_of(hi) upward, and the reverse downward.')));
  const S = (arr, lab) => strip(lab, arr.map((v, i) => ({ ix: i, v: fw(v), cls: v == null ? 'iso' : '' })));
  ap(sec, h('figure', null, h('div', { class: 'pane arrays' }, S(d.seeds_up, 'seeds_up[id]'), S(d.seeds_down, 'seeds_down[id]')),
    h('figcaption', null, 'Written by arc id: seeds[Ax[k]] = val[k]. Fill arcs have no edge, so they start at ∞. Delta entries at this point: ' + d.delta_entries.length + '.')));
  return sec;
}

function stepD2(s) {
  const sec = section(s), d = s.data;
  const T = d.triangles;
  const seeds = ST.D1.data;
  const states = [];
  let up = seeds.seeds_up.slice(), dn = seeds.seeds_down.slice();
  for (const t of T) { up = up.slice(); dn = dn.slice(); up[t.arcs[2]] = t.now[0]; dn[t.arcs[2]] = t.now[1]; states.push([up, dn]); }
  const batchOf = (x) => d.batches.findIndex((b) => x >= b.apex_ranks[0] && x <= b.apex_ranks[1]);
  const improved = T.filter((t) => t.up_won || t.down_won).length;
  ap(sec, stats([['triangles', T.length], ['improve an arc', improved], ['relax batches', d.batches.length], ['equals a full customization', d.equals_full ? 'yes' : 'no']]));
  ap(sec, stepper('st-d2', {
    title: 'Relax, one triangle at a time', n: T.length,
    label: (i) => 'triangle ' + (i + 1) + ' of ' + T.length,
    legend: [['n', 'cur', 'apex x'], ['n', 'ring', 'y and z, the arc’s ends'], ['l', 'apex', 'bottom arcs (x, y) and (x, z)'], ['l', 'hot', 'top arc (y, z), improved'], ['l', 'done', 'top arc, unchanged']],
    render(i, pics, panel) {
      const t = T[i];
      const won = t.up_won || t.down_won;
      const k = new Map([[key2(t.x, t.y), 'e apexwin'], [key2(t.x, t.z), 'e apexwin'], [key2(t.y, t.z), won ? 'e hot' : 'e done']]);
      const arcs = BUILD.arcs.map((a) => { const c = k.get(a.lo + '-' + a.hi); return { lo: a.lo, hi: a.hi, cls: c || 'e faint', z: c ? 3 : 0 }; });
      ap(pics, wrapFig(rankGraph({ nodeOf: BUILD.nodeOf, arcs, rcls: (r) => (r === t.x ? 'cur' : r === t.y || r === t.z ? 'ring' : ''), aria: 'Triangle with apex ' + t.x + ' and top arc (' + t.y + ', ' + t.z + ')' })));
      const [U, Dn] = states[i];
      const hi = new Map([[t.arcs[0], 'hl'], [t.arcs[1], 'hl'], [t.arcs[2], won ? 'cur' : 'ink']]);
      ap(pics, h('figure', null, h('div', { class: 'pane arrays' },
        strip('up[id]', U.map((v, j) => ({ ix: j, v: fw(v), cls: hi.get(j) || (v == null ? 'iso' : '') }))),
        strip('down[id]', Dn.map((v, j) => ({ ix: j, v: fw(v), cls: hi.get(j) || (v == null ? 'iso' : '') })))),
      h('figcaption', null, 'The two weight arrays after this triangle. Tinted: the bottom arcs read; solid: the top arc written.')));
      const bi = batchOf(t.x);
      ap(panel, h('h3', null, 'Apex ' + t.x + ', arc (' + t.y + ', ' + t.z + ')'),
        blk('batch', P('Relax batch ' + (bi + 1) + ' of ' + d.batches.length + ': apex ranks ' + d.batches[bi].apex_ranks.join('–') + ' (' + d.batches[bi].triangles + ' triangles).')),
        blk('arc ids', P('bottom (x, y) = ', C(t.arcs[0]), ', bottom (x, z) = ', C(t.arcs[1]), ', top (y, z) = ', C(t.arcs[2]), '.')),
        blk('detours', mini(['direction', 'via x', 'was', 'now'], [
          ['up(y, z)', { v: 'down(x,y)+up(x,z) = ' + fw(t.try_up), cls: t.up_won ? 'w' : '' }, fw(t.was[0]), fw(t.now[0])],
          ['down(y, z)', { v: 'down(x,z)+up(x,y) = ' + fw(t.try_down), cls: t.down_won ? 'w' : '' }, fw(t.was[1]), fw(t.now[1])]])),
        blk('result', P(won ? ['Strictly smaller in ', [t.up_won ? 'up' : null, t.down_won ? 'down' : null].filter(Boolean).join(' and '), ', so it is written with x = ', B(t.x), ' as the middle.'] : 'Not strictly smaller in either direction: nothing is written.')));
    },
  }));
  const m = d.metric_v7;
  const S = (arr, lab, f) => strip(lab, arr.map((v, i) => ({ ix: i, v: f(v), cls: v == null ? 'iso' : '' })));
  ap(sec, h('figure', null, h('div', { class: 'pane arrays' }, S(m.up, 'up[id]', fw), S(m.down, 'down[id]', fw), S(m.up_middle, 'up_middle[id]', fm), S(m.down_middle, 'down_middle[id]', fm)),
    h('figcaption', null, 'The finished metric at the seed version: four arrays indexed by arc id. A middle is the apex rank the value goes through; a dash means the arc’s own edge.')));
  return sec;
}

function flushBlocks(f, o) {
  o = o || {};
  const out = [];
  if (f.resolved) out.push(table(['pair', 'live edge', 'weights (lo→hi, hi→lo)'], f.resolved.map((r) => [fpair(r.pair), yesno(r.live_edge), (r.weights || []).map(fw).join(', ')]), { caption: 'Resolved at the version (seq ' + f.seq + ')' }));
  if (f.allocation && f.allocation.length) out.push(table(['node', 'degree', 'rank', 'kind'], f.allocation.map((a) => [a.node, a.degree, a.rank, a.kind]), { caption: 'Rank allocation, before classification' }));
  const c = f.classify;
  if (c) out.push(h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'classify (rank pairs)'),
    h('ul', null, h('li', null, 'metric only (the pair has an arc): ' + (c.metric_only.map(farc).join(', ') || 'none')),
      h('li', null, 'S2 insert (no arc yet): ' + (c.s2_insert.map(farc).join(', ') || 'none')),
      c.nothing && c.nothing.length ? h('li', null, 'nothing: ' + c.nothing.map(farc).join(', ')) : null),
    f.path ? P('Path: ', B(f.path), f.valve ? ' (' + f.valve + ')' : '', '.') : null));
  return out;
}
function workTable(w, budget) {
  const cols = [['merge steps', w.merge_steps], ['pushes', w.pushes], ['seed lookups', w.seed_lookups], ['S2', w.s2], ['relax triangles', w.sweep_triangles], ['total', w.total], ['pops (not counted)', w.pops]].filter(([, v]) => v != null);
  return h('div', { style: 'display:grid;gap:6px' }, mini(cols.map((c) => c[0]), [cols.map((c) => c[1])]),
    h('span', { class: 'barlab' }, 'What each unit counts: see ', h('a', { href: '#terms' }, 'words and units'), ' under Toy scale.'),
    budget != null ? [bar(w.total, budget), h('span', { class: 'barlab' }, w.total + ' of ' + budget + ' units (B_gil)')] : null);
}

function stepD3(s) {
  const sec = section(s), d = s.data, f = d.flush;
  const vs = f.seq;
  const rk = BUILD.rankOf;
  const nodes = D.versions[String(vs)].graph.nodes.map((n) => ({ id: n, cls: rk.has(n) ? '' : 'iso', sub: rk.has(n) ? 'r' + rk.get(n) : null }));
  const ch = new Set(s.draw.edges.changed.map(([a, b]) => key2(a, b)));
  const edges = [...bgEdges(relsAt(vs), { faint: true, skip: ch }), ...s.draw.edges.changed.map(([a, b]) => ({ a, b, cls: 'e hot', z: 2 }))];
  ap(sec, h('div', { class: 'two' },
    wrapFig(graph({ nodes, edges, aria: 'The pair whose weight the logged SET changed.' }), 'The pair the logged write changed, at v' + vs + '.'),
    h('div', { style: 'display:grid;gap:10px;align-content:start' }, blk('log at the Seal’s fork', logChips(d.log)), ...flushBlocks(f))));
  ap(sec, recomputeStepper('st-d3-seal', { title: 'The Seal’s scoped recompute (off the lock)', pops: f.recompute, front: f.front, arcs: BUILD.arcs, nodeOf: BUILD.nodeOf, frontWhy: 'The pair already has an arc, so the front is that arc with its new seeds.' }));
  ap(sec, h('div', { class: 'two' }, h('div', null, h('span', { class: 'bh' }, 'work'), workTable(f.work)),
    h('div', null, h('span', { class: 'bh' }, 'then'), P('Equals a full customization at v' + vs + ': ', yesno(d.equals_full), '. Each flat array becomes chunks with WeightArr::from_vec, one array at a time; chunk ids:'),
      arrays(...Object.entries(d.chunks).map(([k, v]) => strip(k, v.map((x) => ({ v: x }))))))));
  const a = d.adoption, af = a.flush;
  ap(sec, h('h3', { class: 'sub-h' }, 'Adoption: absorbing the commit that landed during the Seal'),
    h('div', { class: 'two' }, h('div', { style: 'display:grid;gap:10px;align-content:start' }, blk('relinked log node', logChips(a.log)), ...flushBlocks(af)),
      h('div', null, h('span', { class: 'bh' }, 'work under the lock'), workTable(af.work, af.budget), P('Equals a full customization: ', yesno(a.equals_full), '.'))));
  ap(sec, recomputeStepper('st-d3-adopt', { title: 'Adoption: the v' + af.seq + ' commit absorbed under the lock', pops: af.recompute, front: af.front, arcs: BUILD.arcs, nodeOf: BUILD.nodeOf, budget: af.budget, frontWhy: 'The deleted road still has a ferry the other way, so its arc reseeds to the ferry alone.' }));
  return sec;
}
