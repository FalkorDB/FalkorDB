/* ---------- the running example ---------- */
const NODE_CAP = D.example.node_cap;
const SKEL = ST.B1.draw.edges.skeleton;
function roads(rels) {
  const m = new Map();
  for (const r of rels) {
    if (r.src === r.dst) continue;
    const lo = Math.min(r.src, r.dst), hi = Math.max(r.src, r.dst), k = lo + '-' + hi;
    if (!m.has(k)) m.set(k, { lo, hi, ROAD: {}, FERRY: {} });
    const t = m.get(k)[r.type] || (m.get(k)[r.type] = {});
    t[r.src === lo ? 'up' : 'dn'] = r.weight;
  }
  const out = [];
  for (const p of m.values()) {
    const R = p.ROAD, F = p.FERRY, both = R.up != null || R.dn != null;
    if (R.up != null && R.dn != null) out.push({ a: p.lo, b: p.hi, kind: 'road', tag: R.up === R.dn ? R.up : R.up + '/' + R.dn });
    else if (R.up != null) out.push({ a: p.lo, b: p.hi, kind: 'oneway', arrow: true, tag: R.up });
    else if (R.dn != null) out.push({ a: p.hi, b: p.lo, kind: 'oneway', arrow: true, tag: R.dn });
    if (F.up != null) out.push({ a: p.lo, b: p.hi, kind: 'ferry', arrow: true, tag: F.up, both });
    if (F.dn != null) out.push({ a: p.hi, b: p.lo, kind: 'ferry', arrow: true, tag: F.dn, both });
  }
  return out;
}
const relsAt = (v) => D.versions[String(v)].graph.relationships;
function bgEdges(rels, o) {
  o = o || {};
  const skip = o.skip || new Set();
  return roads(rels).filter((e) => !skip.has(key2(e.a, e.b))).map((e) => ({
    a: e.a, b: e.b, arrow: e.arrow, z: -1,
    cls: 'e' + (e.kind === 'ferry' ? ' fy' : '') + (o.faint ? ' faint' : ''),
    bow: e.kind === 'ferry' && e.both ? -0.44 : 0,
    tag: o.tags ? e.tag : null, tagcls: e.kind === 'ferry' ? 'mut' : '',
  }));
}
function baseArcs(base) {
  const out = [], U = base.Up;
  for (let i = 0; i + 1 < U.Ap.length; i++) for (let k = U.Ap[i]; k < U.Ap[i + 1]; k++) out.push({ lo: i + 1, hi: U.Aj[k], id: U.Ax[k] });
  return out;
}
const BUILD = { arcs: baseArcs(ST.C5.data.base), nodeOf: toMap(ST.C5.data.node_of), rankOf: toMap(ST.C5.data.rank_of) };
function hier(k) {
  const Hh = D.hierarchies[String(k)];
  return { arcs: Hh.arcs.map(([lo, hi, id]) => ({ lo, hi, id })), nodeOf: toMap(Hh.maps.node_of), rankOf: toMap(Hh.maps.rank_of), H: Hh };
}
const FILL0 = new Set(ST.C4.draw.edges.fill.map(([a, b]) => a + '-' + b));
const SIDE_TONE = ['t2', 't1', 'sep'];
const SIDE_BLOB = ['t2', 't1', 'tsep'];
const SIDE_NAME = ['side 0', 'side 1', 'separator'];
const SIDE_TONE91 = ['t0', 't3', 'sep'];
const SIDE_BLOB91 = ['t0', 't3', 'tsep'];

/* rank-space drawing: ranks sit on the node that holds them */
function rankGraph(o) {
  const nodeOf = o.nodeOf;
  const nodes = [], ids = [];
  for (const [r, n] of nodeOf) {
    if (n == null) continue;
    ids.push(n);
    nodes.push({ id: n, big: r, sub: 'n' + n, cls: o.rcls ? o.rcls(r) : '' });
  }
  for (const x of o.extraNodes || []) { nodes.push(x); ids.push(x.id); }
  const edges = o.arcs.map((a) => ({ a: nodeOf.get(a.lo), b: nodeOf.get(a.hi), cls: a.cls, z: a.z, tag: a.tag, tagcls: a.tagcls, arrow: a.arrow }));
  return graph({ nodes, edges, layoutIds: ids, aria: o.aria, cacheKey: 'rank' });
}

/* coarse view: fine nodes as dots inside blobs, coarse vertices at the centroid of their members */
function coarseView(o) {
  const coords = Object.assign({}, o.coords || CO);
  const single = o.verts.every((v) => o.members.get(v).length === 1);
  const cid = (v) => (single ? String(o.members.get(v)[0]) : 'c' + v);
  for (const v of o.verts) {
    const ms = o.members.get(v);
    coords[cid(v)] = [ms.reduce((t, m) => t + coords[m][0], 0) / ms.length, ms.reduce((t, m) => t + coords[m][1], 0) / ms.length];
  }
  const skel = o.skel || SKEL;
  const groups = single ? [] : o.verts.map((v) => {
    const ms = o.members.get(v), set = new Set(ms.map(String));
    return { members: ms, links: skel.filter(([a, b]) => set.has(String(a)) && set.has(String(b))), tone: o.tone ? o.tone(v) : 't4' };
  });
  const nodes = [];
  if (!single) for (const f of o.fine) nodes.push({ id: f, cls: 'dot', r: 4, nolabel: true });
  for (const v of o.verts) nodes.push({ id: cid(v), cls: o.vcls ? o.vcls(v) : '', big: single ? o.members.get(v)[0] : 'c' + v, sub: o.vsub ? o.vsub(v) : null, r: single ? 19 : 18 });
  const wt = (o.A || []).some(([, , w]) => w > 1);
  const edges = [];
  for (const [i, j, w] of o.A || []) if (i < j) edges.push({ a: cid(i), b: cid(j), cls: o.ecls ? o.ecls(i, j) : 'e', tag: wt ? w : null, z: o.ez ? o.ez(i, j) : 0 });
  for (const e of o.edges || []) edges.push(Object.assign({}, e, { a: cid(e.a), b: cid(e.b) }));
  const layoutIds = single ? o.verts.map(cid) : o.fine.map(String).concat(o.verts.map(cid));
  return graph({ coords, nodes, edges, groups, layoutIds, obstacleIds: o.verts.map(cid), aria: o.aria, sx: o.sx, sy: o.sy, cacheKey: o.ck || 'cv' });
}
function greedyTones(verts, A) {
  const adj = new Map(verts.map((v) => [v, new Set()]));
  for (const [i, j] of A) { if (adj.has(i)) adj.get(i).add(j); }
  const tone = new Map(), T = ['t0', 't1', 't2', 't3', 't4'];
  for (const v of verts) {
    const used = new Set([...adj.get(v)].map((u) => tone.get(u)));
    tone.set(v, T.find((t) => !used.has(t)) || 't4');
  }
  return tone;
}
const DEPTH0 = (() => {
  const lv = new Map();
  for (const s of D.steps) if (s.data && s.data.depth === 0 && s.data.rounds) lv.set(s.data.level, s.data);
  const last = [...lv.keys()].sort((a, b) => b - a)[0];
  const top = lv.get(last).next;
  lv.set(last + 1, Object.assign({ level: last + 1 }, top));
  return lv;
})();
const wOf = (dat) => toMap(dat.w);
const memOf = (dat) => toMap(dat.members);
function sideWeights(labels, w) {
  const t = [0, 0, 0];
  for (const [v, l] of labels) t[l] += w.get(v);
  return t;
}
/* replays node FM passes: each move sets the moved vertex's side and pulls its listed neighbours into the separator;
   at the end of a pass the state rolls back to the first `kept` moves */
function fmReplay(start, passes, ctx) {
  let lab = new Map(start);
  const frames = [];
  passes.forEach((ps, pi) => {
    if (!ps.moves.length) return;
    const states = [new Map(lab)];
    ps.moves.forEach((mv, mi) => {
      const nl = new Map(states[states.length - 1]);
      nl.set(mv.move, mv.to_side);
      for (const u of mv.pulled_into_separator) nl.set(u, 2);
      states.push(nl);
      frames.push(Object.assign({ kind: 'move', pass: pi, idx: mi, mv, labels: nl, nmoves: ps.moves.length }, ctx));
    });
    lab = states[ps.kept];
    frames.push(Object.assign({ kind: 'passend', pass: pi, kept: ps.kept, nmoves: ps.moves.length, labels: new Map(lab) }, ctx));
  });
  return { frames, labels: lab };
}
function moveBlock(f, limit) {
  const mv = f.mv;
  return [
    blk('Node FM, pass ' + (f.pass + 1) + ', move ' + (f.idx + 1) + ' of ' + f.nmoves,
      P('Move ', B(f.vname(mv.move)), ' from the separator to ', B(SIDE_NAME[mv.to_side]), '. ',
        mv.pulled_into_separator.length ? ['Its neighbours on the other side, ', C(mv.pulled_into_separator.map(f.vname).join(', ')), ', join the separator.'] : 'No neighbour on the other side, so nothing joins the separator.')),
    mini(['gain', 'separator weight', 'imbalance', 'balance limit'], [[mv.gain, mv.separator_weight, mv.imbalance, limit != null ? fnum(limit) + (mv.over_balance_limit ? ' (over)' : '') : '—']]),
  ];
}
function passEndBlock(f) {
  return blk('End of pass ' + (f.pass + 1),
    P(f.kept === 0 ? 'No move improved on the start of the pass, so every move is undone.' : ['The best state came after move ', B(f.kept), ' of ', f.nmoves, '. Later moves are undone.']));
}
function labelStrip(labels, vname, title) {
  return strip(title || 'label (0, 1, 2 = separator)', [...labels].sort((a, b) => a[0] - b[0]).map(([v, l]) => ({ ix: vname(v), v: l, cls: l === 2 ? 'ink' : l === 1 ? 'pl' : 'ok' })));
}
const sepLegend = [['n', 't2', 'side 0'], ['n', 't1', 'side 1'], ['n', 'sep', 'separator'], ['n', 'ring', 'moved this step'], ['n', 'ring2', 'pulled into the separator']];

/* ===================== Part A ===================== */
function stepA(s) {
  const sec = section(s), d = s.data, dr = s.draw;
  const special = new Set([...dr.edges.pending_add, ...dr.edges.replaced].map(([a, b]) => key2(a, b)));
  const edges = bgEdges(d.graph.relationships, { tags: true, skip: special }).map((e) => Object.assign(e, { z: 0 }));
  const relById = new Map(d.graph.relationships.map((r) => [r.id, r]));
  const wt = (a, b) => { const r = d.graph.relationships.find((x) => x.src === a && x.dst === b); return r ? r.weight : null; };
  const both = (a, b) => { const x = wt(a, b), y = wt(b, a); return x === y ? x : x + '/' + y; };
  for (const [a, b] of dr.edges.pending_add) edges.push({ a, b, cls: 'e add', tag: both(Math.min(a, b), Math.max(a, b)), tagcls: 'acc', z: 1 });
  for (const [a, b] of dr.edges.replaced) edges.push({ a, b, cls: 'e rep', tag: both(Math.min(a, b), Math.max(a, b)), tagcls: 'fil', z: 1 });
  for (const [a, b] of dr.edges.pending_delete) edges.push({ a, b, cls: 'e del', z: 1 });
  const iso = new Set(dr.nodes.isolated), unused = new Set(dr.nodes.unused);
  const nodes = [...dr.nodes.node, ...dr.nodes.unused].map((n) => ({ id: n, cls: iso.has(n) ? 'iso' : unused.has(n) ? 'unused' : '' }));
  const g = graph({ nodes, edges, aria: 'The example graph at the first version: a grid of road junctions, a dead-end road off one corner, one isolated node and one unused id, with the pending edits marked.' });
  const stale = Object.entries(d.stale_m_under_dp).map(([k, v]) => k + ': id ' + v.m_id + ' (weight ' + v.m_weight + ') stays in m; id ' + v.dp_id + ' (weight ' + v.dp_weight + ') is in dp');
  ap(sec, h('div', { class: 'two' },
    wrapFig(g, ['Tags are weights. Where the two directions differ, the tag reads lo→hi / hi→lo. ', stale.join('. '), '.']),
    h('div', { style: 'display:grid;gap:12px;align-content:start' },
      legend([['l', '', 'road, both directions'], ['l', '', 'one-way road (arrow)'], ['l', 'fy', 'ferry, one way (arrow)'], ['l', 'add', 'pending addition: in dp'], ['l', 'del', 'pending deletion: in dm'], ['l', 'rep', 'replaced in one transaction'], ['n', 'iso', 'node with no edge'], ['n', 'unused', 'free id slot']]),
      stats(Object.entries(d.graph.tensors).flatMap(([t, L]) => [[t + ' m', L.m.length, 'committed'], [t + ' dp', L.dp.length, 'added'], [t + ' dm', L.dm.length, 'deleted']])),
      note('The multi-edge layer me: ', d.multi_edge_layer, '.'))));
  // matrix of the layers
  const ids = range(0, NODE_CAP - 1);
  const cells = [], at = new Map();
  const put = (r, c, o) => { const k = r + ',' + c; at.set(k, Object.assign(at.get(k) || { r, c }, o)); };
  const T = d.graph.tensors;
  const staleIds = new Set(Object.values(d.stale_m_under_dp).map((v) => v.m_id));
  const dmSet = new Set(T.ROAD.dm.map(([a, b]) => a + ',' + b));
  for (const [a, b, id] of T.ROAD.m) {
    const r = relById.get(id);
    if (dmSet.has(a + ',' + b)) put(a, b, { cls: '', x: true });
    else if (!staleIds.has(id)) put(a, b, { cls: 'on', text: r ? r.weight : '' });
  }
  for (const [a, b, id] of T.ROAD.dp) put(a, b, { cls: 'a', text: relById.get(id).weight, out: T.ROAD.m.some(([x, y]) => x === a && y === b) });
  for (const [a, b, id] of T.FERRY.m) put(a, b, { cls: 'b', text: relById.get(id).weight });
  for (const v of at.values()) cells.push(v);
  const mx = matrix({ rows: ids, cols: ids, cells, cs: 17, aria: 'Adjacency matrix of node ids: committed road entries, added entries, deleted marks and ferry entries, each with its weight.' });
  ap(sec, h('figure', null, h('div', { class: 'pane scroll' }, mx),
    legend([['q', 'on', 'ROAD m: committed (weight)'], ['q', 'a', 'ROAD dp: added since the last flush'], ['q', 'x', 'ROAD dm: deleted since the last flush (m still holds it)'], ['q', 'b', 'FERRY m'], ['q', 'out', 'dp shadows an older m value']]),
    h('figcaption', null, 'Row is the source node, column the destination. The live value of a pair is its dp entry if there is one, else its m entry unless dm marks it.')));
  return sec;
}

/* ===================== Part B ===================== */
function stepB1(s) {
  const sec = section(s), d = s.data, dr = s.draw;
  const special = new Set([...dr.edges.masked_out, ...dr.edges.pending_add, ...dr.edges.deleted_after_read].map(([a, b]) => key2(a, b)));
  const edges = dr.edges.skeleton.filter(([a, b]) => !special.has(key2(a, b))).map(([a, b]) => ({ a, b, cls: 'e' }));
  for (const [a, b] of dr.edges.masked_out) edges.push({ a, b, cls: 'e del', z: 1 });
  for (const [a, b] of dr.edges.pending_add) edges.push({ a, b, cls: 'e add', z: 1 });
  for (const [a, b] of dr.edges.deleted_after_read) edges.push({ a, b, cls: 'e late', z: 1 });
  const s0 = new Set(dr.nodes.slab0), s1 = new Set(dr.nodes.slab1);
  const nodes = ids0().map((n) => ({ id: n, cls: s0.has(n) ? 't0' : s1.has(n) ? 't1' : 'unused' }));
  const g = graph({ nodes, edges, aria: 'The skeleton S read in two slabs of node ids; the pending deletion is masked out, the pending addition is read, and a road deleted after its slab was read is still present.' });
  const sl = d.slabs;
  ap(sec, h('div', { class: 'two' },
    wrapFig(g, 'Each node is shaded by the slab its row belongs to. A slab reads its rows at the version current when it runs.'),
    h('div', { style: 'display:grid;gap:12px;align-content:start' },
      legend([['n', 't0', 'rows ' + fpair(sl[0].rows) + ', read at v' + sl[0].version], ['n', 't1', 'rows ' + fpair(sl[1].rows) + ', read at v' + sl[1].version],
        ['l', '', 'live skeleton edge'], ['l', 'add', 'pending addition, read from dp'], ['l', 'del', 'masked out: pending deletion in dm'], ['l', 'late', 'deleted after its rows were read']]),
      stats([['entries in S', d.nnz], ['without the dm mask', d.S_without_dm_mask.length]]))));
  const rows = [];
  for (const sb of sl) for (const [t, x] of Object.entries(sb.per_type)) rows.push([fpair(sb.rows), 'v' + sb.version, t, x.m.length, x.dm.length, x.dp.length, fpairs(x.masked_out.filter(([a, b]) => a < b)) , x.live.length]);
  ap(sec, table(['rows', 'read at', 'type', 'm', 'dm', 'dp', 'masked out', 'live'], rows));
  const ids = range(0, NODE_CAP - 1);
  const S = new Set(d.S.map(([a, b]) => a + ',' + b));
  const late = new Set(dr.edges.deleted_after_read.flatMap(([a, b]) => [a + ',' + b, b + ',' + a]));
  const cells = d.S.map(([a, b]) => ({ r: a, c: b, cls: 'on', out: late.has(a + ',' + b) }));
  for (const [a, b] of d.S_without_dm_mask) if (!S.has(a + ',' + b)) cells.push({ r: a, c: b, x: true });
  ap(sec, h('figure', null, h('div', { class: 'pane scroll' }, matrix({ rows: ids, cols: ids, cells, rowRules: [sl[1].rows[0]], aria: 'The pattern of S over node ids, with the masked entries crossed out and the late deletion outlined.' })),
    legend([['q', 'on', 'entry of S'], ['q', 'x', 'dropped by the dm mask'], ['q', 'out', 'deleted after its rows were read; still in S']]),
    h('figcaption', null, 'S = select(OFFDIAG, eWiseAdd(ONEB, P, Pᵀ)) is symmetric. The heavy rule separates the two slabs.')));
  return sec;
}
const ids0 = () => range(0, NODE_CAP - 1);

function stepB2(s) {
  const sec = section(s), d = s.data, dr = s.draw;
  const nr = new Set(dr.nodes.no_rank);
  const nodes = ids0().map((n) => ({ id: n, cls: nr.has(n) ? (D.example.graph_v5.free_ids.includes(n) ? 'unused' : 'iso') : '', sub: dr.labels[n] != null ? 'd ' + dr.labels[n] : null }));
  const g = graph({ nodes, edges: dr.edges.skeleton.map(([a, b]) => ({ a, b, cls: 'e' })), aria: 'Each node labelled with its degree in S; the nodes with no entry get no rank.' });
  const dm = toMap(d.d);
  ap(sec, h('div', { class: 'two' }, wrapFig(g, 'The small number in each node is its degree d.'),
    h('div', { style: 'display:grid;gap:12px;align-content:start' },
      arrays(strip('d[node]', ids0().map((n) => ({ ix: n, v: dm.has(n) ? dm.get(n) : '—', cls: dm.has(n) ? '' : 'iso' }))),
        strip('pool ranks', d.pool_ranks.map((r) => ({ ix: 'r' + r, v: 'none', cls: 'iso' })))),
      stats([['ranked nodes', d.ranked], ['reserve', d.reserve, 'pool ranks'], ['n_r', d.n_r, 'ranked + reserve']]),
      note('A pool rank is created with no holder (03 §2, P). Taking one later never strips a node.'))));
  return sec;
}

function stepB3(s) {
  const sec = section(s), d = s.data, dr = s.draw;
  const R = d.rounds;
  ap(sec, stepper('st-b3', {
    title: 'Peeling, one round at a time', n: R.length, label: (i) => 'round ' + R[i].round + ' of ' + R.length,
    legend: [['n', 'cur', 'in the frontier: degree 1, peeled now'], ['n', 't3', 'peeled earlier'], ['n', '', 'core']],
    render(i, pics, panel) {
      const rk = new Map();
      for (let k = 0; k <= i; k++) for (const [n, r] of R[k].ranks) rk.set(n, r);
      const fr = new Set(R[i].frontier);
      const nodes = [...dr.nodes.core, ...dr.nodes.peeled].map((n) => ({ id: n, cls: fr.has(n) ? 'cur' : rk.has(n) ? 't3' : '', sub: rk.has(n) ? 'r' + rk.get(n) : null }));
      const edges = dr.edges.skeleton.map(([a, b]) => ({ a, b, cls: fr.has(a) || fr.has(b) ? 'e hot' : 'e', z: fr.has(a) || fr.has(b) ? 1 : 0 }));
      ap(pics, wrapFig(graph({ nodes, edges, aria: 'Peeling round ' + R[i].round + ': the frontier node is highlighted.' })));
      const r = R[i];
      ap(panel, h('h3', null, 'Round ' + r.round),
        blk('frontier f', P(C(flist(r.frontier)), ' has degree 1. Each takes the next rank above the pool, in (round, id) order.')),
        blk('ranks given', h('ul', null, r.ranks.map(([n, k]) => h('li', null, 'node ' + n + ' → rank ' + k)))),
        blk('t = S PLUS_PAIR f', P('Frontier neighbours per remaining node: ', C(r.t.map(([n, c]) => n + ': ' + c).join(', ') || '∅'))),
        blk('d after', P(C(r.d_after.map(([n, c]) => 'd[' + n + '] = ' + c).join(', ') || '∅'))),
        blk('next frontier', P(r.next_frontier.length ? C(flist(r.next_frontier)) : 'empty, so peeling stops.')));
    },
  }));
  ap(sec, note('Peeled ranks: ', C(d.peeled_ranks.map(([n, r]) => 'node ' + n + ' → ' + r).join(', ')), '. The core left for nested dissection: ', C(flist(d.core)), '.'));
  return sec;
}

function stepB4(s) {
  const sec = section(s), d = s.data, dr = s.draw;
  const pr = toMap(ST.B3.data.peeled_ranks);
  const comp = new Set(dr.nodes.component);
  const nodes = [...dr.nodes.component, ...dr.nodes.peeled].map((n) => ({ id: n, cls: comp.has(n) ? 't0' : 't3 gone', sub: pr.has(n) ? 'r' + pr.get(n) : null }));
  const ce = new Set(dr.edges.skeleton.map(([a, b]) => key2(a, b)));
  const edges = SKEL.map(([a, b]) => ({ a, b, cls: ce.has(key2(a, b)) ? 'e' : 'e faint' }));
  const cell = d.cells[0];
  ap(sec, h('div', { class: 'two' },
    wrapFig(graph({ nodes, edges, aria: 'The single component left after peeling.' }), 'Faded nodes were peeled in B3 and already hold their ranks.'),
    h('div', { style: 'display:grid;gap:12px;align-content:start' },
      arrays(strip('component[node]', d.component.map(([n, c]) => ({ ix: n, v: c })))),
      stats(d.cells.map((c) => ['component ' + c.label, c.verts.length + ' nodes', 'ranks ' + c.lo + '–' + c.hi])),
      legend([['n', 't0', 'in the component'], ['n', 't3', 'peeled']]))));
  return sec;
}

/* matching: one stepper per coarsening level */
function matchingStepper(key, dat, o) {
  const members = memOf(dat), w = wOf(dat);
  const fine = [...new Set([].concat(...dat.members.map(([, m]) => m)))];
  const pre = dat.level === 0 && o.coarsePrefix !== true ? '' : 'c';
  const vname = (v) => pre + v;
  const capped = new Set((dat.capped_out || []).map(([a, b]) => key2(a, b)));
  const tones = greedyTones(dat.verts, dat.A);
  const R = dat.rounds;
  const n = R.length + 1;
  const cellTone = o.cellTone ? toMap(dat.cell) : null;
  const nextDat = dat.next;
  return stepper(key, {
    title: o.title, n,
    label: (i) => (i < R.length ? 'round ' + R[i].round + ' of ' + R.length : 'merge the pairs'),
    legend: [['l', 'mt', 'mutual pair: matched this round'], ['l', 'mt-old', 'matched in an earlier round'], ['l', 'prop', 'proposal to the heaviest allowed neighbour'], ['l', 'capped', 'blocked by the weight cap'], ['n', 'cur', 'matched this round'], ['n', 'gone', 'already matched']],
    render(i, pics, panel) {
      if (i < R.length) {
        const r = R[i];
        const before = new Set(), oldPairs = [];
        for (let k = 0; k < i; k++) for (const [a, b] of R[k].mutual) { before.add(a); before.add(b); oldPairs.push([a, b]); }
        const now = new Set(r.mutual.flat());
        const prop = toMap(r.proposal), q = toMap(r.q);
        const edges = [];
        for (const [a, b] of oldPairs) edges.push({ a, b, cls: 'e mt-old', z: 2, straight: true });
        for (const [a, b] of r.mutual) edges.push({ a, b, cls: 'e mt', z: 3, straight: true });
        for (const [a, b] of prop) if (!now.has(a)) edges.push({ a, b, cls: 'e prop', arrow: true, z: 4, straight: true, trim: [12, 21] });
        const g = coarseView({
          verts: dat.verts, members, fine, A: dat.A, ck: key,
          ecls: (a, b) => (capped.has(key2(a, b)) ? 'e capped' : 'e faint'), ez: (a, b) => (capped.has(key2(a, b)) ? 1 : 0),
          tone: (v) => tones.get(v), edges,
          vcls: (v) => (now.has(v) ? 'cur' : before.has(v) ? 'gone' : cellTone ? (cellTone.get(v) ? 't1' : 't0') : ''),
          vsub: (v) => (pre ? 'w' + w.get(v) : null),
          aria: o.title + ', round ' + r.round + ': proposals and mutual pairs.',
        });
        ap(pics, wrapFig(g));
        const um = r.unmatched;
        ap(panel, h('h3', null, 'Round ' + r.round),
          i === 0 ? blk('the cap', P('A pair is allowed only if w(u) + w(v) ≤ cap = 1.5·tvwgt/CoarsenTo: ', C(dat.cap.map(([c, x]) => (dat.cap.length > 1 ? 'cell ' + c + ': ' : '') + fnum(x)).join(', ')), '. ',
            dat.capped_out && dat.capped_out.length ? ['It blocks ', B(dat.capped_out.length), ' candidate pairs: ', C(dat.capped_out.map((p) => p.map(vname).join('–')).join(', ')), '.'] : 'It blocks no pair here.')) : null,
          blk('who proposes to whom',
            r.proposal.length ? arrays(
              strip('unmatched u', um.map((v) => ({ v: vname(v) }))),
              strip('p(u) = target(y)', um.map((v) => ({ ix: vname(v), v: prop.has(v) ? vname(prop.get(v)) : '—', cls: prop.has(v) ? '' : 'empty' }))),
              strip('q(u) = p(p(u))', um.map((v) => ({ ix: vname(v), v: q.has(v) ? vname(q.get(v)) : '—', cls: q.has(v) && q.get(v) === v ? 'hl' : q.has(v) ? '' : 'empty' })))) :
              P('Unmatched: ', C(flist(um.map(vname))), '. None of them has an unmatched neighbour it may pair with, so no one proposes.')),
          blk('result', P(r.mutual.length ? ['Mutual where q(u) = u: ', B(r.mutual.map((p) => p.map(vname).join('–')).join(', ')), '.'] : 'No mutual pair. The matching stops after ' + dat.matched_rounds + ' matching rounds' + (dat.round_cap_hit ? ', at the round cap.' : ', under the round cap of ' + thr('matching round cap').toy + '.'))));
      } else {
        const nm = memOf(nextDat), nw = wOf(nextDat);
        const g = coarseView({
          verts: nextDat.verts, members: nm, fine, A: nextDat.A, ck: key + 'n', tone: (v) => greedyTones(nextDat.verts, nextDat.A).get(v),
          vsub: (v) => 'w' + nw.get(v), aria: 'The next, coarser graph after merging the pairs.',
        });
        ap(pics, wrapFig(g, 'The coarser graph: each blob is one new vertex, w is its weight, tags are summed edge weights.'));
        const pairs = dat.pairs;
        const th = dat.two_hop;
        const cellsN = toMap(dat.cell);
        ap(panel, h('h3', null, 'Merge the pairs'),
          blk('pairs', P(C(pairs.map((p) => p.map(vname).join('–')).join(', ')), '. Unmatched vertices carry over alone.')),
          blk('Cᵀ: vertex → coarse id', arrays(strip('Cᵀ', dat.CT.map(([v, c]) => ({ ix: vname(v), v: 'c' + c }))))),
          blk('new weights w = (Cᵀ)ᵀ · w', arrays(strip('w', nextDat.w.map(([v, x]) => ({ ix: 'c' + v, v: x }))))),
          dat.internal_weight_dropped ? blk('dropped by OFFDIAG', P('Edges inside a merged pair land on the diagonal of (Cᵀ)ᵀ·A·Cᵀ and are dropped: ', C(dat.internal_weight_dropped.map(([a, , x]) => 'c' + a + ': ' + x).join(', ')), '.')) : null,
          blk('shrink and the 2-hop pass', h('ul', null, th.map((t) => h('li', null, (th.length > 1 ? 'cell ' + t.cell + ': ' : '') + t.before + ' → ' + t.after_matching + ' vertices, shrink ' + pct(t.shrink) + '. 2-hop pass: ' + (t.fires ? 'fires' : 'does not fire'))))),
          dat.stopped && dat.stopped.length ? blk('coarsening stops', h('ul', null, dat.stopped.map((x) => h('li', null, (x.cell != null && cellsN.size ? 'cell ' + x.cell + ': ' : '') + x.why)))) : null);
        const A1 = dat.A, A2 = nextDat.A;
        const mxA = (A, vs, pfx, aria) => matrix({ rows: vs.map((v) => pfx + v), cols: vs.map((v) => pfx + v), cells: A.map(([a, b, x]) => ({ r: vs.indexOf(a), c: vs.indexOf(b), cls: 'on', text: x })), cs: 18, lw: 28, aria });
        ap(pics, h('div', { class: 'two' },
          h('figure', null, h('div', { class: 'pane scroll' }, mxA(A1, dat.verts, pre, 'A at this level, with edge weights')), h('figcaption', null, 'A at this level')),
          h('figure', null, h('div', { class: 'pane scroll' }, mxA(A2, nextDat.verts, 'c', 'A at the next level, with summed edge weights')), h('figcaption', null, 'select(OFFDIAG, (Cᵀ)ᵀ·A·Cᵀ)'))));
      }
    },
  });
}
function stepB5(s) {
  const sec = section(s), d = s.data;
  ap(sec, matchingStepper('st-' + sid(s.id), d, { title: 'Matching rounds, then the merge' }));
  return sec;
}
function stepB5h(s) {
  const sec = section(s), d = s.data;
  const coords = d.coords;
  const th = d.two_hop[0];
  const anchors = toMap(th.anchors || []);
  const hubs = new Set(anchors.values());
  const frames = ['round', 'stall', 'anchor', 'pairs', 'result'];
  const gstar = (o) => graph(Object.assign({ coords, sx: 120, sy: 105, px: 40, py: 36, cacheKey: 'star' }, o));
  ap(sec, stepper('st-b5h', {
    title: 'Side example: the 2-hop pass on a star', n: frames.length,
    label: (i) => ['round ' + d.rounds[0].round, 'round ' + d.rounds[1].round, 'anchors', '2-hop pairs', 'result'][i],
    legend: [['l', 'mt', 'mutual pair'], ['l', 'hop', 'pair made by the 2-hop pass'], ['l', 'prop', 'proposal or anchor'], ['n', 'cur', 'matched'], ['n', 't3', 'unmatched, grouped under one anchor']],
    render(i, pics, panel) {
      const r1 = d.rounds[0];
      const base = d.A.filter(([a, b]) => a < b).map(([a, b]) => ({ a, b, cls: 'e faint' }));
      const mutual = new Set(r1.mutual.flat());
      const hop = new Set(th.pairs.flat());
      let nodes, edges = base.slice();
      if (frames[i] === 'round') {
        nodes = d.verts.map((v) => ({ id: v, cls: mutual.has(v) ? 'cur' : '' }));
        for (const [a, b] of r1.proposal) if (!(mutual.has(a) && mutual.has(b))) edges.push({ a, b, cls: 'e prop', arrow: true, z: 2, straight: true });
        edges.push({ a: r1.mutual[0][0], b: r1.mutual[0][1], cls: 'e mt', z: 3 });
        ap(panel, h('h3', null, 'Round ' + r1.round), blk('proposals', P('Every leaf has one neighbour, the hub, so every leaf proposes to it. The hub proposes to its heaviest neighbour by key, ', B(fm(toMap(r1.proposal).get(0))), '.')),
          blk('result', P('Only the hub and that leaf are mutual: ', B(fpairs(r1.mutual)), '.')));
      } else if (frames[i] === 'stall') {
        nodes = d.verts.map((v) => ({ id: v, cls: mutual.has(v) ? 'cur' : '' }));
        edges.push({ a: r1.mutual[0][0], b: r1.mutual[0][1], cls: 'e mt-old', z: 3 });
        ap(panel, h('h3', null, 'Round ' + d.rounds[1].round), blk('no proposals', P('The unmatched leaves ', C(flist(d.rounds[1].unmatched)), ' have no unmatched neighbour. The matching stops.')),
          blk('shrink', P(th.before + ' → ' + th.after_matching + ' vertices is a shrink of ' + pct(th.shrink) + ', under the trigger (' + thr('2-hop pass trigger').toy + '), so the 2-hop pass runs.')));
      } else if (frames[i] === 'anchor') {
        nodes = d.verts.map((v) => ({ id: v, cls: hubs.has(v) ? 'cur' : mutual.has(v) ? 'cur gone' : anchors.has(v) ? 't3' : '' }));
        for (const [v, a] of anchors) edges.push({ a: v, b: a, cls: 'e prop', arrow: true, z: 2, straight: true });
        ap(panel, h('h3', null, 'Anchors'), blk('a⟨unmatched⟩ = K MAX_FIRST 1', P('Each unmatched vertex takes its heaviest neighbour as its anchor: ', C([...anchors].map(([v, a]) => v + ' → ' + a).join(', ')), '.')),
          blk('groups', th.groups.map((g) => P('Anchor ', B(g.anchor), ': members in scramble order ', C(flist(g.members_in_scramble_order)), '.'))));
      } else if (frames[i] === 'pairs') {
        nodes = d.verts.map((v) => ({ id: v, cls: hubs.has(v) ? 'cur' : mutual.has(v) ? 'cur gone' : hop.has(v) ? 'cur' : anchors.has(v) ? 't3' : '' }));
        edges.push({ a: r1.mutual[0][0], b: r1.mutual[0][1], cls: 'e mt-old', z: 2 });
        for (const [a, b] of th.pairs) edges.push({ a, b, cls: 'e hop', z: 3, bow: 0.25 });
        const left = (th.unmatched || []).filter((v) => !hop.has(v));
        ap(panel, h('h3', null, '2-hop pairs'), blk('pair neighbours in a row', P('Within the group, neighbours in scramble order pair up while w(u) + w(v) ≤ cap = ', C(fnum(d.cap[0][1])), ': ', B(fpairs(th.pairs)), '.')),
          blk('left over', P(left.length ? ['', C(flist(left)), ' has no partner left and carries over alone.'] : 'Nothing.')));
      } else {
        const nm = memOf(d.next), nw = wOf(d.next);
        const grp = new Map();
        for (const [c, ms] of nm) for (const m of ms) grp.set(m, c);
        const T = ['t0', 't1', 't2', 't3', 't4'];
        nodes = d.verts.map((v) => ({ id: v, cls: T[grp.get(v) % T.length], sub: 'c' + grp.get(v) }));
        edges.push({ a: r1.mutual[0][0], b: r1.mutual[0][1], cls: 'e mt', z: 2 });
        for (const [a, b] of th.pairs) edges.push({ a, b, cls: 'e hop', z: 3, bow: 0.25 });
        const cA = d.next.A.filter(([a, b]) => a < b);
        ap(panel, h('h3', null, 'Result'),
          blk('merged', P(d.verts.length + ' → ' + d.next.verts.length + ' vertices. Each colour is one new vertex; the small label names it.')),
          blk('the coarse graph', mini(['vertex', 'members', 'weight'], d.next.verts.map((v) => ['c' + v, flist(nm.get(v)), nw.get(v)])),
            P('Edges, with summed weights: ', C(cA.map(([a, b, x]) => 'c' + a + '–c' + b + ' (' + x + ')').join(', ')), '.')));
      }
      ap(pics, wrapFig(gstar({ nodes, edges, aria: 'A star: one hub and six leaves.' })));
    },
  }));
  const forced = d.forced_on_running_example;
  ap(sec, h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'On the running example'),
    P('The pass fires on no level: ', d.fires_on_running_example.length ? C(JSON.stringify(d.fires_on_running_example)) : 'none', '. The script also tried salt bases ', d.salt_scan.bases[0], ' to ', d.salt_scan.bases[1], ', and it fired under ', d.salt_scan.fired.length, ' of them.'),
    P('Forced on every level, it pairs no vertex:'),
    table(['depth', 'level', 'cell', 'unmatched', 'anchors', 'pairs'], forced.map((f) => [f.depth, f.level, f.cell, flist(f.unmatched), f.anchors.map(([v, a]) => v + '→' + a).join(', ') || '—', fpairs(f.pairs)]))));
  return sec;
}

function stepB6(s) {
  const sec = section(s), d = s.data, top = d.coarsest;
  const members = memOf(top), w = wOf(top);
  const fine = [...new Set([].concat(...top.members.map(([, m]) => m)))];
  const vname = (v) => 'c' + v;
  const frames = [];
  d.trials.forEach((t) => {
    const grown = new Set(t.grown_side_0);
    const g0 = new Map(top.verts.map((v) => [v, grown.has(v) ? 0 : 1]));
    frames.push({ kind: 'grow', t, labels: g0 });
    const sep = new Set(t.separator_before_fm);
    const start = new Map(top.verts.map((v) => [v, sep.has(v) ? 2 : grown.has(v) ? 0 : 1]));
    frames.push({ kind: 'sep', t, labels: start });
    const rp = fmReplay(start, t.fm_passes, { t, vname });
    frames.push(...rp.frames);
    frames.push({ kind: 'trialend', t, labels: toMap(t.labels) });
  });
  frames.push({ kind: 'best', labels: toMap(d.labels) });
  ap(sec, stepper('st-b6', {
    title: 'Trials and node FM on the coarsest graph', n: frames.length,
    label: (i) => (frames[i].kind === 'best' ? 'best trial' : 'trial ' + frames[i].t.trial + ' · ' + { grow: 'grow', sep: 'separator', move: 'FM move', passend: 'pass end', trialend: 'result' }[frames[i].kind]),
    legend: sepLegend,
    render(i, pics, panel) {
      const f = frames[i], lab = f.labels;
      const moved = f.kind === 'move' ? f.mv.move : null, pulled = new Set(f.kind === 'move' ? f.mv.pulled_into_separator : []);
      const bnd = f.kind === 'grow' ? new Set([...f.t.boundary_0, ...f.t.boundary_1]) : new Set();
      const g = coarseView({
        verts: top.verts, members, fine, A: top.A, ck: 'b6', tone: (v) => SIDE_BLOB[lab.get(v)],
        vcls: (v) => SIDE_TONE[lab.get(v)] + (v === moved ? ' ring' : pulled.has(v) ? ' ring2' : bnd.has(v) ? ' ring' : ''),
        vsub: (v) => 'w' + w.get(v), aria: 'The coarsest graph coloured by side.',
      });
      ap(pics, wrapFig(g));
      const wt = sideWeights(lab, w);
      const head = f.kind === 'best' ? h('h3', null, 'Best trial: ' + d.best) : h('h3', null, 'Trial ' + f.t.trial + ', seed ' + vname(f.t.seed));
      ap(panel, head);
      if (f.kind === 'grow') ap(panel, blk('greedy growing', P('Side 0 grows from the seed until it holds half the weight: ', C(flist(f.t.grown_side_0.map(vname))), '.')),
        blk('boundaries', P('Side 0 boundary ', C(flist(f.t.boundary_0.map(vname))), ', side 1 boundary ', C(flist(f.t.boundary_1.map(vname))), ' (ringed).')));
      if (f.kind === 'sep') ap(panel, blk('separator before FM', P('The lighter boundary becomes the separator: ', C(flist(f.t.separator_before_fm.map(vname))), '.')));
      if (f.kind === 'move') ap(panel, ...moveBlock(Object.assign({ vname }, f), d.balance_limit));
      if (f.kind === 'passend') ap(panel, passEndBlock(f));
      if (f.kind === 'trialend') ap(panel, blk('trial result', P('Weights (side 0, side 1, separator): ', C(flist(f.t.weights)), '. Score (balance kept first, then separator weight, then imbalance): ', C(flist(f.t.score)), '.')));
      if (f.kind === 'best') {
        ap(panel, blk('all trials', mini(['trial', 'seed', 'weights', 'score'], d.trials.map((t) => [t.trial, vname(t.seed), flist(t.weights), flist(t.score)]))),
          blk('why one side is empty', P('The coarsest graph is complete: every coarse vertex touches every other. Any separator that leaves two sides non-empty would cut an edge between them. Uncoarsening moves weight back (B7).')));
        const dr = s.draw, s0 = new Set(dr.nodes.side0), s1 = new Set(dr.nodes.side1);
        ap(pics, wrapFig(graph({ nodes: range(0, 15).map((n) => ({ id: n, cls: s0.has(n) ? 't2' : s1.has(n) ? 't1' : 'sep' })), edges: ST.B4.draw.edges.skeleton.map(([a, b]) => ({ a, b, cls: 'e' })), aria: 'The best trial projected onto the fine nodes.' }), 'The same labels on the fine nodes.'));
      }
      ap(panel, labelStrip(lab, vname), mini(['side 0', 'side 1', 'separator'], [wt]));
    },
  }));
  return sec;
}

function stepB7(s) {
  const sec = section(s), d = s.data;
  const frames = [];
  for (const L of d.levels) {
    const dat = DEPTH0.get(L.level);
    const ctx = { L, dat, vname: (v) => (L.level === 0 ? String(v) : 'c' + v) };
    frames.push(Object.assign({ kind: 'proj', labels: toMap(L.projected) }, ctx));
    let lab = toMap(L.projected);
    for (const c of L.cells) {
      const rp = fmReplay(lab, c.fm_passes, Object.assign({ cell: c }, ctx));
      frames.push(...rp.frames);
      lab = rp.labels;
    }
    frames.push(Object.assign({ kind: 'levelend', labels: toMap(L.labels) }, ctx));
  }
  ap(sec, stepper('st-b7', {
    title: 'Uncoarsening: project, then refine from the separator', n: frames.length,
    label: (i) => 'level ' + frames[i].L.level + ' · ' + { proj: 'project', move: 'FM move', passend: 'pass end', levelend: 'result' }[frames[i].kind],
    legend: sepLegend,
    render(i, pics, panel) {
      const f = frames[i], dat = f.dat, lab = f.labels;
      const members = memOf(dat), w = wOf(dat);
      const fine = range(0, 15);
      const moved = f.kind === 'move' ? f.mv.move : null, pulled = new Set(f.kind === 'move' ? f.mv.pulled_into_separator : []);
      const bnd = f.kind === 'proj' ? new Set(f.L.cells.flatMap((c) => c.boundary)) : new Set();
      ap(pics, wrapFig(coarseView({
        verts: dat.verts, members, fine, A: dat.A, ck: 'b7-' + f.L.level, tone: (v) => SIDE_BLOB[lab.get(v)],
        vcls: (v) => SIDE_TONE[lab.get(v)] + (v === moved ? ' ring' : pulled.has(v) ? ' ring2' : bnd.has(v) ? ' ring' : ''),
        vsub: (v) => (f.L.level === 0 ? null : 'w' + w.get(v)), aria: 'Level ' + f.L.level + ' coloured by side.',
      })));
      ap(panel, h('h3', null, 'Level ' + f.L.level));
      if (f.kind === 'proj') {
        const c = f.L.cells[0];
        ap(panel, blk('labels_ℓ = Cᵀ_ℓ · labels_ℓ₊₁', P('Each vertex takes the label of the coarse vertex it was merged into.')),
          blk('boundary (ringed)', P('Node FM starts from the separator vertices only: ', C(flist(c.boundary.map(f.vname))), '.')),
          blk('neighbour weight per side', P('Each value is the summed vertex weight w of the vertex’s neighbours on that side. Moving a vertex to one side pulls its neighbours on the other side into the separator, so the move’s gain is its own w minus the other side’s value.'), arrays(
            strip('n_0 (side 0)', c.side_0_neighbour_weight.map(([v, x]) => ({ ix: f.vname(v), v: x }))),
            strip('n_1 (side 1)', c.side_1_neighbour_weight.map(([v, x]) => ({ ix: f.vname(v), v: x }))))));
      }
      if (f.kind === 'move') ap(panel, ...moveBlock(f, ST.B6.data.balance_limit));
      if (f.kind === 'passend') ap(panel, passEndBlock(f));
      if (f.kind === 'levelend') ap(panel, blk('level result', P('Weights (side 0, side 1, separator) before: ', C(flist(f.L.cells[0].weights_before)), ', after: ', C(flist(f.L.cells[0].weights_after)), '.')));
      ap(panel, labelStrip(lab, f.vname), mini(['side 0', 'side 1', 'separator'], [sideWeights(lab, w)]));
    },
  }));
  return sec;
}

function stepB8(s) {
  const sec = section(s), d = s.data, dr = s.draw;
  const c0 = new Set(dr.nodes.cell_C0), c1 = new Set(dr.nodes.cell_C1);
  const nodes = range(0, 15).map((n) => ({ id: n, cls: c0.has(n) ? 't2' : c1.has(n) ? 't1' : 'sep', sub: dr.labels[n] || null }));
  const edges = [...dr.edges.kept.map(([a, b]) => ({ a, b, cls: 'e', z: 1 })), ...dr.edges.dropped.map(([a, b]) => ({ a, b, cls: 'e drop' }))];
  const sepr = d.separators[0];
  ap(sec, h('div', { class: 'two' },
    wrapFig(graph({ nodes, edges, aria: 'The split: two cells, the separator with its top ranks, kept and dropped edges.' }), 'Separator nodes are the solid ones (see the key), with their new rank.'),
    h('div', { style: 'display:grid;gap:12px;align-content:start' },
      legend([['n', 't2', d.cells[0].name + ': ranks ' + d.cells[0].lo + '–' + d.cells[0].hi], ['n', 't1', d.cells[1].name + ': ranks ' + d.cells[1].lo + '–' + d.cells[1].hi], ['n', 'sep', 'separator'], ['l', '', 'kept: both ends in one cell'], ['l', 'drop', 'dropped: touches the separator']]),
      arrays(strip('separator rank', sepr.separator_ranks.map(([n, r]) => ({ ix: 'n' + n, v: r, cls: 'ink' })))),
      stats([['entries dropped', d.removed.length], ['entries kept', d.A_next.length]]))));
  const vs = range(0, 15);
  const kept = new Set(d.A_next.map(([a, b]) => a + ',' + b));
  const A0 = DEPTH0.get(0).A;
  const cells = A0.map(([a, b]) => ({ r: a, c: b, cls: kept.has(a + ',' + b) ? 'on' : '', x: !kept.has(a + ',' + b) }));
  const Mrows = d.M.map(([n]) => n);
  ap(sec, h('div', { class: 'two' },
    h('figure', null, h('div', { class: 'pane scroll' }, matrix({ rows: Mrows.map(String), cols: d.cells.map((c) => c.name), cells: d.M.map(([n, c], i) => ({ r: i, c, cls: 'a' })), cs: 18, lw: 28, aria: 'The cell-membership matrix M' })),
      h('figcaption', null, 'M: one row per non-separator node, a 1 in its new cell. Separator nodes get no row.')),
    h('figure', null, h('div', { class: 'pane scroll' }, matrix({ rows: vs, cols: vs, cells, aria: 'A_0 masked by M ANY_PAIR Mᵀ' })),
      legend([['q', 'on', 'kept in A_1'], ['q', 'x', 'dropped']]),
      h('figcaption', null, 'A_1⟨A_0, struct⟩ = M ANY_PAIR Mᵀ keeps an entry only when both ends share a cell.'))));
  return sec;
}

function stepB91(s) {
  const sec = section(s), d = s.data;
  const L0 = d.levels[0];
  const cellOf = toMap(L0.cell);
  const order = [...L0.verts].sort((a, b) => cellOf.get(a) - cellOf.get(b) || a - b);
  const inA = new Set(L0.A.map(([a, b]) => a + ',' + b));
  const blkCells = [];
  for (const a of order) for (const b of order) if (cellOf.get(a) === cellOf.get(b)) blkCells.push({ r: order.indexOf(a), c: order.indexOf(b), cls: inA.has(a + ',' + b) ? (cellOf.get(a) ? 'pl' : 'b') : cellOf.get(a) ? 'blk1' : 'bs2' });
  const nC0 = order.filter((v) => cellOf.get(v) === 0).length;
  ap(sec, h('figure', null, h('div', { class: 'pane scroll' }, matrix({ rows: order, cols: order, cells: blkCells, rowRules: [nC0], colRules: [nC0], aria: 'The matrix at depth ' + d.depth + ': one block per cell.' })),
    legend([['q', 'b', 'edge inside ' + d.cells[0].name], ['q', 'pl', 'edge inside ' + d.cells[1].name], ['q', 'bs2', 'block of ' + d.cells[0].name], ['q', 'p', 'block of ' + d.cells[1].name]]),
    h('figcaption', null, 'Both cells of this depth live in one matrix. Rows and columns are ordered by cell here, so the blocks show; nothing connects them, so one call handles every cell.')));
  // stepper frames
  const frames = [];
  L0.rounds.forEach((r, k) => frames.push({ kind: 'round', k }));
  frames.push({ kind: 'contract' }, { kind: 'sep' }, { kind: 'unc' }, { kind: 'split' });
  const top = d.coarsest, tm = memOf(top), tw = wOf(top);
  const fine = L0.verts;
  const sepLab = new Map();
  for (const sp of d.separators) for (const [v, l] of Object.entries(sp.labels)) sepLab.set(+v, l);
  ap(sec, stepper('st-b91', {
    title: 'Depth ' + d.depth + ': every cell at once', n: frames.length,
    label: (i) => ({ round: 'round ' + (frames[i].k + 1) + ' of ' + L0.rounds.length, contract: 'merge', sep: 'first separators', unc: 'uncoarsen', split: 'split' }[frames[i].kind]),
    legend: [['n', 't2', 'cell ' + d.cells[0].name], ['n', 't1', 'cell ' + d.cells[1].name], ['l', 'mt', 'mutual pair'], ['l', 'prop', 'proposal'], ['n', 't0', 'side 0'], ['n', 't3', 'side 1'], ['n', 'sep', 'separator']],
    render(i, pics, panel) {
      const f = frames[i];
      if (f.kind === 'round') {
        const r = L0.rounds[f.k];
        const before = new Set(), oldPairs = [];
        for (let k = 0; k < f.k; k++) for (const [a, b] of L0.rounds[k].mutual) { before.add(a); before.add(b); oldPairs.push([a, b]); }
        const now = new Set(r.mutual.flat()), prop = toMap(r.proposal), q = toMap(r.q);
        const edges = [...oldPairs.map(([a, b]) => ({ a, b, cls: 'e mt-old', z: 2 })), ...r.mutual.map(([a, b]) => ({ a, b, cls: 'e mt', z: 3 })),
          ...[...prop].filter(([a]) => !now.has(a)).map(([a, b]) => ({ a, b, cls: 'e prop', arrow: true, z: 4, straight: true, trim: [12, 21] }))];
        ap(pics, wrapFig(coarseView({ verts: L0.verts, members: memOf(L0), fine, A: L0.A, ck: 'b91', ecls: () => 'e faint', edges,
          vcls: (v) => (now.has(v) ? 'cur' : before.has(v) ? 'gone' : cellOf.get(v) ? 't1' : 't2'), aria: 'Depth ' + d.depth + ' matching round ' + r.round }), null));
        ap(panel, h('h3', null, 'Round ' + r.round),
          f.k === 0 ? blk('the cap, per cell', P(C(L0.cap.map(([c, x]) => 'cell ' + c + ': ' + fnum(x)).join(', ')), '. Each cell has its own total weight, so its own cap.')) : null,
          r.proposal.length ? blk('proposals', arrays(strip('p(u)', r.unmatched.map((v) => ({ ix: v, v: prop.has(v) ? prop.get(v) : '—', cls: prop.has(v) ? '' : 'empty' }))),
            strip('q(u) = p(p(u))', r.unmatched.map((v) => ({ ix: v, v: q.has(v) ? q.get(v) : '—', cls: q.get(v) === v ? 'hl' : '' }))))) : blk('no proposals', P('Left unmatched: ', C(flist(r.unmatched)), '.')),
          blk('mutual', P(r.mutual.length ? B(fpairs(r.mutual)) : 'none: the matching stops.')));
      } else if (f.kind === 'contract') {
        ap(pics, wrapFig(coarseView({ verts: top.verts, members: tm, fine, A: top.A, ck: 'b91c', tone: (v) => (toMap(top.cell).get(v) ? 't1' : 't2'), vsub: (v) => 'w' + tw.get(v), aria: 'The coarsest graph at depth ' + d.depth })));
        ap(panel, h('h3', null, 'Merge'), blk('Cᵀ', arrays(strip('Cᵀ', L0.CT.map(([v, c]) => ({ ix: v, v: 'c' + c }))))),
          blk('shrink', h('ul', null, L0.two_hop.map((t) => h('li', null, 'cell ' + t.cell + ': ' + t.before + ' → ' + t.after_matching + ', ' + pct(t.shrink) + (t.fires ? ', 2-hop fires' : ', no 2-hop'))))),
          blk('stop', h('ul', null, L0.stopped.map((x) => h('li', null, 'cell ' + x.cell + ': ' + x.why)))));
      } else if (f.kind === 'sep') {
        ap(pics, wrapFig(coarseView({ verts: top.verts, members: tm, fine, A: top.A, ck: 'b91c', tone: (v) => SIDE_BLOB91[sepLab.get(v)], vcls: (v) => SIDE_TONE91[sepLab.get(v)], vsub: (v) => 'w' + tw.get(v), aria: 'First separators of both cells' })));
        ap(panel, h('h3', null, 'First separators'), ...d.separators.map((sp) => blk('cell ' + sp.cell + ' (best trial ' + sp.best + ')',
          mini(['trial', 'seed', 'weights', 'score'], sp.trials.map((t) => [t.trial, 'c' + t.seed, flist(t.weights), flist(t.score)])))));
      } else if (f.kind === 'unc') {
        const U = d.uncoarsen[0], lab = toMap(U.labels);
        ap(pics, wrapFig(graph({ nodes: fine.map((n) => ({ id: n, cls: SIDE_TONE91[lab.get(n)] })), edges: ST.B8.draw.edges.kept.map(([a, b]) => ({ a, b, cls: 'e' })), aria: 'Labels at depth ' + d.depth + ' after node FM' })));
        ap(panel, h('h3', null, 'Uncoarsen to level ' + U.level), ...U.cells.map((c) => blk('cell ' + c.cell + ', boundary ' + flist(c.boundary),
          h('ul', null, c.fm_passes.filter((p) => p.moves.length).map((p, k) => h('li', null, 'pass ' + (k + 1) + ': ' + p.moves.map((m) => m.move + '→' + SIDE_NAME[m.to_side]).join(', ') + '; keeps ' + p.kept))),
          P('weights before ', C(flist(c.weights_before)), ', after ', C(flist(c.weights_after)), '.'))));
      } else {
        const sp = d.split, dr = s.draw;
        const tones = ['t2', 't1', 't0', 't3'];
        const cellTone = new Map();
        sp.cells.forEach((c, k) => { for (const v of c.verts) cellTone.set(v, tones[k % tones.length]); });
        const sr = new Map();
        for (const x of sp.separators) for (const [n, r] of x.separator_ranks) sr.set(n, r);
        const keptE = new Set(sp.A_next.map(([a, b]) => key2(a, b)));
        ap(pics, wrapFig(graph({ nodes: fine.map((n) => ({ id: n, cls: sr.has(n) ? 'sep' : cellTone.get(n), sub: sr.has(n) ? 'r' + sr.get(n) : null })),
          edges: ST.B8.draw.edges.kept.map(([a, b]) => ({ a, b, cls: keptE.has(key2(a, b)) ? 'e' : 'e drop' })), aria: 'The split at depth ' + d.depth + ' into ' + sp.cells.length + ' cells' })));
        ap(panel, h('h3', null, 'Split'), blk('separators', h('ul', null, sp.separators.map((x) => h('li', null, x.cell + ': ' + x.separator_ranks.map(([n, r]) => 'node ' + n + ' → rank ' + r).join(', '))))),
          blk('new cells', mini(['cell', 'nodes', 'ranks'], sp.cells.map((c) => [c.name, flist(c.verts), c.lo + '–' + c.hi])), legend(sp.cells.map((c, k) => ['n', tones[k % tones.length], c.name]))),
          blk('M ANY_PAIR Mᵀ', P(sp.removed.length + ' entries dropped, ' + sp.A_next.length + ' kept.')));
      }
    },
  }));
  return sec;
}

function stepB10(s) {
  const sec = section(s), d = s.data;
  const frames = [];
  for (const lf of d.leaves) lf.steps.forEach((st, k) => frames.push({ lf, st, k }));
  const sepRanks = new Map();
  for (const x of [...ST.B8.data.separators, ...ST['B9.1'].data.split.separators]) for (const [n, r] of x.separator_ranks) sepRanks.set(n, r);
  const tones = ['t2', 't1', 't0', 't3'];
  const toneOf = new Map();
  d.leaves.forEach((lf, k) => { for (const v of lf.verts) toneOf.set(v, tones[k % tones.length]); });
  ap(sec, stepper('st-b10', {
    title: 'Exact minimum degree inside each leaf', n: frames.length,
    label: (i) => frames[i].lf.cell + ' · ' + (frames[i].k + 1) + ' of ' + frames[i].lf.steps.length,
    legend: [['n', 'cur', 'eliminated now'], ['n', 'ring', 'its remaining neighbours in the cell'], ['n', 'sep', 'separator (ranked earlier)'], ['n', 'gone', 'eliminated earlier']],
    render(i, pics, panel) {
      const f = frames[i];
      const done = new Map();
      for (let k = 0; k < i; k++) done.set(frames[k].st.node, frames[k].st.rank);
      const nb = new Set(f.st.neighbours);
      const leafE = new Set(d.leaves.flatMap((l) => l.edges.map(([a, b]) => key2(a, b))));
      const nodes = range(0, 15).map((n) => ({
        id: n, cls: n === f.st.node ? 'cur' : sepRanks.has(n) ? 'sep' : (toneOf.get(n) || '') + (nb.has(n) ? ' ring' : '') + (done.has(n) ? ' gone' : ''),
        sub: n === f.st.node ? 'r' + f.st.rank : sepRanks.has(n) ? 'r' + sepRanks.get(n) : done.has(n) ? 'r' + done.get(n) : null,
      }));
      const edges = ST.B4.draw.edges.skeleton.map(([a, b]) => ({ a, b, cls: leafE.has(key2(a, b)) ? 'e' : 'e faint', z: leafE.has(key2(a, b)) ? 1 : 0 }));
      ap(pics, wrapFig(graph({ nodes, edges, aria: 'Leaf ' + f.lf.cell + ': node ' + f.st.node + ' is eliminated.' })));
      ap(panel, h('h3', null, 'Leaf ' + f.lf.cell + ' (ranks ' + f.lf.lo + '–' + f.lf.hi + ')'),
        blk('pick', P('Node ', B(f.st.node), ' has the fewest remaining neighbours in the cell: ', B(f.st.degree), f.st.neighbours.length ? [' (', C(flist(f.st.neighbours)), ')'] : '', '. Ties go to the lowest id.')),
        blk('fill', P(f.st.fill.length ? ['Joining its neighbours adds ', C(fpairs(f.st.fill)), '.'] : 'Fewer than two neighbours, so nothing to join.')),
        blk('rank', P('It takes rank ', B(f.st.rank), ', the next free one from the bottom of the cell’s range.')));
    },
  }));
  return sec;
}

function stepB11(s) {
  const sec = section(s), d = s.data, dr = s.draw;
  const q = toMap(d.q), moved = new Set([...q].filter(([a, b]) => a !== b).map(([a]) => a));
  const r2n = toMap(d.rank_to_node);
  const rnd = toMap(d.rank_nd), ndr = new Map([...rnd].map(([n, r]) => [r, n]));
  ap(sec, arrays(
    strip('rank before relabel', d.rank_nd.map(([n, r]) => ({ ix: 'n' + n, v: r, cls: moved.has(r) ? 'fill' : '' }))),
    strip('q: old → new', d.q.map(([a, b]) => ({ ix: a, v: b, cls: a !== b ? 'fill' : '' }))),
    strip('rank_to_node', d.rank_to_node.map(([r, n]) => ({ ix: r, v: n == null ? 'none' : n, cls: n == null ? 'iso' : '' })))));
  const par = new Map(d.etree_nd.map(([c, p]) => [c, p]));
  const t = tree({ parent: par, ranks: d.etree_nd.map(([c]) => c), cls: (r) => (ndr.has(r) ? (moved.has(r) ? 't3' : '') : 'iso'), ecls: (c) => (moved.has(c) ? 'te-moved' : 'te'), sub: (r) => (ndr.has(r) ? 'n' + ndr.get(r) : 'pool'), aria: 'Elimination tree before the relabel' });
  const nr = new Set(dr.nodes.no_rank);
  const g = graph({ nodes: ids0().map((n) => ({ id: n, cls: nr.has(n) ? 'iso' : '', sub: dr.labels[n] || null })), edges: dr.edges.skeleton.map(([a, b]) => ({ a, b, cls: 'e' })), aria: 'The final order: each node with its rank.' });
  ap(sec, h('div', { class: 'two' },
    h('figure', null, h('div', { class: 'pane scroll' }, t), legend([['n', 't3', 'rank moved by the postorder relabel'], ['n', 'iso', 'pool rank (no holder)']]), h('figcaption', null, 'Liu’s elimination tree over B, in the ranks before the relabel. The relabel numbers each subtree consecutively, children first.')),
    wrapFig(g, 'The final order: the small label is each node’s rank. Nodes with no edge get none.')));
  return sec;
}
