/* ===================== Design walkthrough: the page =====================
   The page walks through the design proposal: the objects it introduces, how the build makes them
   from the graph, and how they live across versions under MVCC. Every number is read from
   example.json (walkthrough.py's output). The step-by-step algorithm sits in the appendix. */
const MV = D.mvcc, TR = D.track;
const MCOLS = MV.columns;
const MCOL = new Map(MCOLS.map((c) => [c.key, c]));
const TOPO_F = MV.fields.topology, METRIC_F = MV.fields.metric;
const LINK_TEAM = 'https://claude.ai/artifact/1Xs7y3EE3kBMJU8woAv6dw';
const LINK_EXPLAINER = 'https://claude.ai/artifact/SEivswkaiNdpjVejgzdyzd';
const ext = (href, text) => h('a', { href, target: '_blank', rel: 'noopener' }, text);
const BUDGET = D.after_build_events.find((e) => e.budget != null).budget;
const CARD = (pred) => MV.cards.find(pred);
const LB = (k) => MCOL.get(k).label;                 /* a state's label, from the data */
const FIRST = MCOLS[0].label;                         /* the first Built state */
const CHUNK_TXT = () => 'Toy chunks hold ' + thr('metric chunk').toy + '; the engine’s hold ' + thr('metric chunk').real + '.';
const TWO_STATES = () => { const by = new Map(); for (const c of MCOLS) { if (!by.has(c.version)) by.set(c.version, []); by.get(c.version).push(c.label); } return [...by.values()].filter((x) => x.length > 1).map((x) => andList(x)).join('; '); };

/* compress a sorted list of ranks: [5,6,7,8] -> "5–8"; [5,8] -> "5, 8" */
function franks(rs) {
  const xs = rs.slice().sort((a, b) => a - b), out = [];
  for (let i = 0; i < xs.length;) {
    let j = i;
    while (j + 1 < xs.length && xs[j + 1] === xs[j] + 1) j++;
    out.push(j > i + 1 ? xs[i] + '–' + xs[j] : j === i + 1 ? xs[i] + ', ' + xs[j] : String(xs[i]));
    i = j + 1;
  }
  return out.join(', ');
}
const runsOf = (rs) => {
  const xs = rs.slice().sort((a, b) => a - b), out = [];
  for (const x of xs) { if (out.length && out[out.length - 1][1] === x - 1) out[out.length - 1][1] = x; else out.push([x, x]); }
  return out;
};
const plural = (n, w, many) => n + ' ' + (n === 1 ? w : many || w + 's');
const andList = (xs) => (xs.length < 2 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1]);

/* ---------- what each structure is, where it lives, how it is versioned (03 §2) ---------- */
const KM = {
  tensors: { where: 'GraphBLAS matrices of the graph', ver: 'not the index’s', code: 'm, dp, dm', title: 'relationship tensors', lives: 'the graph’s own GraphBLAS matrices, one set per relationship type', rule: 'not part of the index: the build reads their live pattern, and writers log the node pairs they touch' },
  entry: { where: 'MVCC entry', ver: 'one per state', code: 'entry', title: 'index entry', lives: 'the index’s state in one version of the MVCC graph', rule: 'one per published version: Built or Building, a generation, an upkeep job, a log' },
  log: { where: 'persistent list', ver: 'nodes shared and relinked', code: 'CchLog', title: 'CCH log', lives: 'a persistent list of nodes, each with a seq', rule: 'append and relink are one node each; pairs are shared, never copied; a consumer frees what it read' },
  Cch: { where: 'one Arc', ver: 'swapped as one', code: 'Arc<Cch>', title: 'the hierarchy', lives: 'one Arc: a topology and a metric', rule: 'swapped as one Arc, so a reader always holds a matching pair' },
  TopologyVersion: { where: 'one Arc', ver: 'new struct when a field is written', code: 'Arc<TopologyVersion>', title: 'topology', lives: 'one Arc inside the Cch', rule: 'a new struct when a writer writes any field; unwritten fields stay shared' },
  'Up.base': { where: 'Rust arrays, read-only GraphBLAS load', ver: 'shared by Arc', code: 'Up.base', lives: 'Rust-owned CSR arrays Ap, Aj, Ax (value = arc id), loaded read-only into GraphBLAS when an operation needs them', rule: 'shared by Arc for the whole lineage; only a fold replaces it' },
  'Down.base': { where: 'Rust arrays, read-only GraphBLAS load', ver: 'shared by Arc', code: 'Down.base', lives: 'the same arrays for Up transposed, with the same arc ids', rule: 'shared by Arc; only a fold replaces it' },
  'Up.delta': { where: 'B-tree pages', ver: 'path-copied', code: 'Up.delta', lives: 'cow_btree pages; key (lo ≪ 32) | hi, value = arc id', rule: 'a clone bumps the root; an insert copies the pages from the root to its leaf; never folds by itself' },
  'Down.delta': { where: 'B-tree pages', ver: 'path-copied', code: 'Down.delta', lives: 'cow_btree pages; key (hi ≪ 32) | lo, value = arc id', rule: 'as Up.delta' },
  rank_of: { where: 'chunked array', ver: 'chunk copied', code: 'rank_of', lives: 'chunked array over node ids; empty = no rank', rule: 'a write copies its chunk' },
  node_of: { where: 'chunked array', ver: 'chunk copied', code: 'node_of', lives: 'chunked array over ranks; empty = a pool rank', rule: 'a write copies its chunk' },
  free_ranks: { where: 'bitmap', ver: 'copied whole', code: 'free_ranks', lives: 'Arc<RoaringBitmap>: the ranks with no holder', rule: 'copied when it changes' },
  LineageStats: { where: 'counters', ver: 'copied whole', code: 'LineageStats', lives: 'counters that feed the decay signals', rule: 'copied when a counter changes' },
  added_since_fold: { where: 'persistent list', ver: 'new head per append', code: 'added_since_fold', lives: 'persistent list of (seq, arcs S2 added)', rule: 'an append makes a new head; a fold rebuilds it' },
  'metric up': { where: 'chunked array', ver: 'chunk copied, plus append', code: 'up', lives: 'chunked array by arc id: the upward weight', rule: 'a write copies its chunk; new arcs append' },
  'metric down': { where: 'chunked array', ver: 'chunk copied, plus append', code: 'down', lives: 'chunked array by arc id: the downward weight', rule: 'a write copies its chunk; new arcs append' },
  'metric up_middle': { where: 'chunked array', ver: 'chunk copied, plus append', code: 'up_middle', lives: 'chunked array by arc id: the apex rank of the upward value', rule: 'a write copies its chunk; new arcs append' },
  'metric down_middle': { where: 'chunked array', ver: 'chunk copied, plus append', code: 'down_middle', lives: 'chunked array by arc id: the apex rank of the downward value', rule: 'a write copies its chunk; new arcs append' },
};
const kcode = (k) => (KM[k] ? KM[k].code : k);
const BYTES = (k) => { const b = D.design_refs.bytes; return b[k] || (METRIC_F.includes(k) ? b.metric : k === 'log' ? b.log : k === 'entry' ? b.entry : null); };
const inst = (kind, c) => MV.instances[kind][String(c.objects[kind].iid)];
const predCol = (c) => (c.pred ? MCOL.get(c.pred) : null);

function holds(kind, c) {
  const x = c.objects[kind] ? inst(kind, c) : null;
  switch (kind) {
    case 'tensors': return Object.entries(D.versions[String(c.version)].graph.tensors).map(([t, L]) => t + ': m ' + L.m.length + ', dp ' + L.dp.length + ', dm ' + L.dm.length).join(' · ');
    case 'entry': return entryText(c.entry);
    case 'log': return c.log.length ? c.log.map((n) => 'seq ' + n.seq + ': ' + fpairs(n.pairs)).join('; ') : 'empty';
    case 'Cch': return c.entry.state === 'Built' ? 'topology ' + c.objects.TopologyVersion.label + ' and the four metric arrays' : 'the Repair’s base: not served; readers use the Dijkstra fallback';
    case 'TopologyVersion': return c.scalars.n + ' ranks · arc_capacity ' + c.scalars.arc_capacity;
    case 'Up.base': case 'Down.base': return (x.contents.Ap.length - 1) + ' rows · ' + x.contents.Ax.length + ' arc ids';
    case 'Up.delta': case 'Down.delta': return x.root == null ? 'empty' : plural(x.entries.length, 'entry', 'entries') + ' in ' + plural(x.pages.length, 'page');
    case 'rank_of': return x.contents.filter(([, v]) => v != null).length + ' nodes ranked · ' + plural(x.chunks.length, 'chunk');
    case 'node_of': { const free = x.contents.filter(([, v]) => v == null).length; return x.contents.length + ' ranks · ' + free + ' with no holder'; }
    case 'free_ranks': return x.contents.length ? (x.contents.length === 1 ? 'rank ' : 'ranks ') + franks(x.contents) : 'empty';
    case 'LineageStats': return Object.entries(x.contents).map(([k, v]) => k.replace(/_/g, ' ') + ' ' + v).join(' · ');
    case 'added_since_fold': return x.contents.length ? x.contents.map(([q, a]) => 'seq ' + q + ': ' + plural(a.length, 'arc')).join(' · ') : 'empty';
    default: return plural(x.length, 'arc') + ' in ' + plural(x.chunks.length, 'chunk');
  }
}

/* ---------- contents of one structure in one published state ---------- */
function chunkGroups(kind, c, fmt) {
  const x = inst(kind, c), p = predCol(c), pc = p ? inst(kind, p).chunks : [];
  const groups = [];
  x.chunks.forEach((cid, ci) => {
    const vals = MV.metric_chunks[String(cid)], cells = [];
    const isNew = !p || pc[ci] !== cid;
    vals.forEach((v, k) => { const id = ci * MV.chunk + k; if (id < x.length) cells.push({ ix: id, v: fmt(v), cls: v == null ? 'iso' : isNew && p ? 'hl' : '' }); });
    groups.push({ gl: 'chunk ' + ci + ' · id ' + cid + (p && isNew ? (c.metric_rebuilt ? ' · new' : ci < pc.length ? ' · copied' : ' · added') : ''), cells });
  });
  return gstrip('arc id → value', groups);
}
function mapGroups(kind, c) {
  const x = inst(kind, c), p = predCol(c);
  const pc = p ? new Map(inst(kind, p).chunks) : new Map();
  const off = kind === 'node_of' ? 1 : 0;
  const maxKey = Math.max(...x.contents.map(([k]) => k));
  const groups = x.chunks.map(([ci, cid]) => {
    const vals = MV.map_chunks[String(cid)], isNew = p && pc.get(ci) !== cid;
    const cells = [];
    vals.forEach((v, k) => { const key = ci * vals.length + k + off; if (key <= maxKey) cells.push({ ix: key, v: v == null ? 'none' : v, cls: v == null ? 'iso' : isNew ? 'hl' : '' }); });
    return { gl: 'chunk ' + ci + ' · id ' + cid + (isNew ? (pc.has(ci) ? ' · copied' : ' · added') : ''), cells };
  });
  return gstrip(kind === 'rank_of' ? 'node id → rank' : 'rank → node id', groups);
}
function csrCompact(name, A) {
  const g = [];
  for (let i = 0; i + 1 < A.Ap.length; i++) {
    const cells = [];
    for (let k = A.Ap[i]; k < A.Ap[i + 1]; k++) cells.push({ ix: A.Aj[k], v: A.Ax[k] });
    if (cells.length) g.push({ gl: 'r' + (i + 1), cells });
  }
  return h('div', { class: 'arrays compact' }, strip(name + '.Ap, one entry per row plus one', A.Ap.map((v) => ({ v }))),
    gstrip(name + '.Aj above, ' + name + '.Ax in the box, by row', g));
}
function csrStrips(name, A) {
  const gAj = [], gAx = [];
  for (let i = 0; i + 1 < A.Ap.length; i++) {
    const cj = [], cx = [];
    for (let k = A.Ap[i]; k < A.Ap[i + 1]; k++) { cj.push({ ix: k, v: A.Aj[k] }); cx.push({ ix: k, v: A.Ax[k] }); }
    if (cj.length) { gAj.push({ gl: 'row ' + i + ' · r' + (i + 1), cells: cj }); gAx.push({ gl: 'row ' + i, cells: cx }); }
  }
  return h('div', { class: 'pane arrays' }, strip(name + '.Ap', A.Ap.map((v, i) => ({ ix: i, v }))), gstrip(name + '.Aj (column rank)', gAj), gstrip(name + '.Ax (arc id)', gAx));
}
function storageMatrix(g, o) {
  o = o || {};
  const maxId = Math.max(NODE_CAP - 1, ...g.nodes);
  const ids = range(0, maxId);
  const rel = new Map(g.relationships.map((r) => [r.id, r]));
  const at = new Map();
  const put = (r, c, v) => { const k = r + ',' + c; at.set(k, Object.assign(at.get(k) || { r, c }, v)); };
  for (const [t, L] of Object.entries(g.tensors)) {
    const dm = new Set(L.dm.map(([a, b]) => a + ',' + b)), dp = new Set(L.dp.map(([a, b]) => a + ',' + b));
    for (const [a, b, id] of L.m) {
      if (dp.has(a + ',' + b)) continue;
      if (dm.has(a + ',' + b)) put(a, b, { cls: '', x: true });
      else put(a, b, { cls: t === 'FERRY' ? 'b' : 'on', text: rel.has(id) ? rel.get(id).weight : '' });
    }
    for (const [a, b, id] of L.dp) put(a, b, { cls: t === 'FERRY' ? 'b' : 'a', text: rel.get(id).weight, out: L.m.some(([x, y]) => x === a && y === b) });
  }
  return matrix({ rows: ids, cols: ids, cells: [...at.values()], cs: o.cs || 17, aria: 'Adjacency of node ids in the relationship tensors: committed entries, added entries, deleted marks and ferry entries, each with its weight' });
}
const storageLegend = () => legend([['q', 'on', 'ROAD m: committed (weight)'], ['q', 'a', 'ROAD dp: added since the last flush'], ['q', 'x', 'ROAD dm: deleted since the last flush'], ['q', 'b', 'FERRY m'], ['q', 'out', 'dp shadows an older m value']]);

function objDetail(kind, c) {
  const box = h('div', { class: 'detail', 'aria-live': 'polite' });
  const o = c.objects[kind], p = predCol(c);
  const head = h('h3', null, C(kcode(kind)), ' at ' + c.label + (o ? ' · instance ' + o.label : ''));
  ap(box, head, P(h('b', null, 'Lives: '), KM[kind].lives + '. ', h('b', null, 'Versioning: '), KM[kind].rule + '.'));
  if (o) {
    const span = MCOLS.filter((x) => x.objects[kind].iid === o.iid).map((x) => x.label);
    ap(box, P(o.new ? (p ? ['New in this state: ', h('span', { class: 'met' }, o.how), '.'] : [o.how[0].toUpperCase() + o.how.slice(1) + '.']) : 'Shared with the state before it: the same instance, not a copy.', ' Held by ', span.length > 1 ? span[0] + ' to ' + span[span.length - 1] : span[0], '.'));
  }
  const x = o ? inst(kind, c) : null;
  switch (kind) {
    case 'tensors': {
      const g = D.versions[String(c.version)].graph;
      ap(box, P('The graph at v' + c.version + ': rows are source nodes, columns destinations. The index never writes these; the build reads the live pattern ', C('m⟨¬dm⟩ ∪ dp'), ' and every writer adds the node pairs it touches to the CCH log.'),
        h('figure', null, h('div', { class: 'pane scroll' }, storageMatrix(g)), storageLegend()));
      break;
    }
    case 'entry':
      ap(box, mini(['state', 'generation', 'upkeep', 'log'], [[c.entry.state + (c.entry.phase ? ' · ' + c.entry.phase : ''), c.entry.generation, c.entry.upkeep ? c.entry.upkeep.kind + ', ticket ' + c.entry.upkeep.ticket + ', since ' + c.entry.upkeep.since : 'none', c.log.length ? c.log.map((n) => 'seq ' + n.seq).join(', ') : 'empty']]),
        P(c.entry.state === 'Built' ? 'Readers of this version take its Arc<Cch>.' : 'Building: readers of this version use the Dijkstra fallback. The Repair’s base is the hierarchy at the abort, held by this entry.'));
      break;
    case 'log':
      ap(box, logChips(c.log), P(c.entry.state === 'Built' ? (c.entry.upkeep ? 'An upkeep job is in flight, so writers keep their nodes with seq above since = ' + c.entry.upkeep.since + ' (03 §7).' : 'No upkeep job, so a Built entry keeps no log: each writer drops its node at publish (03 §7).') : 'A Building entry keeps its log; the next phase consumes it.'));
      break;
    case 'Cch': case 'TopologyVersion':
      ap(box, mini(['field', 'instance'], Object.entries(x).map(([k, v]) => [kcode(k), String(v)])));
      break;
    case 'Up.base': case 'Down.base':
      ap(box, h('figure', null, h('div', { class: 'pane' }, csrCompact(kind.split('.')[0], x.contents)), h('figcaption', null, 'Row i holds rank i + 1. ' + (kind === 'Up.base' ? 'Aj holds the higher rank of each arc and Ax its arc id.' : 'Down holds Up transposed: Aj is the lower rank, and Ax keeps the same arc id.'))));
      break;
    case 'Up.delta': case 'Down.delta': {
      if (x.root == null) { ap(box, P('Empty: the build put every arc in the base.')); break; }
      const prevPages = p && inst(kind, p).root != null ? new Set(inst(kind, p).pages) : new Set();
      const fresh = new Set(x.pages.filter((pg) => !prevPages.has(pg)));
      ap(box, treeFig({ root: x.root, pages: x.pages.map((pg) => MV.pages[String(pg)]) }, 'Each page shows its id; a leaf lists (row, column) → arc id. Blue pages are new in this state; the rest are shared with ' + (p ? p.label : 'nothing') + '.', { copied: p ? fresh : new Set(), aria: kind + ' pages at ' + c.label }));
      break;
    }
    case 'rank_of': case 'node_of':
      ap(box, h('div', { class: 'pane arrays' }, mapGroups(kind, c)), note(CHUNK_TXT() + ' Blue chunks are new in this state.'));
      break;
    case 'free_ranks':
      ap(box, arrays(strip('free_ranks', x.contents.length ? x.contents.map((r) => ({ v: r })) : [])), P('A rank is here exactly when it has no holder (invariant P), and then it has no arcs.'));
      break;
    case 'LineageStats':
      ap(box, mini(Object.keys(x.contents).map((k) => k.replace(/_/g, ' ')), [Object.values(x.contents)]));
      break;
    case 'added_since_fold':
      ap(box, x.contents.length ? h('div', { style: 'display:grid;gap:6px' }, x.contents.map(([q, a]) => blk('seq ' + q, chips(a.map((t) => ['(' + t[0] + ', ' + t[1] + ') id ' + t[2], 'fil']))))) : P('Empty.'));
      break;
    default: {
      const w = /middle/.test(kind) ? fm : fw;
      ap(box, h('div', { class: 'pane arrays' }, chunkGroups(kind, c, w)), note('The small number over each box is the arc id. ' + CHUNK_TXT() + ' Blue chunks are new in this state.'));
    }
  }
  return box;
}

/* ---------- section scaffold ---------- */
function dsec(id, n, where, title, ...lede) {
  return h('section', { id, class: 'dsec', 'aria-labelledby': id + '-h' },
    h('div', { class: 'fnhead' }, h('span', { class: 'where' }, h('b', null, String(n)), ' · ', where), h('h2', { id: id + '-h' }, title)),
    lede.length ? h('p', { class: 'what' }, lede) : null);
}
function vpicker(label, keys, initial, onPick) {
  const wrap = h('div', { class: 'vpick', role: 'group', 'aria-label': label }, h('span', { class: 'bh' }, label));
  const btns = keys.map((k) => {
    const b = h('button', { type: 'button', 'aria-pressed': k === initial ? 'true' : 'false' }, MCOL.get(k).label);
    b.addEventListener('click', () => { for (const x of btns) x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); onPick(k); });
    return b;
  });
  ap(wrap, btns);
  return wrap;
}

/* ===================== 2. What the index is made of ===================== */
function exampleFigure() {
  const s = ST.A, d = s.data, dr = s.draw;
  const special = new Set([...dr.edges.pending_add, ...dr.edges.replaced].map(([a, b]) => key2(a, b)));
  const edges = bgEdges(d.graph.relationships, { tags: true, skip: special }).map((e) => Object.assign(e, { z: 0 }));
  const wt = (a, b) => { const r = d.graph.relationships.find((x) => x.src === a && x.dst === b); return r ? r.weight : null; };
  const both = (a, b) => { const x = wt(a, b), y = wt(b, a); return x === y ? x : x + '/' + y; };
  for (const [a, b] of dr.edges.pending_add) edges.push({ a, b, cls: 'e add', tag: both(Math.min(a, b), Math.max(a, b)), tagcls: 'acc', z: 1 });
  for (const [a, b] of dr.edges.replaced) edges.push({ a, b, cls: 'e rep', tag: both(Math.min(a, b), Math.max(a, b)), tagcls: 'fil', z: 1 });
  for (const [a, b] of dr.edges.pending_delete) edges.push({ a, b, cls: 'e del', z: 1 });
  const iso = new Set(dr.nodes.isolated), unused = new Set(dr.nodes.unused);
  const nodes = [...dr.nodes.node, ...dr.nodes.unused].map((n) => ({ id: n, cls: iso.has(n) ? 'iso' : unused.has(n) ? 'unused' : '' }));
  const g = graph({ nodes, edges, aria: 'The example road network when the index is created: a grid of junctions, a dead-end road off one corner, one node with no road and one free id slot, with the edits still pending in storage.' });
  const fig = wrapFig(g, 'The example when the index is created (v' + ST.E.data.events[0].version + '). Tags are weights; where the two directions differ the tag reads lo→hi / hi→lo.');
  fig.querySelector('.pane').classList.add('fitg');
  return h('div', { class: 'two' }, fig,
    h('div', { style: 'display:grid;gap:12px;align-content:start' },
      legend([['l', '', 'road, both directions'], ['l', '', 'one-way road (arrow)'], ['l', 'fy', 'ferry, one way (arrow)'], ['l', 'add', 'pending addition (in dp)'], ['l', 'del', 'pending deletion (in dm)'], ['l', 'rep', 'replaced in one transaction'], ['n', 'iso', 'node with no road'], ['n', 'unused', 'free id slot']]),
      h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'the example'), h('ul', null, D.example.notes.slice(0, 2).map((n) => h('li', null, n))))));
}
function objectsSection() {
  const sec = dsec('d-objects', 2, '03 §2 Data structures, §7 Versions', 'What the index is made of',
    'Every version of the graph holds one index entry. A Built entry holds an Arc<Cch>: a topology and a metric that readers take together. Below are the structures the design adds, shown on the example. Each card says what the structure holds in the version you pick, where it lives and how a new version gets its own copy. Pick a card to see its contents.');
  ap(sec, exampleFigure());
  let colKey = '16', kind = 'Up.delta';
  const mapBox = h('div'), detail = h('div');
  const card = (k) => {
    const c = MCOL.get(colKey), o = c.objects[k];
    const how = o && o.new && c.pred ? compactHow(k, c) : '';
    const b = h('button', { type: 'button', class: 'ocard', 'aria-pressed': k === kind ? 'true' : 'false', 'aria-controls': 'd-objects-detail' },
      h('span', { class: 'on' }, C(kcode(k)), o ? h('span', { class: 'ix' }, o.label) : null),
      h('span', { class: 'hold' }, holds(k, c)),
      h('span', { class: 'lv' }, KM[k].where, h('span', { class: 'vr' }, ' · ' + KM[k].ver)),
      BYTES(k) ? h('span', { class: 'by' }, BYTES(k)) : null,
      how ? h('span', { class: 'nw' }, 'new here: ' + how) : null);
    b.addEventListener('click', () => { kind = k; draw(); });
    return b;
  };
  const draw = () => {
    const c = MCOL.get(colKey);
    mapBox.textContent = '';
    ap(mapBox, h('div', { class: 'omap' },
      h('div', { class: 'obox' }, h('span', { class: 'oh' }, 'the graph (storage), at v' + c.version), h('div', { class: 'ogrid' }, card('tensors'))),
      h('div', { class: 'flow-arrow', 'aria-hidden': 'true' }, h('span', { class: 'ar' }), h('span', null, 'the build reads the live pattern; every writer logs the node pairs it touches')),
      h('div', { class: 'obox entry' }, h('span', { class: 'oh' }, 'index entry at ' + c.label),
        h('div', { class: 'ogrid' }, card('entry'), card('log'), card('Cch')),
        h('div', { class: 'obox inner' + (c.entry.state === 'Built' ? '' : ' notserved') }, h('span', { class: 'oh' }, c.entry.state === 'Built' ? 'inside the Arc<Cch>: one topology and one metric, swapped together' : 'inside the Repair’s base (' + MCOL.get('13').label + '’s Arc<Cch>): not served; readers use the Dijkstra fallback'),
          h('div', { class: 'obox' }, h('span', { class: 'oh' }, 'Arc<TopologyVersion>'), h('div', { class: 'ogrid' }, card('TopologyVersion'), TOPO_F.map(card))),
          h('div', { class: 'obox' }, h('span', { class: 'oh' }, 'Metric: four arrays indexed by arc id'), h('div', { class: 'ogrid' }, METRIC_F.map(card)))))));
    detail.textContent = '';
    ap(detail, objDetail(kind, c));
    detail.firstChild.id = 'd-objects-detail';
    focusPanes(detail);
  };
  const R = D.design_refs;
  ap(sec, vpicker('version', MCOLS.map((c) => c.key), colKey, (k) => { colKey = k; draw(); }), mapBox,
    note('Bytes on each card are 03’s, at real scale (', R.bytes.source, '). Against C (', R.against_c.source, '): ', R.against_c.text),
    h('div', { class: 'two' },
      h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'how ranks are given out (' + R.ranks.source + ')'), h('ul', null, R.ranks.items.map((x) => h('li', null, x)))),
      h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'what changes for code that reads the hierarchy (' + R.contracts.source + ')'), h('ul', null,
        R.contracts.items.map(([k, t]) => h('li', null, B(k), ': ' + t)), h('li', null, R.contracts.why_delta), h('li', null, R.contracts.why_arrays)))),
    detail);
  draw();
  ap(sec, note('Instance numbers count the copies of one structure in order: ', C('#2'), ' is the second instance of that structure. A version that does not write a structure holds the same instance as the version before it. Bases are named by the build or fold that made them.'));
  return sec;
}

/* ===================== 3. From graph to index: the construct ===================== */
function cellTree() {
  const T = TR.B.tree, side = TR.B.cells.filter((c) => c.kind === 'pool' || c.kind === 'peeled');
  const kids = new Map(T.map((t) => [t.name, []]));
  for (const t of T) if (t.parent) kids.get(t.parent).push(t.name);
  const pos = new Map();
  let leaf = 0, maxd = 0;
  const visit = (nm, d) => {
    maxd = Math.max(maxd, d);
    const ch = kids.get(nm);
    if (!ch.length) { pos.set(nm, [leaf++, d]); return; }
    for (const c of ch) visit(c, d + 1);
    const xs = ch.map((c) => pos.get(c)[0]);
    pos.set(nm, [(Math.min(...xs) + Math.max(...xs)) / 2, d]);
  };
  for (const t of T.filter((t) => !t.parent)) visit(t.name, 0);
  const bw = 120, bh = 42, sx = 130, sy = 74, pad = 6, top = 58;
  const W = Math.max(pad * 2 + (leaf - 1) * sx + bw, pad * 2 + side.length * (bw + 10));
  const H = top + maxd * sy + bh + pad;
  const svg = sv('svg', { class: 'nat', viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'aria-label': 'The dissection as a tree of cells: each cell with its ranks and its separator' });
  const XY = (nm) => { const [x, d] = pos.get(nm); return [pad + x * sx, top + d * sy]; };
  side.forEach((c, i) => {
    const x = pad + i * (bw + 10);
    svg.append(sv('rect', { class: 'ctb side', x, y: pad, width: bw, height: 38, rx: 5 }));
    svg.append(sv('text', { class: 'bt', x: x + 8, y: pad + 15 }, c.name));
    svg.append(sv('text', { class: 'bs', x: x + 8, y: pad + 30 }, 'ranks ' + franks(c.ranks)));
  });
  for (const t of T) if (t.parent) {
    const [x1, y1] = XY(t.parent), [x2, y2] = XY(t.name);
    svg.append(sv('path', { class: 'pgl', d: 'M' + (x1 + bw / 2) + ',' + (y1 + bh) + 'L' + (x2 + bw / 2) + ',' + y2 }));
  }
  for (const t of T) {
    const [x, y] = XY(t.name);
    svg.append(sv('rect', { class: 'ctb' + (t.leaf ? ' leaf' : ''), x, y, width: bw, height: bh, rx: 5 }));
    svg.append(sv('text', { class: 'bt', x: x + 8, y: y + 16 }, t.leaf ? t.name + ' · a leaf' : t.name + ' · ranks ' + franks(t.ranks)));
    svg.append(sv('text', { class: 'bs', x: x + 8, y: y + 32 }, t.leaf ? 'ranks ' + franks(t.ranks) : 'separator ' + franks(t.separator)));
  }
  return svg;
}
function trackFrames() {
  const S = TR.S, Bt = TR.B, U = TR.Up;
  const b1 = D.bases['base 1'];
  const nArcs = U.arcs.length, nFill = U.fill.length;
  const v9 = D.hierarchies['9'], c16 = MCOL.get('16'), c16f = MCOL.get('16f');
  const slabVers = ST.B1.data.slabs.map((s) => 'rows ' + fpair(s.rows) + ' at v' + s.version);
  const moved = TR.order.moved_by_relabel;
  const frames = [];
  // 1. S
  frames.push({ label: 'S', title: 'S, the masked skeleton, in node ids', pic() {
    const ids = range(0, S.n - 1);
    const cells = S.entries.map(([a, b]) => ({ r: a, c: b, cls: 'on' })).concat(S.masked_by_dm.map(([a, b]) => ({ r: a, c: b, x: true })));
    return [h('figure', null, h('div', { class: 'pane scroll fit' }, matrix({ rows: ids, cols: ids, cells, rowRules: [S.slabs[1][0]], aria: 'S over node ids, read in two slabs' })),
      legend([['q', 'on', 'entry of S'], ['q', 'x', 'pending deletion, masked out']]), h('figcaption', null, 'Rows and columns are node ids. The heavy rule separates the two slabs.'))];
  }, panel: () => [
    blk('object', P('S: builder scratch, one GraphBLAS matrix per slab of node ids (', C('Vec<Arc<Matrix>>'), ').')),
    blk('version', P('Each slab reads its rows at the version current when it runs: ' + slabVers.join('; ') + '.')),
    blk('where it lives', P('Read straight from each type’s tensors as ', C('m⟨¬dm⟩ ∪ dp'), ', then symmetrized. No CSR copy and no whole-matrix merge.')),
    blk('pattern', P(S.entries.length + ' entries. ' + S.masked_by_dm.length + ' entries of a pending deletion are masked out, so no arc is ever made for them.')),
    blk('order', P('Node ids. Nothing is reordered until the one extract in the next frame.'))] });
  // 2. B = S(π,π)
  frames.push({ label: 'B = S(π,π)', title: 'B = S(π,π): the one reorder, with the cells', pic() {
    const n = Bt.n, rk = range(1, n);
    const at = new Map(), boxes = [];
    const tint = (i, j, cls) => { const k = i + ',' + j; if (!at.has(k)) at.set(k, { r: i - 1, c: j - 1, cls }); };
    const T = ['lt0', 'lt1', 'lt2', 'lt3'];
    const leaves = Bt.cells.filter((c) => c.kind === 'leaf');
    const cellRanks = new Map(Bt.tree.map((t) => [t.name, t.ranks]));
    for (const e of Bt.entries) at.set(e[0] + ',' + e[1], { r: e[0] - 1, c: e[1] - 1, cls: 'on' });
    leaves.forEach((c, k) => { for (const i of c.ranks) for (const j of c.ranks) tint(i, j, T[k % T.length]); for (const a of runsOf(c.ranks)) for (const b of runsOf(c.ranks)) boxes.push({ r0: a[0] - 1, r1: a[1] - 1, c0: b[0] - 1, c1: b[1] - 1 }); });
    for (const c of Bt.cells.filter((c) => c.kind === 'peeled')) for (const i of c.ranks) for (const j of c.ranks) tint(i, j, 'lt3');
    for (const s of Bt.cells.filter((c) => c.kind === 'separator')) {
      const span = cellRanks.get(s.cell);
      for (const i of s.ranks) for (const j of span) { tint(i, j, 'lsep'); tint(j, i, 'lsep'); }
      for (const a of runsOf(s.ranks)) for (const b of runsOf(span)) { boxes.push({ r0: a[0] - 1, r1: a[1] - 1, c0: b[0] - 1, c1: b[1] - 1, cls: 'sep' }); boxes.push({ r0: b[0] - 1, r1: b[1] - 1, c0: a[0] - 1, c1: a[1] - 1, cls: 'sep' }); }
    }
    return [h('figure', null, h('div', { class: 'pane scroll fit' }, matrix({ rows: rk, cols: rk, cells: [...at.values()], boxes, aria: 'B over ranks, with the leaf cells and the separator bands outlined' })),
      legend([['q', 'on', 'entry of B'], ['q', 'lt0', 'a leaf cell (tint)'], ['q', 'mbx', 'leaf cell outline'], ['q', 'lsep', 'separator rows and columns'], ['q', 'msep', 'separator band outline']])),
    h('figure', null, h('div', { class: 'pane scroll' }, cellTree()), h('figcaption', null, 'The dissection that gave π, as cells. Separators take the top ranks of their cell; leaves are ordered by exact minimum degree.'))];
  }, panel: () => [
    blk('object', P('B: the one private extract ', C('B = S(π,π)'), ' at the end of Order (03 §3 step 8).')),
    blk('version', P('The last Order batch, forked at v' + ST.E.data.events.find((e) => /^Order: leaves/.test(e.name || '')).fork + '. Order reads no snapshot; it works on the slabs already read.')),
    blk('where it lives', P('A private GraphBLAS matrix on the builder thread. The host unloads its arrays to relabel them, so there is no second extract.')),
    blk('why this is the only reorder', P('During dissection a cell is only a label on node ids; no matrix is permuted. Order builds π from the cells, and this extract is the first time rows sit in rank order.')),
    (() => { const lc = Bt.cells.find((c) => c.kind === 'leaf' && runsOf(c.ranks).length > 1); return blk('the postorder relabel', P('After the extract, ranks are renumbered in postorder so each elimination subtree holds consecutive ranks. A dissection cell need not stay a run of consecutive ranks: here ' + plural(moved.length, 'rank') + ' move (' + moved.map(([a, b]) => a + '→' + b).join(', ') + ')' + (lc ? ', and leaf ' + lc.name + ' holds ranks ' + franks(lc.ranks) + ', drawn as separate squares' : '') + '. Nothing relies on cells being contiguous.')); })(),
    blk('pattern', P(Bt.entries.length + ' entries over ' + Bt.n + ' ranks. The column counts give ' + Bt.arcs_by_counts + ' arcs for B at the build, before the log adds anything.'))] });
  // 3. Up over B′
  frames.push({ label: 'Up over B′', title: 'Up: the supergraph over B′, with its fill', pic() {
    const rk = range(1, U.n);
    const added = new Set(U.added.map(([a, b]) => a + ',' + b));
    const cells = U.B_prime.filter(([a, b]) => a < b).map(([a, b]) => ({ r: a - 1, c: b - 1, cls: added.has(a + ',' + b) ? 'a' : 'on' }));
    for (const [a, b] of U.fill) cells.push({ r: a - 1, c: b - 1, cls: 'f' });
    for (const [a, b] of U.dropped) if (a < b) cells.push({ r: a - 1, c: b - 1, x: true });
    return [h('figure', null, h('div', { class: 'pane scroll fit' }, matrix({ rows: rk, cols: rk, cells, aria: 'Up over ranks: the upper triangle of B′ plus the fill the elimination game adds' })),
      legend([['q', 'on', 'arc from B'], ['q', 'a', 'arc from a logged pair (A)'], ['q', 'x', 'logged pair with no live edge (D): dropped'], ['q', 'f', 'fill: a shortcut no edge supplies']]))];
  }, panel: () => [
    blk('object', P('The supergraph: every arc a weight can ever be written into. It is built once, by the Supergraph batch.')),
    blk('where it lives', P('Host rows on the builder thread, sized exactly by the column counts, until ids are assigned (next frame). The batch keeps its result as an Arc<TopologyVersion>.')),
    blk('version', P('The Supergraph batch forks at v' + ST.C2.data.log.reduce((m, l) => Math.max(m, l.seq), 0) + ' and resolves the log there: ', C('B′ = (B ∖ D) ∪ A'), ' over ' + U.n + ' ranks. ' + (() => { const t = andList(U.new_ranks.map(([r, n, k]) => 'node ' + n + ' takes ' + k + ' rank ' + r)); return t[0].toUpperCase() + t.slice(1); })() + '.')),
    blk('pattern', P('The game over the upper triangle of B′ gives ' + nArcs + ' arcs, ' + nFill + ' of them fill. B′, not B, is the pattern this frame shows; that is why the count differs from the frame before.'))] });
  // 4. Up.base as stored
  frames.push({ label: 'Up.base', title: 'Up.base and Down.base as stored', pic() {
    const rk = range(1, b1.rows);
    const fill = new Set(U.fill.map(([a, b]) => a + ',' + b));
    const cells = TR.base.ids.map(([lo, hi, id]) => ({ r: lo - 1, c: hi - 1, cls: fill.has(lo + ',' + hi) ? 'f' : 'ink', text: id }));
    return [h('figure', null, h('div', { class: 'pane scroll fit' }, matrix({ rows: rk, cols: rk, cells, cs: 19, lw: 26, aria: 'Up.base as a matrix whose values are arc ids' })),
      legend([['q', 'ink', 'arc id of an original arc'], ['q', 'f', 'arc id of a fill arc']]))];
  }, panel: () => [
    blk('object', P(C('Up.base'), ' and ', C('Down.base'), ' (', b1.name, ').')),
    blk('version', P('Built by the Supergraph batch and published with the first Built entry at ' + FIRST + '. Every later version shares them until a fold replaces them.')),
    blk('where it lives', P('Rust-owned arrays (', C('Arc<[u32]>'), '). When a GraphBLAS operation needs the matrix, the arrays are loaded read-only; queries and customization read the same memory as flat slices. There is no second CSR.')),
    blk('arc ids', P('The value of each cell is the arc’s id, assigned row by row, so on a fresh build id = position. ' + b1.Up.Ax.length + ' ids over ' + b1.rows + ' rows. The metric is indexed by these ids.')),
    blk('as stored', csrCompact('Up', b1.Up), csrCompact('Down', b1.Down), note('Row i holds rank i + 1. Down holds Up transposed: Aj is the lower rank, and each arc keeps its id.'))] });
  // 5. the metric
  frames.push({ label: 'metric', title: 'The metric at ' + FIRST + ', drawn on the arcs', pic() {
    const rk = range(1, b1.rows), m = v9.metric;
    const cells = [];
    for (const [lo, hi, id] of v9.arcs) { cells.push({ r: lo - 1, c: hi - 1, cls: 'as', text: fw(m.up[id]) }); cells.push({ r: hi - 1, c: lo - 1, cls: 'bs2', text: fw(m.down[id]) }); }
    return [h('figure', null, h('div', { class: 'pane scroll fit' }, matrix({ rows: rk, cols: rk, cells, cs: 19, lw: 26, aria: 'Upward weights above the diagonal and downward weights below it, at each arc' })),
      legend([['q', 'as', 'up[id] at (lo, hi)'], ['q', 'bs2', 'down[id] at the mirrored cell (hi, lo)']]))];
  }, panel: () => [
    blk('object', P('The metric: four arrays indexed by arc id (', METRIC_F.map((k) => kcode(k)).join(', '), '). It is not a matrix; this frame only draws each weight at its arc’s cell so it can be read.')),
    blk('version', P(FIRST + ', the first Built version: the relax sweep at v' + ST.C2.data.log.reduce((m, l) => Math.max(m, l.seq), 0) + ', then the Seal’s and the adoption’s scoped recomputes.')),
    blk('where it lives', P('Chunked arrays: a write copies only its chunk, and new arcs append. ', MCOL.get('9').objects['metric up'] ? holds('metric up', MCOL.get('9')) + ' per array.' : '')),
    blk('values', P('∞ means no path through lower ranks, for example a fill arc with no detour yet in that direction.'))] });
  // 6. v16: base 1 + Up.delta
  frames.push({ label: LB('16') + ': delta', title: LB('16') + ': ' + b1.name + ' plus Up.delta', pic() {
    const H16 = D.hierarchies['16'], dl = inst('Up.delta', c16), n = c16.scalars.n, rk = range(1, n);
    const inDelta = new Set(dl.entries.map((e) => e[2]));
    const cells = H16.arcs.map(([lo, hi, id]) => ({ r: lo - 1, c: hi - 1, cls: inDelta.has(id) ? 'a' : 'on', text: id }));
    return [h('figure', null, h('div', { class: 'pane scroll fit' }, matrix({ rows: rk, cols: rk, cells, cs: 19, lw: 26, rowRules: [b1.rows], colRules: [b1.rows], aria: LB('16') + ': arc ids from ' + b1.name + ' and from Up.delta' })),
      legend([['q', 'on', 'in ' + b1.name + ' (shared since ' + FIRST + ')'], ['q', 'a', 'in Up.delta ' + c16.objects['Up.delta'].label]]), h('figcaption', null, 'The heavy rules mark the base’s last row and column.'))];
  }, panel: () => {
    const dl = inst('Up.delta', c16), top = c16.scalars.n;
    return [blk('object', P(C('Up.delta'), ' instance ' + c16.objects['Up.delta'].label + ', over ' + b1.name + '.')),
      blk('version', P(LB('16') + ', after ' + plural(MCOLS.slice(1, MCOLS.findIndex((c) => c.key === '16') + 1).filter((c) => c.objects['Up.delta'].new).length, 'state') + ' that inserted arcs. The base is still the one built at ' + FIRST + '.')),
      blk('where it lives', P('A cow_btree keyed by (row, column), whose value is the arc id: ' + holds('Up.delta', c16) + '. Rank ' + top + ' lies above the base’s ' + b1.rows + ' rows, so its arcs live only in the delta; the base is never resized.')),
      blk('arc ids', P('New ids are appended from arc_capacity up, so ids no longer follow row order. Every whole-array write uses the id, never the position.'))];
  } });
  // 7. after the fold
  frames.push({ label: LB('16f'), title: 'After the fold: ' + c16f.objects['Up.base'].label + ' plus the new delta', pic() {
    const Hf = D.hierarchies['16 fold'], dl = inst('Up.delta', c16f), n = c16f.scalars.n, rk = range(1, n);
    const inDelta = new Set(dl.entries.map((e) => e[2])), fromOld = new Set(b1.Up.Ax);
    const cells = Hf.arcs.map(([lo, hi, id]) => ({ r: lo - 1, c: hi - 1, cls: inDelta.has(id) ? 'a' : fromOld.has(id) ? 'on' : 'pl', text: id }));
    return [h('figure', null, h('div', { class: 'pane scroll fit' }, matrix({ rows: rk, cols: rk, cells, cs: 19, lw: 26, aria: 'After the fold: arc ids in base 2 and in the new delta' })),
      legend([['q', 'on', 'from ' + b1.name], ['q', 'pl', 'from the delta at the fold’s fork'], ['q', 'a', 'in the new delta: added after the fork']]))];
  }, panel: () => {
    const b2 = inst('Up.base', c16f).contents, dl = inst('Up.delta', c16f);
    return [blk('object', P(C('Up.base'), ' ', c16f.objects['Up.base'].label, ' and ', C('Up.delta'), ' ', c16f.objects['Up.delta'].label, '.')),
      blk('version', P('The fold job forked at v' + ST.F6.data.fork + ' and published at v' + c16f.version + ' under the lock (the “' + c16f.label + '” state).')),
      blk('where it lives', P('Base 2 is new Rust-owned arrays, unloaded from GraphBLAS after the merge. The new delta is a fresh cow_btree.')),
      blk('what changed', P(c16f.objects['Up.base'].label[0].toUpperCase() + c16f.objects['Up.base'].label.slice(1) + ' holds ' + b2.Ax.length + ' arcs over ' + (b2.Ap.length - 1) + ' rows: base 1 plus the delta at the fork, merged by GraphBLAS with the ids carried as values. The new delta holds ' + plural(dl.entries.length, 'arc') + ' added after the fork. No arc changes its id, so the metric is shared as it is.'))];
  } });
  return frames;
}
function constructSection() {
  const sec = dsec('d-construct', 3, '03 §3 step 8, §4 Supergraph construction, §2', 'From graph to index: the construct',
    'One matrix followed through the build, frame by frame: the skeleton read from storage, the one reorder into rank space, the supergraph with its fill, the base as stored, the metric written on it, and the delta and the fold in later versions. Each frame names the structure, the version and where it lives. How the elimination game and customization compute their results is in ', ext(LINK_EXPLAINER, 'the CCH explainer'), '; step by step on this example, it is in the appendix.');
  const F = trackFrames();
  ap(sec, stepper('st-track', {
    title: 'The construct, one structure at a time', n: F.length,
    label: (i) => (i + 1) + ' of ' + F.length + ': ' + F[i].label,
    render(i, pics, panel) { ap(pics, F[i].pic()); ap(panel, h('h3', null, F[i].title), F[i].panel()); },
  }));
  return sec;
}

/* ===================== 4. Ordering on GraphBLAS ===================== */
function orderingSection() {
  const sec = dsec('d-ordering', 4, '03 §3 Ordering algorithm', 'Ordering on GraphBLAS, compactly',
    'Order runs nested dissection one depth at a time. At each depth every cell of that depth sits in one block-diagonal matrix, so each GraphBLAS call works on all cells at once. Each level shrinks the matrix by matching. The smallest level gets a separator on the host. The labels are then projected back, level by level, to the finest one, and node FM refines them at each level. Last, one masked product splits the matrix into the next depth’s cells.');
  const chip = (cls, t, s) => h('span', { class: 'mxc ' + cls }, h('span', { class: 't' }, t), s ? h('span', { class: 's' }, s) : null);
  const arr = () => h('span', { class: 'ar', 'aria-hidden': 'true' }, '→');
  const lvl = (dat, name) => chip('gb', name + ': ' + dat.verts.length + ' × ' + dat.verts.length, plural(dat.A.length, 'entry', 'entries'));
  const ct = (dat) => chip('gb', 'Cᵀ: ' + dat.verts.length + ' × ' + dat.next.verts.length, plural(dat.CT.length, 'entry', 'entries'));
  const d0 = [ST['B5.0'].data, ST['B5.1'].data, ST['B5.2'].data];
  const sp0 = ST.B8.data, d1 = ST['B9.1'].data, sp1 = d1.split;
  const line = (lab, chips_) => h('div', { class: 'lline' }, h('span', { class: 'll' }, lab), h('div', { class: 'lchips' }, chips_));
  const join = (xs) => xs.map((x, i) => (i ? h('span', { class: 'lpair' }, arr(), x) : x));
  const top0 = d0[d0.length - 1].next;
  const co0 = [];
  d0.forEach((dat, i) => { co0.push(lvl(dat, 'A, level ' + i), ct(dat)); });
  co0.push(chip('gb', 'A, level ' + d0.length + ': ' + top0.verts.length + ' × ' + top0.verts.length, plural(top0.A.length, 'entry', 'entries')));
  const L1 = d1.levels[0];
  const co1 = [chip('gb', 'A, level ' + L1.level + ': ' + L1.verts.length + ' × ' + L1.verts.length, plural(L1.A.length, 'entry', 'entries') + ' in ' + plural(d1.cells.length, 'block')),
    chip('gb', 'Cᵀ: ' + L1.verts.length + ' × ' + d1.coarsest.verts.length, plural(L1.CT.length, 'entry', 'entries')),
    chip('gb', 'A, level ' + d1.levels.length + ': ' + d1.coarsest.verts.length + ' × ' + d1.coarsest.verts.length, plural(d1.coarsest.A.length, 'entry', 'entries'))];
  const leaves = ST.B10.data.leaves;
  ap(sec, h('div', { class: 'ladder' },
    h('div', { class: 'lrow' }, h('span', { class: 'lh' }, 'depth ' + d0[0].depth), h('div', { class: 'lgroup' },
      line('coarsen', join(co0)),
      line('separate', [chip('host', 'host: a separator on the smallest level, then node FM at each finer level', 'labels projected to the finer level by mxv(Cᵀ, labels)')]),
      line('split', join([chip('split', 'M: ' + d0[0].verts.length + ' × ' + sp0.cells.length, plural(sp0.M.length, 'entry', 'entries') + '; separator rows stay empty'), chip('split', 'next A = M ANY_PAIR Mᵀ, masked by A', plural(sp0.A_next.length, 'entry', 'entries') + ' in ' + plural(sp0.cells.length, 'block'))])))),
    h('div', { class: 'lrow' }, h('span', { class: 'lh' }, 'depth ' + d1.depth), h('div', { class: 'lgroup' },
      line('coarsen', join(co1)),
      line('separate', [chip('host', 'host: a separator in every cell, then node FM', 'all cells of the depth in one pass')]),
      line('split', join([chip('split', 'M: ' + L1.verts.length + ' × ' + sp1.cells.length, plural(sp1.M.length, 'entry', 'entries')), chip('split', 'next A', plural(sp1.A_next.length, 'entry', 'entries') + ' in ' + plural(sp1.cells.length, 'block'))])))),
    h('div', { class: 'lrow' }, h('span', { class: 'lh' }, 'leaves'), h('div', { class: 'lgroup' }, line('order', [chip('host', plural(leaves.length, 'cell') + ' of at most ' + Math.max(...leaves.map((l) => l.verts.length)) + ' nodes', 'exact minimum degree on the host')])))),
  legend([['q', 'as', 'GraphBLAS matrix (all cells of the depth at once)'], ['q', 'bs2', 'the split: M and the next depth’s matrix'], ['q', 'out', 'host loop on the builder thread']]));
  ap(sec, note('Sizes are rows × columns; entries are stored values. C maps each vertex to its coarse vertex; the prototype builds Cᵀ directly (00-results Change 6), which has one entry per row. The next level is ', C('A_c = select(OFFDIAG, C·A·Cᵀ)'), ' (03 §3 step 3).'));
  ap(sec, h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'what the prototype does now'),
    P('The prototype now runs the contraction and the split as Rust passes over GraphBLAS’s own CSR arrays: unload the matrix in O(1), make one pass on the host, load the result back in O(1) (', C('prototypes/m1-optimize.md'), '). Also, ' + D.design_refs.d5.text + ' (' + D.design_refs.d5.source + '). ' + D.design_refs.d5.verdict + ' (' + D.design_refs.d5.verdict_source + '). ' + D.design_refs.d5.after + ' (' + D.design_refs.d5.after_source + '). The model does neither: it coarsens every depth from scratch, as 03 §3 writes it.')));
  return sec;
}

/* one matching round, up close (appendix, part B): the keys, the cap and the winners */
function matchingRound() {
  const sec = h('section', { id: 'ap-matching', 'aria-labelledby': 'ap-matching-h' }, h('div', { class: 'fnhead' }, h('span', { class: 'where' }, '03 §3 step 3; 00-results Change 1–3'), h('h2', { id: 'ap-matching-h' }, 'One matching round, up close')));
  const m = ST['B5.2'].data, r1 = m.rounds[0];
  const verts = m.verts, scr = new Map(), w = toMap(m.w);
  for (const [, j, , s] of m.K) scr.set(j, s);
  const capped = new Set(m.capped_out.flatMap(([a, b]) => [a + ',' + b, b + ',' + a]));
  const yv = new Map(r1.y.map(([v, wt, s]) => [v, [wt, s]]));
  const prop = new Map(r1.proposal);
  const cells = m.A.map(([i, j, a]) => ({ r: i, c: j, cls: capped.has(i + ',' + j) ? '' : prop.get(i) === j ? 'a' : 'on', text: capped.has(i + ',' + j) ? null : a, x: capped.has(i + ',' + j) }));
  const hex = (s) => s.toString(16).padStart(8, '0').slice(0, 4);
  const capV = m.cap[0][1];
  ap(sec,
    h('p', { class: 'what' }, 'Matching pairs each vertex with the neighbour it shares the heaviest edge with. To make the choice exact and repeatable, every entry gets a unique key ', C('K = (w ≪ 32) | scramble(j)'), ': the weight decides first, and the scrambled column id breaks ties. ', C('K_cap'), ' drops the entries whose two vertex weights add up to more than the cap. ', C('y⟨u⟩ = K_cap MAX_FIRST u'), ' then gives each unmatched vertex its winning key, and two vertices whose winners point at each other are a pair.'),
    h('div', { class: 'inset2' },
      h('figure', null, h('div', { class: 'pane scroll fit' }, matrix({ rows: verts, cols: verts, cells, cs: 22, lw: 22, aria: 'K over one level: edge weights, the entries the cap removes, and each row’s winner' })),
        legend([['q', 'on', 'edge weight w'], ['q', 'a', 'the row’s winner'], ['q', 'x', 'removed by K_cap']]),
        h('figcaption', null, 'Depth ' + m.depth + ', level ' + m.level + ': ' + verts.length + ' coarse vertices. The cap is ' + fnum(capV) + ': ' + plural(m.capped_out.length, 'pair') + ' would weigh more and are removed.')),
      h('div', { style: 'display:grid;gap:10px;align-content:start' },
        arrays(strip('w (vertex weight)', verts.map((v) => ({ ix: v, v: w.get(v) }))),
          strip('scramble(j), leading hex digits', verts.map((v) => ({ ix: v, v: hex(scr.get(v)) }))),
          strip('y: winning key, w|scramble', verts.map((v) => (yv.has(v) ? { ix: v, v: yv.get(v)[0] + '|' + hex(yv.get(v)[1]), cls: 'hl' } : { ix: v, v: 'none', cls: 'iso' }))),
          strip('proposes', verts.map((v) => (prop.has(v) ? { ix: v, v: '→ ' + prop.get(v) } : { ix: v, v: '—', cls: 'iso' })))),
        P('Round ' + r1.round + ' pairs ', C(r1.mutual.map(fpair).join(', ') || 'nothing'), '. ' + (m.rounds.length > 1 && !m.rounds[1].mutual.length ? 'Round ' + m.rounds[1].round + ' finds no more mutual winners, so the level stops.' : '')),
        (() => {
          const lone = verts.filter((v) => !yv.has(v) && m.A.filter(([i]) => i === v).every(([i, j]) => capped.has(i + ',' + j)));
          return lone.length ? note((lone.length === 1 ? 'Vertex ' : 'Vertices ') + andList(lone) + (lone.length === 1 ? ' has' : ' have') + ' no winner: every edge ' + (lone.length === 1 ? 'it has' : 'they have') + ' is over the cap.') : null;
        })())));
  return sec;
}

/* ===================== 5. Building while writers commit ===================== */
function buildSection() {
  const E = ST.E.data;
  const sec = dsec('d-build', 5, '03 §5 Batching on the native index builder', 'Building while writers commit',
    'The build runs as batches on the builder thread, off the lock. Writers keep committing under the lock; while the entry is Building, each writer appends the node pairs it touched to the CCH log, as one node whose seq is its version. Each batch forks at the published version and, when it publishes, fits itself against what landed since. Time runs down the page; a batch’s box spans the time it ran, so a writer commit beside it landed while it ran. On a narrow screen the lanes stack in time order, and each batch comes after the commits that landed while it ran.');
  const ev = E.events;
  const commitRow = {};
  ev.forEach((e, i) => { if (e.actor === 'writer') commitRow[e.version] = i; });
  let lastPub = -1;
  const span = new Map();
  ev.forEach((e, i) => { if (e.actor !== 'writer') { span.set(i, [Math.max(lastPub + 1, commitRow[e.fork] + 1), i]); lastPub = i; } });
  const builtAt = ev.findIndex((e) => e.entry.state === 'Built');
  const grid = h('div', { class: 'lanes', role: 'group', 'aria-label': 'Swimlanes of the build: writer commits, builder batches, what readers get, and the published log' });
  const R = (i) => i + 2;
  ap(grid, h('span', { class: 'lh', style: 'grid-column:1;grid-row:1' }, 'version'), h('span', { class: 'lh', style: 'grid-column:2;grid-row:1' }, 'writer thread (under the lock)'),
    h('span', { class: 'lh', style: 'grid-column:3;grid-row:1' }, 'builder thread: work off the lock, publish under it'), h('span', { class: 'lh', style: 'grid-column:4;grid-row:1' }, 'readers at this version get'), h('span', { class: 'lh', style: 'grid-column:5;grid-row:1' }, 'log seqs after'));
  const versions = Object.keys(commitRow).map(Number).sort((a, b) => a - b);
  versions.forEach((v, k) => {
    const a = commitRow[v], b = k + 1 < versions.length ? commitRow[versions[k + 1]] - 1 : ev.length - 1;
    ap(grid, h('span', { class: 'vb', style: 'grid-column:1;grid-row:' + R(a) + ' / ' + (R(b) + 1) + ';--o:' + (a * 10) }, 'v' + v));
    const rd = (r0, r1, built) => h('div', { class: 'ev r' + (built ? ' cc' : ''), style: 'grid-column:4;grid-row:' + R(r0) + ' / ' + (R(r1) + 1) + ';--o:' + (r0 * 10 + 1) },
      h('span', { class: 'lane-tag' }, 'readers'),
      h('span', { class: 't' }, built ? 'Arc<Cch> ' + MCOL.get('9').objects.Cch.label : 'Dijkstra fallback'),
      h('span', { class: 's' }, built ? 'the first Built entry' : 'the entry is Building and no older Cch exists: this is the first build'));
    if (builtAt >= a && builtAt <= b) { if (builtAt > a) ap(grid, rd(a, builtAt - 1, false)); ap(grid, rd(builtAt, b, true)); } else ap(grid, rd(a, b, false));
  });
  const detail = h('div');
  const buttons = [];
  const pick = (i, btn) => { for (const x of buttons) x.setAttribute('aria-pressed', x === btn ? 'true' : 'false'); detail.textContent = ''; ap(detail, h('div', { class: 'detail panel', 'aria-live': 'polite' }, eventDetail(ev[i]))); };
  ev.forEach((e, i) => {
    let el;
    if (e.actor === 'writer') {
      el = h('button', { type: 'button', class: 'ev w', style: 'grid-column:2;grid-row:' + R(i) + ';--o:' + (i * 10 + 2) }, h('span', { class: 'lane-tag' }, 'writer, under the lock'),
        h('span', { class: 't' }, 'v' + e.version + ' commits'), h('span', null, e.what.split(';').length > 2 ? e.what.split(';').slice(0, 2).join(';') + '; …' : e.what),
        e.log_node ? h('span', { class: 's' }, 'appends log seq ' + e.log_node.seq + ': ' + fpairs(e.log_node.pairs)) : h('span', { class: 's' }, e.note || ''));
    } else {
      const [a, b] = span.get(i);
      const why = e.fit === 'CatchUp' ? 'CatchUp: a writer committed since the fork; nodes with seq > ' + e.fork + ' are relinked on top' : e.fit === 'Same' ? 'Same: no writer committed since the fork' : e.fit;
      el = h('button', { type: 'button', class: 'ev b' + (e.fit === 'CatchUp' ? ' catch' : ''), style: 'grid-column:3;grid-row:' + R(a) + ' / ' + (R(b) + 1) + ';--o:' + (b * 10 + 3) }, h('span', { class: 'lane-tag' }, 'builder batch'),
        h('span', { class: 't' }, e.name), h('span', { class: 's' }, 'forks at v' + e.fork + (e.reads ? ' · reads ' + e.reads : '')),
        e.consumed && e.consumed.length ? h('span', { class: 's' }, 'consumes log seq ' + e.consumed.join(', ')) : null,
        e.absorbed ? h('span', { class: 's' }, 'absorbs seq ' + e.absorbed.seqs.join(', ') + ' under the lock: ' + e.absorbed.work + ' of ' + e.absorbed.budget + ' units') : null,
        h('span', { class: 's' + (e.fit === 'CatchUp' ? ' fil' : '') }, 'publish ' + why));
    }
    ap(el, h('span', { class: 'lane-log' }, 'log after: ' + (e.published_log.length ? 'seq ' + e.published_log.map((n) => n.seq).join(', ') : 'empty')));
    el.setAttribute('aria-pressed', 'false');
    el.addEventListener('click', () => pick(i, el));
    buttons.push(el);
    ap(grid, el, h('span', { class: 'logcell', style: 'grid-column:5;grid-row:' + R(i) }, e.published_log.length ? e.published_log.map((n) => n.seq).join(', ') : 'empty'));
  });
  ap(sec, h('figure', null, h('div', { class: 'pane scroll' }, grid),
    legend([['q', 'as', 'writer commit'], ['q', 'bs2', 'batch: the box spans the time it ran'], ['q', 'out', 'CatchUp publish'], ['q', 'fb', 'readers: Dijkstra fallback'], ['q', 'cc', 'readers: an Arc<Cch>']]),
    h('figcaption', null, 'Pick a box to see the event in full below.')));
  ap(sec, detail);
  pick(ev.findIndex((e) => e.fit === 'CatchUp'), buttons[ev.findIndex((e) => e.fit === 'CatchUp')]);
  // the entry per version
  const rows = versions.map((v, k) => {
    const b = k + 1 < versions.length ? commitRow[versions[k + 1]] - 1 : ev.length - 1;
    const e = ev[b];
    return [ 'v' + v, entryText(e.entry), e.published_log.length ? e.published_log.map((n) => 'seq ' + n.seq).join(', ') : 'empty', b >= builtAt ? 'Arc<Cch> ' + MCOL.get('9').objects.Cch.label : 'Dijkstra fallback'];
  });
  ap(sec, table(['version', 'entry at the end of the version', 'log', 'readers'], rows, { caption: 'The published entry per version', stack: true }));
  ap(sec, h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'the publish rules used here'), h('ul', null,
    h('li', null, B('Same'), ': no writer committed since the fork. The batch’s copy is published as it is; it already holds every log node of its fork, including writes in the gap between batches.'),
    h('li', null, B('CatchUp'), ': a writer committed since the fork. The published nodes with a higher seq are relinked on top of the batch’s copy, one node each; no pairs are copied.'),
    h('li', null, B('Consume'), ': only the phase that resolves pairs consumes them (Supergraph, the Seal, a Repair). Its copy sets the log’s head to none, and consumed nodes are freed when the last version holding them drops.'),
    h('li', null, B('Adoption'), ': the Seal is the last batch. Its publish absorbs the relinked nodes under the lock, within the lock budget, and the entry becomes Built with no log.'),
    h('li', null, B('Swap'), ': ' + E.swap + '.'))));
  const DRF = D.design_refs;
  ap(sec, h('h3', { class: 'sub-h' }, 'What the example does not exercise: writer rebase, the bound, the hooks'),
    table(['the writer forked from', 'published entry at its publish', 'rebase'], DRF.rebase.rows.map(([a, b, c]) => [a, b, prose(c)]), { caption: 'A writer whose fork predates a batch or job publish is rebased by the published state (' + DRF.rebase.source + ')', stack: true }),
    h('div', { class: 'two' },
      h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'the bound and the Overflow node (' + DRF.bound.source + ')'), h('ul', null, h('li', null, DRF.bound.text), h('li', null, DRF.bound.built), h('li', null, DRF.rebase.note))),
      h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'framework hooks (' + DRF.hooks.source + ')'), P(DRF.hooks.why), h('ul', null, DRF.hooks.items.map((x) => h('li', null, x))))));
  return sec;
}

/* ===================== 6. Versions after the build: shared and copied ===================== */
function chunkChange(a, b) {      /* chunk lists as [index, id] pairs: which indices are copied or added */
  const pb = new Map(b), cp = [], ad = [];
  for (const [i, cid] of a) { if (!pb.has(i)) ad.push(i); else if (pb.get(i) !== cid) cp.push(i); }
  return [cp.length ? (cp.length === 1 ? 'chunk ' : 'chunks ') + franks(cp) + ' copied' : null, ad.length ? (ad.length === 1 ? 'chunk ' : 'chunks ') + franks(ad) + ' added' : null].filter(Boolean).join(', ');
}
function newPages(kind, c) {
  const p = predCol(c), x = inst(kind, c);
  if (x.root == null) return [];
  const old = p && inst(kind, p).root != null ? new Set(inst(kind, p).pages) : new Set();
  return x.pages.filter((pg) => !old.has(pg));
}
function compactHow(kind, c) {
  const o = c.objects[kind];
  if (!o.new) return '';
  if (!c.pred) return /delta/.test(kind) ? '' : 'built';
  const p = predCol(c);
  if (METRIC_F.includes(kind) && c.metric_rebuilt) return 'rebuilt: all ' + inst(kind, c).chunks.length + ' chunks new';
  if (METRIC_F.includes(kind)) return chunkChange(inst(kind, c).chunks.map((cid, i) => [i, cid]), inst(kind, p).chunks.map((cid, i) => [i, cid]));
  if (kind === 'rank_of' || kind === 'node_of') return chunkChange(inst(kind, c).chunks, inst(kind, p).chunks);
  if (/delta/.test(kind)) {
    if (c.key === '16f') return 'rebuilt by the fold';
    const pg = o.pages;
    return plural(pg.new, 'new page') + ' (' + pg.path + ' on paths' + (pg.split ? ', ' + pg.split + ' split ' + (pg.split === 1 ? 'half' : 'halves') : '') + (pg.tree_was_empty ? '; the tree was empty' : '') + ')';
  }
  if (/base/.test(kind)) return 'from the fold';
  if (kind === 'TopologyVersion') return 'new struct';
  if (kind === 'Cch') return 'new Arc';
  if (kind === 'added_since_fold') { if (c.key === '16f') return 'rebuilt by the fold'; const old = new Set(inst(kind, p).contents.map(([q]) => q)); return inst(kind, c).contents.filter(([q]) => !old.has(q)).map(([q]) => 'seq ' + q).join(', ') + ' appended'; }
  return o.how.replace(/^copied: /, '');
}
const HELD_NAME = { 'delta page': ['page', 'pages'], 'metric chunk': ['metric chunk', 'metric chunks'], 'rank_of chunk': ['rank_of chunk', 'rank_of chunks'], 'node_of chunk': ['node_of chunk', 'node_of chunks'], 'log node': ['log node', 'log nodes'] };
/* what only this state holds: freed when its last holder (the entry, a reader, a job) lets go */
function heldText(c) {
  const f = c.held_only, parts = [];
  for (const k of ['Cch', 'TopologyVersion', ...TOPO_F, ...METRIC_F]) if (f[k]) parts.push(kcode(k) + ' ' + f[k].join(' '));
  for (const [k, nm] of Object.entries(HELD_NAME)) if (f[k]) parts.push(f[k].length === 1 && k === 'log node' ? 'log ' + f[k][0] : plural(f[k].length, nm[0], nm[1]));
  return [parts.length ? parts.join(' · ') : 'nothing: every instance it holds is held by another state too', parts.length ? '' : 'dim'];
}
function heldShort(c) {
  const f = c.held_only;
  const objs = [...TOPO_F, ...METRIC_F].filter((k) => f[k]).length, pages = (f['delta page'] || []).length;
  const chunksN = ['metric chunk', 'rank_of chunk', 'node_of chunk'].reduce((t, k) => t + (f[k] || []).length, 0);
  return [f.Cch ? 'Cch ' + f.Cch.join(' ') : null, f.TopologyVersion ? 'topology ' + f.TopologyVersion.join(' ') : null, objs ? plural(objs, 'structure') : null, pages ? plural(pages, 'page') : null, chunksN ? plural(chunksN, 'chunk') : null, f['log node'] ? 'log ' + f['log node'].join(', ') : null].filter(Boolean).join(' · ') || 'nothing';
}
function mvGrid(onPick, initial) {
  const n = MCOLS.length;
  const grid = h('div', { class: 'mv', style: '--n:' + n, role: 'group', 'aria-label': 'Which instance of each structure every published state holds' });
  let row = 1;
  const at = (i, j) => 'grid-row:' + row + ';grid-column:' + (i + 2) + (j != null ? ' / ' + (j + 3) : '');
  const rl = (t, k) => h('span', { class: 'rl' + (k ? ' k' : ''), style: 'grid-row:' + row + ';grid-column:1' }, t);
  const s7 = MCOLS.findIndex((c) => c.s7_met && c.s7_met.length);
  ap(grid, rl('state'), MCOLS.map((c, i) => h('span', { class: 'chd' + (c.entry.state !== 'Built' ? ' bld' : '') + (i === s7 ? ' s7' : ''), style: at(i) }, c.label, i === s7 ? h('span', { class: 's7m' }, 'S7 met *') : null)));
  row++;
  const group = (t) => { ap(grid, h('span', { class: 'grp', style: 'grid-row:' + row }, t)); row++; };
  const text = (label, f) => { ap(grid, rl(label)); MCOLS.forEach((c, i) => { const [t, cls] = f(c); ap(grid, h('span', { class: 'tx' + (cls ? ' ' + cls : '') + (c.entry.state !== 'Built' ? ' bldc' : ''), style: at(i) }, t)); }); row++; };
  const buttons = [];
  const bars = (kind) => {
    ap(grid, rl(kind === 'TopologyVersion' ? h('span', null, 'Arc<', h('wbr'), 'TopologyVersion>') : kcode(kind), true));
    let i = 0;
    while (i < n) {
      let j = i;
      while (j + 1 < n && MCOLS[j + 1].objects[kind].iid === MCOLS[i].objects[kind].iid) j++;
      const c = MCOLS[i], o = c.objects[kind], how = compactHow(kind, c);
      const empty = /delta/.test(kind) && o.iid == null;
      const b = h('button', { type: 'button', class: 'ib ' + (empty ? 'empty' : buttons.length % 2 ? 't1' : 't0'), style: at(i, j), 'aria-pressed': 'false', 'aria-label': kcode(kind) + ' ' + o.label + ', held by ' + c.label + (j > i ? ' to ' + MCOLS[j].label : '') + (how ? '; ' + how : '') },
        h('span', { class: 'il' }, o.label), how ? h('span', { class: 'ih' }, how) : null);
      b.addEventListener('click', () => { for (const x of buttons) x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); onPick(kind, c.key); });
      if (kind === initial.kind && c.key === initial.key) b.setAttribute('aria-pressed', 'true');
      buttons.push(b);
      ap(grid, b);
      if (kind === 'Cch') for (let x = i; x <= j; x++) if (MCOLS[x].entry.state !== 'Built') ap(grid, h('span', { class: 'ib-over', style: at(x), 'aria-hidden': 'true' }, h('span', null, 'not served: the Repair’s base')));
      i = j + 1;
    }
    row++;
  };
  group('the entry');
  text('published by', (c) => [c.by]);
  text('state', (c) => [c.entry.state + (c.entry.phase ? ' · ' + c.entry.phase : '') + ' · generation ' + c.entry.generation, c.entry.state === 'Built' ? '' : 'bad']);
  text('upkeep', (c) => (c.entry.upkeep ? [c.entry.upkeep.kind + ', ticket ' + c.entry.upkeep.ticket + ', since ' + c.entry.upkeep.since] : ['none', 'dim']));
  text('log', (c) => [c.log_why, c.log.length ? '' : 'dim']);
  text('readers get', (c) => (c.serves === 'Cch' ? ['Arc<Cch> ' + c.objects.Cch.label] : ['Dijkstra fallback', 'bad']));
  // jobs: each holds the state it forked from until it publishes or is discarded
  const idx = (k) => MCOLS.findIndex((c) => c.key === k);
  const lanes = [];
  for (const p of MV.pins) {
    const a = idx(p.forked), b = idx(p.until) - (p.inclusive ? 0 : 1);
    let L = lanes.findIndex((ln) => ln.every(([x, y]) => b < x || a > y));
    if (L < 0) { lanes.push([]); L = lanes.length - 1; }
    lanes[L].push([a, b, p]);
  }
  lanes.forEach((ln, k) => {
    ap(grid, rl(k ? '' : 'jobs holding a state'));
    for (const [a, b, p] of ln) ap(grid, h('span', { class: 'ib pin', style: at(a, b) }, h('span', { class: 'il' }, p.job), h('span', { class: 'ih' }, 'holds ' + (p.forked === '14w' ? MCOL.get('13').label + '’s hierarchy as its base' : MCOL.get(p.forked).label) + ' until it ' + p.end)));
    row++;
  });
  group('Arc<Cch> and Arc<TopologyVersion>');
  bars('Cch'); bars('TopologyVersion');
  group('inside the topology');
  for (const k of TOPO_F) bars(k);
  group('the metric (four arrays by arc id)');
  for (const k of METRIC_F) bars(k);
  group('retention');
  ap(grid, rl('held only by this state'));
  MCOLS.forEach((c, i) => {
    const b = h('button', { type: 'button', class: 'ib fr', style: at(i), 'aria-pressed': 'false', 'aria-label': 'What only ' + c.label + ' holds' }, h('span', { class: 'ih' }, heldShort(c)));
    b.addEventListener('click', () => { for (const x of buttons) x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); onPick('held', c.key); });
    buttons.push(b);
    ap(grid, b);
  });
  return grid;
}
function pinInset() {
  const a = MCOL.get('15'), b = MCOL.get('16');
  const rows = [];
  for (const k of ['Cch', 'TopologyVersion', ...TOPO_F, ...METRIC_F]) {
    const oa = a.objects[k], ob = b.objects[k];
    rows.push(h('span', { class: 'rl2' }, kcode(k)));
    if (oa.iid === ob.iid) rows.push(h('span', { class: 'both' }, 'shared: ' + oa.label));
    else rows.push(h('span', { class: 'one' }, oa.label), h('span', { class: 'one cp' }, ob.label + (compactHow(k, b) ? ' · ' + compactHow(k, b) : '')));
  }
  const sharedPages = (k) => { const pa = new Set(inst(k, a).pages); return inst(k, b).pages.filter((x) => pa.has(x)).length; };
  const shared = (k) => { const pa = new Set(inst(k, a).chunks); return inst(k, b).chunks.filter((x) => pa.has(x)).length; };
  const fresh = new Set([...newPages('Up.delta', b), ...newPages('Down.delta', b)]);
  const stripsFor = (k) => {
    const ca = inst(k, a).chunks, cb = inst(k, b).chunks;
    return [strip(kcode(k) + ' at ' + a.label, ca.map((cid, i) => ({ ix: i, v: cid }))), strip(kcode(k) + ' at ' + b.label, cb.map((cid, i) => ({ ix: i, v: cid, cls: ca[i] === cid ? '' : 'hl' })))];
  };
  return h('div', { style: 'display:grid;gap:14px' },
    h('div', { class: 'pin2' }, h('span', { class: 'phl' }), h('span', { class: 'ph' }, 'a reader pinned at ' + a.label), h('span', { class: 'ph' }, 'the writer that publishes ' + b.label), rows),
    P('The ' + b.label + ' writer cloned ' + a.label + '’s hierarchy: every structure was shared. Its inserts then copied the pages on their paths, and its metric writes copied the chunks they touched. Up.delta shares ' + sharedPages('Up.delta') + ' of its ' + inst('Up.delta', b).pages.length + ' pages with ' + a.label + ', and Down.delta ' + sharedPages('Down.delta') + ' of ' + inst('Down.delta', b).pages.length + '; each metric array shares ' + METRIC_F.map((k) => shared(k)).join(', ') + ' of its ' + inst('metric up', b).chunks.length + ' chunks.'),
    h('div', { class: 'arrays' },
      treeFig({ root: inst('Up.delta', a).root, pages: inst('Up.delta', a).pages.map((x) => MV.pages[String(x)]) }, a.label + ' Up.delta, the reader’s root.', { aria: 'Up.delta pages at ' + a.label }),
      treeFig({ root: inst('Up.delta', b).root, pages: inst('Up.delta', b).pages.map((x) => MV.pages[String(x)]) }, b.label + ' Up.delta, the writer’s new root. Blue pages are new; every other page is the same page the reader holds.', { copied: fresh, aria: 'Up.delta pages at ' + b.label + ', new pages highlighted' })),
    h('figure', null, h('div', { class: 'pane arrays' }, ...METRIC_F.slice(0, 2).flatMap(stripsFor)), h('figcaption', null, 'Chunk ids of two metric arrays; the small number is the chunk’s index. A blue id is a chunk the ' + b.label + ' writer copied; every other chunk is the same chunk the reader holds.')));
}
function mvccSection() {
  const sec = dsec('d-mvcc', 6, '03 §2 versioning column, §6b MVCC, §7 MVCC and replication', 'Versions after the build: what is shared and what is copied',
    'Every write goes to a private version, and readers swap one Arc<Cch>. The grid follows each structure across the published states after the build. One bar spans the states that hold the same instance; a new bar starts where a state copied or rebuilt that structure, and says what it copied. Two states share a data version when a job publishes (' + TWO_STATES() + '): a job’s publish keeps the version number. Pick a bar to see that instance’s contents.');
  const detail = h('div');
  const show = (kind, key) => {
    detail.textContent = '';
    const c = MCOL.get(key);
    if (kind === 'held') ap(detail, h('div', { class: 'detail', 'aria-live': 'polite' }, h('h3', null, 'What only ' + c.label + ' holds'), P('The instances no other published state holds. Arc counts references, so they are freed when the last holder of ' + c.label + ' lets go: its entry, a reader still running on it, or a job forked from it. The order states drop in does not matter.'), P(heldText(c)[0] + '.')));
    else ap(detail, objDetail(kind, c));
    focusPanes(detail);
  };
  const init = { kind: 'metric up', key: '14' };
  const s7c = MCOLS.find((c) => c.s7_met && c.s7_met.length);
  ap(sec, h('figure', null, h('div', { class: 'pane scroll' }, mvGrid(show, init)),
    h('figcaption', null, 'Instance numbers count copies of one structure in order; bases are named by the build or fold that made them. Amber text says what the state copied when it made the instance. Bars alternate between two shades only so that neighbouring instances are easy to tell apart; the shade means nothing. The tinted column is the Building state: it holds ' + MCOL.get('13').label + '’s hierarchy as the Repair’s base, and serves nothing.'),
    s7c ? h('div', { class: 'callout warn' }, h('span', { class: 'bh' }, '* S7 met at ' + s7c.label), P('The design re-orders here: S7 starts a new generation through the batched build, and new versions answer from the Dijkstra fallback (' + andList(s7c.s7_met) + '). The toy stays on one lineage to show more commits. See ', h('a', { href: '#d-upkeep' }, 'Upkeep'), '.')) : null));
  ap(sec, detail);
  show(init.kind, init.key);
  const pin = pinInset();
  ap(sec, h('h3', { class: 'sub-h' }, 'A reader pinned at ' + LB('15') + ' while the ' + LB('16') + ' writer commits'), pin);
  focusPanes(pin);
  const c15 = MCOL.get('15');
  ap(sec, h('h3', { class: 'sub-h' }, 'Retention: what a state alone holds, and when the log is kept'),
    h('div', { class: 'two' },
      h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'what a state alone holds'), h('ul', null,
        h('li', null, 'A structure is freed when no state, reader or job holds it any more. Arc counts the references, so the order states drop in does not matter, and a reader can keep an old state, even ' + FIRST + ', alive as long as it runs.'),
        h('li', null, 'The last grid row lists what only that state holds: the instances no other published state shares. They go when that state’s last holder lets go.'),
        h('li', null, 'When the ' + c15.label + ' reader above finishes, and the fold job that forked at ' + c15.label + ' has published, what only ' + c15.label + ' holds goes: ' + heldText(c15)[0] + '.'),
        (() => { const c13 = MCOL.get('13'), cw = MCOL.get('14w'); return h('li', null, c13.label + ' and the hand-off state hold the same hierarchy (the Repair’s base), so neither holds it alone: ' + c13.label + ' alone holds ' + heldText(c13)[0] + ', and the hand-off state alone holds ' + heldText(cw)[0] + '. The hierarchy goes when both are gone.'); })(),
        (() => { const b1 = MCOLS[0].objects['Up.base'], sp = MCOLS.filter((c) => c.objects['Up.base'].iid === b1.iid); return h('li', null, b1.label[0].toUpperCase() + b1.label.slice(1) + ' is held by every state from ' + sp[0].label + ' to ' + sp[sp.length - 1].label + ', so it goes only when the last of them is gone.'); })())),
      h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'when the log is kept (03 §7)'), h('ul', null,
        h('li', null, 'A Building entry keeps its log: the next phase consumes it.'),
        h('li', null, 'A Built entry with no upkeep job drops its log at every writer’s publish.'),
        h('li', null, 'While an upkeep job is in flight (upkeep = Some{since}), writers keep their nodes with seq above since, until the job publishes, is discarded or is given up past the bound.'),
        h('li', null, 'In the grid: the log row shows seq ' + MCOLS.filter((c) => c.log.length && c.entry.state === 'Built').map((c) => c.log.map((x) => x.seq).join(', ') + ' kept at ' + c.label).join(' and seq ') + '; every other Built state has an empty log.')))));
  return sec;
}

/* ===================== 7. What each commit does after the build ===================== */
function classifyText(cl) {
  const out = [];
  if (cl.metric_only.length) out.push(h('li', null, 'metric only: the pair already has arc ', C(cl.metric_only.map(farc).join(', '))));
  if (cl.s2_insert.length) out.push(h('li', null, 'S2 insert: no arc yet for ', C(cl.s2_insert.map(farc).join(', '))));
  for (const x of cl.nothing) out.push(h('li', null, 'nothing for ', C(fpair(x.pair)), ': ' + x.why));
  return h('ul', null, out);
}
function writtenList(ws, col) {
  const by = new Map(ws.map((w) => [w.kind, w]));
  const c = MCOL.get(col);
  const part = (ks) => ks.filter((k) => by.has(k)).map((k) => h('span', null, C(kcode(k)), ' ' + compactHow(k, c)));
  const sepd = (xs) => xs.flatMap((x, i) => (i ? ['; ', x] : [x]));
  const lines = [];
  if (by.has('Cch')) lines.push(['Arc<Cch> ' + by.get('Cch').label + (by.has('TopologyVersion') ? ', with a new topology ' + by.get('TopologyVersion').label : ', over the same topology ' + c.objects.TopologyVersion.label)]);
  for (const ks of [['Up.base', 'Down.base'], ['Up.delta', 'Down.delta'], ['rank_of', 'node_of', 'free_ranks'], ['LineageStats', 'added_since_fold'], METRIC_F]) {
    const xs = part(ks);
    if (xs.length) lines.push(sepd(xs));
  }
  return h('ul', null, lines.map((l) => h('li', null, l)));
}
function commitCard(c) {
  const dl = h('dl');
  const row = (k, ...v) => ap(dl, h('dt', null, k), h('dd', null, v));
  let card;
  if (c.actor === 'writer') {
    const over = c.stopped != null;
    card = h('article', { class: 'card' + (over ? ' over' : '') }, h('div', { class: 'ct' }, h('b', null, 'v' + c.version), h('span', { class: 'tag' }, 'writer, under the lock'), h('span', { class: 'tag' }, over ? 'over the budget' : 'within the budget')));
    row('change', c.what);
    row('classified', classifyText(c.classify));
    if (c.allocation.length) row('ranks', h('ul', null, c.allocation.map((a) => h('li', null, 'node ' + a.node + ' (degree ' + a.degree + ') → ' + a.kind + ' rank ' + a.rank))));
    if (c.new_arcs.length) row('new arcs', 'S2 adds ' + plural(c.new_arcs.length, 'arc') + ', arc ids ' + c.new_arcs[0][2] + '–' + c.new_arcs[c.new_arcs.length - 1][2] + ', preset to ∞');
    row('work', bar(over ? c.stopped.work : c.work, c.budget), h('span', { class: 'barlab' }, over ? 'passed B_gil at pop ' + c.stopped.pop + ': ' + c.stopped.work + ' of ' + c.budget + ' units; finishing would take ' + c.full_cost.work + ' units' : c.work + ' of ' + c.budget + ' units, ' + c.path + ' recompute from a front of ' + plural(c.front, 'arc')));
    if (c.written) row('writes', writtenList(c.written, c.col));
    else if (over) row('writes', 'None of the attempt’s copies is published. The attempt had copied: ', h('ul', null, Object.entries(MV.attempt_v14).map(([k, v]) => h('li', null, C(kcode(k)), ': ' + v))), 'All of it is dropped.');
    row('outcome', c.outcome);
    const col = c.col ? MCOL.get(c.col) : null, firstS7 = MCOLS.find((x) => x.s7_met && x.s7_met.length);
    if (col && firstS7 && col.key === firstS7.key) row('decay', h('span', { class: 'fail' }, 'S7 met: the design re-orders here; the toy stays on one lineage to show more commits.'), ' Signals met: ' + andList(col.s7_met) + '.');
  } else {
    const publishes = !!c.fit;
    card = h('article', { class: 'card job' }, h('div', { class: 'ct' }, h('b', null, c.name), h('span', { class: 'tag' }, publishes ? 'job publish, under the lock' : 'job: runs off the lock')));
    row('fork', 'v' + c.fork);
    if (c.fit) row('publish', B(c.fit), c.why ? ': ' + c.why : '');
    if (publishes) row('work', c.path ? 'off the lock, on the builder thread; only the publish takes the lock' : c.fit === 'Gone' ? 'its work off the lock is thrown away: the publish finds a new generation' : 'built off the lock; the publish takes the lock');
    if (c.path) row('path', c.path + ' (the valve: ' + ST.F5.data.repair.valve + ', so seeds are written by arc id, one relax sweep runs, and the metric is chunked from flat arrays)');
    if (c.note) row('note', c.note);
    if (c.written) row('writes', writtenList(c.written, c.col));
    if (c.outcome) row('outcome', c.outcome);
  }
  ap(dl, h('dt', null, 'entry after'), h('dd', null, entryText(c.entry) + (c.published_log.length ? ' · log ' + c.published_log.map((n) => 'seq ' + n.seq).join(', ') : '')));
  ap(card, dl);
  return card;
}
function commitsSection() {
  const sec = dsec('d-commits', 7, '03 §6 Incremental maintenance, §6b Recomputing modified edges', 'What each commit does after the build',
    'Once Built, the order never changes. Each commit flushes on its private version, under the lock: append its log node, resolve the pairs against the new graph, give ranks to new nodes, classify each pair, add arcs by fill repair (S2) where a pair has none, recompute the affected weights, then publish. One budget of work units, B_gil, covers the whole lock hold (toy: ' + BUDGET + ' units). Jobs run off the lock and publish only if their generation and ticket still match. How the scoped recompute walks its front, pop by pop, is in the appendix.');
  ap(sec, h('div', { class: 'cards' }, MV.cards.map(commitCard)));
  ap(sec, note('Arcs and pairs in the classification are rank pairs (lo, hi). Work units count apex-merge steps, heap pushes and seed lookups, plus S2’s row reads and delta inserts; see ', h('a', { href: '#terms' }, 'words used on this page'), '.'));
  return sec;
}

/* ===================== 8. Over the budget: hand-off and Repair ===================== */
function handoffSection() {
  const d = ST.F5.data, at = d.attempt, rp = d.repair;
  const cw = CARD((c) => c.col === '14w'), cg = CARD((c) => c.fit === 'Gone'), cr = CARD((c) => c.col === '14');
  const before = CARD((c) => c.col === '13');
  const c13 = MCOL.get('13'), c14 = MCOL.get('14');
  const sec = dsec('d-handoff', 8, '03 §6 Hand-off, §5 Phases (Repair), §6b Where it runs, and the bound', 'Over the budget: hand-off and Repair',
    'A commit whose work would pass B_gil does not finish under the lock. It hands the work to a Repair job on the builder thread, and new versions answer from the Dijkstra fallback until the Repair adopts. In the example, the v' + cw.version + ' insert joins the two sides of the top separator (nodes ' + andList(d.top_separator) + '), which gives a large cone.');
  ap(sec, h('ol', { class: 'steps' },
    h('li', { class: 'w' }, h('span', { class: 'sh' }, 'v' + cw.version + ': the writer’s attempt, under the lock'), P(cw.what + '. S2 adds ' + plural(at.new_arcs, 'arc') + ' (' + at.work.s2 + ' units). The recompute checks the running total after every pop; at pop ' + at.stopped.pop + ' it reaches ' + at.stopped.work + ', over B_gil = ' + BUDGET + '. Finishing would have taken ' + at.full_cost.pops + ' pops and ' + at.full_cost.work.total + ' units.')),
    h('li', { class: 'bad' }, h('span', { class: 'sh' }, 'abort'), P('The attempt’s private copies are dropped; none was ever published:'), h('ul', { class: 'plain' }, Object.entries(MV.attempt_v14).map(([k, v]) => h('li', null, C(kcode(k)), ': ' + v)))),
    h('li', { class: 'bad' }, h('span', { class: 'sh' }, 'hand-off: a new generation'), P('The entry becomes ', C('Building{Repair, base, log: [seq ' + cw.version + ']}'), ' under generation ' + d.generation + ' (it was ' + before.entry.generation + '). The base is the hierarchy at the abort: ' + c13.label + '’s Arc<Cch> ' + c13.objects.Cch.label + ', shared, not copied. The new generation is what makes this commit enqueue the Repair job.')),
    h('li', { class: 'j' }, h('span', { class: 'sh' }, 'the fold in flight is discarded'), P(cg.name + ', forked at v' + cg.fork + ': ' + cg.why + ', so its publish fits as ', B(cg.fit), '. ' + cg.outcome[0].toUpperCase() + cg.outcome.slice(1) + '.')),
    h('li', null, h('span', { class: 'sh' }, 'readers meanwhile'), P('A query at v' + cw.version + ' uses the Dijkstra fallback. A reader that started at ' + c13.label + ' keeps Arc<Cch> ' + c13.objects.Cch.label + ': the same instance the Repair uses as its base.')),
    h('li', { class: 'j' }, h('span', { class: 'sh' }, 'the Repair, on the builder thread, off the lock'), P('Repair batches go ahead of column batches on the builder queue. It forks at v' + cr.fork + ', resolves the log there, and runs S2 (' + plural(rp.s2.new_arcs.length, 'arc') + '). ' + rp.valve[0].toUpperCase() + rp.valve.slice(1) + ', so it takes the full path: the full-seed pass writes every seed by arc id (from Ax for the base and from each entry for the delta), then one relax sweep runs over ' + rp.sweep_triangles + ' triangles. As the Seal does, it builds the metric as flat arrays and chunks them with WeightArr::from_vec, so every chunk is new and none is shared with the base. Off the lock there is no budget.')),
    h('li', { class: 'ok' }, h('span', { class: 'sh' }, 'adoption'), P('It publishes ', B(cr.fit), ': ' + cr.why + '. ' + cr.outcome[0].toUpperCase() + cr.outcome.slice(1) + '. Readers of v' + c14.version + ' now take Arc<Cch> ' + c14.objects.Cch.label + '.'))));
  ap(sec, note('This hand-off takes a new generation. The other kind, an adoption that does not fit B_gil, keeps the generation and the running job (Absorbed::Restart); it does not happen in this example.'));
  ap(sec, h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'at real scale: what the design gives up'), h('ul', null, D.design_refs.real_scale.map((x) => h('li', null, B(x.what[0].toUpperCase() + x.what.slice(1)), ': ' + x.value + ' (' + x.source + ').')))));
  return sec;
}

/* ===================== 9. Upkeep: decay signals and the fold ===================== */
function decayTable() {
  const vers = Object.keys(D.decay).sort((a, b) => +a - +b);
  const sig = D.decay[vers[0]].map((x) => x.job + ': ' + x.signal);
  const rows = sig.map((name, k) => [prose(name), ...vers.map((v) => { const x = D.decay[v][k]; return { v: h('span', { class: x.met ? 'met' : 'no' }, fnum(x.value) + ' / ' + fnum(x.threshold)) }; })]);
  return table(['signal', ...vers.map((v) => 'v' + v)], rows, { caption: 'Decay signals after each writer flush: value / threshold; amber = met. At v' + MCOL.get('14w').version + ' the values are the attempt’s.' });
}
function upkeepSection() {
  const f6 = ST.F6.data;
  const c12 = CARD((c) => c.col === '12'), c13 = CARD((c) => c.col === '13'), c15 = CARD((c) => c.col === '15'), c16 = CARD((c) => c.col === '16');
  const jf = CARD((c) => c.col === '16f'), jg = CARD((c) => c.fit === 'Gone');
  const cf = MCOL.get('16f');
  const kept = MV.fields.topology.concat(MV.fields.metric).filter((k) => !cf.objects[k].new);
  const sec = dsec('d-upkeep', 9, '03 §6 Decay, Built-entry jobs (decision D6), Fold job; §7 Retention', 'Upkeep: decay signals and the fold',
    'Each writer flush checks the decay signals. The delta-size signal starts a fold: a Built-entry job that rebuilds the bases from base plus delta while the entry stays Built and keeps serving. The other signals would start S6 (a new lineage from the current order) or S7 (a re-order through the batched build); the model evaluates them but does not run them.');
  const s7c = MCOLS.find((c) => c.s7_met && c.s7_met.length);
  if (s7c) ap(sec, h('div', { class: 'callout warn' }, h('span', { class: 'bh' }, 'from ' + s7c.label + ' on, the design would be re-ordering'), P(D.example.notes.find((n) => /^Decay signals/.test(n)))));
  ap(sec, decayTable());
  ap(sec, h('ol', { class: 'steps' },
    h('li', { class: 'w' }, h('span', { class: 'sh' }, 'v' + c12.version + ': the trigger, and a ticket'), P(c12.outcome + '. The ticket comes from the same counter as generations, and this commit enqueues the job.')),
    h('li', { class: 'j' }, h('span', { class: 'sh' }, jg.name + ': forked, then discarded'), P('It forks at v' + jg.fork + ' and builds off the lock. While it runs, upkeep is set, so v' + c13.version + ' keeps its log node: ' + c13.published_log.map((n) => 'seq ' + n.seq).join(', ') + '. At v' + MCOL.get('14w').version + ' the hand-off takes a new generation, so this job is ', B(jg.fit), ' and publishes nothing.')),
    h('li', { class: 'w' }, h('span', { class: 'sh' }, 'v' + c15.version + ': the trigger again'), P(c15.outcome + '.')),
    h('li', { class: 'j' }, h('span', { class: 'sh' }, 'the fold job, off the lock'), P('It forks at v' + f6.fork + ' and builds ' + cf.objects['Up.base'].label + ' from ' + MCOLS[0].objects['Up.base'].label + ' ⊕ Up.delta with GraphBLAS: an eWiseAdd over read-only loads, then an unload. Ids are carried as values, so no arc changes its id: ' + f6.new_base.Up.Ax.length + ' arcs.')),
    h('li', { class: 'w' }, h('span', { class: 'sh' }, 'v' + c16.version + ': a writer while the fold runs'), P(c16.what + '. S2 adds ' + plural(c16.new_arcs.length, 'arc') + ' to v' + c16.version + '’s own deltas. Upkeep is set, so the log keeps its node: ' + c16.published_log.map((n) => 'seq ' + n.seq).join(', ') + '.')),
    h('li', { class: 'ok' }, h('span', { class: 'sh' }, 'the fold publishes, under the lock'), P('Fit against ' + jf.fit + '. The result is the published ' + LB('16') + ' Arc<Cch> with only the bases, the deltas and added_since_fold replaced. The new deltas hold exactly the arcs added after the fork (seq > ' + f6.fork + '): ' + f6.added_since_fold_after.map(([q, a]) => 'seq ' + q + ', ' + plural(a.length, 'arc')).join('; ') + '. Every other structure is the published copy’s instance: ' + kept.map(kcode).join(', ') + '. Then upkeep and the log are cleared.'))));
  const FD = D.design_refs.fold;
  ap(sec, h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'two rules the example does not reach (' + FD.source + ')'), h('ul', null,
    h('li', null, B('A writer the fold overtakes. '), FD.overtaken),
    h('li', null, B('Why the log is kept while a fold runs. '), FD.retention))));
  const go = h('a', { href: '#d-construct' }, 'the last frame of the construct');
  go.addEventListener('click', () => { const st = document.getElementById('st-track'); const rng = st.querySelector('input[type=range]'); rng.value = rng.max; rng.dispatchEvent(new Event('input', { bubbles: true })); });
  ap(sec, P('The fold’s result, ' + cf.objects['Up.base'].label + ' with the ids that came from the delta, is drawn in ', go, '.'));
  return sec;
}

/* ===================== 10. Deletions ===================== */
function deletionsSection() {
  const b1 = ST.B1.data, c2 = ST.C2.data, f4 = ST.F4.data, f8 = ST.F8.data, A = ST.A.data;
  const dm = A.pending.dm_road, dropped = c2.resolved.filter((r) => !r.live_edge), added = c2.resolved.filter((r) => r.live_edge);
  const f4c = CARD((c) => c.col === '13'), f8c = CARD((c) => c.col === '17');
  const sec = dsec('d-deletions', 10, 'revision 3.1: 03 §3 Input skeleton, §4 step 2; §6 Node delete', 'Deletions',
    'Revision 3.1 makes the build exact: no arc is made for an edge that is already dead. After the build, a deleted node keeps its rank and its arcs, and the recompute sends their weights to ∞.');
  ap(sec, h('div', { class: 'two' },
    h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'pending deletions are masked in the skeleton read'),
      P('Road ' + fpair(dm) + ' is deleted when the index is created, but its m entry is still there with a dm bit set. Each slab reads ', C('m⟨¬dm⟩ ∪ dp'), ', so S has ' + b1.nnz + ' entries instead of ' + b1.S_without_dm_mask.length + ' and no arc is ever made for the dead road. Without the mask, its arc and fill would stay for the life of the lineage.')),
    h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'Supergraph drops logged pairs with no live edge'),
      P('A delete written after its slab was read is in the log. Supergraph resolves each logged pair at its fork: ', C('B′ = (B ∖ D) ∪ A'), '. Dropped (D): ' + (dropped.map((r) => fpair(r.pair)).join(', ') || 'none') + '. Added (A): ' + (added.map((r) => fpair(r.pair)).join(', ') || 'none') + '. So B′ is exactly the live pattern at the fork (check G8).'))));
  ap(sec, h('div', { class: 'two' },
    h('div', { style: 'display:grid;gap:8px;align-content:start' },
      h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'a deleted node keeps its rank and arcs (v' + f4c.version + ')'),
        (() => {
          const rr = f4.arcs_at_rank.map(([lo, hi]) => [lo, hi]).reduce((a, b) => a.filter((x) => b.includes(x)));
          const roads = f4c.what.split(';').filter((x) => /cascade: delete/.test(x)).length;
          return P(f4c.what.split(';').slice(-1)[0].trim()[0].toUpperCase() + f4c.what.split(';').slice(-1)[0].trim().slice(1) + '. The cascade deletes its ' + plural(roads, 'relationship') + ' and logs one pair per neighbour, ' + plural(f4.flush.pairs.length, 'pair') + '. The node keeps rank ' + rr[0] + ' and all ' + plural(f4.arcs_at_rank.length, 'arc') + ' at that rank. The front is the arcs of the logged pairs, and every other arc at the rank reaches ∞ through the triangles. LineageStats counts the deletions, so this commit copies it; the S6 signal reads that count.');
        })()),
      table(['lo', 'hi', 'id', 'up', 'down'], f4.arcs_at_rank.map(([lo, hi, id, u, dn]) => [lo, hi, id, fw(u), fw(dn)]), { caption: 'Every arc at the deleted node’s rank after the recompute' })),
    h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'a reused id takes the rank back (v' + f8c.version + ')'),
      P(f8c.what + '. The new node gets the freed id, and with it rank ' + f8.rank + ' and its ' + plural(f8.arcs_at_rank_before.length, 'arc') + ', still at ∞. Its road lands on arc ' + f8.arc_id + ' ' + farc(f8.fill_arc) + ', a fill arc the build already made, so the change is metric only: no rank, no S2.'))));
  return sec;
}

/* ===================== 11. Replicas and RDB ===================== */
function replicasSection() {
  const sec = dsec('d-replicas', 11, '03 §7 Determinism, Replicas, RDB', 'Replicas and RDB');
  ap(sec, h('div', { class: 'col', style: 'display:grid;gap:10px' },
    P('A replica applies each commit through ', C('GRAPH.EFFECT'), ', which flushes inline under the lock with the same steps and the same shared B_gil as the master. Over the budget it hands off to a Repair the same way.'),
    P('Replicas promise equal distances, not equal hierarchies. Arc ids and ranks depend on how commits are grouped and on fork points: the order S2 numbers new arcs in, which node takes which pool or top rank, and whether a Repair sees an insert that was deleted before its fork. The values are exact on every node; paths of equal cost may differ. ' + D.design_refs.replicas.text + ' (' + D.design_refs.replicas.source + '). The toy’s weights are whole numbers, so its sums are exact.'),
    P('An RDB saves only the index definition, as today, and a load rebuilds the index. Saving rank_of to skip the ordering on load is decision D4.')));
  return sec;
}

/* ===================== 12. Checks and open questions ===================== */
function checksSection() {
  const sec = dsec('d-checks', 12, '03 §9 M5', 'Checks and open design questions',
    'The model checks each invariant where its step runs, then again with one fix removed or one input corrupted. A check that cannot fail proves nothing, so every check below has at least one deliberate break, and each break must make it fail.');
  const g = stepG(ST.G);
  const [statsEl, tableEl] = [...g.children].slice(2);
  ap(sec, statsEl, h('details', { class: 'checks-table' }, h('summary', null, 'Show every check, with its deliberate breaks'), tableEl));
  ap(sec, h('h3', { class: 'sub-h', id: 'questions' }, 'Where the design reads two ways'),
    P('The model had to pick one reading in each of these places. Each says which one it picked.'),
    table(['where', 'question'], D.design_questions.map((q) => [prose(q.where), prose(q.question)]), { stack: true }));
  ap(sec, h('h3', { class: 'sub-h', id: 'toy-scale' }, 'Toy scale'),
    P('The example is tiny, so some thresholds are scaled down. Each row gives the value the model uses, the real value, and its source.'),
    table(['threshold', 'toy', 'real', 'source'], D.thresholds.map((t) => [prose(t.name), prose(String(t.toy)), prose(String(t.real)), prose(t.source)]), { stack: true }),
    h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'where the model is simpler than the design'), h('ul', null,
      ['GraphBLAS calls run as plain host loops with the same inputs and outputs; each step lists the call as the design writes it.',
        'The matching salt is fixed, so the run repeats exactly. The first separator and node FM are simpler than METIS’s.',
        'Work units have no clock. The cow_btree pages and metric chunks are toy sized: ' + thr('cow_btree page').toy + '; metric chunks of ' + thr('metric chunk').toy + '.',
        'The build timeline is scripted. Overflow, writer rebase and Restart at adoption never occur.',
        'S6 and S7 are evaluated but never run, so the later states keep one lineage.',
        'TopologyVersion and Cch identities are derived at each publish from their fields’ identities: a new struct exactly when a field is new.',
        'The full list, each item with why the step still holds, is SIMPLIFICATIONS.md beside the model script.'].map((x) => h('li', null, x)))));
  return sec;
}

/* ===================== appendix: algorithm detail ===================== */
function recomputeAppendix() {
  const out = [];
  const one = (id, title, f, hk, extra) => {
    const Hh = hier(hk);
    const sec = h('section', { id, 'aria-labelledby': id + '-h' }, h('div', { class: 'fnhead' }, h('span', { class: 'where' }, '03 §6, §6b'), h('h2', { id: id + '-h' }, title)));
    if (f.s2) ap(sec, s2Stepper(id + '-s2', f.s2, { oldArcs: Hh.arcs.filter((a) => !f.s2.new_arcs.some((x) => x[2] === a.id)), nodeOf: Hh.nodeOf }));
    ap(sec, recomputeStepper(id + '-rc', Object.assign({ pops: f.recompute, front: f.front, arcs: Hh.arcs, nodeOf: Hh.nodeOf, budget: f.budget, start: (f.work && f.work.s2) || 0,
      frontWhy: f.s2 ? 'The inserted pairs’ arcs plus every arc S2 added (Lemma A: new arcs start at ∞, so only these can start a change).' : f.classify.metric_only.length > 1 ? 'The arcs of the logged pairs, each with its new seed.' : 'The pair already has an arc, so the front is that arc with its new seeds.' }, extra || {})));
    out.push(sec);
  };
  const f = (id) => ST[id].data;
  one('ap-v10', 'v' + f('F1').flush.seq + ': ' + f('F1').journal[0], f('F1').flush, f('F1').flush.seq);
  one('ap-v11', 'v' + f('F2').flush.seq + ': fill repair, then the recompute', f('F2').flush, f('F2').flush.seq);
  one('ap-v12', 'v' + f('F3').flush.seq + ': new nodes, fill repair, the recompute', f('F3').flush, f('F3').flush.seq);
  one('ap-v13', 'v' + f('F4').flush.seq + ': the node delete’s recompute', f('F4').flush, f('F4').flush.seq);
  const d5 = f('F5'), Hh = hier(d5.repair.seq);
  const s5 = h('section', { id: 'ap-v14', 'aria-labelledby': 'ap-v14-h' }, h('div', { class: 'fnhead' }, h('span', { class: 'where' }, '03 §6 Hand-off, §6b'), h('h2', { id: 'ap-v14-h' }, 'v' + d5.repair.seq + ': the attempt that passes B_gil, and the Repair’s fill repair')));
  ap(s5, recomputeStepper('ap-v14-attempt', { title: 'The attempt: the running total is checked after every pop', pops: d5.attempt.recompute_until_stop, front: d5.repair.front, arcs: Hh.arcs, nodeOf: Hh.nodeOf, budget: BUDGET, start: d5.attempt.work.s2, max: d5.attempt.stopped.work,
    frontWhy: 'S2 ran first and put every new arc in the front. The writer pops in (lo, hi) order and stops at the first check over the budget.' }));
  ap(s5, s2Stepper('ap-v14-s2', d5.repair.s2, { title: 'The Repair’s fill repair (S2), off the lock', oldArcs: Hh.arcs.filter((a) => !d5.repair.s2.new_arcs.some((x) => x[2] === a.id)), nodeOf: Hh.nodeOf }));
  out.push(s5);
  one('ap-v15', 'v' + f('F6').v15.flush.seq + ': a weight SET', f('F6').v15.flush, f('F6').v15.flush.seq);
  one('ap-v16', 'v' + f('F6').v16.flush.seq + ': an insert while the fold runs', f('F6').v16.flush, f('F6').v16.flush.seq);
  one('ap-v17', 'v' + f('F8').flush.seq + ': the reused id’s road on a fill arc', f('F8').flush, f('F8').flush.seq);
  return out;
}
function appendix() {
  const det = h('details', { class: 'appx', id: 'appendix' }, h('summary', null, 'Appendix: algorithm detail on the example'));
  ap(det, h('div', { class: 'appx-intro' },
    P('The algorithm step by step, on the same example: ordering (steps B1–B11), the supergraph (steps C1–C5), the weights (steps D1–D3), and the pop-by-pop recompute of every commit after the build. How each algorithm works in general is explained in ', ext(LINK_EXPLAINER, 'the CCH explainer'), '.'),
    h('nav', { class: 'toc', 'aria-label': 'Appendix steps' }, D.steps.filter((s) => /^[BCD]/.test(s.part)).map((s) => h('a', { href: '#' + sid(s.id) }, 'step ' + s.id + ' ' + shortName(s))), h('a', { href: '#ap-matching' }, 'one matching round'), h('a', { href: '#ap-recompute' }, 'recompute, commit by commit'))));
  const byPart = new Map();
  for (const s of D.steps) if (/^[BCD]$/.test(s.part)) { if (!byPart.has(s.part)) byPart.set(s.part, []); byPart.get(s.part).push(s); }
  for (const [p, steps] of byPart) {
    ap(det, h('div', { class: 'part', id: 'part-' + p.toLowerCase() }, h('span', { class: 'eyebrow' }, 'Part ' + p), h('h2', null, PARTS[p][1]), h('p', { class: 'lede col' }, PARTS[p][2])));
    for (const s of steps) {
      let el;
      try { el = (RENDER[s.id] || section)(s); } catch (err) {
        el = section(s);
        el.append(h('p', { class: 'note' }, 'This step’s figures failed to render: ' + err.message));
        if (window.console) console.error(s.id, err);
      }
      ap(det, el);
      if (s.id === 'B5.2') ap(det, matchingRound());
    }
  }
  ap(det, h('div', { class: 'part', id: 'ap-recompute' }, h('span', { class: 'eyebrow' }, 'Part F'), h('h2', null, 'Recompute and fill repair, commit by commit'), h('p', { class: 'lede col' }, 'Each commit after the build, one heap pop at a time: the arc being recomputed, its lower detours, what it pushes, and the work so far against the budget.')));
  ap(det, recomputeAppendix());
  return det;
}

/* ===================== the page ===================== */
function build() {
  const spec = D.spec;
  document.getElementById('spec-design').textContent = spec[0] + ', with the proposed changes in ' + spec[1].split(' ')[0] + ' (not yet merged into 03)';
  document.getElementById('eyebrow').textContent = 'Design proposal · 03-design.md rev 3.1 · toy model';
  document.getElementById('sources').textContent = 'Sources: ' + spec.join('; ') + '. Every toy number on this page is computed by walkthrough.py, a model of the design at toy scale (not FalkorDB code); real-scale values are quoted from 03 with their section.';
  const ch = document.getElementById('changes');
  ap(ch, h('h2', { class: 'sub-h' }, 'What the design changes'),
    h('ul', { class: 'changes' },
      h('li', null, B('Ordering runs on GraphBLAS.'), ' Our own nested dissection: at each depth every cell is processed together, with matching, contraction and the split as GraphBLAS calls over one block-diagonal matrix. Separators, refinement and minimum degree stay small host loops on the builder thread.'),
      h('li', null, B('The hierarchy is a shared base plus a small delta.'), ' The base is CSR arrays owned by Rust, whose values are arc ids, loaded read-only into GraphBLAS when an operation needs them. The delta is a copy-on-write B-tree. Arc ids are never reused within a lineage, and a fold job keeps the delta small.'),
      h('li', null, B('The build runs in batches, off the lock.'), ' Writers record the node pairs they touch in a log owned by the CCH index. Every batch publishes under the lock, by seq rules: Same, or CatchUp relinking the newer nodes. Only the adoption’s absorb is real work under the lock, bounded by B_gil.'),
      h('li', null, B('After the build, each commit updates the index in place of a rebuild.'), ' New arcs come from fill repair, new nodes get a pool or a top rank, and a scoped recompute fixes the weights, all within one work budget per lock hold. Over the budget, a Repair job takes over.'),
      h('li', null, B('Readers never see half a change.'), ' Every write goes to a private version: delta pages and metric chunks are copied only where written. Readers swap one Arc<Cch>, which holds a matching topology and metric. While a Repair or a re-order runs, new versions answer from the Dijkstra fallback.')),
    h('div', { class: 'links' }, h('span', null, 'Team document: ', ext(LINK_TEAM, 'CCH redesign')), h('span', null, 'How the algorithm works: ', ext(LINK_EXPLAINER, 'the CCH explainer')), h('span', null, 'Design: ', C('design/03-design.md'), ' (revision 3.1)')));
  const S = [
    ['changes', 'what the design changes'], ['d-objects', 'what the index is made of'], ['d-construct', 'graph to index'], ['d-ordering', 'ordering on GraphBLAS'], ['d-build', 'building under writes'],
    ['d-mvcc', 'versions: shared and copied'], ['d-commits', 'each commit'], ['d-handoff', 'hand-off and Repair'], ['d-upkeep', 'upkeep and the fold'],
    ['d-deletions', 'deletions'], ['d-replicas', 'replicas and RDB'], ['d-checks', 'checks and questions'], ['terms', 'words used'], ['appendix', 'appendix: algorithm detail'],
  ];
  ap(document.getElementById('toc'), h('nav', { class: 'toc', 'aria-label': 'Sections' }, S.map(([id, t], i) => h('a', { href: '#' + id }, (i < 12 ? (i + 1) + ' ' : '') + t))));
  const main = document.getElementById('main');
  ap(main, objectsSection(), termsBlock(), constructSection(), orderingSection(), buildSection(), mvccSection(), commitsSection(), handoffSection(), upkeepSection(), deletionsSection(), replicasSection(), checksSection(), appendix());
  const openTo = () => {
    const id = decodeURIComponent(location.hash.slice(1));
    const el = id && document.getElementById(id);
    if (el) { const d = el.closest('details'); if (d && !d.open) { d.open = true; requestAnimationFrame(() => el.scrollIntoView()); } }
  };
  window.addEventListener('hashchange', openTo);
  document.addEventListener('click', (e) => { const a = e.target.closest && e.target.closest('a[href^="#"]'); if (a) { const el = document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1))); const d = el && el.closest('details'); if (d) d.open = true; } });
  openTo();
}
function termsBlock() {
  return h('div', { class: 'callout', id: 'terms', style: 'margin-top:28px' }, h('span', { class: 'bh' }, 'words used on this page'), h('ul', null, [
    [B('Version, state'), ': each commit publishes a new data version. A batch’s or a job’s publish keeps the version number, so one version can have several published states: during the build every batch publish is one, and after it two versions have a job’s publish as well (' + TWO_STATES() + ').'],
    [B('Instance'), ': one copy of a structure in memory. Two versions that share an instance hold the very same bytes.'],
    [B('Rel, arc'), ': a relationship in the graph has a relationship id (“rel 70”); an arc of the hierarchy has an arc id (“arc 70”). They are different numbers.'],
    [B('Seq'), ' (03 §5): the data version of the commit that wrote a log node. Every rule on the log compares seqs.'],
    [B('Fork'), ': the published version a batch, job or writer starts from. Its fork is that version’s number.'],
    [B('Relink'), ' (03 §5): put a published log node on top of a batch’s copy of the log. It copies the small node, not its pairs.'],
    [B('V_sg, V_c'), ' (03 §5): the versions the Supergraph batch and the Seed batch fork at. The topology covers V_sg plus the log; the metric covers V_c plus the log. Here both are the same version.'],
    [B('Seal'), ' (03 §5): the last build batch. It resolves the log written since the seed, then publishes.'],
    [B('Adoption'), ' (03 §5): the publish of the last batch (the one with next = None, meaning no batch follows). It absorbs the newer log nodes under the lock, and the entry becomes Built.'],
    [B('Front, cone'), ' (03 §6b): the front is the set of arcs a recompute starts from; the cone is every arc the change reaches from it.'],
    [B('Valve'), ' (03 §6b): off the lock only, a front over n/8 arcs switches to the full path: seeds written by arc id, then one relax sweep.'],
    [B('Gone'), ' (03 §6): what a job’s publish fits as when its generation or ticket has changed; it publishes nothing.'],
    [B('Absorbed::Restart'), ' (03 §5): an adoption or rebase that does not fit B_gil leaves the entry Building{Repair}, so a Repair batch finishes the work.'],
    [B('Work units'), D.conventions.work_units.slice(D.conventions.work_units.indexOf(':'))],
    [B('B_gil'), ' (03 §6b): the work budget of one hold of the lock, shared by every index the hold updates. Toy value: ' + BUDGET + ' units.'],
    [B('Generation'), ' (03 §6): a counter on the index entry. A hand-off to a Repair and an S7 re-order each take a new one; a job whose generation has changed is discarded (Gone).'],
    [B('Ticket'), ' (03 §6, decision D6): the number of a fold or S6 job, from the same counter. Its publish is discarded if the generation or the ticket has changed.'],
    [B('Lineage'), ' (03 §2): one run of arc ids. Within a lineage ids are only appended; a fold keeps every id. S6 and S7 start a new lineage.'],
    [B('S2'), ' (03 §6): fill repair. An insert on a pair with no arc adds that arc and every shortcut it forces, as new arcs with new ids.'],
    [B('S6, S7'), ' (03 §6): decay jobs. S6 rebuilds the arcs and weights from the current order as a new lineage while the index serves; S7 re-orders through the batched build, with readers on the Dijkstra fallback.'],
  ].map((x) => h('li', null, x))));
}
build();
