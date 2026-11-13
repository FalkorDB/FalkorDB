/* ===================== Part E ===================== */
function stepE(s) {
  const sec = section(s), d = s.data;
  ap(sec, timelineStepper('st-e', d.events, 'The build timeline, event by event'));
  ap(sec, note('Swap: ', d.swap, '. Generations: ', d.generations.map((g) => 'generation ' + g.generation + ' from v' + g.from + ' (' + g.why + ')').join('; '), '.'));
  return sec;
}

/* ===================== Part F ===================== */
const ROLE_LEG = {
  set: ['l', 'hot', 'the pair this write sets'], inserted: ['l', 'hot', 'the inserted road'], changed: ['l', 'hot', 'the changed pair'],
  new_arc: ['l', 'new', 'new arc, drawn between the nodes that hold its ranks'], cascade: ['l', 'del', 'edge deleted by the cascade'],
  new_arc_after_fork: ['l', 'new', 'arc added after the fold forked'],
  pool: ['n', 'pool', 'takes a pool rank'], top: ['n', 'top', 'takes a new top rank'], deleted: ['n', 'dead', 'deleted node: keeps its rank and arcs'],
  reused: ['n', 'cur', 'new node on a reused id'], separator: ['n', 'sep', 'top separator'],
};
function fGraph(s, v, hk) {
  const Hh = hier(hk != null ? hk : v);
  const rk = Hh.rankOf, dr = s.draw;
  const roleN = new Map();
  for (const [role, ids] of Object.entries(dr.nodes || {})) for (const n of ids) roleN.set(n, role);
  const NR = { pool: 'pool', top: 'top', deleted: 'dead', reused: 'cur', separator: 'sep' };
  const ids = [...new Set([...D.versions[String(v)].graph.nodes, ...Object.values(dr.nodes || {}).flat()])];
  const nodes = ids.map((n) => ({ id: n, cls: roleN.has(n) ? NR[roleN.get(n)] || '' : rk.has(n) ? '' : 'iso', sub: rk.has(n) ? 'r' + rk.get(n) : null }));
  const ER = { set: 'e hot', inserted: 'e hot', changed: 'e hot', new_arc: 'e new', cascade: 'e del', new_arc_after_fork: 'e new' };
  const hl = [], skip = new Set();
  for (const [role, es] of Object.entries(dr.edges || {})) for (const [a, b] of es) {
    hl.push({ a, b, cls: ER[role] || 'e hot', z: /new_arc/.test(role) ? 2 : 3 });
    if (!/new_arc/.test(role)) skip.add(key2(a, b));
  }
  const edges = [...bgEdges(relsAt(v), { faint: true, skip }), ...hl];
  const leg = [...Object.keys(dr.edges || {}), ...Object.keys(dr.nodes || {})].map((r) => ROLE_LEG[r]).filter(Boolean);
  leg.push(['l', 'faint', 'other live edges']);
  return h('div', { style: 'display:grid;gap:10px;align-content:start' },
    wrapFig(graph({ nodes, edges, aria: s.title }), 'The graph at v' + v + '. The small label is each node’s rank.'), legend(leg));
}
function decayChips(list) {
  return chips(list.map((x) => [x.job + ': ' + x.signal + ' · ' + fnum(x.value) + ' vs ' + fnum(x.threshold) + (x.met ? ' · met' : ''), x.met ? 'fil' : 'muted']));
}
function journalBlock(j) { return blk('the commit', h('ul', null, j.map((x) => h('li', null, x)))); }

function stepFgen(s, extra) {
  const sec = section(s), d = s.data, f = d.flush;
  const v = f.seq, Hh = hier(v);
  ap(sec, h('div', { class: 'two' }, fGraph(s, v),
    h('div', { style: 'display:grid;gap:10px;align-content:start' }, d.journal ? journalBlock(d.journal) : null, ...flushBlocks(f))));
  if (f.s2) ap(sec, s2Stepper('st-' + sid(s.id) + '-s2', f.s2, { oldArcs: Hh.arcs.filter((a) => !f.s2.new_arcs.some((x) => x[2] === a.id)), nodeOf: Hh.nodeOf }));
  ap(sec, recomputeStepper('st-' + sid(s.id) + '-rc', { pops: f.recompute, front: f.front, arcs: Hh.arcs, nodeOf: Hh.nodeOf, budget: f.budget, start: f.work.s2 || 0,
    frontWhy: f.s2 ? 'The inserted pairs’ arcs plus every arc S2 added (Lemma A: new arcs start at ∞, so only these can start a change).' : f.classify.metric_only.length > 1 ? 'The arcs of the logged pairs, each with its new seed.' : 'The pair already has an arc, so the front is that arc with its new seeds.' }));
  ap(sec, h('div', { class: 'two' }, h('div', null, h('span', { class: 'bh' }, 'work under the lock'), workTable(f.work, f.budget)),
    h('div', { style: 'display:grid;gap:6px' }, P('Equals a full customization of the same topology, bit for bit: ', yesno(d.equals_full), '.'),
      d.s2_equals_game != null ? P('S2’s arcs equal the elimination game over the old arcs plus the new pair: ', yesno(d.s2_equals_game), '.') : null)));
  if (extra) extra(sec, d, Hh);
  if (d.decay) ap(sec, blk('decay signals after this flush (03 §6)', decayChips(d.decay)));
  return sec;
}
function btree(delta, o) {
  o = o || {};
  if (delta.root == null) return P(h('span', { class: 'no' }, 'empty'));
  const pages = new Map(delta.pages.map((p) => [p.page, p]));
  const pos = new Map();
  let leaf = 0, maxd = 0;
  const visit = (id, dp) => {
    maxd = Math.max(maxd, dp);
    const p = pages.get(id);
    if (!p || p.kind === 'leaf') { pos.set(id, [leaf++, dp]); return; }
    for (const c of p.children) visit(c, dp + 1);
    const xs = p.children.map((c) => pos.get(c)[0]);
    pos.set(id, [(Math.min(...xs) + Math.max(...xs)) / 2, dp]);
  };
  visit(delta.root, 0);
  const bw = 76, sx = 84, sy = 84, pad = 8, lh = 12.5;
  const W = Math.round(pad * 2 + (leaf - 1) * sx + bw), Hh = Math.round(pad * 2 + maxd * sy + 4 * lh + 22);
  const svg = sv('svg', { class: 'nat', viewBox: '0 0 ' + W + ' ' + Hh, width: W, height: Hh, role: 'img', 'aria-label': o.aria || 'cow_btree pages' });
  const XY = (id) => { const [x, dp] = pos.get(id); return [pad + x * sx, pad + dp * sy]; };
  const lines = (p) => (p.kind === 'leaf' ? p.entries.map((e) => e[0] + ',' + e[1] + '→' + e[2]) : p.separators.map((sp) => '≥ ' + sp.join(',')));
  for (const p of delta.pages) if (p.kind !== 'leaf') {
    const [x, y] = XY(p.page);
    const hgt = lines(p).length * lh + 20;
    for (const c of p.children) { const [cx, cy] = XY(c); svg.append(sv('path', { class: 'pgl', d: 'M' + (x + bw / 2) + ',' + (y + hgt) + 'L' + (cx + bw / 2) + ',' + cy })); }
  }
  svg.dataset.cx = String(Math.round(XY(delta.root)[0] + bw / 2));
  for (const p of delta.pages) {
    const [x, y] = XY(p.page);
    const ls = lines(p);
    const cls = 'pg' + (o.copied && o.copied.has(p.page) ? ' copied' : o.split && o.split.has(p.page) ? ' split' : '');
    svg.append(sv('rect', { class: cls, x, y, width: bw, height: ls.length * lh + 20, rx: 4 }));
    svg.append(sv('text', { class: 'bs', x: x + 6, y: y + 12 }, '#' + p.page + (p.kind === 'leaf' ? '' : ' branch')));
    ls.forEach((t, k) => svg.append(sv('text', { class: 'bt', x: x + 6, y: y + 25 + k * lh }, t)));
  }
  return svg;
}
/* a B-tree figure; on a narrow screen its pane opens scrolled to the root, which btree() records */
const treeFig = (delta, cap, o) => { const svg = btree(delta, o); return h('figure', null, h('div', { class: 'pane scroll', 'data-cx': svg.dataset ? svg.dataset.cx : null }, svg), cap ? h('figcaption', null, cap) : null); };

function stepF1(s) {
  return stepFgen(s, (sec, d) => {
    const cc = d.chunks_copied;
    ap(sec, blk('metric chunks this writer copied (index, old chunk id → new chunk id)', arrays(...Object.entries(cc).map(([k, v]) => strip(k, v.length ? v.map(([i, a, b]) => ({ ix: 'chunk ' + i, v: a + '→' + b, cls: 'hl' })) : [{ v: 'none', cls: 'empty' }])))));
  });
}
function stepF2(s) {
  return stepFgen(s, (sec, d) => {
    ap(sec, h('div', { class: 'two' }, treeFig(d.delta.Up, 'Up.delta: key (lo, hi) → arc id.'), treeFig(d.delta.Down, 'Down.delta: the same arcs as (hi, lo), same ids (I7).')));
    ap(sec, stats([['arc_capacity', d.arc_capacity], ['added_since_fold', d.added_since_fold.map(([q, a]) => 'seq ' + q + ': ' + a.length + ' arcs').join('; ')]]));
  });
}
function stepF3(s) {
  return stepFgen(s, (sec, d) => {
    const m = d.maps;
    ap(sec, arrays(strip('node_of[rank]', m.node_of.map(([r, n]) => ({ ix: r, v: n == null ? 'none' : n, cls: d.flush.allocation.some((a) => a.rank === r) ? 'hl' : n == null ? 'iso' : '' }))),
      strip('free_ranks', m.free_ranks.length ? m.free_ranks.map((r) => ({ v: r })) : [{ v: 'none', cls: 'empty' }])));
    ap(sec, h('div', { class: 'arrays' }, treeFig(d.delta.Up, 'Up.delta after the flush. Rank ' + Math.max(...m.node_of.map(([r]) => r)) + ' lies above the base’s rows, so its arcs exist only here.'), treeFig(d.delta.Down, 'Down.delta.')));
    ap(sec, h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'upkeep set by this writer'), P('kind ', B(d.upkeep.kind), ', ticket ', B(d.upkeep.ticket), ', since ', B(d.upkeep.since), '. The log keeps only nodes with seq above since, so this writer’s own node is not kept (03 §7).')));
  });
}
function stepF4(s) {
  return stepFgen(s, (sec, d) => {
    ap(sec, table(['lo', 'hi', 'id', 'up', 'down'], d.arcs_at_rank.map(([lo, hi, id, u, dn]) => [lo, hi, id, fw(u), fw(dn)]), { caption: 'Every arc at the deleted node’s rank after the recompute (all ∞: ' + (d.all_inf ? 'yes' : 'no') + ')' }));
  });
}
function stepF8(s) {
  return stepFgen(s, (sec, d) => {
    const row = (x) => [x[0], x[1], x[2], fm(x[3]), fw(x[4]), fw(x[5]), fm(x[6]), fm(x[7])];
    const head = ['lo', 'hi', 'id', 'other end (node)', 'up', 'down', 'middle up', 'middle down'];
    ap(sec, h('div', { class: 'two' },
      table(head, d.arcs_at_rank_before.map(row), { caption: 'Arcs at rank ' + d.rank + ' before' }),
      table(head, d.arcs_at_rank_after.map(row), { caption: 'after: arc ' + d.arc_id + ' ' + farc(d.fill_arc) + ' now carries the road' })),
      note('“other end” is the node that holds the arc’s other rank. A middle is the apex rank a value goes through; a dash means none: the value is ∞ or the arc’s own edge.'));
  });
}

function stepF5(s) {
  const sec = section(s), d = s.data, at = d.attempt, rp = d.repair;
  const v = rp.seq, Hh = hier(v);
  ap(sec, h('div', { class: 'two' }, fGraph(s, v), h('div', { style: 'display:grid;gap:10px;align-content:start' }, journalBlock(d.journal),
    blk('top separator', P(C(flist(d.top_separator)), ' (the solid nodes). The inserted road joins the two sides.')))));
  ap(sec, h('h3', { class: 'sub-h' }, 'The writer’s attempt, under the lock'));
  ap(sec, stats([['S2 arcs', at.new_arcs], ['S2 work', at.work.s2], ['stopped at pop', at.stopped.pop], ['units at the stop', at.stopped.work], ['full cost', at.full_cost.work.total, at.full_cost.pops + ' pops'], ['left in the heap', at.work.left_in_heap]]));
  ap(sec, recomputeStepper('st-f5-attempt', { title: 'The attempt: the running total is checked after every pop', pops: at.recompute_until_stop, front: rp.front, arcs: Hh.arcs, nodeOf: Hh.nodeOf, budget: thrBudget(), start: at.work.s2, max: at.stopped.work,
    frontWhy: 'S2 ran first and put every new arc in the front. The writer pops in (lo, hi) order and stops at the first check over the budget.' }));
  ap(sec, h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'hand-off'),
    P('The writer drops its attempt and publishes Building{Repair} under generation ', B(d.generation), '. Readers of this version use the Dijkstra fallback.'),
    P('The new generation discards the in-flight fold (', C(d.fold_discarded.kind + ', ticket ' + d.fold_discarded.ticket + ', since ' + d.fold_discarded.since), ') as Gone.')));
  ap(sec, h('h3', { class: 'sub-h' }, 'The Repair, off the lock'));
  ap(sec, h('div', { style: 'display:grid;gap:10px' }, ...flushBlocks(rp)));
  ap(sec, s2Stepper('st-f5-s2', rp.s2, { oldArcs: Hh.arcs.filter((a) => !rp.s2.new_arcs.some((x) => x[2] === a.id)), nodeOf: Hh.nodeOf }));
  const fs = rp.full_seed;
  const S = (arr, lab) => strip(lab, arr.map((x, i) => ({ ix: i, v: fw(x), cls: x == null ? 'iso' : i >= ST.C5.data.arc_capacity ? 'fill' : '' })));
  ap(sec, h('figure', null, h('div', { class: 'pane arrays' }, S(fs.seeds_up, 'seeds_up[id]'), S(fs.seeds_down, 'seeds_down[id]')),
    h('figcaption', null, 'The full path writes every seed by arc id: from Ax for the base, from each entry for the delta (amber ids). Then one relax sweep runs over ' + rp.sweep_triangles + ' triangles.')));
  ap(sec, table(['lo', 'hi', 'id', 'seed up', 'seed down'], fs.delta_entries.map(([lo, hi, id, u, dn]) => [lo, hi, id, fw(u), fw(dn)]), { caption: 'Delta entries, seeded by id: ' + fs.delta_entries.length }));
  ap(sec, h('div', { class: 'two' }, h('div', null, h('span', { class: 'bh' }, 'Repair work (no budget off the lock)'), workTable(rp.work)),
    h('div', { style: 'display:grid;gap:6px' }, P('Seeds equal the per-pair seeds: ', yesno(d.full_seed_equals_per_pair), '.'), P('Metric equals a full customization: ', yesno(d.equals_full), '.'), P('S2 equals the game: ', yesno(d.s2_equals_game), '.'))));
  ap(sec, timelineStepper('st-f5-ev', d.events, 'Hand-off, the fold discarded, the Repair adopts'));
  ap(sec, h('div', { class: 'arrays' }, treeFig(d.delta.Up, 'Up.delta after the Repair.'), treeFig(d.delta.Down, 'Down.delta after the Repair.')));
  ap(sec, blk('decay signals on the attempt', decayChips(d.decay.attempt)));
  return sec;
}
const thrBudget = () => D.after_build_events.find((e) => e.budget != null).budget;

function stepF6(s) {
  const sec = section(s), d = s.data;
  const H15 = hier(d.v15.flush.seq), H16 = hier(d.v16.flush.seq);
  ap(sec, h('div', { class: 'two' }, fGraph(s, d.v16.flush.seq),
    h('div', { style: 'display:grid;gap:10px;align-content:start' },
      stats([['delta entries at v' + d.v15.flush.seq, d.trigger.delta_entries], ['trigger √|base|', fnum(d.trigger.threshold)], ['fold fork', 'v' + d.fork]]),
      h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'upkeep'), P('kind ', B(d.upkeep.kind), ', ticket ', B(d.upkeep.ticket), ', since ', B(d.upkeep.since), '.')))));
  ap(sec, h('h3', { class: 'sub-h' }, 'v' + d.v15.flush.seq + ': the writer that sets upkeep'), journalBlock(d.v15.journal));
  ap(sec, recomputeStepper('st-f6-v15', { pops: d.v15.flush.recompute, front: d.v15.flush.front, arcs: H15.arcs, nodeOf: H15.nodeOf, budget: d.v15.flush.budget, start: 0, frontWhy: 'A weight SET on a pair with an arc: metric only.' }));
  ap(sec, blk('decay signals', decayChips(d.v15.decay)));
  ap(sec, h('h3', { class: 'sub-h' }, 'v' + d.v16.flush.seq + ': a writer while the fold runs'), journalBlock(d.v16.journal));
  ap(sec, s2Stepper('st-f6-s2', d.v16.flush.s2, { oldArcs: H16.arcs.filter((a) => !d.v16.flush.s2.new_arcs.some((x) => x[2] === a.id)), nodeOf: H16.nodeOf }));
  ap(sec, recomputeStepper('st-f6-v16', { pops: d.v16.flush.recompute, front: d.v16.flush.front, arcs: H16.arcs, nodeOf: H16.nodeOf, budget: d.v16.flush.budget, start: d.v16.flush.work.s2, frontWhy: 'The inserted pair’s arc plus the arcs S2 added.' }));
  ap(sec, h('h3', { class: 'sub-h' }, 'The fold publish'));
  const oldIds = new Set(d.old_base.Up.Ax);
  const grp = (base, fromDelta) => {
    const g = [];
    for (let i = 0; i + 1 < base.Up.Ap.length; i++) {
      const cells = [];
      for (let k = base.Up.Ap[i]; k < base.Up.Ap[i + 1]; k++) cells.push({ ix: base.Up.Aj[k], v: base.Up.Ax[k], cls: fromDelta && !oldIds.has(base.Up.Ax[k]) ? 'fill' : '' });
      if (cells.length) g.push({ gl: 'r' + (i + 1), cells });
    }
    return g;
  };
  ap(sec, h('figure', null, h('div', { class: 'pane arrays' }, gstrip(d.old_base.name + ': Up, column rank above, id in the box', grp(d.old_base, false))),
    h('figcaption', null, d.old_base.name + ': ' + d.old_base.rows + ' rows, ' + d.old_base.Up.Ax.length + ' arcs.')));
  ap(sec, h('figure', null, h('div', { class: 'pane arrays' }, gstrip(d.new_base.name + ': Up, column rank above, id in the box', grp(d.new_base, true))),
    h('figcaption', null, d.new_base.name + ': ' + d.new_base.rows + ' rows, ' + d.new_base.Up.Ax.length + ' arcs. Amber ids came from the delta at the fork; every id is carried as a value, so none changes. ' + d.new_base.row_of_rank + '.')));
  ap(sec, h('div', { class: 'two' }, treeFig(d.new_delta.Up, 'New Up.delta: only the arcs added after the fork (seq > F).'), treeFig(d.new_delta.Down, 'New Down.delta.')));
  ap(sec, h('div', { class: 'callout' }, h('span', { class: 'bh' }, 'checks at the publish'), h('ul', null,
    h('li', null, 'ids unchanged: ', yesno(d.ids_unchanged)), h('li', null, 'metric carried from the published copy: ', yesno(d.metric_carried)), h('li', null, 'metric equals a full customization of the folded topology: ', yesno(d.metric_equals_full)),
    h('li', null, 'added_since_fold after: ' + d.added_since_fold_after.map(([q, a]) => 'seq ' + q + ': ' + a.map((x) => farc(x) + ' id ' + x[2]).join(', ')).join('; ')))));
  ap(sec, timelineStepper('st-f6-ev', d.events, 'Fold job: set, run, overtaken by a writer, published'));
  ap(sec, blk('decay signals after v' + d.v16.flush.seq, decayChips(d.v16.decay)));
  return sec;
}

function stepF7(s) {
  const sec = section(s), d = s.data;
  const [va, vb] = Object.keys(d).filter((k) => /^v\d+$/.test(k)).sort((x, y) => +x.slice(1) - +y.slice(1));
  const A = d[va], Bv = d[vb];
  const split = new Set([...d.copied_split_leaves.Up, ...d.copied_split_leaves.Down]);
  const pathc = new Set([...d.copied_on_paths.Up, ...d.copied_on_paths.Down]);
  ap(sec, stats([['base', d.base, d.base_shared ? 'shared by both versions' : ''], ['pages shared', d.pages_shared.length], ['pages copied by ' + vb, d.pages_copied_by_v16.length, pathc.size + ' on paths, ' + split.size + ' split leaves']]));
  ap(sec, legend([['q', 'as', 'copied by the ' + vb + ' writer: on the root-to-leaf path'], ['q', 'fs', 'new leaf from a split'], ['q', '', 'shared with ' + va]]));
  for (const side of ['Up', 'Down']) {
    ap(sec, h('div', { class: 'arrays' },
      treeFig(A.delta[side], va + ' ' + side + '.delta (root #' + A.delta[side].root + ')', { aria: va + ' ' + side + '.delta pages' }),
      treeFig(Bv.delta[side], vb + ' ' + side + '.delta (root #' + Bv.delta[side].root + ')', { copied: pathc, split, aria: vb + ' ' + side + '.delta pages, copied pages highlighted' })));
  }
  const ch = d.chunks_copied_by_v16;
  ap(sec, h('figure', null, h('div', { class: 'pane arrays' }, ...Object.keys(A.chunks).flatMap((k) => {
    const cp = new Set((ch[k] || []).map((x) => x[0]));
    return [strip(k + ' at ' + va, A.chunks[k].map((c, i) => ({ ix: i, v: c }))), strip(k + ' at ' + vb, Bv.chunks[k].map((c, i) => ({ ix: i, v: c, cls: cp.has(i) ? 'hl' : '' })))];
  })), h('figcaption', null, 'Metric chunk ids per array. A writer copies only the chunk it writes (blue); every other chunk is the same object in both versions.')));
  return sec;
}

/* ===================== Part G ===================== */
function summarizeBreak(b) {
  const out = [];
  for (const [k, v] of Object.entries(b)) {
    if (k === 'break' || k === 'passed') continue;
    const name = k.replace(/_/g, ' ');
    if (Array.isArray(v)) {
      const prim = v.every((x) => typeof x !== 'object');
      out.push(name + ': ' + (prim && v.length <= 12 ? flist(v) : v.length + (v.length === 1 ? ' entry' : ' entries')));
    } else if (typeof v !== 'object') out.push(name + ': ' + v);
  }
  return out.join(' · ');
}
function stepG(s) {
  const sec = section(s), d = s.data;
  const rows = d.invariants.map((iv, k) => ({
    cells: [
      prose(h('span', null, h('b', null, 'G' + (k + 1) + ' '), iv.invariant, h('br'), h('span', { class: 'no' }, iv.design))),
      prose(iv.checked_at),
      { v: h('span', { class: iv.passed ? 'yes' : 'fail' }, iv.passed ? 'holds' : 'fails') },
      prose(h('ul', { class: 'plain' }, iv.breaks.map((b) => h('li', null, b.break, ' → ', h('span', { class: b.passed ? 'fail' : 'yes' }, b.passed ? 'check still passes' : 'check fails, as it should'), summarizeBreak(b) ? h('span', { class: 'no' }, ' (' + summarizeBreak(b) + ')') : null)))),
    ],
  }));
  const nb = d.invariants.reduce((t, iv) => t + iv.breaks.length, 0);
  const nf = d.invariants.reduce((t, iv) => t + iv.breaks.filter((b) => !b.passed).length, 0);
  ap(sec, stats([['invariants', d.invariants.length], ['holding', d.invariants.filter((iv) => iv.passed).length], ['deliberate breaks', nb], ['breaks the check catches', nf]]));
  ap(sec, table(['check', 'checked at', 'holds', 'deliberate break and what the check did'], rows, { stack: true }));
  return sec;
}

/* ===================== page ===================== */
const RENDER = {
  A: stepA, B1: stepB1, B2: stepB2, B3: stepB3, B4: stepB4, 'B5.0': stepB5, 'B5.1': stepB5, 'B5.2': stepB5, 'B5.h': stepB5h,
  B6: stepB6, B7: stepB7, B8: stepB8, 'B9.1': stepB91, B10: stepB10, B11: stepB11,
  C1: stepC1, C2: stepC2, C3: stepC3, C4: stepC4, C5: stepC5, D1: stepD1, D2: stepD2, D3: stepD3, E: stepE,
  F1: stepF1, F2: stepF2, F3: stepF3, F4: stepF4, F5: stepF5, F6: stepF6, F7: stepF7, F8: stepF8, G: stepG,
};
const PARTS = {
  A: ['Example', 'The example', 'One small road network carries every step: a grid of junctions, a dead-end road off one corner, one node with no road and one free id slot. Some edits are still pending in the storage layers when the index is created, so the very first read must already handle them.'],
  B: ['Ordering', 'Ordering', 'Ordering decides which node goes at which rank. It reads the live pattern in slabs, leaves nodes with no edge unranked, peels dead-end chains, then runs nested dissection: shrink the graph by matching, find a small separator on the smallest version, refine it on the way back, give the separator the top ranks, and repeat inside each half. Small pieces are finished by minimum degree.'],
  C: ['Supergraph', 'Supergraph', 'The supergraph is every arc the build knows of: the live edges in rank space plus the shortcuts the elimination game adds. It is built once from the order and the write log, and stored as a read-only base whose values are arc ids, plus an empty delta. Later inserts add more arcs by fill repair (F2).'],
  D: ['Weights', 'Weights', 'Customization writes weights into the arcs. The full-seed pass reads each pair’s live weight straight from the storage layers, the relax sweep improves every arc through lower detours, and the Seal absorbs what writers changed in the meantime.'],
  E: ['Batched build', 'The batched build', 'The build runs in batches on the native index builder, off the lock. Writers keep committing, and record the node pairs they touch in a write log owned by the CCH index. Each batch publish fits itself against the commits that landed since it forked.'],
  F: ['After the build', 'After the build', 'Once Built, every writer updates the hierarchy under the lock, within one work budget per lock hold. The order never changes. A weight change touches only the metric, a new edge adds arcs by fill repair, a new node takes a pool or a top rank, and a commit too big for the budget hands off to a Repair job. A fold keeps the delta small.'],
  G: ['Checks', 'Checks', 'Every invariant is checked by the model script where its step runs, then run again with one fix removed or one input corrupted. A check that cannot fail proves nothing.'],
};
const SHORT = { A: 'storage', B1: 'skeleton read', B2: 'degrees', B3: 'hanging trees', B4: 'components', 'B5.h': 'side example: 2-hop', B6: 'first separator', B7: 'uncoarsen', B8: 'split', 'B9.1': 'next depth', B10: 'leaves', B11: 'final order',
  C1: 'B and counts', C2: 'log and ranks', C3: 'tree over B′', C4: 'elimination game', C5: 'base arrays', D1: 'full-seed pass', D2: 'relax', D3: 'Seal', E: 'timeline',
  F1: 'weight SET', F2: 'insert (S2)', F3: 'new nodes', F4: 'node delete', F5: 'hand-off', F6: 'fold', F7: 'MVCC sharing', F8: 'id reuse', G: 'invariants' };
const shortName = (s) => SHORT[s.id] || (s.data && s.data.depth != null ? 'coarsen d' + s.data.depth + ' l' + s.data.level : s.title.split(':')[0]);

/* the page is assembled in app5.js (design walkthrough); the step renderers above feed its appendix */

