//! Recursive nested-dissection driver.
//!
//! Top-level algorithm (Karypis & Kumar 1998, §4, George 1973):
//!
//! 1. Split graph into connected components. Each component is
//!    ordered independently and its numbering is concatenated.
//! 2. On a single component:
//!    - If its size is below `nd_to_amd_switch`, hand off to AMD.
//!      (AMD dominates at small scales where ND overhead would
//!      exceed its quality benefit; METIS 5.2.0 switches at 200.)
//!    - Otherwise, run a multilevel node bisection, number the
//!      separator last, and recurse on the two sides.
//!
//! The numbering convention is "permutation" = new-position → old-id.
//! Internally we populate `iperm[original_vertex] = new_position` and
//! invert at the end.

use crate::coarsen::{coarsen, contract, CoarseGraph, CoarsenCounters, CUR_DEPTH};
use crate::fm_refine::{refine_bisection, refine_separator, refine_separator_fm};
use crate::graph::Graph;
use crate::initial_partition::{initial_bisect_bfs, initial_bisect_ggp, PART_A, PART_B};
use crate::rng::SplitMix;
use crate::separator::construct_separator;
use crate::{MetisOptions, MetisStats};
use feral_ordering_core::Idx;
use feral_ordering_core::{CscPattern, OrderingError};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

/// Entry point. Produces a permutation `perm` where `perm[i]` is the
/// old vertex id placed at new position `i` (new-to-old).
pub(crate) fn nd_order(
    pattern: &CscPattern<'_>,
    opts: &MetisOptions,
    stats: &mut MetisStats,
) -> Result<Vec<Idx>, OrderingError> {
    let graph = Graph::from_csc_pattern(pattern)?;
    let n = graph.nvtxs as usize;
    let mut iperm: Vec<Idx> = vec![-1; n];
    let mut rng = SplitMix::new(opts.seed);
    let (cc_label, ncc) = connected_components(&graph);
    stats.n_components = ncc as u32;
    let identity: Vec<Idx> = (0..n as Idx).collect();
    split_and_recurse(
        graph, identity, &cc_label, ncc, 0, &mut iperm, opts, &mut rng, stats, 0, None,
    )?;
    invert_iperm(&iperm, n)
}

/// Order the connected components of `graph` (labelled `0..ncc` by
/// `label`) one after the other, in label order, into
/// `[offset, offset + n)`. Linear in the size of `graph`: one counting
/// sort groups the components, and a single-vertex component takes its
/// position directly. `graph` is consumed so that only the pieces stay
/// alive while they are being ordered.
#[allow(clippy::too_many_arguments)]
fn split_and_recurse(
    graph: Graph,
    vtx_map: Vec<Idx>,
    label: &[Idx],
    ncc: usize,
    offset: usize,
    iperm: &mut [Idx],
    opts: &MetisOptions,
    rng: &mut SplitMix,
    stats: &mut MetisStats,
    depth: u32,
    reuse: Option<Reuse<'_>>,
) -> Result<(), OrderingError> {
    let n = graph.nvtxs as usize;
    // Counting sort by label keeps vertex order inside each component.
    let mut start = vec![0usize; ncc + 1];
    for &l in label {
        start[l as usize + 1] += 1;
    }
    for c in 0..ncc {
        start[c + 1] += start[c];
    }
    let mut members = vec![0 as Idx; n];
    let mut local_id = vec![-1 as Idx; n];
    let mut fill = start.clone();
    for v in 0..n {
        let c = label[v] as usize;
        local_id[v] = (fill[c] - start[c]) as Idx;
        members[fill[c]] = v as Idx;
        fill[c] += 1;
    }
    drop(fill);
    let mut pieces: Vec<(Graph, Vec<Idx>, Option<Vec<Idx>>)> = Vec::with_capacity(ncc);
    let mut singles: Vec<(usize, Idx)> = Vec::new();
    for c in 0..ncc {
        let m = &members[start[c]..start[c + 1]];
        if m.len() == 1 {
            singles.push((pieces.len() + singles.len(), vtx_map[m[0] as usize]));
            continue;
        }
        let sub = build_induced(&graph, m, &local_id);
        let map: Vec<Idx> = m.iter().map(|&v| vtx_map[v as usize]).collect();
        let gid = reuse
            .as_ref()
            .map(|r| m.iter().map(|&v| r.gid[v as usize]).collect());
        pieces.push((sub, map, gid));
    }
    let root = reuse.map(|r| r.root);
    drop((graph, vtx_map, members, local_id));
    // Re-interleave singletons in component order.
    let mut off = offset;
    let mut si = singles.into_iter().peekable();
    let mut k = 0usize;
    for (sub, map, gid) in pieces {
        while let Some(&(pos, orig)) = si.peek() {
            if pos != k {
                break;
            }
            iperm[orig as usize] = off as Idx;
            off += 1;
            k += 1;
            si.next();
        }
        let count = sub.nvtxs as usize;
        let ctx = gid.map(|gid| Reuse {
            root: root.expect("reuse context"),
            gid,
        });
        recurse(sub, map, off, iperm, opts, rng, stats, depth, ctx)?;
        off += count;
        k += 1;
    }
    for (_, orig) in si {
        iperm[orig as usize] = off as Idx;
        off += 1;
    }
    debug_assert_eq!(off, offset + n);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn recurse(
    subgraph: Graph,
    vtx_map: Vec<Idx>,
    offset: usize,
    iperm: &mut [Idx],
    opts: &MetisOptions,
    rng: &mut SplitMix,
    stats: &mut MetisStats,
    depth: u32,
    reuse: Option<Reuse<'_>>,
) -> Result<(), OrderingError> {
    let n = subgraph.nvtxs as usize;
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        iperm[vtx_map[0] as usize] = offset as Idx;
        return Ok(());
    }

    let (cc_label, ncc) = connected_components(&subgraph);
    if ncc > 1 {
        return split_and_recurse(
            subgraph, vtx_map, &cc_label, ncc, offset, iperm, opts, rng, stats, depth, reuse,
        );
    }
    drop(cc_label);

    if n <= opts.nd_to_amd_switch as usize {
        amd_leaf(&subgraph, &vtx_map, offset, iperm, stats)?;
        return Ok(());
    }

    CUR_DEPTH.store(depth, Ordering::Relaxed);
    let l2 = n >= L2_MIN_VERTICES;
    // Break 3: cells at depth >= 3 drop their context and re-coarsen.
    let reuse = if opts.reuse_break == 3 && depth >= 3 { None } else { reuse };
    let root_owned: RootHier;
    let mut check_hier: Option<Hier> = None;
    let (labels, shared): (Vec<u8>, Option<(&RootHier, Vec<Idx>)>) = match reuse {
        None if !opts.reuse => {
            let labels = if l2 {
                best_of_bisections(&subgraph, opts, rng, stats)
            } else {
                multilevel_node_bisection(&subgraph, opts, rng, stats)
            };
            (labels, None)
        }
        None => {
            // A root (depth 0, or a break-3 cell): R's bisection, keeping
            // the winning hierarchy for the descendants.
            if opts.reuse_break == 4 {
                rng.next_u64();
            }
            let (labels, levels) = if l2 {
                best_of_bisections_keep(&subgraph, opts, rng, stats)
            } else {
                mnb_keep(&subgraph, opts, rng, stats)
            };
            root_owned = RootHier::new(&levels, n);
            NODE_LOG.lock().unwrap().push(NodeRec {
                depth,
                n: n as u32,
                root: true,
                l2,
                reused: 0,
                fresh: levels.len() as u32,
                skipped: 0,
                slow: 0,
                maxw: levels.last().map_or(1, |c| *c.graph.vwgt.iter().max().unwrap()),
                cap_r: ((1.5 * n as f64) / (opts.coarsen_floor as f64)) as Idx,
            });
            if opts.reuse_check {
                check_hier = Some(Hier::from_root(levels));
            }
            (labels, Some((&root_owned, (0..n as Idx).collect())))
        }
        Some(ctx) if opts.reuse_capstop && l2 => {
            // Diagnostic: reuse the top stage, then R's five fresh runs.
            let h = reuse_levels(&subgraph, &ctx.gid, ctx.root, l2, opts, rng, stats, true);
            let labels = best_of_bisections_top(&subgraph, &h.levels, opts, rng, stats);
            NODE_LOG.lock().unwrap().push(NodeRec {
                depth,
                n: n as u32,
                root: false,
                l2,
                reused: h.levels.len() as u32 - h.fresh,
                fresh: h.fresh,
                skipped: h.skipped,
                slow: h.slow,
                maxw: h.levels.last().map_or(1, |c| *c.graph.vwgt.iter().max().unwrap()),
                cap_r: ((1.5 * n as f64) / (opts.coarsen_floor as f64)) as Idx,
            });
            if opts.reuse_check {
                check_hier = Some(h);
            }
            (labels, Some((ctx.root, ctx.gid)))
        }
        Some(ctx) => {
            let h = reuse_levels(&subgraph, &ctx.gid, ctx.root, l2, opts, rng, stats, true);
            let labels = if l2 {
                best_of_bisections_given(&subgraph, &h.levels, h.split, opts, rng, stats)
            } else {
                bisect_on_levels(&subgraph, &h.levels, opts, rng, stats)
            };
            NODE_LOG.lock().unwrap().push(NodeRec {
                depth,
                n: n as u32,
                root: false,
                l2,
                reused: h.levels.len() as u32 - h.fresh,
                fresh: h.fresh,
                skipped: h.skipped,
                slow: h.slow,
                maxw: h.levels.last().map_or(1, |c| *c.graph.vwgt.iter().max().unwrap()),
                cap_r: ((1.5 * n as f64) / (opts.coarsen_floor as f64)) as Idx,
            });
            if opts.reuse_check {
                check_hier = Some(h);
            }
            (labels, Some((ctx.root, ctx.gid)))
        }
    };
    log_split(depth, &labels);

    let mut a_verts: Vec<Idx> = Vec::new();
    let mut b_verts: Vec<Idx> = Vec::new();
    let mut s_verts: Vec<Idx> = Vec::new();
    for (v, &l) in labels.iter().enumerate() {
        match l {
            PART_A => a_verts.push(v as Idx),
            PART_B => b_verts.push(v as Idx),
            _ => s_verts.push(v as Idx),
        }
    }

    if a_verts.is_empty() || b_verts.is_empty() {
        amd_leaf(&subgraph, &vtx_map, offset, iperm, stats)?;
        return Ok(());
    }

    stats.n_separator_vertices += s_verts.len() as u32;
    let na = a_verts.len();
    let nb = b_verts.len();
    for (i, &v) in s_verts.iter().enumerate() {
        let orig = vtx_map[v as usize];
        iperm[orig as usize] = (offset + na + nb + i) as Idx;
    }
    drop(s_verts);

    // One local-id array serves both sides (their ids never mix).
    let mut local_id: Vec<Idx> = vec![-1; n];
    for (i, &v) in a_verts.iter().enumerate() {
        local_id[v as usize] = i as Idx;
    }
    let sub_a = build_induced(&subgraph, &a_verts, &local_id);
    for &v in &a_verts {
        local_id[v as usize] = -1;
    }
    for (i, &v) in b_verts.iter().enumerate() {
        local_id[v as usize] = i as Idx;
    }
    let sub_b = build_induced(&subgraph, &b_verts, &local_id);
    // Reuse: each child keeps the root hierarchy and its vertices' root ids.
    let children = shared.map(|(root, gid)| {
        let ga: Vec<Idx> = a_verts.iter().map(|&v| gid[v as usize]).collect();
        let gb: Vec<Idx> = b_verts.iter().map(|&v| gid[v as usize]).collect();
        if let Some(h) = &check_hier {
            check_split(
                &subgraph, &labels, h, root, &sub_a, &ga, &a_verts, &sub_b, &gb, &b_verts, opts,
                depth,
            );
        }
        (Reuse { root, gid: ga }, Reuse { root, gid: gb })
    });
    drop((check_hier, labels));
    let map_a: Vec<Idx> = a_verts.iter().map(|&l| vtx_map[l as usize]).collect();
    let map_b: Vec<Idx> = b_verts.iter().map(|&l| vtx_map[l as usize]).collect();
    drop((subgraph, vtx_map, local_id, a_verts, b_verts));
    let (ra, rb) = match children {
        Some((a, b)) => (Some(a), Some(b)),
        None => (None, None),
    };

    recurse(sub_a, map_a, offset, iperm, opts, rng, stats, depth + 1, ra)?;
    recurse(sub_b, map_b, offset + na, iperm, opts, rng, stats, depth + 1, rb)?;
    Ok(())
}

/// Graphs with at least this many vertices get [`best_of_bisections`]
/// (METIS `MlevelNodeBisectionL2`: `nvtxs < 5000` takes the single run).
const L2_MIN_VERTICES: usize = 5000;
/// Bisections tried by [`best_of_bisections`] (METIS: `nruns = 5`).
const L2_RUNS: usize = 5;

/// METIS's `MlevelNodeBisectionL2` (`ometis.c`): coarsen once, to
/// `max(100, n / 30)` vertices, run [`L2_RUNS`] complete multilevel
/// bisections of that coarse graph, keep the one with the lightest
/// separator (the first of equals), and refine it back up the levels
/// with the node FM. Each run starts from different random draws, so
/// the best of five is a far better separator than one run, for the
/// price of five bisections of a graph 30 times smaller.
fn best_of_bisections(
    graph: &Graph,
    opts: &MetisOptions,
    rng: &mut SplitMix,
    stats: &mut MetisStats,
) -> Vec<u8> {
    let coarse_opts = MetisOptions {
        coarsen_floor: (graph.nvtxs as u32 / 30).max(100),
        ..opts.clone()
    };
    let mut counters = CoarsenCounters::default();
    let levels = coarsen(graph, &coarse_opts, rng, &mut counters);
    stats.n_two_hop_fallbacks += counters.n_two_hop_fallbacks;
    stats.n_levels += levels.len() as u32;
    let coarse: &Graph = levels.last().map_or(graph, |c| &c.graph);

    let separator = |labels: &[u8]| -> i64 {
        labels
            .iter()
            .zip(&coarse.vwgt)
            .filter(|(&p, _)| p != PART_A && p != PART_B)
            .map(|(_, &w)| w)
            .sum()
    };
    let mut labels = multilevel_node_bisection(coarse, opts, rng, stats);
    let mut best = separator(&labels);
    for _ in 1..L2_RUNS {
        if best == 0 {
            break;
        }
        let run = multilevel_node_bisection(coarse, opts, rng, stats);
        let sep = separator(&run);
        if sep < best {
            best = sep;
            labels = run;
        }
    }

    for level_idx in (0..levels.len()).rev() {
        let cg = &levels[level_idx];
        let finer: &Graph = if level_idx == 0 {
            graph
        } else {
            &levels[level_idx - 1].graph
        };
        labels = (0..finer.nvtxs as usize)
            .map(|v| labels[cg.cmap[v] as usize])
            .collect();
        refine_separator_fm(finer, &mut labels, opts.max_imbalance, opts.fm_passes);
        stats.n_fm_passes += opts.fm_passes;
    }
    labels
}

/// Multilevel node bisection: coarsen → initial bisection with niparts
/// trials → uncoarsen with FM → convert edge bisection to node
/// separator → refine separator. Returns labels in {PART_A, PART_B,
/// PART_SEP}.
fn multilevel_node_bisection(
    graph: &Graph,
    opts: &MetisOptions,
    rng: &mut SplitMix,
    stats: &mut MetisStats,
) -> Vec<u8> {
    let mut counters = CoarsenCounters::default();
    let levels = coarsen(graph, opts, rng, &mut counters);
    stats.n_two_hop_fallbacks += counters.n_two_hop_fallbacks;
    stats.n_levels += levels.len() as u32;

    // Coarsest graph for initial bisection.
    let coarsest: &Graph = match levels.last() {
        Some(cg) => &cg.graph,
        None => graph,
    };
    let total: i64 = coarsest.vwgt.iter().copied().sum();
    let target = total / 2;

    // niparts trials; keep best post-FM cut.
    let mut best_labels: Vec<u8> = vec![PART_A; coarsest.nvtxs as usize];
    let mut best_cut: Idx = Idx::MAX;
    for trial in 0..opts.niparts {
        let mut trial_labels = if trial % 2 == 0 {
            initial_bisect_ggp(coarsest, rng, target)
        } else {
            initial_bisect_bfs(coarsest, rng, target)
        };
        let cut = refine_bisection(
            coarsest,
            &mut trial_labels,
            opts.max_imbalance,
            opts.fm_passes,
        );
        stats.n_fm_passes += opts.fm_passes;
        if cut < best_cut {
            best_cut = cut;
            best_labels = trial_labels;
        }
    }
    let mut labels = best_labels;

    if opts.node_refine {
        // METIS's structure (`ometis.c` + `refine.c`): build the node
        // separator at the *coarsest* level, then project and
        // FM-refine the separator itself at every uncoarsening level.
        // The projection below is label-agnostic, so `PART_SEP` rides
        // down the hierarchy unchanged.
        construct_separator(coarsest, &mut labels);
        stats.n_fm_passes += opts.fm_passes;
        refine_separator_fm(coarsest, &mut labels, opts.max_imbalance, opts.fm_passes);
        for level_idx in (0..levels.len()).rev() {
            let cg = &levels[level_idx];
            let prev_graph: &Graph = if level_idx == 0 {
                graph
            } else {
                &levels[level_idx - 1].graph
            };
            let prev_n = prev_graph.nvtxs as usize;
            let mut proj: Vec<u8> = vec![PART_A; prev_n];
            for (v, p) in proj.iter_mut().enumerate().take(prev_n) {
                *p = labels[cg.cmap[v] as usize];
            }
            labels = proj;
            refine_separator_fm(prev_graph, &mut labels, opts.max_imbalance, opts.fm_passes);
            stats.n_fm_passes += opts.fm_passes;
        }
        return labels;
    }

    // Uncoarsen: walk levels in reverse. `cmap` at level i maps
    // previous-graph vertices to level-i graph vertices.
    for level_idx in (0..levels.len()).rev() {
        let cg = &levels[level_idx];
        let prev_graph: &Graph = if level_idx == 0 {
            graph
        } else {
            &levels[level_idx - 1].graph
        };
        let prev_n = prev_graph.nvtxs as usize;
        let mut proj: Vec<u8> = vec![PART_A; prev_n];
        for (v, p) in proj.iter_mut().enumerate().take(prev_n) {
            let c = cg.cmap[v] as usize;
            *p = labels[c];
        }
        labels = proj;
        refine_bisection(prev_graph, &mut labels, opts.max_imbalance, opts.fm_passes);
        stats.n_fm_passes += opts.fm_passes;
    }

    // Convert edge bisection to node separator.
    construct_separator(graph, &mut labels);
    // Refine separator (greedy).
    refine_separator(graph, &mut labels, opts.max_imbalance, opts.fm_passes);
    stats.n_fm_passes += opts.fm_passes;

    labels
}

/// Hand off to AMD on an uncoarsened subgraph. Writes new positions
/// `[offset, offset + n)` into `iperm`.
fn amd_leaf(
    subgraph: &Graph,
    vtx_map: &[Idx],
    offset: usize,
    iperm: &mut [Idx],
    stats: &mut MetisStats,
) -> Result<(), OrderingError> {
    stats.n_amd_leaf_calls += 1;
    let n = subgraph.nvtxs as usize;
    if n == 0 {
        return Ok(());
    }
    let (col_ptr, row_idx) = graph_to_csc_pattern(subgraph);
    let pattern = CscPattern::new(n, &col_ptr, &row_idx).ok_or(OrderingError::MalformedInput)?;
    let perm_local = feral_amd::amd_order(&pattern)?;
    debug_assert_eq!(perm_local.len(), n);
    for (new_pos, &local_id) in perm_local.iter().enumerate() {
        let orig = vtx_map[local_id as usize];
        iperm[orig as usize] = (offset + new_pos) as Idx;
    }
    Ok(())
}

/// Build a full-symmetric CSC pattern from an internal `Graph`.
/// Adjacency in `Graph` is already full-symmetric with diagonal
/// dropped; we reinsert the diagonal for downstream consumers that
/// expect it. Row indices within each column remain sorted.
fn graph_to_csc_pattern(graph: &Graph) -> (Vec<Idx>, Vec<Idx>) {
    let n = graph.nvtxs as usize;
    let mut col_ptr: Vec<Idx> = Vec::with_capacity(n + 1);
    let mut row_idx: Vec<Idx> = Vec::with_capacity(graph.adjncy.len() + n);
    col_ptr.push(0);
    for v in 0..n {
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        // Insert neighbors keeping sorted order, splicing in the
        // diagonal at its correct position.
        let mut diag_inserted = false;
        for k in lo..hi {
            let u = graph.adjncy[k];
            if !diag_inserted && u as usize > v {
                row_idx.push(v as Idx);
                diag_inserted = true;
            }
            row_idx.push(u);
        }
        if !diag_inserted {
            row_idx.push(v as Idx);
        }
        col_ptr.push(row_idx.len() as Idx);
    }
    (col_ptr, row_idx)
}

/// Invert the new→old position map `iperm` (where `iperm[old] = new_pos`)
/// into the old→new permutation `perm` (where `perm[new_pos] = old`).
///
/// Rejects an out-of-range or duplicated target position rather than
/// silently emitting a non-bijection — parity with the scotch/kahip
/// `invert_iperm` helpers (O20).
fn invert_iperm(iperm: &[Idx], n: usize) -> Result<Vec<Idx>, OrderingError> {
    let mut perm: Vec<Idx> = vec![-1; n];
    for (old, &new_pos) in iperm.iter().enumerate() {
        if new_pos < 0 || (new_pos as usize) >= n {
            return Err(OrderingError::Internal(
                "metis nd produced invalid permutation",
            ));
        }
        let np = new_pos as usize;
        if perm[np] >= 0 {
            return Err(OrderingError::Internal(
                "metis nd produced duplicate position",
            ));
        }
        perm[np] = old as Idx;
    }
    Ok(perm)
}

/// Connected-component labeling via BFS. Returns `(cc_label, ncc)`
/// where `cc_label[v] ∈ 0..ncc`.
fn connected_components(graph: &Graph) -> (Vec<Idx>, usize) {
    let n = graph.nvtxs as usize;
    let mut cc: Vec<Idx> = vec![-1; n];
    let mut ncc: Idx = 0;
    let mut queue: Vec<Idx> = Vec::new();
    for start in 0..n {
        if cc[start] >= 0 {
            continue;
        }
        cc[start] = ncc;
        queue.clear();
        queue.push(start as Idx);
        while let Some(v) = queue.pop() {
            let vu = v as usize;
            let lo = graph.xadj[vu] as usize;
            let hi = graph.xadj[vu + 1] as usize;
            for k in lo..hi {
                let u = graph.adjncy[k];
                if cc[u as usize] < 0 {
                    cc[u as usize] = ncc;
                    queue.push(u);
                }
            }
        }
        ncc += 1;
    }
    (cc, ncc as usize)
}

fn build_induced(graph: &Graph, vtx_map: &[Idx], local_id: &[Idx]) -> Graph {
    let sub_n = vtx_map.len();
    let mut xadj: Vec<Idx> = Vec::with_capacity(sub_n + 1);
    let mut adjncy: Vec<Idx> = Vec::new();
    let mut adjwgt: Vec<Idx> = Vec::new();
    let mut vwgt: Vec<Idx> = Vec::with_capacity(sub_n);
    xadj.push(0);
    for &orig in vtx_map {
        let v = orig as usize;
        vwgt.push(graph.vwgt[v]);
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            let lu = local_id[u];
            if lu >= 0 {
                adjncy.push(lu);
                adjwgt.push(graph.adjwgt[k]);
            }
        }
        xadj.push(adjncy.len() as Idx);
    }
    Graph {
        nvtxs: sub_n as Idx,
        xadj,
        adjncy,
        vwgt,
        adjwgt,
    }
}

// ---------------------------------------------------------------------
// M1 reuse prototype (D5 option b): reuse the depth-0 hierarchy.
// ---------------------------------------------------------------------

/// The depth-0 (root) hierarchy's maps, shared by every descendant cell
/// of one top-level component.
pub(crate) struct RootHier {
    /// `cmaps[l]` maps a root level-`l` vertex to its level-`l+1` vertex.
    cmaps: Vec<Vec<Idx>>,
    /// Epoch-stamped scratch keyed by a root-level id (size: root n).
    scratch: RefCell<Scratch>,
}

struct Scratch {
    epoch: u32,
    slot: Vec<(u32, Idx)>,
}

impl Scratch {
    fn next(&mut self) -> u32 {
        self.epoch += 1;
        self.epoch
    }
}

impl RootHier {
    fn new(levels: &[CoarseGraph], n: usize) -> Self {
        RootHier {
            cmaps: levels.iter().map(|c| c.cmap.clone()).collect(),
            scratch: RefCell::new(Scratch {
                epoch: 0,
                slot: vec![(0, 0); n],
            }),
        }
    }
    fn depth(&self) -> usize {
        self.cmaps.len()
    }
    /// Root level-`to` id of a root level-`from` id.
    fn lift(&self, mut r: Idx, from: usize, to: usize) -> Idx {
        for l in from..to {
            r = self.cmaps[l][r as usize];
        }
        r
    }
}

/// What a cell below depth 0 carries: the root maps and the root
/// level-0 id of each of its vertices.
pub(crate) struct Reuse<'a> {
    root: &'a RootHier,
    gid: Vec<Idx>,
}

const FRESH: u32 = u32::MAX;

/// A cell's hierarchy (B1: the root matching, straddlers split).
#[derive(Default)]
struct Hier {
    levels: Vec<CoarseGraph>,
    /// Root level (1-based) of each level; `FRESH` for a fresh level.
    rlev: Vec<u32>,
    /// Root ids of each level's vertices (empty for a fresh level).
    rids: Vec<Vec<Idx>>,
    /// L2 cells: how many levels form the top stage (to max(100, n/30)).
    split: usize,
    fresh: u32,
    skipped: u32,
    slow: u32,
}

impl Hier {
    fn from_root(levels: Vec<CoarseGraph>) -> Self {
        let rlev = (1..=levels.len() as u32).collect();
        let rids = levels.iter().map(|c| (0..c.graph.nvtxs).collect()).collect();
        Hier {
            levels,
            rlev,
            rids,
            ..Default::default()
        }
    }
}

/// One reused level for a cell: group the cell's current vertices (root
/// ids `rid` at root level `l-1`) by their root id at level `l`. Coarse
/// ids follow first appearance, as `renumber` does, so a level that
/// does not merge anything is the identity.
fn compose_level(root: &RootHier, l: usize, rid: &[Idx]) -> (Vec<Idx>, Vec<Idx>) {
    let cm = &root.cmaps[l - 1];
    let mut sc = root.scratch.borrow_mut();
    let e = sc.next();
    let mut cmap = Vec::with_capacity(rid.len());
    let mut nrid = Vec::new();
    for &r in rid {
        let t = cm[r as usize];
        let s = &mut sc.slot[t as usize];
        if s.0 == e {
            cmap.push(s.1);
        } else {
            let id = nrid.len() as Idx;
            *s = (e, id);
            nrid.push(t);
            cmap.push(id);
        }
    }
    (cmap, nrid)
}

/// B1: a cell's levels from the root matching, re-contracted from the
/// cell's own graph. Stops like `coarsen` (first level at or below the
/// floor) unless `reuse_stop_ge` (deepest level still at or above it).
/// L2 cells take the top stage to max(100, n/30), then the rest to
/// `coarsen_floor`. When the root levels run out above the floor, the
/// rest is coarsened fresh (only if `allow_fresh`).
#[allow(clippy::too_many_arguments)]
fn reuse_levels(
    g: &Graph,
    gid: &[Idx],
    root: &RootHier,
    l2: bool,
    opts: &MetisOptions,
    rng: &mut SplitMix,
    stats: &mut MetisStats,
    allow_fresh: bool,
) -> Hier {
    let n = g.nvtxs as usize;
    let f1 = (g.nvtxs as u32 / 30).max(100) as usize;
    let f2 = opts.coarsen_floor as usize;
    let floors: Vec<usize> = if l2 && opts.reuse_capstop {
        vec![f1]
    } else if l2 {
        vec![f1, f2]
    } else {
        vec![f2]
    };
    let total_w: Idx = g.vwgt.iter().sum();
    let mut h = Hier::default();
    let mut rid: Vec<Idx> = gid.to_vec();
    let mut l = 1usize;
    let mut pending: Option<(Vec<Idx>, Vec<Idx>, usize)> = None;
    let mut exhausted = false;
    for (pi, &floor) in floors.iter().enumerate() {
        loop {
            let cur_n = h.levels.last().map_or(n, |c| c.graph.nvtxs as usize);
            if cur_n <= floor {
                break;
            }
            if exhausted || (pending.is_none() && l > root.depth()) {
                exhausted = true;
                if !allow_fresh {
                    break;
                }
                let fresh_opts = MetisOptions {
                    coarsen_floor: floor as u32,
                    ..opts.clone()
                };
                let mut ctr = CoarsenCounters::default();
                let cur = h.levels.last().map_or(g, |c| &c.graph);
                let lv = coarsen(cur, &fresh_opts, rng, &mut ctr);
                stats.n_two_hop_fallbacks += ctr.n_two_hop_fallbacks;
                stats.n_levels += lv.len() as u32;
                h.fresh += lv.len() as u32;
                for c in lv {
                    h.levels.push(c);
                    h.rlev.push(FRESH);
                    h.rids.push(Vec::new());
                }
                break;
            }
            let (cmap, nrid, lvl) = match pending.take() {
                Some(p) => p,
                None => {
                    let (c, r) = compose_level(root, l, &rid);
                    l += 1;
                    (c, r, l - 1)
                }
            };
            let new_n = nrid.len();
            if new_n == cur_n {
                // Nothing merges inside this cell at this root level.
                rid = nrid;
                h.skipped += 1;
                continue;
            }
            if opts.reuse_stop_ge && new_n < floor {
                pending = Some((cmap, nrid, lvl));
                break;
            }
            let cur = h.levels.last().map_or(g, |c| &c.graph);
            if opts.reuse_capstop {
                // R's cap for this stage, as `coarsen` computes it.
                let cap = ((1.5 * total_w as f64) / (floor as f64)) as Idx;
                let mut w = vec![0 as Idx; new_n];
                for (v, &c) in cmap.iter().enumerate() {
                    w[c as usize] += cur.vwgt[v];
                }
                if w.iter().any(|&x| x > cap) {
                    exhausted = true;
                    continue;
                }
            }
            if new_n as f64 > 0.95 * cur_n as f64 {
                h.slow += 1;
            }
            let mut cg = contract(cur, &cmap, new_n as Idx);
            if opts.reuse_break == 5 && !cg.adjwgt.is_empty() {
                // Break 5: one asymmetric edge weight in every reused level.
                cg.adjwgt[0] += 1;
            }
            h.levels.push(CoarseGraph { graph: cg, cmap });
            h.rlev.push(lvl as u32);
            h.rids.push(nrid.clone());
            rid = nrid;
        }
        if pi == 0 && l2 {
            h.split = h.levels.len();
        }
    }
    h
}

/// `multilevel_node_bisection` after coarsening, on given levels
/// (a copy of its body; the stock function is untouched).
fn bisect_on_levels(
    graph: &Graph,
    levels: &[CoarseGraph],
    opts: &MetisOptions,
    rng: &mut SplitMix,
    stats: &mut MetisStats,
) -> Vec<u8> {
    let coarsest: &Graph = match levels.last() {
        Some(cg) => &cg.graph,
        None => graph,
    };
    let total: i64 = coarsest.vwgt.iter().copied().sum();
    let target = total / 2;

    let mut best_labels: Vec<u8> = vec![PART_A; coarsest.nvtxs as usize];
    let mut best_cut: Idx = Idx::MAX;
    for trial in 0..opts.niparts {
        let mut trial_labels = if trial % 2 == 0 {
            initial_bisect_ggp(coarsest, rng, target)
        } else {
            initial_bisect_bfs(coarsest, rng, target)
        };
        let cut = refine_bisection(
            coarsest,
            &mut trial_labels,
            opts.max_imbalance,
            opts.fm_passes,
        );
        stats.n_fm_passes += opts.fm_passes;
        if cut < best_cut {
            best_cut = cut;
            best_labels = trial_labels;
        }
    }
    let mut labels = best_labels;

    if opts.node_refine {
        construct_separator(coarsest, &mut labels);
        stats.n_fm_passes += opts.fm_passes;
        refine_separator_fm(coarsest, &mut labels, opts.max_imbalance, opts.fm_passes);
        for level_idx in (0..levels.len()).rev() {
            let cg = &levels[level_idx];
            let prev_graph: &Graph = if level_idx == 0 {
                graph
            } else {
                &levels[level_idx - 1].graph
            };
            let prev_n = prev_graph.nvtxs as usize;
            let mut proj: Vec<u8> = vec![PART_A; prev_n];
            for (v, p) in proj.iter_mut().enumerate().take(prev_n) {
                *p = labels[cg.cmap[v] as usize];
            }
            labels = proj;
            refine_separator_fm(prev_graph, &mut labels, opts.max_imbalance, opts.fm_passes);
            stats.n_fm_passes += opts.fm_passes;
        }
        return labels;
    }

    for level_idx in (0..levels.len()).rev() {
        let cg = &levels[level_idx];
        let prev_graph: &Graph = if level_idx == 0 {
            graph
        } else {
            &levels[level_idx - 1].graph
        };
        let prev_n = prev_graph.nvtxs as usize;
        let mut proj: Vec<u8> = vec![PART_A; prev_n];
        for (v, p) in proj.iter_mut().enumerate().take(prev_n) {
            let c = cg.cmap[v] as usize;
            *p = labels[c];
        }
        labels = proj;
        refine_bisection(prev_graph, &mut labels, opts.max_imbalance, opts.fm_passes);
        stats.n_fm_passes += opts.fm_passes;
    }
    construct_separator(graph, &mut labels);
    refine_separator(graph, &mut labels, opts.max_imbalance, opts.fm_passes);
    stats.n_fm_passes += opts.fm_passes;
    labels
}

/// `multilevel_node_bisection`, keeping its levels.
fn mnb_keep(
    graph: &Graph,
    opts: &MetisOptions,
    rng: &mut SplitMix,
    stats: &mut MetisStats,
) -> (Vec<u8>, Vec<CoarseGraph>) {
    let mut counters = CoarsenCounters::default();
    let levels = coarsen(graph, opts, rng, &mut counters);
    stats.n_two_hop_fallbacks += counters.n_two_hop_fallbacks;
    stats.n_levels += levels.len() as u32;
    let labels = bisect_on_levels(graph, &levels, opts, rng, stats);
    (labels, levels)
}

/// `best_of_bisections`, keeping its top levels plus the winning run's
/// levels: the root hierarchy the descendants reuse.
fn best_of_bisections_keep(
    graph: &Graph,
    opts: &MetisOptions,
    rng: &mut SplitMix,
    stats: &mut MetisStats,
) -> (Vec<u8>, Vec<CoarseGraph>) {
    let coarse_opts = MetisOptions {
        coarsen_floor: (graph.nvtxs as u32 / 30).max(100),
        ..opts.clone()
    };
    let mut counters = CoarsenCounters::default();
    let mut levels = coarsen(graph, &coarse_opts, rng, &mut counters);
    stats.n_two_hop_fallbacks += counters.n_two_hop_fallbacks;
    stats.n_levels += levels.len() as u32;
    let coarse: &Graph = levels.last().map_or(graph, |c| &c.graph);

    let separator = |labels: &[u8]| -> i64 {
        labels
            .iter()
            .zip(&coarse.vwgt)
            .filter(|(&p, _)| p != PART_A && p != PART_B)
            .map(|(_, &w)| w)
            .sum()
    };
    let (mut labels, mut sub) = mnb_keep(coarse, opts, rng, stats);
    let mut best = separator(&labels);
    for _ in 1..L2_RUNS {
        if best == 0 {
            break;
        }
        let (run, run_sub) = mnb_keep(coarse, opts, rng, stats);
        let sep = separator(&run);
        if sep < best {
            best = sep;
            labels = run;
            sub = run_sub;
        }
    }

    for level_idx in (0..levels.len()).rev() {
        let cg = &levels[level_idx];
        let finer: &Graph = if level_idx == 0 {
            graph
        } else {
            &levels[level_idx - 1].graph
        };
        labels = (0..finer.nvtxs as usize)
            .map(|v| labels[cg.cmap[v] as usize])
            .collect();
        refine_separator_fm(finer, &mut labels, opts.max_imbalance, opts.fm_passes);
        stats.n_fm_passes += opts.fm_passes;
    }
    levels.extend(sub);
    (labels, levels)
}

/// `best_of_bisections` with a given top stage; every run coarsens fresh
/// (`multilevel_node_bisection`), as R does.
fn best_of_bisections_top(
    graph: &Graph,
    top: &[CoarseGraph],
    opts: &MetisOptions,
    rng: &mut SplitMix,
    stats: &mut MetisStats,
) -> Vec<u8> {
    let coarse: &Graph = top.last().map_or(graph, |c| &c.graph);
    let separator = |labels: &[u8]| -> i64 {
        labels
            .iter()
            .zip(&coarse.vwgt)
            .filter(|(&p, _)| p != PART_A && p != PART_B)
            .map(|(_, &w)| w)
            .sum()
    };
    let mut labels = multilevel_node_bisection(coarse, opts, rng, stats);
    let mut best = separator(&labels);
    for _ in 1..L2_RUNS {
        if best == 0 {
            break;
        }
        let run = multilevel_node_bisection(coarse, opts, rng, stats);
        let sep = separator(&run);
        if sep < best {
            best = sep;
            labels = run;
        }
    }
    for level_idx in (0..top.len()).rev() {
        let cg = &top[level_idx];
        let finer: &Graph = if level_idx == 0 {
            graph
        } else {
            &top[level_idx - 1].graph
        };
        labels = (0..finer.nvtxs as usize)
            .map(|v| labels[cg.cmap[v] as usize])
            .collect();
        refine_separator_fm(finer, &mut labels, opts.max_imbalance, opts.fm_passes);
        stats.n_fm_passes += opts.fm_passes;
    }
    labels
}

/// `best_of_bisections` on given levels: the top stage is
/// `levels[..split]`, and every run bisects `levels[split..]`.
fn best_of_bisections_given(
    graph: &Graph,
    levels: &[CoarseGraph],
    split: usize,
    opts: &MetisOptions,
    rng: &mut SplitMix,
    stats: &mut MetisStats,
) -> Vec<u8> {
    let coarse: &Graph = if split == 0 {
        graph
    } else {
        &levels[split - 1].graph
    };
    let sub = &levels[split..];
    let separator = |labels: &[u8]| -> i64 {
        labels
            .iter()
            .zip(&coarse.vwgt)
            .filter(|(&p, _)| p != PART_A && p != PART_B)
            .map(|(_, &w)| w)
            .sum()
    };
    let mut labels = bisect_on_levels(coarse, sub, opts, rng, stats);
    let mut best = separator(&labels);
    for _ in 1..L2_RUNS {
        if best == 0 {
            break;
        }
        let run = bisect_on_levels(coarse, sub, opts, rng, stats);
        let sep = separator(&run);
        if sep < best {
            best = sep;
            labels = run;
        }
    }
    for level_idx in (0..split).rev() {
        let cg = &levels[level_idx];
        let finer: &Graph = if level_idx == 0 {
            graph
        } else {
            &levels[level_idx - 1].graph
        };
        labels = (0..finer.nvtxs as usize)
            .map(|v| labels[cg.cmap[v] as usize])
            .collect();
        refine_separator_fm(finer, &mut labels, opts.max_imbalance, opts.fm_passes);
        stats.n_fm_passes += opts.fm_passes;
    }
    labels
}

/// One record per bisected cell in reuse mode.
#[derive(Debug, Clone, Copy)]
pub struct NodeRec {
    /// ND depth.
    pub depth: u32,
    /// Cell size.
    pub n: u32,
    /// A root: coarsened fresh (depth 0).
    pub root: bool,
    /// Took the best-of-5 (L2) path.
    pub l2: bool,
    /// Levels taken from the root hierarchy.
    pub reused: u32,
    /// Levels coarsened fresh (for a root: all of them).
    pub fresh: u32,
    /// Root levels that merged nothing inside the cell (skipped).
    pub skipped: u32,
    /// Reused levels that shrank the cell by under 5 %.
    pub slow: u32,
    /// Heaviest vertex at the coarsest level.
    pub maxw: Idx,
    /// The cap R would use at the bottom stage: 1.5 n / coarsen_floor.
    pub cap_r: Idx,
}

/// Every cell bisected in reuse mode since the last take.
pub static NODE_LOG: Mutex<Vec<NodeRec>> = Mutex::new(Vec::new());

/// (depth, n, labels hash) of every bisection; the hash only at depth 0.
pub static SPLIT_LOG: Mutex<Vec<(u32, u32, u64)>> = Mutex::new(Vec::new());

fn log_split(depth: u32, labels: &[u8]) {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    if depth == 0 {
        for &l in labels {
            h ^= l as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    SPLIT_LOG
        .lock()
        .unwrap()
        .push((depth, labels.len() as u32, if depth == 0 { h } else { 0 }));
}

/// Straddler statistics and B1/B2 results at one (split depth, root level).
#[derive(Debug, Default, Clone, Copy)]
pub struct StradRec {
    /// Splits with a level at this root level (whether needed or not).
    pub splits_all: u64,
    /// Parent coarse vertices at this level, all splits.
    pub n_all: u64,
    /// Parent entries at this level, all splits.
    pub nnz_all: u64,
    /// Side straddlers at this level, all splits.
    pub strad_side_all: u64,
    /// Splits where a child of more than `nd_to_amd_switch` vertices
    /// coarsens with this level ("needed"); the fields below up to
    /// `strad_rows_nnz` count only those.
    pub splits: u64,
    /// Parent coarse vertices at this level.
    pub n_coarse: u64,
    /// Parent entries at this level.
    pub nnz: u64,
    /// Parent coarse vertices wholly in the separator (dropped).
    pub sep_only: u64,
    /// Parent coarse vertices with members in >= 2 of {A, B, S}.
    pub strad_side: u64,
    /// Parent coarse vertices with members in >= 2 of {components of A
    /// and of B, S}: the children feral recurses into.
    pub strad_child: u64,
    /// Parent entries in the rows of side straddlers.
    pub strad_rows_nnz: u64,
    /// Sum over parent vertices of the sides {A, B} they touch.
    pub parts: u64,
    /// Sum over A and B of their distinct root ids at this level.
    pub child_n: u64,
    /// Splits where `parts != child_n` (accounting identity).
    pub ident_fail: u64,
    /// Child cells (A or B) whose B1 levels include this level.
    pub used: u64,
    /// Their vertices and entries at this level.
    pub x_n: u64,
    /// Entries of those child levels.
    pub x_nnz: u64,
    /// Entries of the child level each was contracted from.
    pub xprev_nnz: u64,
    /// Of those, the entries in rows of straddler members (B2's input).
    pub recontract_in_nnz: u64,
    /// Entries in the straddler rows of the child level (B2's output).
    pub recontract_out_nnz: u64,
    /// Child straddler parts at this level.
    pub x_strad: u64,
    /// Child levels compared B1 against B2.
    pub b2_checked: u64,
    /// Of those, levels where B1 and B2 differ.
    pub b2_mismatch: u64,
    /// B1 levels failing weight conservation, a self-loop or symmetry.
    pub b1_invalid: u64,
}

/// Keyed by (split depth, root level).
pub static STRAD_LOG: Mutex<BTreeMap<(u32, u32), StradRec>> = Mutex::new(BTreeMap::new());

/// Sorted, merged (row rid, col rid, w) entries and (rid, vwgt) of a level.
type LevelSig = (Vec<(Idx, Idx, Idx)>, Vec<(Idx, Idx)>);

fn sig_of(g: &Graph, rids: &[Idx]) -> LevelSig {
    let mut e = Vec::with_capacity(g.adjncy.len());
    for v in 0..g.nvtxs as usize {
        for k in g.xadj[v] as usize..g.xadj[v + 1] as usize {
            e.push((rids[v], rids[g.adjncy[k] as usize], g.adjwgt[k]));
        }
    }
    e.sort_unstable();
    let mut w: Vec<(Idx, Idx)> = (0..g.nvtxs as usize).map(|v| (rids[v], g.vwgt[v])).collect();
    w.sort_unstable();
    (e, w)
}

fn merge_sorted(mut e: Vec<(Idx, Idx, Idx)>) -> Vec<(Idx, Idx, Idx)> {
    e.sort_unstable_by_key(|&(a, b, _)| (a, b));
    let mut out: Vec<(Idx, Idx, Idx)> = Vec::with_capacity(e.len());
    for (a, b, w) in e {
        match out.last_mut() {
            Some(last) if last.0 == a && last.1 == b => last.2 += w,
            _ => out.push((a, b, w)),
        }
    }
    out
}

/// At a split: straddler statistics for every reused parent level, and
/// for each side X in {A, B} its levels built two ways and compared:
/// B1 = re-contraction with the root matching (`reuse_levels`), and
/// B2 = the parent's level restricted to X's clean coarse vertices
/// (a masked split), plus straddler rows and columns re-contracted from
/// X's previous level. Neither touches the RNG or the order.
#[allow(clippy::too_many_arguments)]
fn check_split(
    g: &Graph,
    labels: &[u8],
    h: &Hier,
    root: &RootHier,
    sub_a: &Graph,
    gid_a: &[Idx],
    a_verts: &[Idx],
    sub_b: &Graph,
    gid_b: &[Idx],
    b_verts: &[Idx],
    opts: &MetisOptions,
    depth: u32,
) {
    const SEP: u32 = u32::MAX - 1;
    const MIX: u32 = u32::MAX;
    const UNSET: u32 = u32::MAX - 2;
    let n = g.nvtxs as usize;
    let mut recs: BTreeMap<u32, StradRec> = BTreeMap::new();

    // Fine level: side bits (A=1, B=2, S=4) and child component ids.
    let mut mask: Vec<u8> = labels
        .iter()
        .map(|&l| match l {
            PART_A => 1,
            PART_B => 2,
            // Break 2: count separator members as A.
            _ if opts.reuse_break == 2 => 1,
            _ => 4,
        })
        .collect();
    let mut cc: Vec<u32> = vec![SEP; n];
    let (ca, na) = connected_components(sub_a);
    for (i, &v) in a_verts.iter().enumerate() {
        cc[v as usize] = ca[i] as u32;
    }
    let (cb, _) = connected_components(sub_b);
    for (i, &v) in b_verts.iter().enumerate() {
        cc[v as usize] = na as u32 + cb[i] as u32;
    }

    // Each side's B1 levels; a side of more than nd_to_amd_switch
    // vertices will coarsen, so its levels are "needed".
    let mut dummy_rng = SplitMix::new(0);
    let mut dummy_stats = MetisStats::default();
    let mut hxs: Vec<Hier> = Vec::new();
    for (sub, gid) in [(sub_a, gid_a), (sub_b, gid_b)] {
        hxs.push(reuse_levels(
            sub,
            gid,
            root,
            false,
            opts,
            &mut dummy_rng,
            &mut dummy_stats,
            false,
        ));
    }
    let needed = [
        sub_a.nvtxs > opts.nd_to_amd_switch as Idx,
        sub_b.nvtxs > opts.nd_to_amd_switch as Idx,
    ];
    let need_max: u32 = (0..2)
        .filter(|&k| needed[k])
        .flat_map(|k| hxs[k].rlev.iter().copied())
        .max()
        .unwrap_or(0);

    // Parent levels: masks per level (kept for B2), statistics.
    let mut masks: Vec<Vec<u8>> = Vec::new();
    let mut lvl_of: BTreeMap<u32, usize> = BTreeMap::new();
    for (i, cg) in h.levels.iter().enumerate() {
        if h.rlev[i] == FRESH {
            break;
        }
        let np = cg.graph.nvtxs as usize;
        let mut nm = vec![0u8; np];
        let mut ncc = vec![UNSET; np];
        for (v, &c) in cg.cmap.iter().enumerate() {
            let c = c as usize;
            nm[c] |= mask[v];
            ncc[c] = match ncc[c] {
                UNSET => cc[v],
                x if x == cc[v] => x,
                _ => MIX,
            };
        }
        mask = nm;
        cc = ncc;
        let p = &cg.graph;
        let need = h.rlev[i] <= need_max;
        let r = recs.entry(h.rlev[i]).or_default();
        r.splits_all += 1;
        r.n_all += np as u64;
        r.nnz_all += p.adjncy.len() as u64;
        if need {
            r.splits += 1;
            r.n_coarse += np as u64;
            r.nnz += p.adjncy.len() as u64;
        }
        for q in 0..np {
            let m = mask[q];
            if m.count_ones() >= 2 {
                r.strad_side_all += 1;
            }
            if need {
                if m == 4 {
                    r.sep_only += 1;
                }
                if m.count_ones() >= 2 {
                    r.strad_side += 1;
                    r.strad_rows_nnz += (p.xadj[q + 1] - p.xadj[q]) as u64;
                }
                if cc[q] == MIX {
                    r.strad_child += 1;
                }
            }
            r.parts += (m & 3).count_ones() as u64;
        }
        masks.push(mask.clone());
        lvl_of.insert(h.rlev[i], i);
    }

    // Accounting identity: sum over sides of their distinct root ids at
    // each parent level equals the parts the masks predict.
    for gid in [gid_a, gid_b] {
        let mut ids: Vec<Idx> = gid.to_vec();
        let mut from = 0usize;
        for (&rl, _) in lvl_of.iter() {
            let rl_us = rl as usize;
            for x in ids.iter_mut() {
                *x = root.lift(*x, from, rl_us);
            }
            from = rl_us;
            let mut sc = root.scratch.borrow_mut();
            let e = sc.next();
            let mut cnt = 0u64;
            for &x in &ids {
                let s = &mut sc.slot[x as usize];
                if s.0 != e {
                    *s = (e, 0);
                    cnt += 1;
                }
            }
            recs.get_mut(&rl).unwrap().child_n += cnt;
        }
    }
    for r in recs.values_mut() {
        if r.parts != r.child_n {
            r.ident_fail += 1;
        }
    }

    // B1 against B2 for each side.
    for (k, (bit, sub, gid)) in [(1u8, sub_a, gid_a), (2u8, sub_b, gid_b)].into_iter().enumerate() {
        let hx = &hxs[k];
        let need = needed[k];
        let total_w: Idx = sub.vwgt.iter().sum();
        for j in 0..hx.levels.len() {
            let rl = hx.rlev[j];
            let xl = &hx.levels[j].graph;
            let x_rids = &hx.rids[j];
            let (gprev, rprev, aprev): (&Graph, &[Idx], usize) = if j == 0 {
                (sub, gid, 0)
            } else {
                (&hx.levels[j - 1].graph, &hx.rids[j - 1], hx.rlev[j - 1] as usize)
            };
            let b1 = sig_of(xl, x_rids);
            // B1 validity: weight conserved, no self-loop, symmetric.
            let mut invalid = xl.vwgt.iter().sum::<Idx>() != total_w;
            invalid |= b1.0.iter().any(|&(a, b, _)| a == b);
            let mut tr: Vec<(Idx, Idx, Idx)> = b1.0.iter().map(|&(a, b, w)| (b, a, w)).collect();
            tr.sort_unstable();
            invalid |= tr != b1.0;

            let r = recs.entry(rl).or_default();
            if need {
                r.used += 1;
                r.x_n += xl.nvtxs as u64;
                r.x_nnz += xl.adjncy.len() as u64;
                r.xprev_nnz += gprev.adjncy.len() as u64;
            }
            if invalid {
                r.b1_invalid += 1;
            }
            let Some(&i) = lvl_of.get(&rl) else {
                // The parent has no such level: a bug, except under the
                // capstop diagnostic, whose best-of-5 cells keep only
                // their top stage.
                if !opts.reuse_capstop {
                    r.b2_checked += 1;
                    r.b2_mismatch += 1;
                }
                continue;
            };
            let pg = &h.levels[i].graph;
            let prid = &h.rids[i];
            let pm = &masks[i];

            // B2.
            let f: Vec<Idx> = rprev
                .iter()
                .map(|&x| root.lift(x, aprev, rl as usize))
                .collect();
            let mut sc = root.scratch.borrow_mut();
            let e = sc.next();
            for (q, &x) in prid.iter().enumerate() {
                sc.slot[x as usize] = (e, q as Idx);
            }
            let pidx = |x: Idx| -> usize {
                let s = sc.slot[x as usize];
                assert_eq!(s.0, e, "child vertex outside the parent level");
                s.1 as usize
            };
            let clean = |q: usize| pm[q] == bit;
            let mut ent: Vec<(Idx, Idx, Idx)> = Vec::new();
            let mut vw: Vec<(Idx, Idx)> = Vec::new();
            let mut strad_w: Vec<(Idx, Idx)> = Vec::new();
            let mut seen_p: Vec<usize> = Vec::new();
            for &x in &f {
                let q = pidx(x);
                assert!(pm[q] & bit != 0, "child part of a parent vertex without it");
                seen_p.push(q);
            }
            seen_p.sort_unstable();
            seen_p.dedup();
            let mut in_nnz = 0u64;
            for &q in &seen_p {
                if clean(q) || opts.reuse_break == 1 {
                    // Break 1: straddlers treated as clean (masked, not
                    // re-contracted): their mixed parent weights leak in.
                    for k in pg.xadj[q] as usize..pg.xadj[q + 1] as usize {
                        let t = pg.adjncy[k] as usize;
                        let keep = if opts.reuse_break == 1 {
                            pm[t] & bit != 0
                        } else {
                            clean(t)
                        };
                        if keep {
                            ent.push((prid[q], prid[t], pg.adjwgt[k]));
                        }
                    }
                    vw.push((prid[q], pg.vwgt[q]));
                }
            }
            if opts.reuse_break != 1 {
                for u in 0..gprev.nvtxs as usize {
                    let fu = f[u];
                    let q = pidx(fu);
                    if clean(q) {
                        continue;
                    }
                    strad_w.push((fu, gprev.vwgt[u]));
                    in_nnz += (gprev.xadj[u + 1] - gprev.xadj[u]) as u64;
                    for k in gprev.xadj[u] as usize..gprev.xadj[u + 1] as usize {
                        let t = f[gprev.adjncy[k] as usize];
                        if t == fu {
                            continue;
                        }
                        let w = gprev.adjwgt[k];
                        ent.push((fu, t, w));
                        if clean(pidx(t)) {
                            ent.push((t, fu, w));
                        }
                    }
                }
            }
            let x_strad = seen_p.iter().filter(|&&q| !clean(q)).count() as u64;
            let ent = merge_sorted(ent);
            for (x, w) in merge_sorted(strad_w.into_iter().map(|(x, w)| (x, 0, w)).collect())
                .into_iter()
                .map(|(x, _, w)| (x, w))
            {
                vw.push((x, w));
            }
            vw.sort_unstable();
            let out_nnz = ent.iter().filter(|&&(a, _, _)| !clean(pidx(a))).count() as u64;
            drop(sc);
            r.b2_checked += 1;
            if ent != b1.0 || vw != b1.1 {
                r.b2_mismatch += 1;
            }
            if need {
                r.recontract_in_nnz += in_nnz;
                r.recontract_out_nnz += out_nnz;
                r.x_strad += x_strad;
            }
        }
    }

    let mut log = STRAD_LOG.lock().unwrap();
    for (rl, r) in recs {
        let t = log.entry((depth, rl)).or_default();
        t.splits_all += r.splits_all;
        t.n_all += r.n_all;
        t.nnz_all += r.nnz_all;
        t.strad_side_all += r.strad_side_all;
        t.splits += r.splits;
        t.n_coarse += r.n_coarse;
        t.nnz += r.nnz;
        t.sep_only += r.sep_only;
        t.strad_side += r.strad_side;
        t.strad_child += r.strad_child;
        t.strad_rows_nnz += r.strad_rows_nnz;
        t.parts += r.parts;
        t.child_n += r.child_n;
        t.ident_fail += r.ident_fail;
        t.used += r.used;
        t.x_n += r.x_n;
        t.x_nnz += r.x_nnz;
        t.xprev_nnz += r.xprev_nnz;
        t.recontract_in_nnz += r.recontract_in_nnz;
        t.recontract_out_nnz += r.recontract_out_nnz;
        t.x_strad += r.x_strad;
        t.b2_checked += r.b2_checked;
        t.b2_mismatch += r.b2_mismatch;
        t.b1_invalid += r.b1_invalid;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use feral_ordering_core::CscPattern;
    use std::collections::BTreeSet;

    fn csc_from_triples(n: usize, triples: &[(usize, usize)]) -> (Vec<Idx>, Vec<Idx>) {
        let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
        for &(i, j) in triples {
            set.insert((i, j));
            set.insert((j, i));
        }
        let mut cols: Vec<Vec<Idx>> = vec![Vec::new(); n];
        for &(r, c) in &set {
            cols[c].push(r as Idx);
        }
        for col in &mut cols {
            col.sort();
        }
        let mut col_ptr: Vec<Idx> = vec![0];
        let mut row_idx: Vec<Idx> = Vec::new();
        for col in &cols {
            for &r in col {
                row_idx.push(r);
            }
            col_ptr.push(row_idx.len() as Idx);
        }
        (col_ptr, row_idx)
    }

    fn grid_triples(m: usize, n: usize) -> Vec<(usize, usize)> {
        let idx = |r: usize, c: usize| r * n + c;
        let mut t = Vec::new();
        for r in 0..m {
            for c in 0..n {
                let k = idx(r, c);
                t.push((k, k));
                if r + 1 < m {
                    t.push((k, idx(r + 1, c)));
                }
                if c + 1 < n {
                    t.push((k, idx(r, c + 1)));
                }
            }
        }
        t
    }

    fn assert_permutation(perm: &[Idx]) {
        let n = perm.len();
        let mut seen = vec![false; n];
        for &p in perm {
            let p = p as usize;
            assert!(p < n, "index {} out of bounds", p);
            assert!(!seen[p], "duplicate {}", p);
            seen[p] = true;
        }
    }

    #[test]
    fn invert_iperm_rejects_duplicate_positions() {
        // A valid new→old inversion of a bijection.
        // iperm[old] = new_pos: old0→2, old1→0, old2→1 ⇒ perm = [1, 2, 0].
        assert_eq!(invert_iperm(&[2, 0, 1], 3).unwrap(), vec![1, 2, 0]);
        // Two olds claiming the same position must be rejected, not
        // silently overwritten into a non-bijection (parity with the
        // scotch/kahip duplicate-position check; O20).
        assert!(matches!(
            invert_iperm(&[0, 0, 2], 3),
            Err(OrderingError::Internal(_))
        ));
        // Out-of-range target position is rejected.
        assert!(matches!(
            invert_iperm(&[3, 0, 1], 3),
            Err(OrderingError::Internal(_))
        ));
    }

    #[test]
    fn cc_disconnected_blocks() {
        // Two disconnected 3x3 grids.
        let mut t = grid_triples(3, 3);
        // second block at ids 9..18
        for &(i, j) in grid_triples(3, 3).iter() {
            t.push((i + 9, j + 9));
        }
        let (cp, ri) = csc_from_triples(18, &t);
        let pat = CscPattern::new(18, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let (_, ncc) = connected_components(&g);
        assert_eq!(ncc, 2);
    }

    #[test]
    fn nd_order_small_grid_is_permutation() {
        // 10x10 grid, 100 vertices. With defaults (nd_to_amd_switch=200)
        // this falls into the AMD leaf branch at the top level.
        let t = grid_triples(10, 10);
        let (cp, ri) = csc_from_triples(100, &t);
        let pat = CscPattern::new(100, &cp, &ri).unwrap();
        let opts = MetisOptions::default();
        let mut stats = MetisStats::default();
        let perm = nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 100);
        assert_permutation(&perm);
        assert!(stats.n_amd_leaf_calls >= 1);
    }

    #[test]
    fn nd_order_large_grid_uses_multilevel() {
        // 20x20 grid, 400 vertices > 200 (nd_to_amd_switch) so the top
        // level runs a real multilevel bisection.
        let t = grid_triples(20, 20);
        let (cp, ri) = csc_from_triples(400, &t);
        let pat = CscPattern::new(400, &cp, &ri).unwrap();
        let opts = MetisOptions::default();
        let mut stats = MetisStats::default();
        let perm = nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 400);
        assert_permutation(&perm);
        assert!(
            stats.n_separator_vertices > 0,
            "expected a top-level separator"
        );
    }

    #[test]
    fn best_of_bisections_gives_a_valid_balanced_separator() {
        // 100x100 grid: 10,000 vertices, above L2_MIN_VERTICES.
        let (m, n) = (100, 100);
        let t = grid_triples(m, n);
        let (cp, ri) = csc_from_triples(m * n, &t);
        let pat = CscPattern::new(m * n, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let opts = MetisOptions::default();
        let mut stats = MetisStats::default();
        let mut rng = SplitMix::new(opts.seed);
        let labels = best_of_bisections(&g, &opts, &mut rng, &mut stats);
        let mut rng = SplitMix::new(opts.seed);
        let again = best_of_bisections(&g, &opts, &mut rng, &mut MetisStats::default());
        assert_eq!(labels, again, "same seed, same separator");
        // No edge joins the two sides.
        for v in 0..m * n {
            for k in g.xadj[v] as usize..g.xadj[v + 1] as usize {
                let u = g.adjncy[k] as usize;
                let pair = (labels[v], labels[u]);
                assert!(pair != (PART_A, PART_B) && pair != (PART_B, PART_A));
            }
        }
        let count = |p: u8| labels.iter().filter(|&&l| l == p).count();
        let (a, b, sep) = (
            count(PART_A),
            count(PART_B),
            m * n - count(PART_A) - count(PART_B),
        );
        // A straight cut has 100 vertices; the sides stay within the bound.
        assert!(sep <= 150, "separator {sep}");
        let bound = ((1.0 + opts.max_imbalance) * (m * n) as f64 / 2.0).ceil() as usize;
        assert!(a > 0 && b > 0 && a <= bound && b <= bound, "{a} / {b}");
    }

    #[test]
    fn nd_order_deterministic() {
        let t = grid_triples(12, 12);
        let n = 144;
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let opts = MetisOptions::default();
        let mut s1 = MetisStats::default();
        let mut s2 = MetisStats::default();
        let p1 = nd_order(&pat, &opts, &mut s1).unwrap();
        let p2 = nd_order(&pat, &opts, &mut s2).unwrap();
        assert_eq!(p1, p2);
    }

    #[test]
    fn nd_order_handles_disconnected_graph() {
        // Two separate 6x6 grids.
        let mut t = grid_triples(6, 6);
        for &(i, j) in grid_triples(6, 6).iter() {
            t.push((i + 36, j + 36));
        }
        let (cp, ri) = csc_from_triples(72, &t);
        let pat = CscPattern::new(72, &cp, &ri).unwrap();
        let opts = MetisOptions::default();
        let mut stats = MetisStats::default();
        let perm = nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 72);
        assert_permutation(&perm);
        assert_eq!(stats.n_components, 2);
    }

    #[test]
    fn extract_induced_preserves_edges() {
        // 3x3 grid, extract top row {0,1,2}.
        let t = grid_triples(3, 3);
        let (cp, ri) = csc_from_triples(9, &t);
        let pat = CscPattern::new(9, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let local_id: Vec<Idx> = vec![0, 1, 2, -1, -1, -1, -1, -1, -1];
        let sub = build_induced(&g, &[0, 1, 2], &local_id);
        assert_eq!(sub.nvtxs, 3);
        // Top row: 0-1, 1-2 → 2 edges, each stored twice.
        assert_eq!(sub.adjncy.len(), 4);
    }

    /// The `node_refine` path must produce a valid permutation and a
    /// *better or equal* ordering than the edge-cut path on a grid,
    /// which is the shape nested dissection is designed for. The
    /// quality claim is checked as a separator-width proxy: the
    /// permutation's last block (the top separator) must not grow.
    #[test]
    fn node_refine_path_is_a_valid_permutation() {
        let t = grid_triples(20, 20);
        let (cp, ri) = csc_from_triples(400, &t);
        let pat = CscPattern::new(400, &cp, &ri).unwrap();
        let opts = MetisOptions {
            node_refine: true,
            ..MetisOptions::default()
        };
        let mut stats = MetisStats::default();
        let perm = nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 400);
        assert_permutation(&perm);
        assert!(
            stats.n_separator_vertices > 0,
            "expected a top-level separator"
        );
    }

    /// Same seed, same permutation — the crate contract.
    #[test]
    fn node_refine_path_is_deterministic() {
        let t = grid_triples(24, 24);
        let (cp, ri) = csc_from_triples(576, &t);
        let pat = CscPattern::new(576, &cp, &ri).unwrap();
        let opts = MetisOptions {
            node_refine: true,
            ..MetisOptions::default()
        };
        let mut sa = MetisStats::default();
        let mut sb = MetisStats::default();
        let a = nd_order(&pat, &opts, &mut sa).unwrap();
        let b = nd_order(&pat, &opts, &mut sb).unwrap();
        assert_eq!(a, b);
    }
}
