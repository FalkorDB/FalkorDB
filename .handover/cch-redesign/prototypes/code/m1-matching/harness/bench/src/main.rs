// M1 prototype: ordering quality of the design's mutual-max matching against stock feral SHEM.
//
// usage:
//   m1bench order <graph.txt.gz> <variant> <seed> <reps>
//   m1bench coarsen <graph.txt.gz> <variant> <seed>       (largest component, coarsen to <= 100)
//   m1bench same <graph.txt.gz>                           (stock copy vs original feral: identical perm?)
//   m1bench checkbreak <graph.txt.gz>                     (closure checker must reject a broken row)
//
// variants: shem | shem-cap | mm | mm-nocap | mm-r<K> | mm-inv | mm-seq2hop | swap<K> (feral order + K random swaps)
//           mm-reuse | mm-reuse-ge | mm-reuse-b<K> (M1 reuse prototype; b<K> = deliberate break K)
//   m1bench samebase <graph.txt.gz> <variant> <seed>       (this crate vs the pre-reuse copy: identical perm?)
//
// The elimination tree and chordal supergraph are the harness's (cchbench/src/bin/phase3.rs, copied
// from the stack's order.rs): Liu's etree, then child absorption in increasing rank.

use std::io::{BufRead, BufReader};
use std::time::Instant;

use feral_metis::internals::coarsen::{coarsen, take_level_log, CoarsenCounters, LevelRec};
use feral_metis::internals::reuse::{NODE_LOG, SPLIT_LOG, STRAD_LOG};
use feral_metis::internals::graph::Graph;
use feral_metis::internals::rng::SplitMix;
use feral_metis::{CscPattern, Matching, MetisOptions};

const NONE: u64 = u64::MAX;

struct Csr {
    n: usize,
    p: Vec<u64>,
    i: Vec<u64>,
}
impl Csr {
    fn row(&self, u: usize) -> &[u64] {
        &self.i[self.p[u] as usize..self.p[u + 1] as usize]
    }
}

fn load(path: &str) -> Csr {
    let f = std::fs::File::open(path).unwrap();
    let r = BufReader::new(flate2::read::GzDecoder::new(f));
    let mut edges = Vec::new();
    let mut n = 0u64;
    for line in r.lines() {
        let line = line.unwrap();
        if line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let u: u64 = it.next().unwrap().parse().unwrap();
        let v: u64 = it.next().unwrap().parse().unwrap();
        n = n.max(u + 1).max(v + 1);
        edges.push((u, v));
    }
    // undirected skeleton, self-loops dropped, one entry per direction
    edges.retain(|&(u, v)| u != v);
    let mut both = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        both.push((u, v));
        both.push((v, u));
    }
    both.sort_unstable();
    both.dedup();
    let n = n as usize;
    let mut p = vec![0u64; n + 1];
    for &(u, _) in &both {
        p[u as usize + 1] += 1;
    }
    for k in 0..n {
        p[k + 1] += p[k];
    }
    Csr { n, p, i: both.into_iter().map(|(_, v)| v).collect() }
}

fn variant_opts(v: &str, seed: u64) -> MetisOptions {
    let mut o = MetisOptions { seed, ..MetisOptions::default() };
    match v {
        "shem" => {}
        "shem-cap" => o.vwgt_cap = true,
        "mm" => {
            o.matching = Matching::MutualMax;
            o.vwgt_cap = true;
        }
        "mm-nocap" => o.matching = Matching::MutualMax,
        "mm-reuse" | "mm-reuse-ge" | "mm-reuse-capstop" | "mm-reuse-b1" | "mm-reuse-b2" | "mm-reuse-b3" | "mm-reuse-b4" | "mm-reuse-b5" => {
            o.matching = Matching::MutualMax;
            o.vwgt_cap = true;
            o.reuse = true;
            o.reuse_stop_ge = v == "mm-reuse-ge";
            o.reuse_capstop = v == "mm-reuse-capstop";
            if let Some(k) = v.strip_prefix("mm-reuse-b") {
                o.reuse_break = k.parse().unwrap();
            }
        }
        "mm-inv" => {
            o.matching = Matching::MutualMax;
            o.vwgt_cap = true;
            o.mm_invert_weight = true;
        }
        "mm-seq2hop" => {
            o.matching = Matching::MutualMax;
            o.vwgt_cap = true;
            o.mm_seq_two_hop = true;
        }
        _ if v.starts_with("mm-r") => {
            o.matching = Matching::MutualMax;
            o.vwgt_cap = true;
            o.mm_round_cap = v[4..].parse().unwrap();
        }
        _ if v.starts_with("swap") => {}
        _ => panic!("unknown variant {v}"),
    }
    o
}

/// Isolated nodes first (id order), then the backend's order of the rest: nd.rs's shape.
fn connected_pattern(s: &Csr) -> (Vec<u64>, Vec<u64>, Vec<i64>, Vec<i64>) {
    let mut renumber = vec![u64::MAX; s.n];
    let mut nodes = Vec::new();
    let mut iso = Vec::new();
    for u in 0..s.n {
        if s.row(u).is_empty() {
            iso.push(u as u64);
        } else {
            renumber[u] = nodes.len() as u64;
            nodes.push(u as u64);
        }
    }
    let mut xadj = vec![0i64];
    let mut adj = Vec::with_capacity(s.i.len());
    for &u in &nodes {
        for &v in s.row(u as usize) {
            adj.push(renumber[v as usize] as i64);
        }
        xadj.push(adj.len() as i64);
    }
    (iso, nodes, xadj, adj)
}

fn order(s: &Csr, opts: &MetisOptions) -> (Vec<u64>, f64, feral_metis::MetisStats) {
    let (mut iso, nodes, xadj, adj) = connected_pattern(s);
    let pat = CscPattern::new(nodes.len(), &xadj, &adj).unwrap();
    let t = Instant::now();
    let (perm, _, stats) = feral_metis::metis_order_full(&pat, opts).unwrap();
    let secs = t.elapsed().as_secs_f64();
    iso.extend(perm.into_iter().map(|v| nodes[v as usize]));
    (iso, secs, stats)
}

fn invert(perm: &[u64], n: usize) -> Vec<u64> {
    let mut iperm = vec![NONE; n];
    for (rank, &node) in perm.iter().enumerate() {
        assert_eq!(iperm[node as usize], NONE, "not a permutation");
        iperm[node as usize] = rank as u64;
    }
    assert!(iperm.iter().all(|&r| r != NONE), "not a permutation");
    iperm
}

fn elimination_tree(skeleton: &Csr, perm: &[u64], iperm: &[u64]) -> Vec<u64> {
    let n = skeleton.n;
    let mut parent = vec![NONE; n];
    let mut ancestor = vec![NONE; n];
    for rank in 0..n as u64 {
        let u = perm[rank as usize] as usize;
        for &v in skeleton.row(u) {
            let mut w = iperm[v as usize];
            while w < rank {
                let next = ancestor[w as usize];
                ancestor[w as usize] = rank;
                if next == NONE {
                    parent[w as usize] = rank;
                }
                w = next;
            }
        }
    }
    parent
}

fn chordal_supergraph(skeleton: &Csr, perm: &[u64], iperm: &[u64], parent: &[u64]) -> Csr {
    let n = skeleton.n;
    // children lists
    let mut cp = vec![0u64; n + 1];
    for &p in parent {
        if p != NONE {
            cp[p as usize + 1] += 1;
        }
    }
    for r in 0..n {
        cp[r + 1] += cp[r];
    }
    let mut ci = vec![0u64; cp[n] as usize];
    let mut fill = cp.clone();
    for (c, &p) in parent.iter().enumerate() {
        if p != NONE {
            ci[fill[p as usize] as usize] = c as u64;
            fill[p as usize] += 1;
        }
    }
    let mut up_p: Vec<u64> = Vec::with_capacity(n + 1);
    up_p.push(0);
    let mut up_i: Vec<u64> = Vec::with_capacity(skeleton.i.len());
    let mut mark: Vec<u64> = vec![0; n];
    let mut row: Vec<u64> = Vec::new();
    for rank in 0..n as u64 {
        row.clear();
        let stamp = rank + 1;
        let u = perm[rank as usize] as usize;
        for &v in skeleton.row(u) {
            let w = iperm[v as usize];
            if w > rank && mark[w as usize] != stamp {
                mark[w as usize] = stamp;
                row.push(w);
            }
        }
        for &child in &ci[cp[rank as usize] as usize..cp[rank as usize + 1] as usize] {
            let lo = up_p[child as usize] as usize;
            let hi = up_p[child as usize + 1] as usize;
            for k in lo..hi {
                let w = up_i[k];
                if w != rank && mark[w as usize] != stamp {
                    mark[w as usize] = stamp;
                    row.push(w);
                }
            }
        }
        row.sort_unstable();
        up_i.extend_from_slice(&row);
        up_p.push(up_i.len() as u64);
    }
    Csr { n, p: up_p, i: up_i }
}

/// Lower triangles: for each rank x, every pair {y, z} of up(x) is a triangle with apex x
/// (it is an arc by closure, which this checks: up(x) above y must be a subset of up(y)).
/// Returns (triangles, closure violations).
fn triangles_and_closure(up: &Csr) -> (u64, u64) {
    let n = up.n;
    let mut mark = vec![NONE; n];
    let mut tri = 0u64;
    let mut bad = 0u64;
    let mut stamp = 0u64; // unique per (x, y)
    for x in 0..n {
        let row = up.row(x);
        let d = row.len() as u64;
        tri += d * d.saturating_sub(1) / 2;
        for (k, &y) in row.iter().enumerate() {
            stamp += 1;
            for &z in up.row(y as usize) {
                mark[z as usize] = stamp;
            }
            for &z in &row[k + 1..] {
                if mark[z as usize] != stamp {
                    bad += 1;
                }
            }
        }
    }
    (tri, bad)
}

struct Quality {
    arcs: u64,
    tri: u64,
    bad: u64,
    height: u64,
    mean_depth: f64,
    cover_ok: bool,
    tree_mismatch: u64,
}

/// Implicit tree check: ranks r with parent(r) != up(r)[0].
fn tree_mismatches(up: &Csr, parent: &[u64]) -> u64 {
    (0..up.n)
        .filter(|&r| up.row(r).first().copied().unwrap_or(NONE) != parent[r])
        .count() as u64
}

/// Coverage: skeleton entries whose pair is not an arc.
fn cover_misses(s: &Csr, iperm: &[u64], up: &Csr) -> u64 {
    let mut miss = 0u64;
    for u in 0..s.n {
        for &v in s.row(u) {
            let (a, b) = (iperm[u], iperm[v as usize]);
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            if up.row(lo as usize).binary_search(&hi).is_err() {
                miss += 1;
            }
        }
    }
    miss
}

fn quality(s: &Csr, perm: &[u64]) -> Quality {
    let iperm = invert(perm, s.n);
    let parent = elimination_tree(s, perm, &iperm);
    let up = chordal_supergraph(s, perm, &iperm, &parent);
    let tree_mismatch = tree_mismatches(&up, &parent);
    let cover_ok = cover_misses(s, &iperm, &up) == 0;
    let (tri, bad) = triangles_and_closure(&up);
    // height: nodes on the longest leaf-to-root path (a root alone is 1)
    let mut depth = vec![0u32; s.n];
    let mut height = 0u64;
    let mut sum = 0u64;
    for r in (0..s.n).rev() {
        depth[r] = if parent[r] == NONE { 1 } else { depth[parent[r] as usize] + 1 };
        height = height.max(depth[r] as u64);
        sum += depth[r] as u64;
    }
    Quality { arcs: up.i.len() as u64, tri, bad, height, mean_depth: sum as f64 / s.n as f64, cover_ok, tree_mismatch }
}

fn summarize_levels(log: &[LevelRec]) -> String {
    // the top call: the one whose level 0 has the largest n_fine
    let top = log.iter().filter(|r| r.level == 0).max_by_key(|r| r.n_fine).map(|r| r.call);
    let mut out = String::new();
    if let Some(c) = top {
        let lv: Vec<&LevelRec> = log.iter().filter(|r| r.call == c).collect();
        out += &format!("top_call levels={} :", lv.len());
        for r in &lv {
            out += &format!(
                " [{}->{} ({:.3}){}{}{}]",
                r.n_fine,
                r.n_coarse,
                r.n_coarse as f64 / r.n_fine as f64,
                if r.rounds > 0 { format!(" r{}", r.rounds) } else { String::new() },
                if r.two_hop { " 2hop" } else { "" },
                if r.cap_refusals > 0 { format!(" capref{}", r.cap_refusals) } else { String::new() }
            );
        }
    }
    let calls = log.iter().map(|r| r.call).max().map_or(0, |c| c + 1);
    let nf: i64 = log.iter().map(|r| r.n_fine).sum();
    let nc: i64 = log.iter().map(|r| r.n_coarse).sum();
    let nm: i64 = log.iter().map(|r| r.n_after_match).sum();
    let twohop = log.iter().filter(|r| r.two_hop).count();
    let capped = log.iter().filter(|r| r.capped).count();
    let mut rounds: Vec<u32> = log.iter().filter(|r| r.rounds > 0).map(|r| r.rounds).collect();
    rounds.sort_unstable();
    let rq = |q: f64| if rounds.is_empty() { 0 } else { rounds[((rounds.len() - 1) as f64 * q) as usize] };
    out += &format!(
        "\n  all_calls: calls={calls} levels={} weighted_shrink(after_match)={:.4} weighted_shrink(final)={:.4} two_hop_levels={twohop} round_capped_levels={capped} rounds p50={} p99={} max={}",
        log.len(),
        nm as f64 / nf.max(1) as f64,
        nc as f64 / nf.max(1) as f64,
        rq(0.5),
        rq(0.99),
        rq(1.0)
    );
    out
}

fn clear_logs() {
    take_level_log();
    NODE_LOG.lock().unwrap().clear();
    SPLIT_LOG.lock().unwrap().clear();
    STRAD_LOG.lock().unwrap().clear();
}

/// One line per ND depth: splits, fresh coarsening (all variants), and
/// for reuse cells the levels reused, fresh, skipped, and the weight of
/// their heaviest coarsest vertex against the cap R would have used.
fn summarize_depths(
    log: &[LevelRec],
    nodes: &[feral_metis::internals::reuse::NodeRec],
    splits: &[(u32, u32, u64)],
) -> String {
    let maxd = splits.iter().map(|x| x.0).max().unwrap_or(0);
    let mut out = String::new();
    for d in 0..=maxd {
        let sp: Vec<_> = splits.iter().filter(|x| x.0 == d).collect();
        let lv: Vec<_> = log.iter().filter(|r| r.depth == d).collect();
        let nd: Vec<_> = nodes.iter().filter(|r| r.depth == d).collect();
        let mut ratios: Vec<f64> = nd
            .iter()
            .filter(|r| !r.root)
            .map(|r| r.maxw as f64 / r.cap_r.max(1) as f64)
            .collect();
        ratios.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let q = |f: f64| if ratios.is_empty() { 0.0 } else { ratios[((ratios.len() - 1) as f64 * f) as usize] };
        out += &format!(
            "  DEPTH d={d} splits={} n_sum={} coarsen_levels={} coarsen_nfine={} reuse_cells={} roots={} reused={} fresh={} skipped={} slow={} overcap={} wratio_p50={:.2} wratio_p90={:.2} wratio_max={:.2} l2_cells={}\n",
            sp.len(),
            sp.iter().map(|x| x.1 as u64).sum::<u64>(),
            lv.len(),
            lv.iter().map(|r| r.n_fine as u64).sum::<u64>(),
            nd.iter().filter(|r| !r.root).count(),
            nd.iter().filter(|r| r.root).count(),
            nd.iter().filter(|r| !r.root).map(|r| r.reused as u64).sum::<u64>(),
            nd.iter().filter(|r| !r.root).map(|r| r.fresh as u64).sum::<u64>(),
            nd.iter().map(|r| r.skipped as u64).sum::<u64>(),
            nd.iter().map(|r| r.slow as u64).sum::<u64>(),
            nd.iter().filter(|r| !r.root && r.maxw > r.cap_r).count(),
            q(0.5),
            q(0.9),
            q(1.0),
            nd.iter().filter(|r| !r.root && r.l2).count(),
        );
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a[1].as_str();
    let s = load(&a[2]);
    let isolated = (0..s.n).filter(|&u| s.row(u).is_empty()).count();
    eprintln!("graph {}: n={} skeleton_entries={} isolated={}", a[2], s.n, s.i.len(), isolated);
    match mode {
        "order" => {
            let v = a[3].as_str();
            let seed: u64 = a[4].parse().unwrap();
            let reps: usize = a[5].parse().unwrap();
            let opts = variant_opts(v, seed);
            let mut times = Vec::new();
            let mut first: Option<Vec<u64>> = None;
            let mut levels = String::new();
            let mut depth_lines = String::new();
            let mut check_lines = String::new();
            let mut check_tot = String::from("b2_checked=- b2_mismatch=- ident_fail=- b1_invalid=- check_s=-");
            let mut st = None;
            let mut mech = String::new();
            clear_logs();
            if opts.reuse && std::env::var("M1_NOCHECK").is_err() {
                // Untimed check rep: B1 == B2 on every level, straddler statistics.
                let copts = MetisOptions { reuse_check: true, ..opts.clone() };
                let (perm, secs, _) = order(&s, &copts);
                let strad = std::mem::take(&mut *STRAD_LOG.lock().unwrap());
                clear_logs();
                let (mut c, mut m, mut idf, mut inv) = (0u64, 0u64, 0u64, 0u64);
                for ((d, l), r) in &strad {
                    c += r.b2_checked;
                    m += r.b2_mismatch;
                    idf += r.ident_fail;
                    inv += r.b1_invalid;
                    check_lines += &format!(
                        "  STRAD d={d} l={l} splits_all={} n_all={} nnz_all={} strad_side_all={} splits={} n={} nnz={} sep_only={} strad_side={} strad_child={} strad_rows_nnz={} parts={} child_n={} ident_fail={} used={} x_n={} x_nnz={} xprev_nnz={} rin={} rout={} x_strad={} b2_checked={} b2_mismatch={} b1_invalid={}\n",
                        r.splits_all, r.n_all, r.nnz_all, r.strad_side_all, r.splits, r.n_coarse, r.nnz, r.sep_only, r.strad_side, r.strad_child, r.strad_rows_nnz, r.parts, r.child_n, r.ident_fail, r.used, r.x_n, r.x_nnz, r.xprev_nnz, r.recontract_in_nnz, r.recontract_out_nnz, r.x_strad, r.b2_checked, r.b2_mismatch, r.b1_invalid
                    );
                }
                check_tot = format!("b2_checked={c} b2_mismatch={m} ident_fail={idf} b1_invalid={inv} check_s={secs:.3}");
                first = Some(perm);
            }
            for rep in 0..reps {
                clear_logs();
                // Break: a different seed on the second repetition.
                let ropts = if rep == 1 && std::env::var("M1_REP_BREAK").is_ok() {
                    MetisOptions { seed: seed + 1, ..opts.clone() }
                } else {
                    opts.clone()
                };
                let (mut perm, secs, stats) = order(&s, &ropts);
                let log = take_level_log();
                let nodes = std::mem::take(&mut *NODE_LOG.lock().unwrap());
                let splits = std::mem::take(&mut *SPLIT_LOG.lock().unwrap());
                if v.starts_with("swap") {
                    let k: usize = v[4..].parse().unwrap();
                    let mut r = SplitMix::new(seed ^ 0xABCD);
                    for _ in 0..k {
                        let i = r.gen_range(perm.len() as u64) as usize;
                        let j = r.gen_range(perm.len() as u64) as usize;
                        perm.swap(i, j);
                    }
                }
                times.push(secs);
                if rep == 0 {
                    levels = summarize_levels(&log);
                    depth_lines = summarize_depths(&log, &nodes, &splits);
                    // Mechanism: matching after depth 0 only where reuse ran out.
                    let after0 = log.iter().filter(|r| r.depth >= 1).count();
                    let fresh_nodes: u32 = nodes.iter().filter(|r| !r.root).map(|r| r.fresh).sum();
                    let roots_after0 = nodes.iter().filter(|r| r.root && r.depth >= 1).count();
                    let mut h: u64 = 0;
                    for &(d, n, x) in &splits {
                        if d == 0 {
                            h = h.rotate_left(7) ^ x ^ (n as u64);
                        }
                    }
                    let roots: Vec<String> = splits.iter().filter(|x| x.0 == 0).map(|x| format!("{}:{:016x}", x.1, x.2)).collect();
                    mech = format!("coarsen_levels_after_d0={after0} reuse_fresh_levels={fresh_nodes} roots_after_d0={roots_after0} root_hash={h:016x} roots={}", roots.join(","));
                    st = Some(stats);
                }
                match &first {
                    None => first = Some(perm),
                    Some(p) => assert_eq!(p, &perm, "order differs between repetitions"),
                }
            }
            let perm = first.unwrap();
            let q = quality(&s, &perm);
            let mut ts = times.clone();
            ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let st = st.unwrap();
            println!(
                "RESULT graph={} variant={v} seed={seed} arcs={} triangles={} closure_violations={} cover_ok={} tree_mismatch={} etree_height={} mean_depth={:.1} order_s_median={:.3} order_s_all={:?} nd_levels_total={} two_hop_fallbacks={} sep_vertices={} {mech} {check_tot}",
                a[2].rsplit('/').next().unwrap(),
                q.arcs,
                q.tri,
                q.bad,
                q.cover_ok,
                q.tree_mismatch,
                q.height,
                q.mean_depth,
                ts[ts.len() / 2],
                times.iter().map(|t| (t * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
                st.n_levels,
                st.n_two_hop_fallbacks,
                st.n_separator_vertices
            );
            println!("  {levels}");
            print!("{depth_lines}");
            print!("{check_lines}");
        }
        "strad" => {
            // The untimed check rep alone (B1 == B2, straddler statistics), with quality.
            let v = a[3].as_str();
            let seed: u64 = a[4].parse().unwrap();
            let opts = MetisOptions { reuse_check: true, ..variant_opts(v, seed) };
            assert!(opts.reuse, "strad: a reuse variant");
            clear_logs();
            let (perm, secs, _) = order(&s, &opts);
            let strad = std::mem::take(&mut *STRAD_LOG.lock().unwrap());
            clear_logs();
            let q = quality(&s, &perm);
            let (mut c, mut m, mut idf, mut inv) = (0u64, 0u64, 0u64, 0u64);
            let mut lines = String::new();
            for ((d, l), r) in &strad {
                c += r.b2_checked;
                m += r.b2_mismatch;
                idf += r.ident_fail;
                inv += r.b1_invalid;
                lines += &format!(
                    "  STRAD d={d} l={l} splits_all={} n_all={} nnz_all={} strad_side_all={} splits={} n={} nnz={} sep_only={} strad_side={} strad_child={} strad_rows_nnz={} parts={} child_n={} ident_fail={} used={} x_n={} x_nnz={} xprev_nnz={} rin={} rout={} x_strad={} b2_checked={} b2_mismatch={} b1_invalid={}\n",
                    r.splits_all, r.n_all, r.nnz_all, r.strad_side_all, r.splits, r.n_coarse, r.nnz, r.sep_only, r.strad_side, r.strad_child, r.strad_rows_nnz, r.parts, r.child_n, r.ident_fail, r.used, r.x_n, r.x_nnz, r.xprev_nnz, r.recontract_in_nnz, r.recontract_out_nnz, r.x_strad, r.b2_checked, r.b2_mismatch, r.b1_invalid
                );
            }
            println!(
                "STRADRUN graph={} variant={v} seed={seed} arcs={} triangles={} etree_height={} closure_violations={} cover_ok={} tree_mismatch={} b2_checked={c} b2_mismatch={m} ident_fail={idf} b1_invalid={inv} check_s={secs:.3}",
                a[2].rsplit('/').next().unwrap(), q.arcs, q.tri, q.height, q.bad, q.cover_ok, q.tree_mismatch
            );
            print!("{lines}");
        }
        "coarsen" => {
            // largest connected component, coarsened to <= 100 (the design's cell floor)
            let v = a[3].as_str();
            let seed: u64 = a[4].parse().unwrap();
            let mut comp = vec![u32::MAX; s.n];
            let mut best = (0usize, 0u32);
            let mut stack = Vec::new();
            let mut c = 0u32;
            for r in 0..s.n {
                if comp[r] != u32::MAX || s.row(r).is_empty() {
                    continue;
                }
                let mut size = 0;
                comp[r] = c;
                stack.push(r);
                while let Some(u) = stack.pop() {
                    size += 1;
                    for &w in s.row(u) {
                        if comp[w as usize] == u32::MAX {
                            comp[w as usize] = c;
                            stack.push(w as usize);
                        }
                    }
                }
                if size > best.0 {
                    best = (size, c);
                }
                c += 1;
            }
            let mut local = vec![-1i64; s.n];
            let mut members = Vec::new();
            for u in 0..s.n {
                if comp[u] == best.1 {
                    local[u] = members.len() as i64;
                    members.push(u);
                }
            }
            let mut xadj = vec![0i64];
            let mut adj = Vec::new();
            for &u in &members {
                for &w in s.row(u) {
                    adj.push(local[w as usize]);
                }
                xadj.push(adj.len() as i64);
            }
            let pat = CscPattern::new(members.len(), &xadj, &adj).unwrap();
            let g = Graph::from_csc_pattern(&pat).unwrap();
            let opts = MetisOptions { coarsen_floor: 100, ..variant_opts(v, seed) };
            let mut rng = SplitMix::new(seed);
            let mut ctr = CoarsenCounters::default();
            take_level_log();
            let t = Instant::now();
            let lv = coarsen(&g, &opts, &mut rng, &mut ctr);
            let secs = t.elapsed().as_secs_f64();
            let log = take_level_log();
            println!(
                "COARSEN graph={} variant={v} component_n={} levels={} final_n={} coarsen_s={:.3}",
                a[2].rsplit('/').next().unwrap(),
                members.len(),
                lv.len(),
                lv.last().map_or(g.nvtxs, |l| l.graph.nvtxs),
                secs
            );
            let maxw = lv.last().map_or(1, |l| *l.graph.vwgt.iter().max().unwrap());
            println!("  max coarse vertex weight={maxw}");
            println!("  {}", summarize_levels(&log));
        }
        "same" => {
            let (mut iso, nodes, xadj, adj) = connected_pattern(&s);
            let pat = CscPattern::new(nodes.len(), &xadj, &adj).unwrap();
            let mine = if std::env::var("M1_SAME_BREAK").is_ok() {
                // Break: the copy with seed 2 must not match feral's default (seed 1).
                feral_metis::metis_order_full(&pat, &MetisOptions { seed: 2, ..MetisOptions::default() }).unwrap().0
            } else {
                feral_metis::metis_order(&pat).unwrap()
            };
            let pat2 = feral_orig::CscPattern::new(nodes.len(), &xadj, &adj).unwrap();
            let orig = feral_orig::metis_order(&pat2).unwrap();
            println!("SAME graph={} break={} stock-copy == original feral f127189: {}", a[2].rsplit('/').next().unwrap(), std::env::var("M1_SAME_BREAK").is_ok(), mine == orig);
            iso.extend(mine.into_iter().map(|v| nodes[v as usize]));
        }
        "checkbreak" => {
            let v = a.get(3).map_or("shem", |x| x.as_str());
            let opts = variant_opts(v, 1);
            let (perm, _, _) = order(&s, &opts);
            let iperm = invert(&perm, s.n);
            let parent = elimination_tree(&s, &perm, &iperm);
            let up = chordal_supergraph(&s, &perm, &iperm, &parent);
            let report = |tag: &str, up: &Csr, parent: &[u64]| {
                let (t, b) = triangles_and_closure(up);
                println!(
                    "CHECKBREAK variant={v} {tag}: tree_mismatch={} cover_misses={} closure_violations={b} triangles={t}",
                    tree_mismatches(up, parent),
                    cover_misses(&s, &iperm, up)
                );
            };
            report("intact", &up, &parent);
            // c = first rank with >= 3 upper neighbours, r = its parent, z = its next one:
            // closure needs z in up(r); drop it there.
            let c = (0..up.n).find(|&c| up.row(c).len() >= 3).unwrap();
            let r = up.row(c)[0] as usize;
            let z = up.row(c)[1];
            let drop_arc = |up: &Csr, r: usize, z: u64| {
                let mut b = Csr { n: up.n, p: up.p.clone(), i: up.i.clone() };
                let pos = b.p[r] as usize + b.row(r).binary_search(&z).unwrap();
                b.i.remove(pos);
                for k in r + 1..=b.n {
                    b.p[k] -= 1;
                }
                b
            };
            report(&format!("closure break: arc ({r},{z}) dropped"), &drop_arc(&up, r, z), &parent);
            // cover: drop the arc of a skeleton edge that is not a tree edge
            let (u, w) = (0..s.n)
                .flat_map(|u| s.row(u).iter().map(move |&v| (u, v as usize)))
                .map(|(u, v)| {
                    let (x, y) = (iperm[u], iperm[v]);
                    if x < y { (x as usize, y) } else { (y as usize, x) }
                })
                .find(|&(lo, hi)| parent[lo] != hi)
                .unwrap();
            report(&format!("cover break: skeleton arc ({u},{w}) dropped"), &drop_arc(&up, u, w), &parent);
            // tree: re-point one parent to its second upper neighbour
            let t = (0..up.n).find(|&t| up.row(t).len() >= 2).unwrap();
            let mut bad_parent = parent.clone();
            bad_parent[t] = up.row(t)[1];
            report(&format!("tree break: parent({t}) := up({t})[1]"), &up, &bad_parent);
        }
        "samebase" => {
            // This crate against the untouched pre-reuse copy, same variant and seed.
            let v = a[3].as_str();
            let seed: u64 = a[4].parse().unwrap();
            let (_, nodes, xadj, adj) = connected_pattern(&s);
            let pat = CscPattern::new(nodes.len(), &xadj, &adj).unwrap();
            let bump = if std::env::var("M1_SAMEBASE_BREAK").is_ok() { 1 } else { 0 };
            let mine = feral_metis::metis_order_full(&pat, &variant_opts(v, seed + bump)).unwrap().0;
            let pat2 = feral_base::CscPattern::new(nodes.len(), &xadj, &adj).unwrap();
            let mut bo = feral_base::MetisOptions { seed, ..feral_base::MetisOptions::default() };
            match v {
                "shem" => {}
                "mm" => {
                    bo.matching = feral_base::Matching::MutualMax;
                    bo.vwgt_cap = true;
                }
                _ => panic!("samebase: shem or mm"),
            }
            let base = feral_base::metis_order_full(&pat2, &bo).unwrap().0;
            println!("SAMEBASE graph={} variant={v} seed={seed} break={bump} identical={}", a[2].rsplit('/').next().unwrap(), mine == base);
        }
        "export" => {
            let (_, nodes, xadj, adj) = connected_pattern(&s);
            let mut out: Vec<u8> = Vec::new();
            out.extend((nodes.len() as i64).to_le_bytes());
            out.extend((adj.len() as i64).to_le_bytes());
            for x in xadj.iter().chain(adj.iter()) {
                out.extend(x.to_le_bytes());
            }
            std::fs::write(&a[3], out).unwrap();
        }
        "evalperm" => {
            let (mut iso, nodes, _, _) = connected_pattern(&s);
            let raw = std::fs::read(&a[3]).unwrap();
            assert_eq!(raw.len(), nodes.len() * 8);
            for c in raw.chunks(8) {
                iso.push(nodes[i64::from_le_bytes(c.try_into().unwrap()) as usize]);
            }
            let q = quality(&s, &iso);
            println!(
                "RESULT graph={} variant=cmetis seed=0 arcs={} triangles={} closure_violations={} cover_ok={} etree_height={} mean_depth={:.1} order_s_median=0 nd_levels_total=0 two_hop_fallbacks=0 sep_vertices=0",
                a[2].rsplit('/').next().unwrap(), q.arcs, q.tri, q.bad, q.cover_ok, q.height, q.mean_depth
            );
        }
        _ => panic!("mode"),
    }
}
