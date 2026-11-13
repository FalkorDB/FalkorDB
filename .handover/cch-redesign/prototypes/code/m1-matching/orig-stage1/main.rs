// M1 prototype: ordering quality of the design's mutual-max matching against stock feral SHEM.
//
// usage:
//   m1bench order <graph.txt.gz> <variant> <seed> <reps>
//   m1bench coarsen <graph.txt.gz> <variant> <seed>       (largest component, coarsen to <= 100)
//   m1bench same <graph.txt.gz>                           (stock copy vs original feral: identical perm?)
//   m1bench checkbreak <graph.txt.gz>                     (closure checker must reject a broken row)
//
// variants: shem | shem-cap | mm | mm-nocap | mm-r<K> | mm-inv | mm-seq2hop | swap<K> (feral order + K random swaps)
//
// The elimination tree and chordal supergraph are the harness's (cchbench/src/bin/phase3.rs, copied
// from the stack's order.rs): Liu's etree, then child absorption in increasing rank.

use std::io::{BufRead, BufReader};
use std::time::Instant;

use feral_metis::internals::coarsen::{coarsen, take_level_log, CoarsenCounters, LevelRec};
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
}

fn quality(s: &Csr, perm: &[u64]) -> Quality {
    let iperm = invert(perm, s.n);
    let parent = elimination_tree(s, perm, &iperm);
    let up = chordal_supergraph(s, perm, &iperm, &parent);
    // implicit tree check: parent(r) = up(r)[0]
    for r in 0..s.n {
        assert_eq!(up.row(r).first().copied().unwrap_or(NONE), parent[r]);
    }
    // coverage: every skeleton edge is an arc
    let mut cover_ok = true;
    for u in 0..s.n {
        for &v in s.row(u) {
            let (a, b) = (iperm[u], iperm[v as usize]);
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            if up.row(lo as usize).binary_search(&hi).is_err() {
                cover_ok = false;
            }
        }
    }
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
    Quality { arcs: up.i.len() as u64, tri, bad, height, mean_depth: sum as f64 / s.n as f64, cover_ok }
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
            let mut st = None;
            for _ in 0..reps {
                take_level_log();
                let (mut perm, secs, stats) = order(&s, &opts);
                let log = take_level_log();
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
                match &first {
                    None => {
                        levels = summarize_levels(&log);
                        st = Some(stats);
                        first = Some(perm);
                    }
                    Some(p) => assert_eq!(p, &perm, "order differs between repetitions"),
                }
            }
            let perm = first.unwrap();
            let q = quality(&s, &perm);
            let mut ts = times.clone();
            ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let st = st.unwrap();
            println!(
                "RESULT graph={} variant={v} seed={seed} arcs={} triangles={} closure_violations={} cover_ok={} etree_height={} mean_depth={:.1} order_s_median={:.3} order_s_all={:?} nd_levels_total={} two_hop_fallbacks={} sep_vertices={}",
                a[2].rsplit('/').next().unwrap(),
                q.arcs,
                q.tri,
                q.bad,
                q.cover_ok,
                q.height,
                q.mean_depth,
                ts[ts.len() / 2],
                times.iter().map(|t| (t * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
                st.n_levels,
                st.n_two_hop_fallbacks,
                st.n_separator_vertices
            );
            println!("  {levels}");
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
            let mine = feral_metis::metis_order(&pat).unwrap();
            let pat2 = feral_orig::CscPattern::new(nodes.len(), &xadj, &adj).unwrap();
            let orig = feral_orig::metis_order(&pat2).unwrap();
            println!("SAME stock-copy == original feral f127189: {}", mine == orig);
            iso.extend(mine.into_iter().map(|v| nodes[v as usize]));
        }
        "checkbreak" => {
            let opts = variant_opts("shem", 1);
            let (perm, _, _) = order(&s, &opts);
            let iperm = invert(&perm, s.n);
            let parent = elimination_tree(&s, &perm, &iperm);
            let mut up = chordal_supergraph(&s, &perm, &iperm, &parent);
            let (t0, b0) = triangles_and_closure(&up);
            // c = first rank with >= 3 upper neighbours, r = its parent, z = its next one:
            // closure needs z in up(r); drop it there.
            let c = (0..up.n).find(|&c| up.row(c).len() >= 3).unwrap();
            let r = up.row(c)[0] as usize;
            let z = up.row(c)[1];
            let pos = up.p[r] as usize + up.row(r).binary_search(&z).unwrap();
            up.i.remove(pos);
            for k in r + 1..=up.n {
                up.p[k] -= 1;
            }
            let (t1, b1) = triangles_and_closure(&up);
            println!("CHECKBREAK intact: triangles={t0} violations={b0}; one arc dropped from rank {r}: triangles={t1} violations={b1}");
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
