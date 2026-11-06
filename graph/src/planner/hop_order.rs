//! Ordering of the hops of one connected pattern component.

/// The parts of a hop that [`self_loops_after_binders`] looks at.
#[derive(Clone, Copy)]
pub struct HopShape<N> {
    pub from: N,
    pub to: N,
    /// `(n)-[]->(n)`.
    pub self_loop: bool,
    /// Neither variable-length nor a shortest-path hop.
    pub fixed: bool,
}

/// Past the first hop, a fixed-length self-loop on a node that is not
/// `bound` and that no earlier hop touches could only scan that node again
/// and cross it with the hops before it. Move every such self-loop to just
/// after the first later hop to another node that touches its node (its
/// "binder"), so it plans as an `ExpandInto` on the bound node. Self-loops
/// moved behind the same binder are placed latest-first; all other hops keep
/// their relative order. A self-loop with no later binder stays where it is.
///
/// One pass over `hops`: each hop is emitted in place, or deferred once and
/// emitted right after its binder; no hop is looked at twice and nothing is
/// shifted. Only a fixed self-loop scans its neighbours.
pub fn self_loops_after_binders<T, N: Copy + Eq>(
    hops: Vec<T>,
    shape: impl Fn(&T) -> HopShape<N>,
    bound: impl Fn(&N) -> bool,
) -> Vec<T> {
    let shapes: Vec<HopShape<N>> = hops.iter().map(shape).collect();
    let movable = |k: usize| k > 0 && shapes[k].self_loop && shapes[k].fixed;
    if !(1..shapes.len()).any(movable) {
        return hops;
    }
    let touches = |x: &HopShape<N>, n: N| x.from == n || x.to == n;
    let mut deferred: Vec<(N, T)> = vec![];
    let mut out = Vec::with_capacity(hops.len());
    for (i, (hop, s)) in hops.into_iter().zip(&shapes).enumerate() {
        // Earlier movable self-loops on the same node are deferred too, so
        // they do not bind it.
        if movable(i)
            && !bound(&s.from)
            && !(0..i).any(|k| touches(&shapes[k], s.from) && !movable(k))
            && shapes[i + 1..]
                .iter()
                .any(|x| !x.self_loop && touches(x, s.from))
        {
            deferred.push((s.from, hop));
            continue;
        }
        out.push(hop);
        if !s.self_loop {
            // Latest-first, across both endpoints.
            for k in (0..deferred.len()).rev() {
                if touches(s, deferred[k].0) {
                    out.push(deferred.remove(k).1);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// (from, to, fixed); a self-loop when from == to.
    type Hop = (u32, u32, bool);

    fn shape(h: &(usize, Hop)) -> HopShape<u32> {
        let (_, (from, to, fixed)) = *h;
        HopShape {
            from,
            to,
            self_loop: from == to,
            fixed,
        }
    }

    /// The loop this replaced (#3102 review), kept verbatim in shape: it
    /// revisits every hop it moves.
    fn reference(
        mut rels: Vec<(usize, Hop)>,
        bound: &HashSet<u32>,
    ) -> Vec<(usize, Hop)> {
        let mut i = 1;
        while i < rels.len() {
            let r = shape(&rels[i]);
            let node = r.from;
            let touches = |x: &(usize, Hop)| {
                let x = shape(x);
                x.from == node || x.to == node
            };
            let unbound_fixed_loop =
                r.self_loop && r.fixed && !bound.contains(&node) && !rels[..i].iter().any(touches);
            if unbound_fixed_loop
                && let Some(j) = rels[i + 1..]
                    .iter()
                    .position(|x| !shape(x).self_loop && touches(x))
            {
                let r = rels.remove(i);
                rels.insert(i + j + 1, r);
                continue;
            }
            i += 1;
        }
        rels
    }

    fn check(
        hops: &[Hop],
        bound: &[u32],
    ) -> Vec<Hop> {
        let tagged: Vec<(usize, Hop)> = hops.iter().copied().enumerate().collect();
        let bound: HashSet<u32> = bound.iter().copied().collect();
        let new = self_loops_after_binders(tagged.clone(), shape, |n| bound.contains(n));
        assert_eq!(
            new,
            reference(tagged, &bound),
            "hops {hops:?} bound {bound:?}"
        );
        new.into_iter().map(|(_, h)| h).collect()
    }

    const F: bool = true;
    const V: bool = false;

    #[test]
    fn pr_cases() {
        // (a)-[:R]->(b)-[:S*]->(c)-[:L]->(c): sorted R, L, S*.
        assert_eq!(
            check(&[(0, 1, F), (2, 2, F), (1, 2, V)], &[]),
            [(0, 1, F), (1, 2, V), (2, 2, F)]
        );
        // Two self-loops on c: latest-first after the binder.
        assert_eq!(
            check(&[(0, 1, F), (2, 2, F), (2, 2, F), (1, 2, V)], &[]),
            [(0, 1, F), (1, 2, V), (2, 2, F), (2, 2, F)]
        );
        // Node bound by an earlier clause: stays.
        check(&[(0, 1, F), (2, 2, F), (1, 2, V)], &[2]);
    }

    #[test]
    fn extra_chains() {
        // Self-loop on the first node: never moved.
        check(&[(0, 0, F), (0, 1, F), (1, 2, V)], &[]);
        // Self-loop on the last node of a long var-length tail.
        check(
            &[
                (0, 1, F),
                (5, 5, F),
                (1, 2, V),
                (2, 3, V),
                (3, 4, V),
                (4, 5, V),
            ],
            &[],
        );
        // Several self-loops on different nodes, binders sharing nodes.
        check(
            &[
                (0, 1, F),
                (3, 3, F),
                (2, 2, F),
                (3, 3, F),
                (1, 2, V),
                (2, 3, V),
            ],
            &[],
        );
        // Var-length self-loop: not moved, but it binds later fixed loops.
        check(
            &[(0, 1, F), (2, 2, F), (2, 2, V), (2, 2, F), (1, 2, V)],
            &[],
        );
        // No binder later: stays.
        check(&[(0, 1, F), (2, 2, F)], &[]);
    }

    #[test]
    fn matches_reference_on_random_chains() {
        let mut seed = 0x2999_u64;
        let mut next = |m: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((seed >> 33) % m) as u32
        };
        let mut reordered = 0;
        for _ in 0..20_000 {
            let len = next(10) as usize;
            let hops: Vec<Hop> = (0..len)
                .map(|_| {
                    let from = next(5);
                    let to = if next(2) == 0 { from } else { next(5) };
                    (from, to, next(3) != 0)
                })
                .collect();
            let bound: Vec<u32> = (0..5).filter(|_| next(4) == 0).collect();
            if check(&hops, &bound) != hops {
                reordered += 1;
            }
        }
        assert!(reordered > 1_000, "only {reordered} chains reordered");
    }

    /// `cargo test --release -p graph --lib hop_order::tests::bench -- --ignored --nocapture`
    #[test]
    #[ignore = "timing only"]
    fn bench() {
        use std::time::Instant;
        for (name, len) in [
            ("chain", 50u32),
            ("chain", 500),
            ("unbound", 50),
            ("unbound", 500),
        ] {
            let mut hops: Vec<Hop> = if name == "chain" {
                // A chain 0-1-..., alternating fixed and var-length hops,
                // every fourth node also with a self-loop.
                (0..len)
                    .map(|i| (i, i + 1, i % 2 == 0))
                    .chain((0..len).step_by(4).map(|i| (i + 1, i + 1, true)))
                    .collect()
            } else {
                // Worst case for the old loop: every fourth node has a
                // self-loop and is reached only through var-length hops, so
                // every self-loop is moved.
                (0..len)
                    .map(|i| (i, i + 1, i < 2))
                    .chain((4..len).step_by(4).map(|i| (i, i, true)))
                    .collect()
            };
            // Fixed hops before var-length ones, as plan_match sorts them.
            hops.sort_by_key(|h| !h.2);
            let tagged: Vec<(usize, Hop)> = hops.iter().copied().enumerate().collect();
            let bound = HashSet::new();
            let iters = 20_000_000 / (len * len) as usize;
            let t = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(reference(tagged.clone(), &bound));
            }
            let old = t.elapsed() / iters as u32;
            let t = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(self_loops_after_binders(tagged.clone(), shape, |n| {
                    bound.contains(n)
                }));
            }
            let new = t.elapsed() / iters as u32;
            println!("{name} {} hops: old {old:?}  new {new:?}", hops.len());
        }
    }
}
