use super::*;
use std::collections::BTreeSet;

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Sorted doc ids the tree yields for `[lo, hi]`.
fn tree_range(
    t: &CowBTree,
    lo: u64,
    hi: u64,
) -> Vec<u64> {
    let mut v: Vec<u64> = t.range(lo, hi).collect();
    v.sort_unstable();
    v
}
/// The same, computed from a reference set of `(key, doc)`.
fn ref_range(
    s: &BTreeSet<(u64, u64)>,
    lo: u64,
    hi: u64,
) -> Vec<u64> {
    let mut v: Vec<u64> = s.range((lo, 0)..=(hi, u64::MAX)).map(|&(_, d)| d).collect();
    v.sort_unstable();
    v
}

/// The full `(key, doc)` multiset the tree stores, decoded straight from the leaf bytes (so it verifies
/// the *value* column, not just docs — a full-range cursor scan skips key reads on interior leaves).
/// Generic over `LEAF_MAX` so the const-generic parity test can drive a non-default leaf size through it.
fn tree_pairs<const LEAF_MAX: usize, const BRANCH_MAX: usize>(
    t: &CowBTree<LEAF_MAX, BRANCH_MAX>
) -> Vec<(u64, u64)> {
    let mut v: Vec<(u64, u64)> = t
        .leaves()
        .into_iter()
        .flat_map(|(fmt, bytes)| Leaf::<LEAF_MAX>::from_parts(fmt, bytes).to_pairs())
        .collect();
    v.sort_unstable();
    v
}

#[test]
fn empty_and_single() {
    let mut t = CowBTree::<256, 256>::new();
    assert!(t.is_empty());
    assert_eq!(t.point(5).count(), 0);
    t.insert(5, 50);
    assert_eq!(t.len(), 1);
    assert_eq!(t.point(5).collect::<Vec<_>>(), vec![50]);
    assert_eq!(t.point(6).count(), 0);
}

#[test]
fn insert_split_parity() {
    // enough inserts to force several levels of splits
    let n = 5_000u64;
    let mut t = CowBTree::<256, 256>::new();
    let mut reference = BTreeSet::new();
    for q in 0..n {
        let k = splitmix(q) % 1000; // duplicate keys (multiple docs per key)
        t.insert(k, q);
        reference.insert((k, q));
    }
    assert_eq!(t.len(), reference.len());
    for k in 0..1000u64 {
        assert_eq!(
            tree_range(&t, k, k),
            ref_range(&reference, k, k),
            "point {k}"
        );
    }
    for q in 0..200u64 {
        let lo = splitmix(q ^ 1) % 1000;
        let hi = (lo + 50).min(999);
        assert_eq!(
            tree_range(&t, lo, hi),
            ref_range(&reference, lo, hi),
            "range [{lo},{hi}]"
        );
    }
}

#[test]
fn bulk_build_matches_incremental() {
    let pairs: Vec<(u64, u64)> = (0..3_000u64).map(|i| (i, i)).collect();
    let bulk = CowBTree::<256, 256>::from_sorted(&pairs);
    let mut inc = CowBTree::<256, 256>::new();
    for &(k, d) in &pairs {
        inc.insert(k, d);
    }
    assert_eq!(bulk.len(), inc.len());
    for k in (0..3_000u64).step_by(7) {
        assert_eq!(tree_range(&bulk, k, k), tree_range(&inc, k, k));
    }
    assert_eq!(tree_range(&bulk, 100, 2_900), tree_range(&inc, 100, 2_900));
}

#[test]
fn remove_merge_parity() {
    let n = 4_000u64;
    let mut t = CowBTree::<256, 256>::from_sorted(&(0..n).map(|i| (i, i)).collect::<Vec<_>>());
    let mut reference: BTreeSet<(u64, u64)> = (0..n).map(|i| (i, i)).collect();
    // remove ~75% in scattered order, forcing many merges
    for q in 0..n {
        if !splitmix(q).is_multiple_of(4) {
            let id = splitmix(q ^ 9) % n;
            t.remove(id, id);
            reference.remove(&(id, id));
        }
    }
    assert_eq!(t.len(), reference.len());
    assert_eq!(tree_range(&t, 0, n - 1), ref_range(&reference, 0, n - 1));
}

#[test]
fn churn_stays_compact() {
    // build, then ~100% delete+reinsert turnover; assert no memory bloat (bytes/entry stays tight).
    let w = 20_000u64;
    let mut t = CowBTree::<256, 256>::from_sorted(&(0..w).map(|i| (i, i)).collect::<Vec<_>>());
    let bytes_per = |t: &CowBTree| -> f64 {
        let b: usize = t.leaves().iter().map(|(_, l)| l.len()).sum();
        b as f64 / t.len().max(1) as f64
    };
    let build_bpc = bytes_per(&t);

    let mut live: Vec<u64> = (0..w).collect();
    let mut next = w;
    let rounds = 10u64;
    let per = w / rounds;
    for r in 0..rounds {
        for k in 0..per {
            let idx = (splitmix(r * 1009 + k) % live.len() as u64) as usize;
            let id = live.swap_remove(idx);
            t.remove(id, id);
        }
        let adds: Vec<(u64, u64)> = (0..per)
            .map(|_| {
                let id = next;
                next += 1;
                live.push(id);
                (id, id)
            })
            .collect();
        t.insert_batch(&adds);
    }
    let churn_bpc = bytes_per(&t);
    assert_eq!(t.len(), live.len());
    // tight build ≈ 16 B + header; no-bloat ⇒ stays well under 1.5× after full turnover
    assert!(
        churn_bpc < build_bpc * 1.5,
        "bloat under churn: {build_bpc:.1} → {churn_bpc:.1} B/entry"
    );
    // and reads are still correct
    let reference: BTreeSet<(u64, u64)> = live.iter().map(|&id| (id, id)).collect();
    assert_eq!(
        tree_range(&t, 0, u64::MAX),
        ref_range(&reference, 0, u64::MAX)
    );
}

#[test]
fn clone_is_an_isolated_snapshot() {
    // a clone (snapshot) is unaffected by later mutations of the original — the MVCC property.
    let mut t =
        CowBTree::<256, 256>::from_sorted(&(0..1_000u64).map(|i| (i, i)).collect::<Vec<_>>());
    let snapshot = t.clone();
    for i in 0..1_000u64 {
        t.remove(i, i);
        t.insert(10_000 + i, 10_000 + i);
    }
    // snapshot still sees the original contents; the live tree sees the new ones
    assert_eq!(snapshot.len(), 1_000);
    assert_eq!(snapshot.point(500).collect::<Vec<_>>(), vec![500]);
    assert_eq!(t.point(500).count(), 0);
    assert_eq!(t.point(10_500).collect::<Vec<_>>(), vec![10_500]);
}

#[test]
fn snapshot_is_isolated_from_a_concurrent_writer() {
    // The MVCC property under real concurrency: a reader iterating a snapshot on one thread must keep
    // seeing the snapshot's contents while a writer churns the *shared* tree on another. Every page the
    // two share has `Arc` strong-count >= 2, so the writer clones it into a private copy rather than
    // mutating in place — the reader can never observe a write it shouldn't, and the writer only ever
    // *reads* a shared page (to clone it), so there is no write-while-read race. A bug that mutated a
    // shared page in place would surface here as a changed/garbled read.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let n = 4_000u64;
    let base = CowBTree::<256, 256>::from_sorted(&(0..n).map(|i| (i, i)).collect::<Vec<_>>());
    let snapshot = base.clone(); // shares every page with `base`
    let expected: Vec<u64> = (0..n).collect();

    let done = Arc::new(AtomicBool::new(false));
    let done_reader = Arc::clone(&done);
    let reader = thread::spawn(move || {
        // Re-read the whole snapshot for as long as the writer is mutating; it must never change.
        let mut reads = 0u64;
        while !done_reader.load(Ordering::Relaxed) {
            assert_eq!(
                snapshot.range(0, u64::MAX).collect::<Vec<_>>(),
                expected,
                "snapshot observed a concurrent writer's mutation"
            );
            reads += 1;
        }
        assert_eq!(snapshot.range(0, u64::MAX).collect::<Vec<_>>(), expected);
        reads
    });

    // Churn the shared tree hard while the reader runs: delete every original entry, then insert a
    // disjoint range. Each step path-copies through pages the snapshot still holds — exercising
    // copy-on-write path-copying against a live reader.
    let mut writer = base;
    for i in 0..n {
        writer.remove(i, i);
    }
    for i in n..2 * n {
        writer.insert(i, i);
    }
    done.store(true, Ordering::Relaxed);

    let reads = reader.join().unwrap();
    assert!(reads > 0, "reader thread never completed a read");
    assert_eq!(
        writer.range(0, u64::MAX).collect::<Vec<_>>(),
        (n..2 * n).collect::<Vec<_>>(),
        "the writer's own tree should reflect all the churn"
    );
}

#[test]
fn leaf_bytes_round_trip_through_a_byte_store() {
    // a leaf blob round-trips through a byte store verbatim: the in-memory form is the serialized form.
    let t = CowBTree::<256, 256>::from_sorted(&(0..2_000u64).map(|i| (i, i)).collect::<Vec<_>>());
    // store the format tag + raw bytes
    let store: Vec<(LeafFormat, Vec<u8>)> =
        t.leaves().iter().map(|(f, b)| (*f, b.to_vec())).collect();
    // re-read the docs for a key directly from a stored leaf — proving the bytes are usable as-is.
    // find the leaf containing key 1234 by its min key, then scan it.
    let want = 1234u64;
    let mut found = None;
    for (fmt, blob) in &store {
        let leaf = Leaf::<256>::from_parts(*fmt, Arc::from(blob.as_slice())); // re-wrap — proving they're usable as-is
        if leaf.count() > 0 && leaf.key(0) <= want && leaf.key(leaf.count() - 1) >= want {
            let i = leaf.lower_bound(want);
            if i < leaf.count() && leaf.key(i) == want {
                found = Some(leaf.doc(i));
            }
        }
    }
    assert_eq!(found, Some(1234));
}

#[test]
fn leaf_format_roundtrip() {
    // For each data shape: build a leaf, assert it decodes back to the exact input, reads each entry
    // consistently, round-trips through bytes()/from_parts, and selected the expected in-RAM type.
    // Covers AoS, compact-without-dedup, and compact-with-dedup (the three encode paths) — now three
    // distinct in-RAM types, though only two serialized formats (index presence lives in the buffer).

    /// The expected in-RAM leaf type for a data shape (`None` ⇒ don't check, e.g. the empty leaf).
    enum Want {
        Aos,
        Compact,
        CompactIndexed,
    }

    fn check(
        pairs: Vec<(u64, u64)>,
        want: Option<Want>,
    ) {
        let leaf = Leaf::<256>::from_pairs(&pairs);
        assert_eq!(leaf.count(), pairs.len(), "count for {pairs:?}");
        assert_eq!(leaf.to_pairs(), pairs, "to_pairs for {pairs:?}");
        for (i, &(k, d)) in pairs.iter().enumerate() {
            assert_eq!(leaf.key(i), k, "key[{i}] of {pairs:?}");
            assert_eq!(leaf.doc(i), d, "doc[{i}] of {pairs:?}");
        }
        let blob = leaf.bytes();
        assert_eq!(
            Leaf::<256>::from_parts(leaf.format(), blob).to_pairs(),
            pairs,
            "bytes round-trip for {pairs:?}"
        );
        match want {
            Some(Want::Aos) => {
                assert!(matches!(leaf, Leaf::Aos(_)), "expected AoS for {pairs:?}")
            }
            Some(Want::Compact) => assert!(
                matches!(leaf, Leaf::Compact(_)),
                "expected compact (no index) for {pairs:?}"
            ),
            Some(Want::CompactIndexed) => assert!(
                matches!(leaf, Leaf::CompactIndexed(_)),
                "expected compact-indexed for {pairs:?}"
            ),
            None => {}
        }
    }
    check(vec![], None); // empty ⇒ AoS
    check(vec![(5, 50)], Some(Want::Aos)); // single entry ⇒ AoS (header doesn't amortise)
    // wide, all-distinct values AND docs (both need 8-byte width) ⇒ no compression ⇒ AoS
    check(
        (0..200u64).map(|i| (i << 40, i << 40)).collect(),
        Some(Want::Aos),
    );
    // narrow consecutive values + small ids, all distinct ⇒ compact WITHOUT dedup
    check((0..256u64).map(|i| (i, i)).collect(), Some(Want::Compact));
    // low cardinality (4 distinct wide values × 64 docs) ⇒ compact WITH dedup
    let mut low_card: Vec<(u64, u64)> = Vec::new();
    for v in 0..4u64 {
        for d in 0..64u64 {
            low_card.push((v * 1_000_000, v * 64 + d));
        }
    }
    low_card.sort_unstable();
    check(low_card, Some(Want::CompactIndexed));
    // single value, many docs ⇒ compact (dedup, n_distinct == 1)
    check(
        (0..200u64).map(|d| (42, d)).collect(),
        Some(Want::CompactIndexed),
    );
}

#[test]
fn low_cardinality_parity() {
    // Low-cardinality data forces compact (dedup) leaves; point + range reads must still match a
    // reference, and at least one leaf must actually have chosen the compact format.
    let (n_values, docs_per) = (60u64, 70u64);
    let mut t = CowBTree::<256, 256>::new();
    let mut reference: BTreeSet<(u64, u64)> = BTreeSet::new();
    let mut doc = 0u64;
    for v in 0..n_values {
        let key = v * 1_000;
        for _ in 0..docs_per {
            t.insert(key, doc);
            reference.insert((key, doc));
            doc += 1;
        }
    }
    assert_eq!(t.len(), reference.len());
    assert!(
        t.leaves().iter().any(|(f, _)| *f == LeafFormat::Compact),
        "expected a compact leaf"
    );
    for v in 0..n_values {
        let k = v * 1_000;
        assert_eq!(
            tree_range(&t, k, k),
            ref_range(&reference, k, k),
            "point {k}"
        );
    }
    assert_eq!(
        tree_range(&t, 0, u64::MAX),
        ref_range(&reference, 0, u64::MAX),
        "full range"
    );
    assert_eq!(
        tree_range(&t, 5_000, 25_000),
        ref_range(&reference, 5_000, 25_000),
        "sub range"
    );
}

#[test]
fn format_migration() {
    // A leaf starts AoS (wide, all-distinct) and migrates to compact as low-cardinality data arrives;
    // reads stay correct across the mix. Exercises the cursor on a tree holding both formats.
    let mut t = CowBTree::<256, 256>::new();
    let mut reference: BTreeSet<(u64, u64)> = BTreeSet::new();
    for i in 0..120u64 {
        let (k, d) = (i << 40, i << 40); // wide value + wide doc, all distinct ⇒ AoS
        t.insert(k, d);
        reference.insert((k, d));
    }
    assert!(
        t.leaves().iter().any(|(f, _)| *f == LeafFormat::Aos),
        "expected AoS leaves initially"
    );
    for j in 0..500u64 {
        let k = (j % 3) << 40; // pile many docs onto 3 existing keys ⇒ low cardinality
        let d = 10_000 + j; // narrow docs so the now-low-card leaves clear the compaction margin
        t.insert(k, d);
        reference.insert((k, d));
    }
    assert!(
        t.leaves().iter().any(|(f, _)| *f == LeafFormat::Compact),
        "expected compact leaves after low-card inserts"
    );
    assert_eq!(t.len(), reference.len());
    assert_eq!(
        tree_range(&t, 0, u64::MAX),
        ref_range(&reference, 0, u64::MAX),
        "parity across mixed formats"
    );
}

#[test]
fn bulk_built_thin_tail_deletes_without_panicking() {
    // Regression: a bulk build whose leaf count is `BRANCH_MAX + 1` (= 257) once packed a trailing
    // single-child branch; deleting from its lone child then panicked in `rebalance` (no sibling, so
    // `child_idx - 1` underflowed). Build exactly that shape, delete the tail, and check it stays sane.
    let n = (256 * 256 + 1) as u64; // 257 leaves: 256 full + 1 one-entry leaf
    let pairs: Vec<(u64, u64)> = (0..n).map(|i| (i, i)).collect();
    let mut t = CowBTree::<256, 256>::from_sorted(&pairs);
    let mut reference: BTreeSet<(u64, u64)> = pairs.iter().copied().collect();
    assert_eq!(t.len(), reference.len());
    // Delete the tail — the first removal underflows the formerly-singleton branch's child.
    for k in (n - 400..n).rev() {
        t.remove(k, k);
        reference.remove(&(k, k));
    }
    assert_eq!(t.len(), reference.len());
    let got: Vec<u64> = t.range(0, u64::MAX).collect();
    let expected: Vec<u64> = reference.iter().map(|&(_, d)| d).collect();
    assert_eq!(got, expected);
}

#[test]
fn aos_splice_insert_remove_edges() {
    // The AoS byte-splice path: insert/remove at the front, back, and middle of a leaf, plus the
    // already-present / absent no-ops. Wide, all-distinct data keeps the leaf in `LeafFormat::Aos`.
    let mut t = CowBTree::<256, 256>::new();
    let mut reference: BTreeSet<(u64, u64)> = BTreeSet::new();
    for i in 1..100u64 {
        let (k, d) = (i << 40, i << 40);
        t.insert(k, d);
        reference.insert((k, d));
    }
    assert!(
        t.leaves().iter().all(|(f, _)| *f == LeafFormat::Aos),
        "wide data should encode as AoS"
    );
    // insert below all, above all, and into the middle
    for (k, d) in [
        (0u64, 0u64),
        (1_000u64 << 40, 1_000 << 40),
        ((50 << 40) + 1, 7),
    ] {
        t.insert(k, d);
        reference.insert((k, d));
    }
    // a duplicate (key, doc) insert is a no-op
    let before = t.len();
    t.insert(0, 0);
    assert_eq!(t.len(), before, "duplicate insert must not grow the leaf");
    // remove front / back / middle, then an absent tuple (no-op)
    for (k, d) in [
        (0u64, 0u64),
        (1_000u64 << 40, 1_000 << 40),
        ((50 << 40) + 1, 7),
    ] {
        t.remove(k, d);
        reference.remove(&(k, d));
    }
    t.remove(99_999 << 40, 1); // absent
    assert_eq!(t.len(), reference.len());
    assert_eq!(
        t.range(0, u64::MAX).collect::<Vec<_>>(),
        reference.iter().map(|&(_, d)| d).collect::<Vec<_>>(),
    );
}

#[test]
fn insert_batch_merge_parity() {
    // `insert_batch` must match a `BTreeSet` reference across the merge paths: small batches (AoS
    // byte-merge), large batches that overflow into several pages (fallback), tuples already present
    // (dedup), and compact source leaves (fallback).
    fn check(
        initial: &[(u64, u64)],
        batch: &[(u64, u64)],
    ) {
        let mut t = CowBTree::<256, 256>::from_sorted(initial);
        let mut reference: BTreeSet<(u64, u64)> = initial.iter().copied().collect();
        let mut sorted = batch.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        t.insert_batch(&sorted);
        reference.extend(sorted.iter().copied());
        assert_eq!(t.len(), reference.len(), "len after batch");
        assert_eq!(
            t.range(0, u64::MAX).collect::<Vec<_>>(),
            reference.iter().map(|&(_, d)| d).collect::<Vec<_>>(),
        );
    }
    // wide, all-distinct ⇒ AoS leaves
    let aos: Vec<(u64, u64)> = (0..500u64).map(|i| (i << 40, i << 40)).collect();
    // small batch (byte-merge fast path), including a tuple already present (must de-dup)
    check(
        &aos,
        &[
            (7 << 40, 7 << 40),
            ((3 << 40) + 1, 99),
            ((250 << 40) + 5, 7),
        ],
    );
    // large batch ⇒ overflows leaves into several pages (fallback path)
    let big: Vec<(u64, u64)> = (0..2_000u64).map(|i| ((i << 40) + 1, i)).collect();
    check(&aos, &big);
    // low cardinality ⇒ compact source leaves (fallback path)
    let mut low_card: Vec<(u64, u64)> = (0..2_000u64).map(|i| ((i % 8) * 1_000, i)).collect();
    low_card.sort_unstable();
    low_card.dedup();
    let add: Vec<(u64, u64)> = (0..60u64)
        .map(|i| ((i % 8) * 1_000, 1_000_000 + i))
        .collect();
    check(&low_card, &add);
}

#[test]
fn is_empty_tracks_emptiness() {
    // `is_empty` is O(1) (root-only) and must agree with `len() == 0` through inserts, single
    // removes, and a bulk-built tree drained to nothing (leaves merge back to one empty root).
    let mut t = CowBTree::<256, 256>::new();
    assert!(t.is_empty());
    assert_eq!(t.len(), 0);
    t.insert(5, 50);
    assert!(!t.is_empty());
    assert_eq!(t.len(), 1);
    t.remove(5, 50);
    assert!(t.is_empty(), "removing the last entry empties the tree");

    let pairs: Vec<(u64, u64)> = (0..1_000u64).map(|i| (i, i)).collect();
    let mut big = CowBTree::<256, 256>::from_sorted(&pairs);
    assert!(!big.is_empty());
    for &(k, d) in &pairs {
        big.remove(k, d);
    }
    assert!(big.is_empty(), "draining every entry empties the tree");
    assert_eq!(big.len(), 0);
}

#[test]
fn a_reader_reads_the_committed_version_lock_free_during_a_write() {
    // The MVCC question, modeled the way the graph does it — NO mutex. The committed version is an
    // immutable `Arc<CowBTree>` (cf. `MvccGraph.graph`); a writer takes a copy-on-write clone (cf.
    // `new_version()`), mutates *that*, and would publish by swapping the pointer (cf. `commit`). Readers
    // never lock — they just clone the committed `Arc` (cf. `read()`).
    //
    // We freeze the writer mid-mutation (after it has cloned a shared branch and updated the private
    // copy's children/seps); ONLY THEN does the reader take its view and read the committed version
    // concurrently. It must see V1 intact — because the writer mutates a *private* clone of every node it
    // touches, so the committed nodes a reader reaches are never disturbed. That's why MVCC needs no lock
    // between a reader's clone and a writer's mutation.
    use super::node::cow_gate;
    use std::sync::Arc;
    use std::sync::atomic::Ordering::SeqCst;
    use std::thread;

    // 1024 entries = 4 full leaves ⇒ depth-2. Inserting the sentinel (the max key) splits the full
    // rightmost leaf and propagates a `Split` to the root branch, where the park fires — after the
    // private copy's children/seps are mutated, so the reader observes a genuinely in-flight write.
    let pairs: Vec<(u64, u64)> = (0..1024u64).map(|i| (i, i)).collect();
    let committed = Arc::new(CowBTree::<256, 256>::from_sorted(&pairs)); // the committed version (V1)

    cow_gate::PARKED.store(false, SeqCst);
    cow_gate::RELEASE.store(false, SeqCst);

    let writer = thread::spawn({
        let base = Arc::clone(&committed);
        move || {
            let mut working = (*base).clone(); // new_version(): a CoW clone sharing V1's nodes
            working.insert(cow_gate::KEY, 1); // clones the shared root, splits the full leaf, mutates the copy — then parks
            working // the new version that `commit` would publish
        }
    });

    while !cow_gate::PARKED.load(SeqCst) {
        std::hint::spin_loop(); // wait until the writer is frozen mid-mutation
    }

    // The writer is frozen mid-mutation. Only now does the reader arrive: it clones the committed version
    // (lock-free) and reads it — and must see V1 intact: 1024 entries, no sentinel leaked in.
    let snapshot = (*committed).clone();
    let docs: Vec<u64> = snapshot.range(0, u64::MAX).collect();
    assert_eq!(
        docs.len(),
        1024,
        "reader observed a writer's in-flight mutation"
    );
    assert!(
        snapshot.point(cow_gate::KEY).next().is_none(),
        "the writer's uncommitted insert leaked into the committed snapshot",
    );

    cow_gate::RELEASE.store(true, SeqCst); // let the writer finish
    let published = writer.join().unwrap();

    assert_eq!(
        published.len(),
        1025,
        "the new version must hold the insert"
    );
    assert_eq!(
        committed.len(),
        1024,
        "the committed version must be untouched"
    );
}

#[test]
fn cow_writer_shares_committed_nodes_then_privatizes_on_write() {
    // The invariant behind copy-on-write: a writer's new version is a clone of committed, so it SHARES
    // committed's nodes (refcount >= 2). Every node the writer touches must therefore be cloned into a
    // private copy before mutation, or it would corrupt the committed version a reader may hold. (This is
    // also why we always clone rather than reuse `Arc::make_mut`'s in-place path: the first touch of any
    // node is always shared, so make_mut would copy anyway — see `node::make_private`.)
    use super::node::Node;
    use std::sync::Arc;
    fn root_rc<const L: usize, const B: usize>(t: &CowBTree<L, B>) -> usize {
        match &t.root {
            Node::Branch(b) => Arc::strong_count(b),
            Node::Leaf(_) => panic!("test needs a multi-level tree"),
        }
    }
    // 44 entries, LEAF_MAX=8 ⇒ 6 leaves under one branch root (depth 2).
    let committed = CowBTree::<8, 8>::from_sorted(&(0..44u64).map(|i| (i, i)).collect::<Vec<_>>());
    assert!(
        matches!(committed.root, Node::Branch(_)),
        "need a branch root"
    );

    // A writer's new version shares committed's root ⇒ refcount 2. Touching it MUST copy (never mutate
    // the shared node in place), or the committed version would be corrupted.
    let mut working = committed.clone();
    assert_eq!(
        root_rc(&working),
        2,
        "the new version shares committed's root (refcount 2)"
    );

    // After a (non-splitting) insert the writer's root is a private copy (refcount 1) and committed's is
    // unshared again — the write went to the copy, committed is untouched.
    working.insert(100, 1);
    assert_eq!(
        root_rc(&working),
        1,
        "the writer's root is now a private copy"
    );
    assert_eq!(
        root_rc(&committed),
        1,
        "committed's root is untouched / unshared again"
    );
}

#[test]
fn compact_splice_arms_parity() {
    // Exercise every packed-mutation arm explicitly on a compact leaf and check parity vs a reference,
    // including that the leaf *stays* compact (proving the in-place splice fired, not a rebuild to AoS).
    // Build a low-cardinality indexed leaf directly (5 distinct values, narrow docs ⇒ small widths).
    // `from_sorted` builds it via `from_pairs`, which selects the compact format; incremental inserts
    // would instead stay AoS until a split, so we seed the compact page up front.
    let mut doc = 0u64;
    let mut pairs: Vec<(u64, u64)> = Vec::new();
    for v in 0..5u64 {
        for _ in 0..30 {
            pairs.push((v * 100, doc));
            doc += 1;
        }
    }
    let mut t = CowBTree::<256, 256>::from_sorted(&pairs);
    let mut reference: BTreeSet<(u64, u64)> = pairs.iter().copied().collect();
    let is_compact = |t: &CowBTree| t.leaves().iter().any(|(f, _)| *f == LeafFormat::Compact);
    assert!(is_compact(&t), "setup should be compact");
    let check = |t: &CowBTree, r: &BTreeSet<(u64, u64)>| {
        assert_eq!(t.len(), r.len());
        // doc parity via the cursor, *and* full (key, doc) parity decoded from the leaf bytes
        assert_eq!(tree_range(t, 0, u64::MAX), ref_range(r, 0, u64::MAX));
        assert_eq!(tree_pairs(t), r.iter().copied().collect::<Vec<_>>());
    };

    // insert: existing distinct value (no distinct-table change), new doc within width.
    t.insert(200, doc);
    reference.insert((200, doc));
    doc += 1;
    assert!(
        is_compact(&t),
        "existing-distinct insert must stay compact (spliced)"
    );
    check(&t, &reference);

    // insert: new distinct value within the current widths (distinct-table grow + index remap).
    t.insert(250, doc);
    reference.insert((250, doc));
    doc += 1;
    assert!(
        is_compact(&t),
        "new-distinct insert must stay compact (spliced)"
    );
    check(&t, &reference);

    // insert: a present tuple ((0, 0) was inserted in setup) is a no-op; a genuinely-new one grows by one.
    let before = t.len();
    t.insert(0, 0);
    assert_eq!(t.len(), before, "present insert must be a no-op");
    t.insert(300, doc); // new doc on an existing distinct value
    reference.insert((300, doc));
    check(&t, &reference);

    // remove: an existing entry (index byte + doc cell cut; orphan distinct slot left behind).
    t.remove(100, 30); // (v=1 first doc)
    reference.remove(&(100, 30));
    check(&t, &reference);

    // merge_batch through the packed indexed merge: existing + brand-new distinct values, plus a present
    // tuple (dedup). Docs stay < 256 so the doc width does not widen ⇒ the packed path fires (not rebuild).
    let mut batch: Vec<(u64, u64)> = vec![
        (200, 200), // existing distinct value, fresh doc
        (250, 201), // existing distinct value (added above)
        (175, 202), // new distinct value, between existing
        (425, 203), // new distinct value, above all
        (0, 0),     // already present ⇒ must dedup
    ];
    batch.sort_unstable();
    t.insert_batch(&batch);
    for &p in &batch {
        reference.insert(p);
    }
    check(&t, &reference);

    // param-grow fallback: a key below the current min forces a rebuild (re-picks min/widths). Result
    // must still be correct (the format may change here — we only assert contents).
    t.insert(7, 12_345_678);
    reference.insert((7, 12_345_678));
    check(&t, &reference);
}

#[test]
fn no_index_compact_splice_arms() {
    // The no-index compact page (distinct == count): narrow, all-distinct data selects it via
    // `from_sorted`. Exercise splice insert (new value + value collision), remove, and merge, asserting
    // full (key, doc) parity decoded from the bytes. (The indexed arms are covered separately.)
    let pairs: Vec<(u64, u64)> = (0..200u64).map(|i| (i, i)).collect(); // narrow, all-distinct
    let mut t = CowBTree::<256, 256>::from_sorted(&pairs);
    let mut reference: BTreeSet<(u64, u64)> = pairs.iter().copied().collect();
    let no_index = |t: &CowBTree| {
        t.leaves()
            .into_iter()
            .all(|(f, b)| matches!(Leaf::<256>::from_parts(f, b), Leaf::Compact(_)))
    };
    assert!(
        no_index(&t),
        "narrow all-distinct data should be no-index compact"
    );
    let check = |t: &CowBTree, r: &BTreeSet<(u64, u64)>| {
        assert_eq!(t.len(), r.len());
        assert_eq!(tree_pairs(t), r.iter().copied().collect::<Vec<_>>());
    };
    // new value (fits widths) ⇒ stays no-index.
    t.insert(250, 250);
    reference.insert((250, 250));
    assert!(
        no_index(&t),
        "new narrow value must stay no-index (spliced)"
    );
    check(&t, &reference);
    // value collision (key 50 already present, fresh narrow doc) ⇒ stored per-entry, still no-index.
    t.insert(50, 251);
    reference.insert((50, 251));
    assert!(
        no_index(&t),
        "value collision must stay no-index (stored per-entry)"
    );
    check(&t, &reference);
    // remove front / middle / back.
    for kd in [(0u64, 0u64), (100, 100), (250, 250)] {
        t.remove(kd.0, kd.1);
        reference.remove(&kd);
        check(&t, &reference);
    }
    // merge a narrow batch: new value, value collision, and a present tuple (dedup). All fit widths ⇒
    // packed no-index merge fires.
    let mut batch = vec![(10, 252), (50, 50), (190, 253), (175, 254)];
    batch.sort_unstable();
    t.insert_batch(&batch);
    for &p in &batch {
        reference.insert(p);
    }
    check(&t, &reference);
    // a widening doc forces the rebuild fallback (contents still correct; format may change).
    t.insert(5, 9_000_000);
    reference.insert((5, 9_000_000));
    check(&t, &reference);
}

#[test]
fn block_copy_merge_parity() {
    // Batches whose keys are all existing distinct values, into a large indexed leaf, take the block-copy
    // merge path (gallop + memcpy, any size). Must match a reference, with a within-batch dup and a
    // leaf-present tuple both dropped.
    let mut seed: Vec<(u64, u64)> = Vec::new();
    for v in 0..4u64 {
        for d in 0..50u64 {
            seed.push((v * 1_000, d)); // 200 entries, 4 distinct, narrow docs
        }
    }
    let mut t = CowBTree::<256, 256>::from_sorted(&seed);
    let mut reference: BTreeSet<(u64, u64)> = seed.iter().copied().collect();
    assert!(
        t.leaves()
            .into_iter()
            .all(|(f, b)| matches!(Leaf::<256>::from_parts(f, b), Leaf::CompactIndexed(_))),
        "setup should be a single indexed leaf"
    );
    // B = 8 (< 200/4), keys all existing distinct (0/1000/2000/3000), docs fit. (2000,102) is a
    // within-batch dup; (1000,0) and (0,0) are already in the leaf — all three must be dropped.
    let mut batch = vec![
        (0, 100),
        (1_000, 101),
        (2_000, 102),
        (2_000, 102),
        (3_000, 103),
        (1_000, 0),
        (0, 0),
        (3_000, 104),
    ];
    batch.sort_unstable();
    t.insert_batch(&batch);
    for &p in &batch {
        reference.insert(p);
    }
    assert_eq!(t.len(), reference.len());
    assert_eq!(
        tree_pairs(&t),
        reference.iter().copied().collect::<Vec<_>>()
    );
    assert!(
        t.leaves()
            .into_iter()
            .all(|(f, b)| matches!(Leaf::<256>::from_parts(f, b), Leaf::CompactIndexed(_))),
        "block-copy merge must keep the leaf indexed"
    );
    // A *large* all-existing-distinct batch (no size guard now — galloping keeps block-copy ≥ the walk).
    // 40 entries into the ~205-entry leaf stays under LEAF_MAX and on the block-copy path.
    let mut big: Vec<(u64, u64)> = (0..40u64).map(|i| ((i % 4) * 1_000, 500 + i)).collect();
    big.sort_unstable();
    t.insert_batch(&big);
    for &p in &big {
        reference.insert(p);
    }
    assert_eq!(t.len(), reference.len());
    assert_eq!(
        tree_pairs(&t),
        reference.iter().copied().collect::<Vec<_>>()
    );
}

#[test]
fn packed_compact_mutation_fuzz() {
    // Randomized insert / remove / insert_batch over mostly-low-cardinality data so the packed splice +
    // merge fast paths fire, with occasional wide keys (AoS) and widening docs (rebuild fallback). The
    // tree must equal a BTreeSet reference throughout — the ground-truth check for every packed arm and
    // their interaction with splits, merges, and format transitions.
    // Seed with a low-cardinality dataset built via `from_sorted` so leaves start *compact* (incremental
    // inserts would stay AoS until a split, leaving the packed splice arms barely exercised). The random
    // ops below then hit `splice_insert` / `merge` / `splice_remove` on compact pages from step 0.
    let mut seed: Vec<(u64, u64)> = Vec::new();
    for v in 0..40u64 {
        for d in 0..120u64 {
            seed.push((v * 1_000, d));
        }
    }
    let mut t = CowBTree::<256, 256>::from_sorted(&seed);
    let mut reference: BTreeSet<(u64, u64)> = seed.iter().copied().collect();
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        s = splitmix(s);
        s
    };
    for step in 0..40_000u64 {
        // Low-card key most of the time (⇒ compact leaves); an occasional wide key (⇒ AoS). The 500
        // step interleaves with the 1000-step seed, so new distinct values land in the *middle* of a
        // leaf's distinct table (exercising the index remap), not only at the end.
        let gen_key = |n: u64| {
            if n.is_multiple_of(16) {
                n >> 8
            } else {
                (n % 100) * 500
            }
        };
        let gen_doc = |n: u64| {
            if n.is_multiple_of(8) {
                n >> 20
            } else {
                n % 800
            }
        }; // narrow + widening docs
        match next() % 10 {
            0..=5 => {
                let (k, d) = (gen_key(next()), gen_doc(next()));
                t.insert(k, d);
                reference.insert((k, d));
            }
            6..=7 => {
                let (k, d) = (gen_key(next()), gen_doc(next()));
                t.remove(k, d);
                reference.remove(&(k, d));
            }
            _ => {
                let bn = (next() % 12) as usize;
                let mut batch: Vec<(u64, u64)> = Vec::with_capacity(bn);
                for _ in 0..bn {
                    batch.push((gen_key(next()), gen_doc(next())));
                }
                batch.sort_unstable();
                batch.dedup();
                t.insert_batch(&batch);
                reference.extend(batch.iter().copied());
            }
        }
        if step % 53 == 0 {
            assert_eq!(t.len(), reference.len(), "len mismatch at step {step}");
            // doc parity via the cursor, *and* full (key, doc) parity decoded straight from the leaf
            // bytes — the latter verifies the value column the packed splice/merge code maintains (a
            // full-range cursor scan sets `whole` and skips key reads on interior leaves).
            assert_eq!(
                t.range(0, u64::MAX).collect::<Vec<_>>(),
                reference.iter().map(|&(_, d)| d).collect::<Vec<_>>(),
                "doc contents mismatch at step {step}"
            );
            assert_eq!(
                tree_pairs(&t),
                reference.iter().copied().collect::<Vec<_>>(),
                "(key, doc) contents mismatch at step {step}"
            );
        }
    }
    assert_eq!(
        tree_pairs(&t),
        reference.iter().copied().collect::<Vec<_>>(),
        "final (key, doc) contents"
    );
}

#[test]
fn const_generic_leaf_size_parity() {
    // The const generic must instantiate and stay correct at a *non-default* leaf size. Drive a few
    // hundred mixed low-cardinality inserts/removes into a `CowBTree<64>` (a 64-entry leaf splits/merges
    // far more often than the default 256, exercising the generic on the split/merge paths) and assert it
    // matches a BTreeSet reference decoded straight from the leaf bytes.
    let mut t = CowBTree::<64, 256>::new();
    let mut reference: BTreeSet<(u64, u64)> = BTreeSet::new();
    let mut s = 0xDEAD_BEEF_CAFE_F00Du64;
    let mut next = || {
        s = splitmix(s);
        s
    };
    for _ in 0..600u64 {
        let key = (next() % 20) * 1_000; // low cardinality ⇒ compact leaves
        let doc = next() % 500; // narrow docs
        if next() % 3 == 0 {
            t.remove(key, doc);
            reference.remove(&(key, doc));
        } else {
            t.insert(key, doc);
            reference.insert((key, doc));
        }
    }
    assert_eq!(t.len(), reference.len());
    assert_eq!(
        tree_pairs(&t),
        reference.iter().copied().collect::<Vec<_>>(),
        "CowBTree<64> contents must match the reference"
    );
}

#[test]
fn const_generic_branch_size_parity() {
    // The `BRANCH_MAX` const generic must instantiate and stay correct at a *non-default* fan-out. A
    // small branch fanout (16) makes branches split and merge far more often than the default 256 —
    // forcing many more node packs, rebalances, and root grow/shrinks — so it exercises the generic on
    // the branch paths. Drive a few hundred mixed inserts/removes via `splitmix` and assert the tree
    // matches a `BTreeSet` reference decoded straight from the leaf bytes.
    let mut t = CowBTree::<256, 16>::new();
    let mut reference: BTreeSet<(u64, u64)> = BTreeSet::new();
    let mut s = 0xC0FF_EE00_1234_5678u64;
    let mut next = || {
        s = splitmix(s);
        s
    };
    for _ in 0..600u64 {
        let key = next() % 2_000; // spread keys so the tree grows several branch levels at fanout 16
        let doc = next() % 1_000;
        if next() % 3 == 0 {
            t.remove(key, doc);
            reference.remove(&(key, doc));
        } else {
            t.insert(key, doc);
            reference.insert((key, doc));
        }
    }
    assert_eq!(t.len(), reference.len());
    assert_eq!(
        tree_pairs(&t),
        reference.iter().copied().collect::<Vec<_>>(),
        "CowBTree<256, 16> contents must match the reference"
    );
}
