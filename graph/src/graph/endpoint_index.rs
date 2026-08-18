//! `edge_id -> (src, dst)`, in the narrowest field type the endpoints need.
//!
//! Every materialised relationship reads this, so it has to be O(1); it holds
//! one slot per edge id ever reserved, so its width is multiplied by the edge
//! count and is worth minimising.
//!
//! It used to be a `Vec<u64>` holding `(src << 32) | dst`. That fixed 8 bytes
//! per edge whatever the graph's size, and imposed a hard ceiling of 2^32 per
//! endpoint — a limit checked on every write that no graph could grow past.
//!
//! Both go away by choosing the field *type* from the endpoints instead of
//! fixing it:
//!
//! | nodes up to | field | bytes/edge | vs the old 8 |
//! |---|---|---|---|
//! | 65 534         | `u16` | 4  | −50% |
//! | 16.7 M         | `U24` | 6  | −25% |
//! | 4.29 B         | `u32` | 8  |   0% |
//! | 18.4 E         | `u64` | 16 | *(no previous form)* |
//!
//! The last row has no comparison to make: the old packing could not represent
//! those graphs at all, so its 16 bytes are not a doubling of anything. They buy
//! a range that previously ended in a panic.
//!
//! The interesting row is the second: graphs in the millions of nodes — the
//! common case — pay 6 bytes where they used to pay 8, and graphs past 4.29
//! billion nodes now work at all, which is what the old packing refused.
//!
//! Each width is a distinct `Vec` of a concrete pair type rather than one
//! buffer with a runtime width. That keeps this free of byte arithmetic: a read
//! is an ordinary indexed load the compiler bounds-checks, and the empty slot is
//! a plain constant of the field type rather than a mask re-derived whenever the
//! width changes.
//!
//! # Why all four tiers are held at once
//!
//! Holding one tier and converting on overflow would be simpler, and it throws
//! away most of the saving. Edge ids are allocated in increasing order and the
//! width only ever grows, so the tiers partition the id space into contiguous
//! ranges: everything written while node ids were under 65,535 is in `w16`,
//! everything after in `w24`, and so on. A single-tier index has to repack all
//! of that to the widest width the graph ever reaches, and then keep paying it.
//!
//! Whether that matters depends entirely on *when* edges are written relative to
//! node growth. Measured on this engine: a bulk load — all nodes, then all edges
//! — writes every edge after the ids reach their final width, so tiering saves
//! nothing. A graph grown incrementally across the 65,535 boundary wrote **30%**
//! of its edges while ids were still narrow, and those keep 4 bytes per edge
//! instead of 6.
//!
//! It also removes the repack. A single-tier index rewrites every slot when it
//! widens; here a widening promotes only the tiers *below* the new width, and
//! since a slot can be promoted at most three times in the life of a graph the
//! total work is `O(3|E|)` spread out rather than three whole-index pauses.

use std::sync::Arc;

/// A field type an endpoint can be stored in.
pub trait Endpoint: Copy + PartialEq + Sized {
    /// The value marking an empty or deleted slot.
    const EMPTY: Self;
    /// One past the largest id this type can hold, [`Self::EMPTY`] excluded.
    const CEILING: u64;
    fn get(self) -> u64;
    fn put(v: u64) -> Self;

    /// Whether this field is the empty marker. A slot is empty when *both* of
    /// its fields are, so this is per field rather than per slot.
    fn vacant(self) -> bool {
        self == Self::EMPTY
    }
}

/// Three bytes, little-endian. There is no primitive for it, and it is the
/// width that matters most: it covers 16.7 M nodes, which is where a great many
/// graphs sit, and it is the one a `u16`/`u32`-only ladder would skip straight
/// past — costing those graphs 8 bytes per edge instead of 6.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct U24([u8; 3]);

impl Endpoint for u16 {
    const EMPTY: Self = Self::MAX;
    const CEILING: u64 = Self::MAX as u64;
    fn get(self) -> u64 {
        u64::from(self)
    }
    fn put(v: u64) -> Self {
        v as Self
    }
}

impl Endpoint for U24 {
    const EMPTY: Self = Self([0xFF; 3]);
    const CEILING: u64 = 0x00FF_FFFF;
    fn get(self) -> u64 {
        u64::from(u32::from_le_bytes([self.0[0], self.0[1], self.0[2], 0]))
    }
    fn put(v: u64) -> Self {
        let b = v.to_le_bytes();
        Self([b[0], b[1], b[2]])
    }
}

impl Endpoint for u32 {
    const EMPTY: Self = Self::MAX;
    const CEILING: u64 = Self::MAX as u64;
    fn get(self) -> u64 {
        u64::from(self)
    }
    fn put(v: u64) -> Self {
        v as Self
    }
}

impl Endpoint for u64 {
    const EMPTY: Self = Self::MAX;
    const CEILING: u64 = Self::MAX;
    fn get(self) -> u64 {
        self
    }
    fn put(v: u64) -> Self {
        v
    }
}

/// One slot per edge id, in a fixed field type.
type Slots<T> = Vec<(T, T)>;

/// A tier, shared between MVCC versions until one of them writes to it.
///
/// The sharing is per tier rather than per index, and that is the point. A write
/// transaction takes a new version of the whole graph, and the first edge it
/// mutates reaches `Arc::make_mut` with the committed snapshot still holding a
/// reference — a deep clone. Copying the *whole* index there costs about 0.9
/// instructions per edge in the graph, so a one-edge transaction against 10 M
/// edges paid 9.15 M instructions before touching anything.
///
/// Almost every write appends, and appends land in the widest tier; the narrow
/// tiers are history and are not touched. Holding each behind its own `Arc`
/// means a version clones only the tier it writes to, and the history stays
/// shared.
type Tier<T> = Option<Arc<Slots<T>>>;

/// `edge_id -> (src, dst)`, split into contiguous id ranges by field width.
///
/// The tiers are ordered: `w16` holds the lowest edge ids, then `w24`, `w32`,
/// `w64`. A tier is `None` until the graph needs it. See the module docs for why
/// they are held simultaneously rather than converted.
#[derive(Clone, Default)]
pub struct EndpointIndex {
    w16: Tier<u16>,
    w24: Tier<U24>,
    w32: Tier<u32>,
    w64: Tier<u64>,
}

/// Move every slot of `from` to the front of `into`, re-emitting empty slots as
/// the wider type's own empty value rather than copying a pattern that means
/// something else at the new width.
fn promote<A: Endpoint, B: Endpoint>(
    from: &mut Tier<A>,
    into: &mut Slots<B>,
) {
    let Some(old) = from.take() else { return };
    // Reuse the buffer when this version owns it outright; a version that still
    // shares it with a snapshot has to copy.
    let old = Arc::try_unwrap(old).unwrap_or_else(|shared| (*shared).clone());
    if old.is_empty() {
        return;
    }
    let mut merged: Slots<B> = Vec::with_capacity(old.len() + into.len());
    merged.extend(old.into_iter().map(|(s, d)| {
        if s.vacant() && d.vacant() {
            (B::EMPTY, B::EMPTY)
        } else {
            (B::put(s.get()), B::put(d.get()))
        }
    }));
    merged.append(into);
    *into = merged;
}

/// Run `$body` for each tier in id order, with `$v` bound to `&Option<Slots<_>>`.
macro_rules! each_tier {
    ($self:expr, |$v:ident| $body:block) => {{
        {
            let $v = &$self.w16;
            $body
        }
        {
            let $v = &$self.w24;
            $body
        }
        {
            let $v = &$self.w32;
            $body
        }
        {
            let $v = &$self.w64;
            $body
        }
    }};
}

impl EndpointIndex {
    /// Which tier a value of this size needs: 0 = `w16` … 3 = `w64`.
    const fn rank_for(v: u64) -> u8 {
        if v < u16::CEILING {
            0
        } else if v < U24::CEILING {
            1
        } else if v < u32::CEILING {
            2
        } else {
            3
        }
    }

    /// The tier new edge ids append to: the widest one that already holds
    /// anything, since a narrower range can never follow a wider one.
    const fn append_rank(&self) -> u8 {
        if self.w64.is_some() {
            3
        } else if self.w32.is_some() {
            2
        } else if self.w24.is_some() {
            1
        } else {
            0
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        let mut n = 0;
        each_tier!(self, |v| {
            n += v.as_ref().map_or(0, |v| v.len());
        });
        n
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes held, for `GRAPH.MEMORY`.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.w16.as_ref().map_or(0, |v| v.capacity() * 4)
            + self.w24.as_ref().map_or(0, |v| v.capacity() * 6)
            + self.w32.as_ref().map_or(0, |v| v.capacity() * 8)
            + self.w64.as_ref().map_or(0, |v| v.capacity() * 16)
    }

    /// Mean bytes per stored slot, which is the quantity the tiering is for.
    #[must_use]
    pub fn bytes_per_edge(&self) -> f64 {
        let n = self.len();
        if n == 0 {
            return 0.0;
        }
        (self.w16.as_ref().map_or(0, |v| v.len() * 4)
            + self.w24.as_ref().map_or(0, |v| v.len() * 6)
            + self.w32.as_ref().map_or(0, |v| v.len() * 8)
            + self.w64.as_ref().map_or(0, |v| v.len() * 16)) as f64
            / n as f64
    }

    /// The endpoints of `edge_id`, or `None` if the slot is empty, deleted, or
    /// past the end.
    #[must_use]
    pub fn get(
        &self,
        edge_id: u64,
    ) -> Option<(u64, u64)> {
        let mut slot = usize::try_from(edge_id).ok()?;
        each_tier!(self, |v| {
            if let Some(v) = v {
                if slot < v.len() {
                    let (s, d) = v[slot];
                    return if s.vacant() && d.vacant() {
                        None
                    } else {
                        Some((s.get(), d.get()))
                    };
                }
                slot -= v.len();
            }
        });
        None
    }

    /// Promote every tier below `rank` into it, so `rank` holds a contiguous
    /// range starting at edge id 0.
    ///
    /// Only the reused-id path may call this. Appending must *not*: promoting on
    /// append would rewrite every earlier edge to the new width and throw away
    /// the whole point of keeping the tiers apart.
    fn raise_to(
        &mut self,
        rank: u8,
    ) {
        match rank {
            1 => promote(
                &mut self.w16,
                Arc::make_mut(self.w24.get_or_insert_with(Arc::default)),
            ),
            2 => {
                let into = Arc::make_mut(self.w32.get_or_insert_with(Arc::default));
                promote(&mut self.w24, into);
                promote(&mut self.w16, into);
            }
            3 => {
                let into = Arc::make_mut(self.w64.get_or_insert_with(Arc::default));
                promote(&mut self.w32, into);
                promote(&mut self.w24, into);
                promote(&mut self.w16, into);
            }
            _ => {
                self.w16.get_or_insert_with(Arc::default);
            }
        }
    }

    /// Grow the append tier so slot `needed - 1` exists, and make sure that tier
    /// is wide enough for `max_endpoint`.
    ///
    /// Sizing for a whole batch here is what keeps bulk insertion linear: the
    /// index reserves exactly rather than doubling, so growing per edge is a
    /// reallocation per edge — measured at **87,370 instructions per edge
    /// created against 8,927** before this existed, and a bulk-load test timing
    /// out.
    pub fn prepare(
        &mut self,
        max_edge_id: u64,
        max_endpoint: u64,
    ) {
        let rank = Self::rank_for(max_endpoint).max(self.append_rank());
        let needed = usize::try_from(max_edge_id).expect("edge id exceeds usize") + 1;
        let below = self.len() - self.tier_len(rank);
        let want = needed.saturating_sub(below);
        self.with_tier(rank, |v| {
            if want > v.len() {
                v.reserve_exact(want - v.len());
            }
        });
        self.resize_tier(rank, want);
    }

    fn tier_len(
        &self,
        rank: u8,
    ) -> usize {
        match rank {
            0 => self.w16.as_ref().map_or(0, |v| v.len()),
            1 => self.w24.as_ref().map_or(0, |v| v.len()),
            2 => self.w32.as_ref().map_or(0, |v| v.len()),
            _ => self.w64.as_ref().map_or(0, |v| v.len()),
        }
    }

    fn with_tier<F>(
        &mut self,
        rank: u8,
        f: F,
    ) where
        F: FnOnce(&mut dyn TierOps),
    {
        // `make_mut` here rather than on the index as a whole: this copies one
        // tier, and only if a snapshot still shares it.
        // Bound before the call so `make_mut` resolves against the concrete
        // vector; coercing to `dyn TierOps` first would resolve it against the
        // trait object, which is not `Clone`.
        match rank {
            0 => {
                let v = Arc::make_mut(self.w16.get_or_insert_with(Arc::default));
                f(v);
            }
            1 => {
                let v = Arc::make_mut(self.w24.get_or_insert_with(Arc::default));
                f(v);
            }
            2 => {
                let v = Arc::make_mut(self.w32.get_or_insert_with(Arc::default));
                f(v);
            }
            _ => {
                let v = Arc::make_mut(self.w64.get_or_insert_with(Arc::default));
                f(v);
            }
        }
    }

    fn resize_tier(
        &mut self,
        rank: u8,
        len: usize,
    ) {
        self.with_tier(rank, |v| v.grow_to(len));
    }

    /// Record `edge_id`'s endpoints, promoting and growing as needed.
    pub fn set(
        &mut self,
        edge_id: u64,
        src: u64,
        dst: u64,
    ) {
        let slot = usize::try_from(edge_id).expect("edge id exceeds usize");
        // An id inside an existing tier is an update in place. If it no longer
        // fits that tier — which happens when a deleted edge id is reused for an
        // edge with wider endpoints — every tier up to the needed width is
        // promoted, which keeps the ranges contiguous.
        let needed = Self::rank_for(src.max(dst));
        let mut base = 0usize;
        for rank in 0..4u8 {
            let n = self.tier_len(rank);
            if slot < base + n {
                if needed > rank {
                    self.raise_to(needed);
                    // Promotion moved this slot into `needed`, whose range now
                    // starts at 0, so the offset is the raw id again.
                    self.with_tier(needed, |v| v.put(slot, src, dst));
                } else {
                    self.with_tier(rank, |v| v.put(slot - base, src, dst));
                }
                return;
            }
            base += n;
        }
        // Past the end: append to the widest active tier, or a wider one if this
        // edge needs it.
        let rank = needed.max(self.append_rank());
        let below = self.len() - self.tier_len(rank);
        let at = slot - below;
        self.with_tier(rank, |v| {
            v.grow_to(at + 1);
            v.put(at, src, dst);
        });
    }

    /// Tombstone `edge_id`'s slot. Vectors never shrink, as before.
    pub fn clear(
        &mut self,
        edge_id: u64,
    ) {
        let Ok(slot) = usize::try_from(edge_id) else {
            return;
        };
        let mut base = 0usize;
        for rank in 0..4u8 {
            let n = self.tier_len(rank);
            if slot < base + n {
                self.with_tier(rank, |v| v.vacate(slot - base));
                return;
            }
            base += n;
        }
    }

    /// Build from `(edge_id, src, dst)` triples, sizing one tier exactly. Used
    /// by the load path, which knows the whole set and writes it at one width.
    #[must_use]
    pub fn build(triples: &[(u64, u64, u64)]) -> Self {
        let max_endpoint = triples.iter().map(|&(_, s, d)| s.max(d)).max().unwrap_or(0);
        let slots = triples.iter().map(|&(e, _, _)| e).max().map_or(0, |m| {
            usize::try_from(m).expect("edge id exceeds usize") + 1
        });
        let mut ix = Self::default();
        let rank = Self::rank_for(max_endpoint);
        ix.with_tier(rank, |v| {
            v.reserve_exact_to(slots);
            v.grow_to(slots);
            for &(e, s, d) in triples {
                v.put(e as usize, s, d);
            }
        });
        ix
    }
}

/// The operations a tier needs, so the four concrete vectors can be reached
/// through one `dyn` without repeating the body four times. Only used on write
/// paths, which are not per-row; [`EndpointIndex::get`] indexes the vectors
/// directly.
trait TierOps {
    fn grow_to(
        &mut self,
        len: usize,
    );
    fn reserve_exact_to(
        &mut self,
        len: usize,
    );
    fn put(
        &mut self,
        at: usize,
        src: u64,
        dst: u64,
    );
    fn vacate(
        &mut self,
        at: usize,
    );
    fn len(&self) -> usize;
    fn reserve_exact(
        &mut self,
        extra: usize,
    );
}

impl<T: Endpoint> TierOps for Slots<T> {
    fn grow_to(
        &mut self,
        len: usize,
    ) {
        if len > self.len() {
            self.resize(len, (T::EMPTY, T::EMPTY));
        }
    }
    fn reserve_exact_to(
        &mut self,
        len: usize,
    ) {
        if len > self.len() {
            Vec::reserve_exact(self, len - self.len());
        }
    }
    fn put(
        &mut self,
        at: usize,
        src: u64,
        dst: u64,
    ) {
        self[at] = (T::put(src), T::put(dst));
    }
    fn vacate(
        &mut self,
        at: usize,
    ) {
        if let Some(s) = self.get_mut(at) {
            *s = (T::EMPTY, T::EMPTY);
        }
    }
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn reserve_exact(
        &mut self,
        extra: usize,
    ) {
        Vec::reserve_exact(self, extra);
    }
}

#[cfg(test)]
mod tests {
    use super::{Endpoint, EndpointIndex, U24};

    /// Each width has to keep its all-ones value spare, or a legitimate pair
    /// reads back as a deleted slot. The boundaries are where an off-by-one
    /// lands.
    #[test]
    fn the_widest_id_of_each_tier_is_not_an_empty_slot() {
        for (id, want) in [
            (0u64, 4.0),
            (u64::from(u16::MAX) - 1, 4.0),
            (u64::from(u16::MAX), 6.0),
            (U24::CEILING - 1, 6.0),
            (U24::CEILING, 8.0),
            (u64::from(u32::MAX) - 1, 8.0),
            (u64::from(u32::MAX), 16.0),
        ] {
            let mut ix = EndpointIndex::default();
            ix.set(0, id, id);
            assert!(
                (ix.bytes_per_edge() - want).abs() < f64::EPSILON,
                "tier for id {id}: {} != {want}",
                ix.bytes_per_edge()
            );
            assert_eq!(ix.get(0), Some((id, id)), "id {id} read back as empty");
        }
    }

    /// **The property the tiering exists for.** Edges written while ids were
    /// narrow must *stay* narrow when later edges force a wider tier — that is
    /// the whole difference from converting the index on overflow.
    #[test]
    fn early_edges_keep_their_narrow_tier() {
        let mut ix = EndpointIndex::default();
        for e in 0..1000u64 {
            ix.set(e, e, e + 1);
        }
        assert!(
            (ix.bytes_per_edge() - 4.0).abs() < f64::EPSILON,
            "small ids should be in the 4-byte tier"
        );

        // Now the graph outgrows u16. The early thousand must not be repacked.
        for e in 1000..2000u64 {
            ix.set(e, 100_000 + e, 100_001 + e);
        }
        let mean = ix.bytes_per_edge();
        assert!(
            (mean - 5.0).abs() < 0.001,
            "half at 4 bytes and half at 6 should average 5, got {mean}"
        );
        assert_eq!(ix.get(0), Some((0, 1)), "an early edge was lost");
        assert_eq!(ix.get(999), Some((999, 1000)));
        assert_eq!(ix.get(1000), Some((101_000, 101_001)));
        assert_eq!(ix.get(1999), Some((101_999, 102_000)));
    }

    /// A reused edge id whose new endpoints no longer fit its tier is the case
    /// that forces a promotion of everything below. Contents and tombstones
    /// have to survive it.
    #[test]
    fn reusing_a_low_id_with_wide_endpoints_promotes_and_preserves() {
        let mut ix = EndpointIndex::default();
        for e in 0..100u64 {
            ix.set(e, e, e + 1);
        }
        ix.clear(50);
        // id 7 is reused for an edge needing three bytes
        ix.set(7, 1 << 20, (1 << 20) + 3);

        assert_eq!(ix.get(7), Some((1 << 20, (1 << 20) + 3)));
        assert_eq!(ix.get(50), None, "tombstone lost across a promotion");
        assert_eq!(ix.get(0), Some((0, 1)));
        assert_eq!(ix.get(99), Some((99, 100)));
        assert_eq!(ix.len(), 100, "promotion changed the slot count");
    }

    /// Endpoints and tombstones must survive every promotion in turn.
    #[test]
    fn promotion_preserves_pairs_and_tombstones_at_every_tier() {
        let mut ix = EndpointIndex::default();
        ix.set(0, 1, 2);
        ix.set(1, 3, 4);
        ix.clear(1);
        ix.set(2, 300, 400);

        for (id, src, dst) in [
            (3u64, 70_000u64, 4u64),
            (4, 1 << 25, 5),
            (5, 1 << 33, (1 << 34) + 7),
        ] {
            ix.set(id, src, dst);
            assert_eq!(ix.get(0), Some((1, 2)), "pair lost at ({src}, {dst})");
            assert_eq!(ix.get(1), None, "tombstone lost at ({src}, {dst})");
            assert_eq!(ix.get(2), Some((300, 400)));
            assert_eq!(ix.get(id), Some((src, dst)));
        }
    }

    /// Gaps left by non-contiguous edge ids read as absent, not as `(0, 0)`.
    #[test]
    fn gaps_read_as_absent() {
        let mut ix = EndpointIndex::default();
        ix.set(5, 7, 8);
        assert_eq!(ix.get(0), None);
        assert_eq!(ix.get(4), None);
        assert_eq!(ix.get(5), Some((7, 8)));
        assert_eq!(ix.get(6), None, "past the end");
    }

    /// `build` and `prepare` + `set` must agree, since the load path uses one
    /// and every other path the other.
    #[test]
    fn build_agrees_with_prepared_sets() {
        let triples = [(0u64, 1u64, 2u64), (3, 70_000, 4), (7, 5, 6)];
        let mut a = EndpointIndex::default();
        a.prepare(7, 70_000);
        for &(e, s, d) in &triples {
            a.set(e, s, d);
        }
        let b = EndpointIndex::build(&triples);
        assert_eq!(a.len(), b.len());
        for e in 0..a.len() as u64 {
            assert_eq!(a.get(e), b.get(e), "slot {e}");
        }
    }
}

#[cfg(test)]
mod cow_bench {
    use super::EndpointIndex;
    use crate::graph::graphblas::instr::read_instr;

    /// What an MVCC version costs when the writer touches one edge.
    ///
    /// `Graph` holds this behind an `Arc`, and the committed snapshot keeps a
    /// second reference, so the first edge mutation of a version reaches
    /// `Arc::make_mut` with a refcount above one — a deep clone. The question
    /// this answers is how much of the index that clone has to copy: all of it,
    /// or only the tier the write lands in.
    ///
    /// Run with:
    ///   cargo test --release -p graph cow_cost -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cow_cost_of_a_version() {
        // A graph grown across the u16 boundary: most edges are history in the
        // narrow tier, and a write appends to the wide one.
        for edges in [100_000u64, 1_000_000, 10_000_000] {
            let mut ix = EndpointIndex::default();
            let narrow = edges * 9 / 10;
            for e in 0..narrow {
                ix.set(e, e % 60_000, (e + 1) % 60_000);
            }
            for e in narrow..edges {
                ix.set(e, 100_000 + e % 1000, 100_001 + e % 1000);
            }

            // What a write transaction actually pays: take a version (the
            // clone), then write one edge — which is where a copy-on-write copy
            // happens, and the whole question is how much of the index it
            // copies. The original stays alive throughout, so the refcount is
            // above one exactly as a committed snapshot makes it.
            let reps = 20u64;
            let i0 = read_instr();
            for e in 0..reps {
                let mut version = ix.clone();
                version.set(edges + e, 1, 2);
                std::hint::black_box(&version);
            }
            let i1 = read_instr();
            let per_version = match (i0, i1) {
                (Some(a), Some(b)) => (b - a) as f64 / reps as f64,
                _ => f64::NAN,
            };
            println!(
                "{edges:>10} edges  mean {:.2} B/edge  version + 1 write {:>12.0} instr",
                ix.bytes_per_edge(),
                per_version
            );
        }
    }
}
