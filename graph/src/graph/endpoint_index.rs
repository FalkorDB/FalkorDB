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
/// The largest a page grows to. A write copies at most one page, so this bounds
/// the copy-on-write cost: 4096 slots is 16 KB of `(u16, u16)` and 64 KB of
/// `(u64, u64)`, against 16 bytes of page pointer per page.
const PAGE_SLOTS: usize = 4096;

/// A tier's slots, in pages that are shared individually.
///
/// One flat `Vec` behind the tier's `Arc` was the whole cost this type exists to
/// avoid: see [`Tier`]. Sharing the pages rather than the vector means the write
/// that follows a snapshot copies the page it lands in instead of every edge in
/// the tier, and appends touch only the last one.
///
/// Every page but the last holds exactly `PAGE_SLOTS` slots, which keeps
/// addressing to a shift and a mask. **The last page is only as long as the tier
/// needs**, doubling as it fills. That matters more than it sounds: a fixed
/// `PAGE_SLOTS` tail cost every graph holding a single edge a whole page, which
/// measured at **+20.4 KB per graph** against a flat `Vec` — 27% of the
/// per-graph footprint, on a server that may hold thousands of small graphs, in
/// exchange for a win that only appears on large ones. Sizing the tail removes
/// that floor and keeps the bound, because the copy is still one page.
type Page<T> = Arc<[(T, T)]>;

struct Paged<T> {
    pages: Vec<Page<T>>,
    len: usize,
}

impl<T> Clone for Paged<T> {
    /// Clones the page *pointers*, not the slots — the point of the type.
    fn clone(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            len: self.len,
        }
    }
}

impl<T> Default for Paged<T> {
    fn default() -> Self {
        Self {
            pages: Vec::new(),
            len: 0,
        }
    }
}

/// A page of `slots` empty slots.
///
/// `Arc<[_]>` rather than `Arc<Vec<_>>` so the slots live *inline* in the `Arc`
/// allocation: a `Vec` behind an `Arc` costs the read path a second dependent
/// load to reach them, and a bounds check against a length it also has to load.
fn empty_page<T: Endpoint>(slots: usize) -> Page<T> {
    Arc::from(vec![(T::EMPTY, T::EMPTY); slots])
}

impl<T: Endpoint> Paged<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The slot at `at`, which must be within `len`.
    fn at(
        &self,
        at: usize,
    ) -> (T, T) {
        self.pages[at / PAGE_SLOTS][at % PAGE_SLOTS]
    }

    fn iter(&self) -> impl Iterator<Item = (T, T)> + '_ {
        (0..self.len).map(|i| self.at(i))
    }

    /// Bytes actually allocated, tail page included at its real length.
    fn allocated_bytes(&self) -> usize {
        self.pages.iter().map(|p| p.len()).sum::<usize>() * size_of::<(T, T)>()
    }

    /// Bytes the *used* slots occupy, which is what a per-edge figure wants.
    /// Taken from the layout the compiler chose rather than a literal that has
    /// to be kept in step with it: `(U24, U24)` is 6 bytes only because `U24` is
    /// `[u8; 3]` and so alignment 1, and a field type that gained alignment
    /// would otherwise make `GRAPH.MEMORY` drift silently.
    fn used_bytes(&self) -> usize {
        self.len * size_of::<(T, T)>()
    }

    /// Replace page `p` with a longer one, empty past what it already held.
    fn regrow(
        &mut self,
        p: usize,
        slots: usize,
    ) {
        let mut next = vec![(T::EMPTY, T::EMPTY); slots];
        let held = self.pages[p].len().min(slots);
        next[..held].copy_from_slice(&self.pages[p][..held]);
        self.pages[p] = Arc::from(next);
    }

    /// Make the pages address at least `slots` positions.
    ///
    /// Only the last page may be short, so a page about to stop being last is
    /// filled out to `PAGE_SLOTS` first. The tail then doubles rather than
    /// growing by the shortfall, which keeps appending amortised.
    fn ensure_pages(
        &mut self,
        slots: usize,
    ) {
        if slots == 0 {
            return;
        }
        let want = slots.div_ceil(PAGE_SLOTS);
        let tail = slots - (want - 1) * PAGE_SLOTS;

        if self.pages.len() < want {
            if let Some(last) = self.pages.len().checked_sub(1)
                && self.pages[last].len() < PAGE_SLOTS
            {
                self.regrow(last, PAGE_SLOTS);
            }
            while self.pages.len() + 1 < want {
                self.pages.push(empty_page(PAGE_SLOTS));
            }
            self.pages.push(empty_page(tail));
            return;
        }

        let last = want - 1;
        if self.pages[last].len() < tail {
            let grown = tail.max(self.pages[last].len() * 2).min(PAGE_SLOTS);
            self.regrow(last, grown);
        }
    }

    /// The page holding `at`, unshared so it can be written.
    ///
    /// This is `Arc::make_mut` by hand, because that does not exist for an
    /// unsized `Arc<[_]>`. `get_mut` returning `None` is the same test
    /// `make_mut` makes — unique in both strong and weak counts — so a page a
    /// snapshot still holds is copied here, and only that page.
    fn page_mut(
        &mut self,
        at: usize,
    ) -> &mut [(T, T)] {
        let p = at / PAGE_SLOTS;
        if Arc::get_mut(&mut self.pages[p]).is_none() {
            self.pages[p] = Arc::from(self.pages[p].to_vec());
        }
        Arc::get_mut(&mut self.pages[p]).expect("unique after the copy above")
    }

    fn push(
        &mut self,
        value: (T, T),
    ) {
        self.ensure_pages(self.len + 1);
        let at = self.len;
        self.page_mut(at)[at % PAGE_SLOTS] = value;
        self.len += 1;
    }
}

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
type Tier<T> = Option<Arc<Paged<T>>>;

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
    into: &mut Paged<B>,
) {
    let Some(old) = from.take() else { return };
    // Reuse the buffer when this version owns it outright; a version that still
    // shares it with a snapshot has to copy. Only the page pointers are cloned
    // here — the slots are read out below either way.
    let old = Arc::try_unwrap(old).unwrap_or_else(|shared| (*shared).clone());
    if old.is_empty() {
        return;
    }
    let mut merged: Paged<B> = Paged::default();
    merged.ensure_pages(old.len() + into.len());
    for (s, d) in old.iter() {
        merged.push(if s.vacant() && d.vacant() {
            (B::EMPTY, B::EMPTY)
        } else {
            (B::put(s.get()), B::put(d.get()))
        });
    }
    for i in 0..into.len() {
        merged.push(into.at(i));
    }
    *into = merged;
}

/// Run `$body` for each tier in id order, with `$v` bound to `&Option<Paged<_>>`.
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
    ///
    /// Public because the restore path recovers the tier boundaries with it.
    #[must_use]
    pub const fn rank_for(v: u64) -> u8 {
        match v {
            0..u16::CEILING => 0,
            u16::CEILING..U24::CEILING => 1,
            U24::CEILING..u32::CEILING => 2,
            _ => 3,
        }
    }

    /// The tier new edge ids append to: the widest one that already holds
    /// anything, since a narrower range can never follow a wider one.
    const fn append_rank(&self) -> u8 {
        match (&self.w64, &self.w32, &self.w24) {
            (Some(_), ..) => 3,
            (_, Some(_), _) => 2,
            (_, _, Some(_)) => 1,
            (None, None, None) => 0,
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
        let mut n = 0;
        each_tier!(self, |v| {
            n += v.as_ref().map_or(0, |v| v.allocated_bytes());
        });
        n
    }

    /// Mean bytes per stored slot, which is the quantity the tiering is for.
    #[must_use]
    pub fn bytes_per_edge(&self) -> f64 {
        let n = self.len();
        if n == 0 {
            return 0.0;
        }
        let mut bytes = 0;
        each_tier!(self, |v| {
            bytes += v.as_ref().map_or(0, |v| v.used_bytes());
        });
        bytes as f64 / n as f64
    }

    /// The endpoints of `edge_id`, or `None` if the slot is empty, deleted, or
    /// past the end.
    ///
    /// Indexes the concrete vectors rather than going through [`Self::locate`]
    /// and `TierOps`: this is the per-row read path, and a `dyn` call plus a
    /// second walk of the tier lengths would show up on it. The last tier's
    /// `slot -=` is dead by construction — there is nothing after it — which is
    /// what the allow covers.
    #[allow(unused_assignments)]
    #[must_use]
    pub fn get(
        &self,
        edge_id: u64,
    ) -> Option<(u64, u64)> {
        let mut slot = usize::try_from(edge_id).ok()?;
        each_tier!(self, |v| {
            if let Some(v) = v {
                if slot < v.len() {
                    let (s, d) = v.at(slot);
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

    /// Lay out all four tiers directly, for a caller that already knows where
    /// the boundaries fall.
    ///
    /// `starts[r]` is the first edge id belonging to tier `r + 1`, and `len` is
    /// one past the largest edge id. Both are slot indices, so the caller — who
    /// knows its ids fit — does the narrowing and this stays total. Every slot
    /// is created empty.
    ///
    /// This exists so a restore can *rebuild* the tiering rather than flatten
    /// it. Sizing with [`Self::prepare`] and the whole graph's widest endpoint
    /// would put every edge at the widest width, undoing on each load exactly
    /// what the tiers are for. The boundaries are recoverable without keeping
    /// anything per edge: under incremental growth an edge lands in the widest
    /// tier any earlier edge needed, so tier `r` begins at the smallest edge id
    /// whose own endpoints need at least `r` — three scalars, found in the same
    /// scan that finds the maximum id.
    pub fn prepare_tiers(
        &mut self,
        starts: [usize; 3],
        len: usize,
    ) {
        let [s1, s2, s3] = starts.map(|s| s.min(len));
        debug_assert!(s1 <= s2 && s2 <= s3, "tier starts must be ordered");
        for (rank, n) in [(0u8, s1), (1, s2 - s1), (2, s3 - s2), (3, len - s3)] {
            if n == 0 {
                continue;
            }
            self.with_tier(rank, |v| {
                v.reserve_exact(n);
                v.grow_to(n);
            });
        }
    }

    /// The tier holding `slot` and its offset within that tier, or `None` if
    /// the id is past every tier. The tiers partition the id space into
    /// contiguous ranges, so this walks at most four of them.
    fn locate(
        &self,
        slot: usize,
    ) -> Option<(u8, usize)> {
        let mut base = 0usize;
        for rank in 0..4u8 {
            let n = self.tier_len(rank);
            if slot < base + n {
                return Some((rank, slot - base));
            }
            base += n;
        }
        None
    }

    /// Record `edge_id`'s endpoints, promoting and growing as needed.
    pub fn set(
        &mut self,
        edge_id: u64,
        src: u64,
        dst: u64,
    ) {
        let slot = usize::try_from(edge_id).expect("edge id exceeds usize");
        let needed = Self::rank_for(src.max(dst));
        match self.locate(slot) {
            // A reused id whose new endpoints no longer fit its tier. Promoting
            // up to `needed` re-bases that tier's range to 0, so the offset just
            // computed is stale and the raw id is the right index.
            Some((rank, _)) if needed > rank => {
                self.raise_to(needed);
                self.with_tier(needed, |v| v.put(slot, src, dst));
            }
            // An id inside an existing tier is an update in place.
            Some((rank, at)) => self.with_tier(rank, |v| v.put(at, src, dst)),
            // Past the end: append to the widest active tier, or a wider one if
            // this edge needs it.
            None => {
                let rank = needed.max(self.append_rank());
                let at = slot - (self.len() - self.tier_len(rank));
                self.with_tier(rank, |v| {
                    v.grow_to(at + 1);
                    v.put(at, src, dst);
                });
            }
        }
    }

    /// The identity of each page backing the `u16` tier.
    ///
    /// Pointers, not contents: whether two versions *share* a page is exactly
    /// what the copy-on-write bound is about, and it is invisible in anything
    /// derived from the slots or from the page count.
    #[cfg(test)]
    fn w16_page_ids(&self) -> Vec<usize> {
        self.w16
            .as_ref()
            .map(|t| {
                t.pages
                    .iter()
                    .map(|p| Arc::as_ptr(p).cast::<()>() as usize)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Tombstone `edge_id`'s slot. Vectors never shrink, as before.
    pub fn clear(
        &mut self,
        edge_id: u64,
    ) {
        let Ok(slot) = usize::try_from(edge_id) else {
            return;
        };
        if let Some((rank, at)) = self.locate(slot) {
            self.with_tier(rank, |v| v.vacate(at));
        }
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

impl<T: Endpoint> TierOps for Paged<T> {
    fn grow_to(
        &mut self,
        len: usize,
    ) {
        if len > self.len {
            self.ensure_pages(len);
            self.len = len;
        }
    }
    fn put(
        &mut self,
        at: usize,
        src: u64,
        dst: u64,
    ) {
        // One page copies here if a snapshot still shares it, not the tier.
        self.page_mut(at)[at % PAGE_SLOTS] = (T::put(src), T::put(dst));
    }
    fn vacate(
        &mut self,
        at: usize,
    ) {
        if at < self.len {
            self.page_mut(at)[at % PAGE_SLOTS] = (T::EMPTY, T::EMPTY);
        }
    }
    fn len(&self) -> usize {
        self.len
    }
    fn reserve_exact(
        &mut self,
        extra: usize,
    ) {
        // `saturating_sub` because the page count is only ever *at least* what
        // the logical length needs: a caller reserving less than is already
        // allocated asks for nothing, not for a wrapped-around capacity.
        self.pages.reserve_exact(
            (self.len + extra)
                .div_ceil(PAGE_SLOTS)
                .saturating_sub(self.pages.len()),
        );
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

    /// A restore must rebuild the same tiering the graph had in memory, or
    /// every reload would flatten the index to its widest width.
    #[test]
    fn prepare_tiers_rebuilds_the_layout_growth_produced() {
        // Grown incrementally: 1000 narrow edges, then 1000 wide ones.
        let mut grown = EndpointIndex::default();
        for e in 0..1000u64 {
            grown.set(e, e, e + 1);
        }
        for e in 1000..2000u64 {
            grown.set(e, 100_000 + e, 100_001 + e);
        }

        // Restored: the boundary is the first edge id needing the wider tier.
        let mut restored = EndpointIndex::default();
        restored.prepare_tiers([1000, 2000, 2000], 2000);
        for e in 0..1000u64 {
            restored.set(e, e, e + 1);
        }
        for e in 1000..2000u64 {
            restored.set(e, 100_000 + e, 100_001 + e);
        }

        assert_eq!(restored.len(), grown.len());
        assert!(
            (restored.bytes_per_edge() - grown.bytes_per_edge()).abs() < f64::EPSILON,
            "restore flattened the tiering: {} vs {}",
            restored.bytes_per_edge(),
            grown.bytes_per_edge()
        );
        for e in 0..2000u64 {
            assert_eq!(restored.get(e), grown.get(e), "slot {e}");
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

    /// `prepare` exists only to stop bulk insertion being quadratic, so it must
    /// not change what the index ends up holding. The load path prepares and
    /// every other path does not, and both have to agree.
    #[test]
    fn preparing_first_changes_nothing_but_the_cost() {
        let triples = [
            (0u64, 1u64, 2u64),
            (3, 70_000, 4),
            (7, 5, 6),
            (9, 1 << 25, 2),
        ];
        let mut prepared = EndpointIndex::default();
        prepared.prepare(9, 1 << 25);
        for &(e, s, d) in &triples {
            prepared.set(e, s, d);
        }
        let mut plain = EndpointIndex::default();
        for &(e, s, d) in &triples {
            plain.set(e, s, d);
        }
        assert_eq!(prepared.len(), plain.len());
        for e in 0..plain.len() as u64 {
            assert_eq!(prepared.get(e), plain.get(e), "slot {e}");
        }
        // The *contents* agree; the layouts deliberately do not. `prepare`
        // sizes one tier to the batch's widest endpoint, so it is wider on
        // average than letting each edge pick its own tier. That is the trade
        // for not reallocating per edge, and it is why the restore path uses
        // `prepare_tiers` — which rebuilds the boundaries — rather than this.
        assert!(
            prepared.bytes_per_edge() >= plain.bytes_per_edge(),
            "preparing should never be narrower than growing edge by edge"
        );
    }

    /// A deterministic xorshift, so a failure here is reproducible rather than
    /// a Heisenbug that vanishes on re-run.
    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Endpoint values chosen to sit exactly on the tier boundaries, where the
    /// empty-slot sentinel of each width lives. `65_535` is `u16::EMPTY`, so an
    /// edge with that endpoint must be held one tier wider or it would read
    /// back as a deleted slot.
    const BOUNDARIES: [u64; 8] = [
        0,
        65_534,
        65_535,
        16_777_214,
        16_777_215,
        4_294_967_294,
        4_294_967_295,
        4_294_967_296,
    ];

    /// Random `set`/`clear`/snapshot sequences must agree with a flat
    /// `Vec<Option<(src, dst)>>` that models nothing but the observable answer.
    ///
    /// The paging exists to make an MVCC version cheap, which means the slots a
    /// version reads are reached through page pointers it may share with a
    /// snapshot. A wrong `Arc::make_mut` there does not fail loudly: it writes
    /// through into a snapshot's page, or drops a write into a copy nobody
    /// reads, and the index quietly returns the wrong endpoints for an edge.
    /// So the sequence takes snapshots as it goes and checks every one of them
    /// at the end, against the model as it stood when the snapshot was taken.
    #[test]
    fn random_edits_and_snapshots_agree_with_a_flat_reference() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut ix = EndpointIndex::default();
        let mut model: Vec<Option<(u64, u64)>> = Vec::new();
        let mut snapshots: Vec<(EndpointIndex, Vec<Option<(u64, u64)>>)> = Vec::new();
        let mut writes = 0u32;
        let mut clears = 0u32;

        for _ in 0..40_000 {
            let r = xorshift(&mut state);
            // Ids dense enough to reuse slots and to cross page boundaries
            // repeatedly: several pages' worth at every width.
            let id = r % 9_000;
            match (r >> 21) & 7 {
                0..=4 => {
                    // Half the endpoints land on a tier boundary; the rest are
                    // spread across the four widths so promotion happens in
                    // every direction the tiers allow.
                    let (src, dst) = if r & 1 == 0 {
                        (
                            BOUNDARIES[((r >> 40) % 8) as usize],
                            BOUNDARIES[((r >> 44) % 8) as usize],
                        )
                    } else {
                        let cap = [60_000_u64, 16_000_000, 4_000_000_000, u64::MAX / 2]
                            [((r >> 40) % 4) as usize];
                        ((r >> 3) % cap, (r >> 25) % cap)
                    };
                    ix.set(id, src, dst);
                    if model.len() <= id as usize {
                        model.resize(id as usize + 1, None);
                    }
                    model[id as usize] = Some((src, dst));
                    writes += 1;
                }
                5 | 6 => {
                    ix.clear(id);
                    if (id as usize) < model.len() {
                        model[id as usize] = None;
                    }
                    clears += 1;
                }
                _ => {
                    // A version, so the *next* write reaches a shared page and
                    // has to copy it rather than scribble on this snapshot.
                    snapshots.push((ix.clone(), model.clone()));
                }
            }
        }

        // The sequence has to have actually exercised all three, or the test
        // asserts agreement about nothing.
        assert!(writes > 10_000, "writes: {writes}");
        assert!(clears > 5_000, "clears: {clears}");
        assert!(snapshots.len() > 1_000, "snapshots: {}", snapshots.len());

        let check = |ix: &EndpointIndex, model: &[Option<(u64, u64)>], what: &str| {
            for id in 0..9_000u64 {
                let want = model.get(id as usize).copied().flatten();
                assert_eq!(ix.get(id), want, "{what}: edge {id}");
            }
        };
        check(&ix, &model, "live");
        for (i, (snap, snap_model)) in snapshots.iter().enumerate() {
            check(snap, snap_model, &format!("snapshot {i}"));
        }
    }

    /// A small tier allocates the slots it uses, not a whole page.
    ///
    /// The first cut of the paged layout gave every page exactly `PAGE_SLOTS`
    /// slots, tail included, so a graph holding a single edge paid a whole page
    /// for it: **+20.4 KB per graph** measured against a flat `Vec` over 200
    /// graphs, 27% of the per-graph footprint, on a server that may hold
    /// thousands of small graphs. That is the wrong trade for a copy-on-write
    /// win that only shows up on large ones, so the tail page is sized to the
    /// tier and this keeps it that way.
    #[test]
    fn a_small_tier_does_not_allocate_a_whole_page() {
        for edges in [1_u64, 10, 100] {
            let mut ix = EndpointIndex::default();
            for e in 0..edges {
                ix.set(e, e % 60_000, (e + 1) % 60_000);
            }
            // 4 bytes a slot in the `u16` tier, so the used bytes are the floor
            // and a doubling tail is the ceiling.
            let used = usize::try_from(edges).unwrap() * 4;
            let allocated = ix.memory_usage();
            assert!(
                allocated < 4 * used.max(64),
                "{edges} edges allocated {allocated} bytes for {used} bytes of slots"
            );
        }

        // And the bound still holds once the tier is genuinely large: whole
        // pages then, so allocation tracks the edges rather than the page size.
        let mut ix = EndpointIndex::default();
        for e in 0..50_000u64 {
            ix.set(e, e % 60_000, (e + 1) % 60_000);
        }
        let allocated = ix.memory_usage();
        assert!(
            (200_000..280_000).contains(&allocated),
            "50k edges at 4 bytes a slot allocated {allocated} bytes"
        );
    }

    /// The whole point, as a bound rather than a benchmark: the first write
    /// after a version must leave every page but one shared with the snapshot.
    ///
    /// Stated as page *identity* rather than bytes. A deep copy of the pages
    /// produces an index of exactly the same size and the same page count — so
    /// `memory_usage`, and anything else computed from the slots, cannot tell
    /// the two apart. Sharing is only observable in the pointers.
    #[test]
    fn a_version_that_writes_one_edge_copies_one_page() {
        let mut ix = EndpointIndex::default();
        for e in 0..200_000u64 {
            ix.set(e, e % 60_000, (e + 1) % 60_000);
        }
        let before = ix.w16_page_ids();
        assert!(
            before.len() > 40,
            "want many pages to share: {}",
            before.len()
        );

        // The committed snapshot stays alive, so the version's first write
        // reaches `Arc::make_mut` with a refcount above one — the deep-copy
        // case this type exists to bound.
        let mut version = ix.clone();
        version.set(0, 1, 2);

        let after = version.w16_page_ids();
        assert_eq!(after.len(), before.len(), "a write should not add pages");
        let copied = after.iter().zip(&before).filter(|(a, b)| a != b).count();
        assert_eq!(
            copied,
            1,
            "a one-edge version copied {copied} of {} pages",
            before.len()
        );

        // And the copy is a copy: the snapshot still reads what it always did.
        assert_eq!(ix.get(0), Some((0, 1)));
        assert_eq!(version.get(0), Some((1, 2)));
    }
}

#[cfg(test)]
mod cow_bench {
    use super::EndpointIndex;
    use crate::graph::graphblas::instr::read_instr;

    /// What paging costs the read path.
    ///
    /// `get` used to be one bounds-checked load from a flat vector; a page
    /// lookup makes it a shift, a mask, a load of the page pointer and a second
    /// bounds check. Instructions will show that; the thing instructions cannot
    /// show is the extra indirection's cache behaviour, so this reports time as
    /// well, for a sequential scan and for random access over an index far
    /// larger than L2.
    ///
    /// Run with:
    ///   cargo test --release -p graph get_cost -- --ignored --nocapture
    #[test]
    #[ignore]
    fn get_cost_of_paging() {
        use std::time::Instant;

        let edges = 10_000_000u64;
        let mut ix = EndpointIndex::default();
        for e in 0..edges {
            ix.set(e, e % 60_000, (e + 1) % 60_000);
        }

        // Sequential: every page pointer is used 4096 times in a row.
        let i0 = read_instr();
        let t0 = Instant::now();
        let mut acc = 0u64;
        for e in 0..edges {
            if let Some((s, d)) = ix.get(e) {
                acc = acc.wrapping_add(s ^ d);
            }
        }
        let seq_t = t0.elapsed();
        let seq_i = read_instr().zip(i0).map(|(a, b)| a - b);
        std::hint::black_box(acc);

        // Random: the page-pointer array is 8 bytes per 4096 slots, so for this
        // index it is ~20 KB and stays cached even when the slots do not. That
        // is the case where an extra indirection would hurt if it were going to.
        let reps = 10_000_000u64;
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let i1 = read_instr();
        let t1 = Instant::now();
        let mut acc2 = 0u64;
        for _ in 0..reps {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if let Some((s, d)) = ix.get(state % edges) {
                acc2 = acc2.wrapping_add(s ^ d);
            }
        }
        let rnd_t = t1.elapsed();
        let rnd_i = read_instr().zip(i1).map(|(a, b)| a - b);
        std::hint::black_box(acc2);

        let per = |i: Option<u64>, n: u64| {
            i.map_or("n/a".to_string(), |v| format!("{:.2}", v as f64 / n as f64))
        };
        println!(
            "  sequential {edges} gets   {} instr/get   {:.1} ns/get",
            per(seq_i, edges),
            seq_t.as_nanos() as f64 / edges as f64
        );
        println!(
            "  random     {reps} gets   {} instr/get   {:.1} ns/get",
            per(rnd_i, reps),
            rnd_t.as_nanos() as f64 / reps as f64
        );
    }

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
