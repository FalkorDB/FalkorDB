//! Input-order emission for the batched optional sub-plan (`OPTIONAL MATCH`,
//! and the optional arm of `Apply`).
//!
//! The batched operators run one sub-plan over a whole input batch, and an
//! input row that produced nothing gets a NULL-fallback row. Which rows those
//! are is known only once the sub-plan is exhausted, so emitting every result
//! batch as it arrives puts all of the fallback rows after all of the matched
//! ones — an order neither the input nor the C engine has (#3070).
//!
//! [`FallbackOrder`] keeps the streaming path for the case that needs nothing
//! else: a result batch whose origins are ascending, follow everything already
//! emitted, and have no unmatched origin below them goes straight out. Only
//! once a gap appears (an origin below the batch that has produced nothing
//! *yet*) are the results held, and on exhaustion the held results and the
//! fallback rows are interleaved by origin.
//!
//! Held results are bounded by [`HOLD_LIMIT`]: past it, the held batches are
//! released in arrival order and the rest of the input batch streams as
//! before, fallback rows last. Ordering is best effort there, never row
//! content: a fallback row is only ever produced at exhaustion, for an origin
//! that produced nothing.

use std::collections::VecDeque;

use crate::parser::ast::Variable;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, Column},
    value::Value,
};

/// Most result rows held back for one input batch before ordering is given up.
const HOLD_LIMIT: usize = 8 * BATCH_SIZE;

/// Per-input-batch ordering state for an optional sub-plan's output.
pub(crate) struct FallbackOrder<'a> {
    /// Dense per-origin match flags (`matched[i]` == origin `i` produced a
    /// result); the unmatched origins get the NULL fallback.
    matched: Vec<bool>,
    /// Smallest origin that has produced no result so far.
    frontier: usize,
    /// Largest origin already emitted.
    last_emitted: usize,
    /// Result batches waiting for exhaustion, because a smaller origin may
    /// still need a fallback row in front of them.
    held: Vec<Batch<'a>>,
    /// The origin of every active row of `held`, in order.
    held_origins: Vec<usize>,
    held_rows: usize,
    /// False once [`HOLD_LIMIT`] was exceeded: everything streams from then on.
    in_order: bool,
}

impl<'a> FallbackOrder<'a> {
    pub(crate) fn new(input_len: usize) -> Self {
        Self {
            matched: vec![false; input_len],
            frontier: 0,
            last_emitted: 0,
            held: Vec::new(),
            held_origins: Vec::new(),
            held_rows: 0,
            in_order: true,
        }
    }

    /// Record a merged result batch whose k-th active row came from
    /// `origins[k]` (non-empty), queueing on `out` whatever can be emitted now.
    pub(crate) fn push(
        &mut self,
        merged: Batch<'a>,
        origins: &[usize],
        out: &mut VecDeque<Batch<'a>>,
    ) {
        for &o in origins {
            self.matched[o] = true;
        }
        while self.matched.get(self.frontier).copied().unwrap_or(false) {
            self.frontier += 1;
        }
        if self.held.is_empty() {
            let first = origins[0];
            let last = origins[origins.len() - 1];
            if !self.in_order
                || (first >= self.last_emitted && last < self.frontier && origins.is_sorted())
            {
                self.last_emitted = self.last_emitted.max(last);
                out.push_back(merged);
                return;
            }
        }
        self.held_rows += merged.active_len();
        self.held_origins.extend_from_slice(origins);
        self.held.push(merged);
        if self.held_rows > HOLD_LIMIT {
            self.in_order = false;
            self.held_rows = 0;
            self.held_origins.clear();
            out.extend(self.held.drain(..));
        }
    }

    /// The sub-plan is exhausted: queue the held results and a NULL-fallback
    /// row (from `input`, with `vars` set to NULL) for every origin that
    /// produced nothing, in origin order.
    pub(crate) fn finish(
        &mut self,
        input: &Batch<'a>,
        vars: &[Variable],
        out: &mut VecDeque<Batch<'a>>,
    ) {
        let unmatched: Vec<usize> = (self.frontier..self.matched.len())
            .filter(|&i| !self.matched[i])
            .collect();
        let fallback = (!unmatched.is_empty()).then(|| {
            let mut fb = input.gather(&unmatched);
            for v in vars {
                fb.set_column(v.id, Column::Values(vec![Value::Null; unmatched.len()]));
            }
            fb
        });
        if self.held.is_empty() {
            // Everything emitted so far precedes the first unmatched origin.
            out.extend(fallback);
            return;
        }
        let mut parts = std::mem::take(&mut self.held);
        // Every (batch, row) of the held results with its origin, then the
        // fallback rows (whose origins are ascending), sorted by origin into
        // input order. The sort is stable, so a sub-plan's own order within an
        // origin is kept.
        let mut picks: Vec<(usize, usize, usize)> = Vec::with_capacity(
            self.held_origins.len() + fallback.as_ref().map_or(0, Batch::len),
        );
        let mut k = 0;
        for (b, batch) in parts.iter().enumerate() {
            for r in batch.active_indices() {
                picks.push((self.held_origins[k], b, r));
                k += 1;
            }
        }
        if let Some(fb) = fallback {
            let b = parts.len();
            picks.extend(unmatched.iter().enumerate().map(|(r, &o)| (o, b, r)));
            parts.push(fb);
        }
        self.held_origins.clear();
        picks.sort_by_key(|&(o, _, _)| o);
        let picks: Vec<(usize, usize)> = picks.into_iter().map(|(_, b, r)| (b, r)).collect();
        out.extend(
            picks
                .chunks(BATCH_SIZE)
                .map(|chunk| Batch::interleave(&parts, chunk)),
        );
    }
}
