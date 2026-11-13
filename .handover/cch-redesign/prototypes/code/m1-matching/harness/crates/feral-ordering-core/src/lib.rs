//! Shared contract types for FERAL's fill-reducing ordering crates.
//!
//! This crate exists so that `feral-amd`, `feral-metis`, `feral-scotch`,
//! and `feral-kahip` all accept the same input type and emit the same
//! stats / error types without a type-conversion layer at their
//! boundary. The contract itself is documented in
//! `dev/plans/ordering-crate-contract.md`.
//!
//! The public surface is deliberately minimal:
//!
//! - [`CscPattern`] — borrowed, full-symmetric, 0-based, `Idx`-indexed.
//! - [`OrderingStats`] — producer-agnostic diagnostic counters.
//! - [`OrderingError`] — shared error shape.
//! - [`CONTRACT_VERSION`] — bumped on any breaking change.
//!
//! Each ordering crate exposes exactly one contract-conforming
//! function with the signature:
//!
//! ```ignore
//! pub fn xxx_order(
//!     pattern: &feral_ordering_core::CscPattern<'_>,
//!     opts: &XxxOptions,
//! ) -> Result<
//!     (Vec<Idx>, feral_ordering_core::OrderingStats, XxxStats),
//!     feral_ordering_core::OrderingError,
//! >;
//! ```
//!
//! `perm[k] = j` means new index `k` corresponds to old index `j`
//! (new-to-old). Callers that need the inverse compute it with a
//! trivial helper.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod quotient_graph;

use core::fmt;

/// Version of the shared ordering-crate contract.
///
/// Bumped on any breaking change to [`CscPattern`], [`OrderingStats`],
/// [`OrderingError`], or the per-crate producer-function signature.
/// Consumers can assert at build time that all linked ordering crates
/// target the same contract version.
pub const CONTRACT_VERSION: u32 = 1;

/// Signed index type of every pattern, permutation and workspace array.
///
/// Upstream uses `i32`; this copy is widened to `i64` so a graph is
/// bounded by memory rather than by 2^31 vertices or entries. Indices
/// stay signed because `-1` sentinels and AMD's `FLIP` encoding rely
/// on it.
pub type Idx = i64;

/// Borrowed symmetric sparsity pattern in CSC form.
///
/// The pattern must be **full-symmetric** — both the upper and lower
/// halves are present. Row indices within each column must be sorted
/// in ascending order. Indices are 0-based.
///
/// Invariants (checked by [`CscPattern::new`]):
///
/// - `col_ptr.len() == n + 1`
/// - `col_ptr[0] == 0`, `col_ptr` is non-decreasing
/// - `row_idx.len() == col_ptr[n]`
/// - every row index is in `0..n` (non-negative and `< n`)
/// - row indices within each column are sorted ascending
///
/// Structural symmetry is the caller's responsibility and is not
/// checked here; individual ordering crates may debug-assert it.
#[derive(Debug, Clone, Copy)]
pub struct CscPattern<'a> {
    /// Matrix dimension.
    pub n: usize,
    /// Column pointers. Length `n + 1`.
    pub col_ptr: &'a [Idx],
    /// Row indices. Length `col_ptr[n]`.
    pub row_idx: &'a [Idx],
}

impl<'a> CscPattern<'a> {
    /// Construct a validated pattern.
    ///
    /// Returns `None` if the structural invariants above are
    /// violated. Does not check symmetry.
    pub fn new(n: usize, col_ptr: &'a [Idx], row_idx: &'a [Idx]) -> Option<Self> {
        if col_ptr.len() != n + 1 {
            return None;
        }
        if col_ptr[0] != 0 {
            return None;
        }
        let nnz_i32 = *col_ptr.last()?;
        if nnz_i32 < 0 {
            return None;
        }
        let nnz = nnz_i32 as usize;
        if row_idx.len() != nnz {
            return None;
        }
        for w in col_ptr.windows(2) {
            if w[1] < w[0] {
                return None;
            }
        }
        let n_i32: Idx = match Idx::try_from(n) {
            Ok(v) => v,
            Err(_) => return None,
        };
        for &r in row_idx {
            if r < 0 || r >= n_i32 {
                return None;
            }
        }
        // Row indices within each column must be sorted ascending. This is a
        // documented precondition that downstream consumers silently rely on:
        // feral-metis's adjacency builder dedups only *adjacent* duplicates
        // (graph.rs), and feral-scotch's compress step inserts neighbours with
        // `partition_point` assuming sorted runs. Unsorted rows would let a
        // non-adjacent duplicate survive as a spurious edge, corrupting the
        // graph. Enforce it here (O(nnz)) so every consumer can trust it.
        for w in col_ptr.windows(2) {
            let lo = w[0] as usize;
            let hi = w[1] as usize;
            if row_idx[lo..hi].windows(2).any(|p| p[1] < p[0]) {
                return None;
            }
        }
        Some(Self {
            n,
            col_ptr,
            row_idx,
        })
    }

    /// Number of stored nonzeros.
    pub fn nnz(&self) -> usize {
        self.row_idx.len()
    }
}

/// Diagnostic counters shared by every ordering producer.
///
/// Crate-specific counters live in the crate's own stats struct
/// (e.g. `AmdStats`, `MetisStats`) returned alongside this one, not
/// inside it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct OrderingStats {
    /// Wall-clock ordering time, microseconds.
    pub time_us: u64,
    /// Predicted non-zeros in L (upper bound if known).
    ///
    /// `None` when the algorithm does not produce an estimate.
    /// AMD can populate this; METIS / Scotch / KaHIP typically
    /// cannot without a follow-up symbolic pass.
    pub fill_estimate: Option<u64>,
    /// Predicted factorization flops. `None` when not produced.
    pub flop_estimate: Option<u64>,
}

/// Shared error shape for the ordering-crate contract.
///
/// Crate-specific failure modes are carried via [`OrderingError::Internal`]
/// rather than a wrapped crate-specific enum, to avoid an error-type
/// dependency tree between the sibling crates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderingError {
    /// Input failed [`CscPattern::new`] validation (wrong lengths,
    /// out-of-range row indices, etc.).
    MalformedInput,
    /// Input pattern was not structurally symmetric. Debug-only
    /// detection; release builds trust the caller.
    NonSymmetric,
    /// Index overflow in the crate's internal workspace (for
    /// example, `Idx` overflow on a very large matrix).
    IndexOverflow,
    /// Graph is disconnected and the crate does not handle
    /// disconnected components in its current form.
    DisconnectedGraph,
    /// Crate-specific failure with a short static message. Keep
    /// short — this is a status channel, not a rich diagnostic.
    Internal(&'static str),
}

impl fmt::Display for OrderingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedInput => f.write_str("ordering input failed structural validation"),
            Self::NonSymmetric => {
                f.write_str("ordering input pattern was not structurally symmetric")
            }
            Self::IndexOverflow => f.write_str("ordering workspace exceeded the index range"),
            Self::DisconnectedGraph => f.write_str("ordering input graph is disconnected"),
            Self::Internal(msg) => write!(f, "ordering internal error: {msg}"),
        }
    }
}

impl std::error::Error for OrderingError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_version_is_one() {
        assert_eq!(CONTRACT_VERSION, 1);
    }

    #[test]
    fn empty_pattern_ok() {
        let cp = [0 as Idx];
        let ri: [Idx; 0] = [];
        let p = CscPattern::new(0, &cp, &ri).expect("n=0 pattern");
        assert_eq!(p.n, 0);
        assert_eq!(p.nnz(), 0);
    }

    #[test]
    fn diagonal_2x2_ok() {
        let cp = [0 as Idx, 1, 2];
        let ri = [0 as Idx, 1];
        let p = CscPattern::new(2, &cp, &ri).unwrap();
        assert_eq!(p.nnz(), 2);
    }

    #[test]
    fn rejects_bad_col_ptr_length() {
        let cp = [0 as Idx, 1];
        let ri = [0 as Idx];
        assert!(CscPattern::new(2, &cp, &ri).is_none());
    }

    #[test]
    fn rejects_oob_row_index() {
        let cp = [0 as Idx, 1];
        let ri = [5 as Idx];
        assert!(CscPattern::new(1, &cp, &ri).is_none());
    }

    #[test]
    fn rejects_negative_row_index() {
        let cp = [0 as Idx, 1];
        let ri = [-1 as Idx];
        assert!(CscPattern::new(1, &cp, &ri).is_none());
    }

    #[test]
    fn rejects_nonzero_first_col_ptr() {
        let cp = [1 as Idx, 2];
        let ri = [0 as Idx];
        assert!(CscPattern::new(1, &cp, &ri).is_none());
    }

    #[test]
    fn rejects_nonmonotone_col_ptr() {
        let cp = [0 as Idx, 2, 1];
        let ri = [0 as Idx, 0];
        assert!(CscPattern::new(2, &cp, &ri).is_none());
    }

    #[test]
    fn rejects_row_idx_length_mismatch() {
        let cp = [0 as Idx, 1, 2];
        let ri = [0 as Idx];
        assert!(CscPattern::new(2, &cp, &ri).is_none());
    }

    #[test]
    fn rejects_unsorted_rows_within_column() {
        // Column 0 has rows [1, 0] — descending, violating the documented
        // ascending-order invariant. `new` must reject it: feral-metis's
        // adjacency builder dedups only *adjacent* duplicates, so unsorted
        // rows would let a non-adjacent duplicate survive as a spurious edge.
        let cp = [0 as Idx, 2, 2];
        let ri = [1 as Idx, 0];
        assert!(CscPattern::new(2, &cp, &ri).is_none());
    }

    #[test]
    fn accepts_sorted_rows_with_adjacent_duplicate() {
        // Ascending order permits adjacent duplicates; feral-metis drops them
        // in its adjacency builder. `new` enforces sortedness, not
        // strict-increase, so this must still be accepted.
        let cp = [0 as Idx, 3, 3];
        let ri = [0 as Idx, 0, 1];
        assert!(CscPattern::new(2, &cp, &ri).is_some());
    }

    #[test]
    fn rejects_negative_col_ptr_tail() {
        // col_ptr.last() is negative → nnz would be invalid.
        let cp = [0 as Idx, -1];
        let ri: [Idx; 0] = [];
        assert!(CscPattern::new(1, &cp, &ri).is_none());
    }

    #[test]
    fn ordering_error_display_is_non_empty() {
        for e in [
            OrderingError::MalformedInput,
            OrderingError::NonSymmetric,
            OrderingError::IndexOverflow,
            OrderingError::DisconnectedGraph,
            OrderingError::Internal("boom"),
        ] {
            assert!(!format!("{e}").is_empty());
        }
    }

    #[test]
    fn ordering_stats_default_is_none_fields() {
        let s = OrderingStats::default();
        assert_eq!(s.time_us, 0);
        assert!(s.fill_estimate.is_none());
        assert!(s.flop_estimate.is_none());
    }
}
