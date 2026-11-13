//! Diagnostic counters exposed alongside the permutation.

/// Diagnostic counters collected during AMD ordering.
///
/// In release builds only `ncmpa` has non-zero cost; the other
/// populated fields add no branches to the hot loop. In debug builds
/// every field is populated except `n_clear_flag`, which is a
/// not-yet-wired constant `0` (see its field doc).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AmdStats {
    /// Number of garbage-collection compactions fired.
    pub ncmpa: u32,
    /// Number of mark-array generation-counter resets.
    ///
    /// Currently always `0`: not wired to a backing counter. The reset
    /// it would count (`clear_flag`) only fires when the generation
    /// counter `wflg` reaches `wbig = Idx::MAX - n`, which during
    /// elimination requires `n` on the order of tens of thousands, so
    /// the true count is `0` on every practically testable input.
    pub n_clear_flag: u32,
    /// Number of variables absorbed by mass elimination
    /// (Slice B).
    pub n_mass_elim: u32,
    /// Number of supervariable merges detected (Slice B).
    pub n_supervar_merge: u32,
    /// Number of variables placed into the dense-deferred bucket
    /// at initialization.
    pub n_dense_deferred: u32,
    /// Flop counter: divisions (faer amd.rs:547-566).
    pub ndiv: u64,
    /// Flop counter: LU multiply-subtracts.
    pub nms_lu: u64,
    /// Flop counter: LDLᵀ multiply-subtracts.
    pub nms_ldl: u64,
}
