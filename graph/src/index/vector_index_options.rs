/// Most dimensions a vector index may declare. Index creation refuses more, and a
/// persisted index declaring more is refused on load (the dimension sizes the
/// vector library's storage, so it cannot be taken from a payload unchecked).
pub const MAX_VECTOR_DIMENSION: u64 = 1 << 24;

/// Most a vector index's `M`, `efConstruction` or `efRuntime` may be, for the same
/// reason; creation and load hold to the same limit so a saved graph reloads.
pub const MAX_VECTOR_GRAPH_PARAM: usize = 1 << 20;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct VectorIndexOptions {
    /// `u64`, matching C's `size_t`, the v19 RDB's field and the effects
    /// wire. A `u32` here meant every RDB load narrowed and every wire read
    /// had to check.
    pub dimension: u64,
    pub similarity_function: Option<String>,
    pub m: Option<usize>,
    pub ef_construction: Option<usize>,
    pub ef_runtime: Option<usize>,
}
