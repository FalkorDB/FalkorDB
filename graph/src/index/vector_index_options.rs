/// Most a vector index's `M`, `efConstruction` or `efRuntime` may be.
///
/// These size the vector library's graph (C with `M:2000000` reaches 16 GB on two vectors),
/// so a persisted index declaring more is refused on load, and creation holds to
/// the same limit so a saved graph reloads.
///
/// The dimension has no such limit: as in C, any dimension is accepted, and it
/// costs nothing until a vector of that dimension is stored, which the vector's
/// own size bounds.
pub const MAX_VECTOR_GRAPH_PARAM: usize = 1 << 20;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
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
