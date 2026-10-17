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
