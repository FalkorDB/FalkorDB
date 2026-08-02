#[derive(Debug, Default, Clone)]
pub struct VectorIndexOptions {
    pub dimension: u32,
    pub similarity_function: Option<String>,
    pub m: Option<usize>,
    pub ef_construction: Option<usize>,
    pub ef_runtime: Option<usize>,
}
