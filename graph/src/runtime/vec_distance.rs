//! SIMD-backed distance functions over dense `f32` slices.
//!
//! Single source of truth for the math used by:
//! - The `vec.euclideanDistance` / `vec.cosineDistance` cypher
//!   functions (`runtime::functions::spatial`).
//! - Per-result distance computation when materialising KNN results
//!   from a vector index (`graph::vector_query_nodes` / `_edges`).
//!
//! The work is delegated to [`simsimd`], which runtime-dispatches to
//! AVX-512 / AVX2 / NEON / scalar implementations under the hood. The
//! conventions here match `RediSearch`'s `VecSimMetric_*` so the
//! distance you read back from the user-facing function and from the
//! KNN procedure agree exactly:
//!
//! - `euclidean`: L2 distance, `sqrt(sum((a - b)^2))`. Computed as
//!   `sqrt(sqeuclidean)` — `simsimd::sqeuclidean` returns the
//!   squared form, which avoids a square root in the inner kernel.
//! - `cosine`: `1 - cosine_similarity`. Zero-vectors collapse to 1.0
//!   (kept total — same behaviour as the prior scalar impl).
//! - `ip`: negated dot product, so smaller-is-closer holds for
//!   downstream sorting.
//!
//! Returns `None` on dimension mismatch or unrecognised metric. The
//! caller is responsible for surfacing a proper error message; both
//! call sites already do that.

use simsimd::SpatialSimilarity;

/// L2 distance between two equal-length `f32` slices.
pub fn euclidean(
    a: &[f32],
    b: &[f32],
) -> Option<f64> {
    f32::sqeuclidean(a, b).map(f64::sqrt)
}

/// `1 - cosine_similarity`. Matches `RediSearch`'s `COSINE` metric.
/// `simsimd` returns 1.0 when either operand is the zero vector.
#[must_use]
pub fn cosine(
    a: &[f32],
    b: &[f32],
) -> Option<f64> {
    f32::cosine(a, b)
}

/// Negated inner product — smaller is closer.
#[must_use]
pub fn inner_product(
    a: &[f32],
    b: &[f32],
) -> Option<f64> {
    f32::dot(a, b).map(|d| -d)
}

/// Dispatch by metric name. `metric == None` defaults to `"euclidean"`
/// to mirror `VectorIndexOptions::similarity_function`'s default.
#[must_use]
pub fn distance(
    metric: Option<&str>,
    a: &[f32],
    b: &[f32],
) -> Option<f64> {
    match metric.unwrap_or("euclidean") {
        "euclidean" => euclidean(a, b),
        "cosine" => cosine(a, b),
        "ip" => inner_product(a, b),
        _ => None,
    }
}
