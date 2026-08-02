//! The shared storage seam's error type.

/// Result alias for the shared storage seam.
pub type Result<T> = core::result::Result<T, StorageError>;

/// Errors surfaced by the shared storage seam.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StorageError {
    /// A durable backend operation failed (I/O, serialization).
    #[error("storage backend error: {0}")]
    Backend(String),
}
