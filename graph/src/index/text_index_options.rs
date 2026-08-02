//! Configuration options for full-text search indexes.
//!
//! [`TextIndexOptions`] controls how RediSearch processes text when building
//! and querying a fulltext index.  These options map directly to RediSearch
//! field and index settings:
//!
//! | Field       | RediSearch effect                                    |
//! |-------------|------------------------------------------------------|
//! | `weight`    | Scoring multiplier for the field (default 1.0)       |
//! | `nostem`    | Disable stemming (e.g. keep "running" as-is)        |
//! | `phonetic`  | Enable phonetic matching (sounds-like queries)       |
//! | `language`  | Stemmer language (e.g. "english", "spanish")         |
//! | `stopwords` | Custom stop-word list (overrides RediSearch default) |
//!
//! `language` and `stopwords` are index-wide settings applied once when the
//! RediSearch index is created.  `weight`, `nostem`, and `phonetic` are
//! per-field settings applied when registering each fulltext field.

use std::sync::Arc;

#[derive(Debug, Default, Clone)]
pub struct TextIndexOptions {
    pub weight: Option<f64>,
    pub nostem: Option<bool>,
    pub phonetic: Option<bool>,
    pub language: Option<Arc<String>>,
    pub stopwords: Option<Vec<Arc<String>>>,
}
