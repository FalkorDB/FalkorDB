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

#[derive(Debug, Default, Clone, PartialEq)]
pub struct TextIndexOptions {
    pub weight: Option<f64>,
    pub nostem: Option<bool>,
    /// The phonetic algorithm code, e.g. `"dm:en"`, or `None` when the
    /// statement said nothing. Empty, or C's `"no"`, means off: see
    /// [`Self::phonetic_enabled`].
    ///
    /// A code and not a `bool`: C stores `char *phonetic` and accepts
    /// `dm:fr`/`dm:pt`/`dm:es`, the v19 RDB persists the string, and the
    /// effects wire carries the string. A `bool` here was the only narrow
    /// representation in the chain, and it silently rewrote any other
    /// algorithm to `dm:en` on an RDB round trip.
    pub phonetic: Option<String>,
    pub language: Option<Arc<String>>,
    pub stopwords: Option<Vec<Arc<String>>>,
}

/// C's code for "phonetic matching off" (`INDEX_FIELD_DEFAULT_PHONETIC`). C stores it
/// on every field it did not enable and writes it to the RDB as-is.
pub const PHONETIC_OFF: &str = "no";

impl TextIndexOptions {
    /// Whether this field gets RediSearch's phonetic flag. C enables it for any code
    /// but `"no"`; we also read empty (or absent) as off. Treating `"no"` as a code
    /// turned phonetic matching on for every fulltext index a C-written RDB carried.
    #[must_use]
    pub fn phonetic_enabled(&self) -> bool {
        self.phonetic
            .as_deref()
            .is_some_and(|p| !p.is_empty() && p != PHONETIC_OFF)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(p: Option<&str>) -> TextIndexOptions {
        TextIndexOptions {
            phonetic: p.map(str::to_owned),
            ..Default::default()
        }
    }

    #[test]
    fn phonetic_off_spellings() {
        assert!(!with(None).phonetic_enabled());
        assert!(!with(Some("")).phonetic_enabled());
        assert!(!with(Some(PHONETIC_OFF)).phonetic_enabled());
        assert!(with(Some("dm:en")).phonetic_enabled());
        assert!(with(Some("dm:fr")).phonetic_enabled());
    }
}
