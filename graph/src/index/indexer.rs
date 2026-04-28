//! Index lifecycle management for property-based graph lookups.
//!
//! The [`Indexer`] is the top-level coordinator for all indexes in a graph.
//! It owns one [`Index`](super::Index) per label and exposes methods for
//! creating, dropping, querying, and populating indexes.
//!
//! # Responsibilities
//!
//! - **Create / drop** indexes for (label, attribute, type) triples.
//! - **Route queries** -- delegates [`IndexQuery`] execution to the correct
//!   per-label [`Index`](super::Index).
//! - **Commit mutations** -- batches of added/removed documents are flushed
//!   to RediSearch during transaction commit.
//! - **Background population** -- tracks progress, serializes background
//!   batches with writes via a shared `write_lock`, and supports
//!   cancellation.
//!
//! # Internal layout
//!
//! ```text
//! Indexer
//!    |
//!    +-- index: RwLock<HashMap<Label, Index>>
//!    |      One Index per label; each Index wraps a single
//!    |      RSIndex handle and its field definitions.
//!    |
//!    +-- write_lock: Mutex<()>
//!    |      Serializes background population with commit_index
//!    |      so they never run concurrently.
//!    |
//!    +-- graph: Mutex<Option<Arc<Graph>>>
//!           Latest committed graph snapshot shared with
//!           background index population threads.
//! ```
//!
//! # Concurrency
//!
//! Read-side queries (`query`, `fulltext_query`, `is_label_indexed`, ...)
//! acquire a `read()` lock.  Write-side mutations (`create_index`,
//! `drop_index`, `commit`, ...) acquire a `write()` lock.  Background
//! population uses `write_lock` to avoid racing with per-transaction
//! commit calls.

use std::{
    collections::HashMap,
    ffi::CString,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use atomic_refcell::AtomicRefCell;
use parking_lot::{Mutex, RwLock};
use roaring::RoaringTreemap;

use super::Index;
pub use super::{
    Document, Field, IdIter, IndexInfo, IndexQuery, IndexResultsIter, IndexType, ScoredIdIter,
    TextIndexOptions, VectorIndexOptions,
};
use crate::{graph::graph::Graph, runtime::value::Value};

pub enum IndexOptions {
    Text(TextIndexOptions),
    Vector(VectorIndexOptions),
}

impl IndexOptions {
    /// Extract language from the options (only applicable for Text index options).
    #[must_use]
    pub const fn language(&self) -> &Option<Arc<String>> {
        match self {
            Self::Text(opts) => &opts.language,
            Self::Vector(_) => &None,
        }
    }

    /// Extract stopwords from the options (only applicable for Text index options).
    #[must_use]
    pub const fn stopwords(&self) -> &Option<Vec<Arc<String>>> {
        match self {
            Self::Text(opts) => &opts.stopwords,
            Self::Vector(_) => &None,
        }
    }

    /// Extract per-field text options (weight, nostem, phonetic).
    #[must_use]
    pub fn field_options(&self) -> Option<TextIndexOptions> {
        match self {
            Self::Text(opts) => {
                if opts.weight.is_some() || opts.nostem.is_some() || opts.phonetic.is_some() {
                    Some(TextIndexOptions {
                        weight: opts.weight,
                        nostem: opts.nostem,
                        phonetic: opts.phonetic,
                        ..Default::default()
                    })
                } else {
                    None
                }
            }
            Self::Vector(_) => None,
        }
    }

    /// Extract vector index options (only applicable for Vector index options).
    #[must_use]
    pub const fn vector_options(&self) -> Option<&VectorIndexOptions> {
        match self {
            Self::Vector(opts) => Some(opts),
            Self::Text(_) => None,
        }
    }
}

#[derive(Default, Clone)]
pub struct Indexer {
    index: Arc<RwLock<HashMap<Arc<String>, Index>>>,
    /// Serializes background index population batches with write-path
    /// `commit_index` calls so they never run concurrently.
    write_lock: Arc<Mutex<()>>,
    cancelled: Arc<AtomicBool>,
    /// Latest committed graph, shared with background index population.
    /// Updated by `MvccGraph::commit()` so background batches see fresh data.
    graph: Arc<Mutex<Option<Arc<AtomicRefCell<Graph>>>>>,
}

unsafe impl Send for Indexer {}
unsafe impl Sync for Indexer {}

impl Indexer {
    #[must_use]
    pub fn has_indices(&self) -> bool {
        !self.index.read().is_empty()
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.index.read().values().map(Index::memory_usage).sum()
    }

    pub fn create_index(
        &mut self,
        index_type: &IndexType,
        label: &Arc<String>,
        attrs: &Vec<Arc<String>>,
        total: u64,
        options: Option<IndexOptions>,
    ) -> Result<(), String> {
        let mut index = self.index.write();

        let (language, stopwords, field_options, vector_options) = match options {
            Some(IndexOptions::Text(text_opts)) => {
                let language = text_opts.language.clone();
                let stopwords = text_opts.stopwords.clone();
                (language, stopwords, Some(text_opts), None)
            }
            Some(IndexOptions::Vector(vec_opts)) => (None, None, None, Some(vec_opts)),
            None => (None, None, None, None),
        };

        // Pre-validate against the existing entry (if any) *before*
        // creating one. Using `entry(...).or_default()` here would
        // insert an empty `Index` for a previously-unseen label, and
        // a later validation error would leave that empty entry
        // behind — `has_index()` / `has_indices()` would then lie
        // about the label being indexed.
        let existing = index.get(label);

        // Validate language/stopwords are not already set for existing fulltext indexes
        let has_fulltext = existing.is_some_and(Index::has_fulltext_field);

        if has_fulltext {
            if language.is_some() {
                return Err(format!(
                    "Can not override index configuration: Language is already set for label '{label}'"
                ));
            }

            if stopwords.is_some() {
                return Err(format!(
                    "Can not override index configuration: Stopwords are already set for label '{label}'"
                ));
            }
        }

        // For now, field_options is match against full text indexes only
        if field_options.is_some() && *index_type != IndexType::Fulltext {
            return Err("Text index options are only valid for fulltext indexes".into());
        }

        // Pre-validate: check all attrs for conflicts BEFORE inserting any,
        // so that a conflict on a later attribute does not leave earlier
        // attributes partially registered. `attrs` is always small, so
        // the O(n²) prefix scan for intra-request duplicates is fine.
        for (i, attr) in attrs.iter().enumerate() {
            if attrs[..i].contains(attr) {
                // Distinguish from the "already indexed" case below:
                // the attribute isn't indexed yet — the caller
                // listed it twice in the same statement.
                return Err(format!(
                    "Attribute '{attr}' is duplicated in the same request"
                ));
            }
            if existing.is_some_and(|idx| idx.has_field_with_type(attr, index_type)) {
                return Err(format!("Attribute '{attr}' is already indexed"));
            }
        }

        // Validation passed — now it's safe to materialize the entry.
        let label_indexes = index.entry(label.clone()).or_default();

        let mut new_fields: HashMap<Arc<String>, Vec<Arc<Field>>> = HashMap::new();

        for attr in attrs {
            let field_name = match index_type {
                IndexType::Range => Arc::new(format!("range:{attr}")),
                IndexType::Fulltext => attr.clone(),
                IndexType::Vector => Arc::new(format!("vector:{attr}")),
            };

            let field = if let Some(ref vopts) = vector_options {
                Arc::new(Field::new_with_vector_options(
                    CString::new(field_name.as_str()).map_err(|e| e.to_string())?,
                    index_type.clone(),
                    vopts.clone(),
                ))
            } else {
                Arc::new(Field::new(
                    CString::new(field_name.as_str()).map_err(|e| e.to_string())?,
                    index_type.clone(),
                    field_options.clone(),
                ))
            };

            new_fields
                .entry(attr.clone())
                .or_default()
                .push(field.clone());

            if label_indexes.contains_field(attr) {
                label_indexes.add_field_to_existing(attr, field);
            } else {
                label_indexes.insert_field(attr.clone(), field);
            }
        }
        if !label_indexes.has_rs_index() {
            let effective_stopwords = stopwords
                .clone()
                .or_else(|| label_indexes.stopwords().cloned());
            let effective_language = language
                .clone()
                .or_else(|| label_indexes.language().cloned());
            label_indexes.create_rs_index(
                label,
                effective_stopwords.as_ref(),
                effective_language.as_ref(),
            )?;
        }

        label_indexes.register_fields(&new_fields, field_options.as_ref())?;

        // Update the label indexes with global settings
        // Default to "english" for fulltext indexes when no language is specified,
        // matching RediSearch's default behavior.
        if label_indexes.language().is_none() && *index_type == IndexType::Fulltext {
            match language {
                Some(lang) => label_indexes.set_language(Some(lang)),
                None => label_indexes.set_language(Some(Arc::new(String::from("english")))),
            }
        } else if language.is_some() && label_indexes.language().is_none() {
            label_indexes.set_language(language);
        }
        if stopwords.is_some() && label_indexes.stopwords().is_none() {
            label_indexes.set_stopwords(stopwords);
        }

        label_indexes.set_progress(0, total);
        label_indexes.increment_pending();
        Ok(())
    }

    /// Drop index fields and return (dropped_count, remaining_count).
    /// Returns `None` if the label has no index.
    pub fn drop_index(
        &mut self,
        label: &Arc<String>,
        attrs: &Vec<Arc<String>>,
        index_type: &IndexType,
        total: u64,
    ) -> Option<(usize, usize)> {
        let mut index = self.index.write();
        if let Some(index) = index.get_mut(label) {
            let before = index.index_count();
            let mut removed = false;
            for attr in attrs {
                let (has_type, field_count) = if let Some(fields) = index.get_fields(attr) {
                    (fields.iter().any(|f| f.ty == *index_type), fields.len())
                } else {
                    continue;
                };
                if has_type {
                    if field_count == 1 {
                        index.remove_field(attr);
                    } else {
                        index.retain_fields(attr, index_type);
                    }
                    removed = true;
                }
            }
            if removed {
                index.set_progress(0, total);
                index.increment_pending();
            }
            let after = index.index_count();
            return Some((before - after, after));
        }
        drop(index);
        None
    }

    pub fn remove(
        &mut self,
        label: &Arc<String>,
    ) {
        self.index.write().remove(label);
    }

    #[must_use]
    pub fn has_field_for_label(
        &self,
        label: &Arc<String>,
        field: &Arc<String>,
        index_type: &IndexType,
    ) -> bool {
        if let Some(index) = self.index.read().get(label) {
            return index.has_field_with_type(field, index_type);
        }
        false
    }

    #[must_use]
    pub fn is_label_indexed(
        &self,
        label: &Arc<String>,
        field: &Arc<String>,
        index_type: &IndexType,
    ) -> bool {
        if let Some(index) = self.index.read().get(label)
            && index.is_operational()
        {
            return index.has_field_with_type(field, index_type);
        }
        false
    }

    #[must_use]
    pub fn is_attr_indexed(
        &self,
        label: &Arc<String>,
        field: &Arc<String>,
    ) -> bool {
        if let Some(index) = self.index.read().get(label)
            && index.is_operational()
        {
            return index.contains_field(field);
        }
        false
    }

    #[must_use]
    pub fn query(
        &self,
        label: &Arc<String>,
        query: IndexQuery<Value>,
    ) -> IdIter {
        if let Some(index) = self.index.read().get(label) {
            return index.query(query);
        }
        IndexResultsIter::empty()
    }

    /// Like `query`, but for edge indexes: yields `(src, dst, edge_id)`
    /// triples read from the 24-byte document key.
    #[must_use]
    pub fn query_edges(
        &self,
        label: &Arc<String>,
        query: IndexQuery<Value>,
    ) -> super::EdgeTripleIter {
        if let Some(index) = self.index.read().get(label) {
            return index.query_edges(query);
        }
        super::EdgeTripleIter::empty()
    }

    pub fn fulltext_query(
        &self,
        label: &Arc<String>,
        query: &str,
    ) -> Result<ScoredIdIter, String> {
        if let Some(index) = self.index.read().get(label) {
            return index.fulltext_query(query);
        }
        Ok(IndexResultsIter::empty_scored())
    }

    /// Like [`fulltext_query`], but for *edge* indexes: yields
    /// `(src, dst, edge_id, score)` tuples read from the 24-byte
    /// document key plus the RediSearch relevance score.
    pub fn fulltext_query_edges(
        &self,
        label: &Arc<String>,
        query: &str,
    ) -> Result<super::ScoredEdgeTripleIter, String> {
        if let Some(index) = self.index.read().get(label) {
            return index.fulltext_query_edges(query);
        }
        Ok(super::ScoredEdgeTripleIter::empty())
    }

    pub fn enable(
        &mut self,
        label: &Arc<String>,
    ) -> bool {
        let index = self.index.read();
        if let Some(index) = index.get(label) {
            let res = index.decrement_pending();
            debug_assert!(res > 0);
            return res == 1;
        }
        drop(index);
        false
    }

    pub fn disable(
        &mut self,
        label: &Arc<String>,
    ) {
        let mut index = self.index.write();
        if let Some(index) = index.get_mut(label) {
            index.increment_pending();
        }
    }

    #[must_use]
    pub fn enabled(
        &self,
        label: &Arc<String>,
    ) -> bool {
        if let Some(index) = self.index.read().get(label) {
            return index.pending_count() == 0;
        }
        false
    }

    #[must_use]
    pub fn pending_changes(
        &self,
        label: &Arc<String>,
    ) -> i32 {
        if let Some(index) = self.index.read().get(label) {
            return index.pending_count();
        }
        0
    }

    pub fn commit(
        &mut self,
        add_docs: &mut HashMap<Arc<String>, Vec<Document>>,
        remove_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
    ) {
        let mut index = self.index.write();
        for (label, add_docs) in add_docs {
            let Some(index) = index.get_mut(label) else {
                continue;
            };
            for doc in add_docs.drain(..) {
                index.add_document(&doc);
            }
        }
        for (label, remove_docs) in remove_docs {
            let Some(index) = index.get_mut(label) else {
                continue;
            };
            for id in remove_docs.iter() {
                index.delete_document(id);
            }
        }
        drop(index);
    }

    /// Edge-index variant of `commit`: adds documents built with
    /// `Document::new_edge`, deletes by the 24-byte `[src, dst, edge_id]`
    /// key. Callers pass the delete set as
    /// `relationship-type-name → { edge_id → (src, dst) }`; the
    /// type-id → name conversion is done upstream in
    /// `Graph::commit_edge_index` before this call.
    pub fn commit_edge(
        &mut self,
        add_docs: &mut HashMap<Arc<String>, Vec<Document>>,
        remove_docs: &mut HashMap<Arc<String>, std::collections::HashMap<u64, (u64, u64)>>,
    ) {
        let mut index = self.index.write();
        for (label, add_docs) in add_docs {
            let Some(index) = index.get_mut(label) else {
                continue;
            };
            for doc in add_docs.drain(..) {
                index.add_document(&doc);
            }
        }
        for (label, edges) in remove_docs {
            let Some(index) = index.get_mut(label) else {
                continue;
            };
            for (&edge_id, &(src, dst)) in edges.iter() {
                index.delete_edge_document(src, dst, edge_id);
            }
            edges.clear();
        }
        drop(index);
    }

    #[must_use]
    pub fn get_fields(
        &self,
        label: &Arc<String>,
    ) -> HashMap<Arc<String>, Vec<Arc<Field>>> {
        self.index
            .read()
            .get(label)
            .map(|index| index.fields().clone())
            .unwrap_or_default()
    }

    /// Get fields for all labels with pending population.
    #[must_use]
    pub fn get_all_pending_fields(
        &self
    ) -> Vec<(Arc<String>, HashMap<Arc<String>, Vec<Arc<Field>>>)> {
        self.index
            .read()
            .iter()
            .filter(|(_, index)| index.pending_count() > 0)
            .map(|(label, index)| (label.clone(), index.fields().clone()))
            .collect()
    }

    #[must_use]
    pub fn index_info(&self) -> Vec<IndexInfo> {
        self.index
            .read()
            .iter()
            .filter(|(_, index)| !index.is_empty())
            .map(|(label, index)| {
                let (progress, total) = index.progress();
                IndexInfo {
                    label: label.clone(),
                    pending: index.pending_count(),
                    progress,
                    total,
                    fields: index.fields().clone(),
                    language: index.language().cloned(),
                    stopwords: index.stopwords().cloned(),
                    entity_type: String::new(),
                }
            })
            .collect()
    }

    #[must_use]
    pub fn has_index(
        &self,
        label: &Arc<String>,
    ) -> bool {
        self.index.read().contains_key(label)
    }

    #[must_use]
    pub fn has_indexed_attr(
        &self,
        label: &Arc<String>,
        field: &Arc<String>,
    ) -> bool {
        if let Some(index) = self.index.read().get(label) {
            return index.contains_field(field);
        }
        false
    }

    pub fn update_progress(
        &self,
        label: &Arc<String>,
        progress: u64,
    ) {
        let mut index = self.index.write();
        if let Some(index) = index.get_mut(label) {
            let (_, total) = index.progress();
            index.set_progress(progress, total);
        }
    }

    /// Get a clone of the serialization lock for index mutations.
    ///
    /// Used by background index population and `commit_index` to serialize
    /// their index mutations so they never run concurrently.  Returns a
    /// cloned `Arc` so the caller can lock it without borrowing `self`.
    #[must_use]
    pub fn write_lock(&self) -> Arc<Mutex<()>> {
        self.write_lock.clone()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        // Clear the graph reference to break the circular Arc:
        // MvccGraph → Arc<Graph> → Indexer → Arc<Graph>
        *self.graph.lock() = None;
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    pub fn recreate_index(
        &mut self,
        label: &Arc<String>,
    ) -> Result<(), String> {
        let mut index = self.index.write();
        if let Some(index) = index.get_mut(label) {
            index.recreate_index(label)?;
        }
        drop(index);
        Ok(())
    }

    pub fn set_graph(
        &mut self,
        graph: Arc<AtomicRefCell<Graph>>,
    ) {
        *self.graph.lock() = Some(graph);
    }

    #[must_use]
    pub fn get_graph(&self) -> Option<Arc<AtomicRefCell<Graph>>> {
        self.graph.lock().clone()
    }
}
