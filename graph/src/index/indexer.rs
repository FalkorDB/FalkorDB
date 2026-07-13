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
//!    +-- index: ArcSwap<HashMap<Label, Arc<Index>>>
//!    |      One Index per label; each Index wraps a single
//!    |      RSIndex handle and its field definitions.
//!    |      Readers `load()` a lock-free snapshot; schema changes
//!    |      clone the map, modify the copy, and `store()` it.
//!    |
//!    +-- write_lock: Mutex<()>
//!    |      Serializes map mutations (create/drop/remove/recreate)
//!    |      and background population batches with write-path
//!    |      commit_index calls so they never run concurrently.
//!    |
//!    +-- graph: Mutex<Option<Arc<Graph>>>
//!           Latest committed graph snapshot shared with
//!           background index population threads.
//! ```
//!
//! # Concurrency
//!
//! Read-side queries (`query`, `fulltext_query`, `is_label_indexed`, ...)
//! `load()` the current map snapshot without locking. Schema mutations
//! (`create_index`, `drop_index`, ...) publish a new map via clone-and-swap
//! while holding `write_lock`; per-document mutations (`commit`, ...) call
//! `&self` methods on the internally-synchronized RediSearch spec.
//! Background population uses `write_lock` to avoid racing with
//! per-transaction commit calls and schema changes.

use std::{
    collections::HashMap,
    ffi::CString,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use arc_swap::{ArcSwap, ArcSwapOption};
use atomic_refcell::AtomicRefCell;
use parking_lot::Mutex;
use roaring::RoaringTreemap;

use super::Index;
pub use super::{
    Document, Field, IdIter, IndexInfo, IndexQuery, IndexResultsIter, IndexType, ScoredIdIter,
    TextIndexOptions, VectorIndexOptions, VectorScoredEdgeTripleIter, VectorScoredIdIter,
};
use crate::{graph::graph::Graph, runtime::value::Value};

pub enum IndexOptions {
    Text(TextIndexOptions),
    Vector(VectorIndexOptions),
}

#[derive(Clone, Debug)]
pub struct PopulationTicket {
    label: Arc<String>,
    generation_id: u64,
}

#[derive(Clone, Debug)]
pub struct PopulationSnapshot {
    pub fields: HashMap<Arc<String>, Vec<Arc<Field>>>,
    pub ticket: PopulationTicket,
}

impl PopulationTicket {
    #[must_use]
    pub const fn generation_id(&self) -> u64 {
        self.generation_id
    }

    #[must_use]
    pub const fn label(&self) -> &Arc<String> {
        &self.label
    }
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
    /// Lock-free snapshot of the per-label indexes. Readers `load()` the
    /// current map; schema mutations clone the map, modify the copy (via
    /// [`Index::clone_for_update`]), and `store()` it under `write_lock`.
    index: Arc<ArcSwap<HashMap<Arc<String>, Arc<Index>>>>,
    /// Serializes map mutations and background index population batches with
    /// write-path `commit_index` calls so they never run concurrently.
    write_lock: Arc<Mutex<()>>,
    cancelled: Arc<AtomicBool>,
    /// Latest committed graph, shared with background index population.
    /// Updated by `MvccGraph::commit()` so background batches see fresh data.
    graph: Arc<ArcSwapOption<AtomicRefCell<Graph>>>,
}

unsafe impl Send for Indexer {}
unsafe impl Sync for Indexer {}

impl Indexer {
    #[must_use]
    pub fn has_indices(&self) -> bool {
        !self.index.load().is_empty()
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.index
            .load()
            .values()
            .map(|index| index.memory_usage())
            .sum()
    }

    pub fn create_index(
        &self,
        index_type: &IndexType,
        label: &Arc<String>,
        attrs: &Vec<Arc<String>>,
        total: u64,
        options: Option<IndexOptions>,
    ) -> Result<(), String> {
        // Serialize with any in-flight populate batch and other schema
        // mutations. Without this, a recreate (e.g. adding a second vector
        // field) can land between a batch's id check and its commit, causing
        // the batch to flush docs built against a stale field set into
        // the freshly recreated rs_idx — corrupting HNSW state.
        let _guard = self.write_lock.lock();
        let map = self.index.load_full();

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
        // materializing a new one, so a validation error publishes nothing —
        // `has_index()` / `has_indices()` never lie about the label being
        // indexed.
        let existing = map.get(label).map(Arc::as_ref);

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

        // Validation passed — materialize a private copy of the entry
        // (sharing the RS spec handle and pending counters with the
        // published generation) to mutate and publish below.
        let mut label_indexes = existing.map_or_else(Index::default, Index::clone_for_update);

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
        if label_indexes.has_rs_index() {
            // Adding fields to an existing RediSearch index. For most
            // field types we append to the live spec, but adding a
            // vector field forces a drop+rebuild: RediSearch's HNSW
            // state on the existing vector field gets corrupted when
            // the subsequent populate phase re-adds documents with
            // ADD_REPLACE, leaving KNN queries returning fewer hits
            // than the data warrants. Matches the C FalkorDB behavior
            // (Index_Disable + Index_ConstructStructure).
            let adds_vector = *index_type == IndexType::Vector;
            if adds_vector {
                label_indexes.recreate_index(label)?;
            } else {
                label_indexes.register_fields(&new_fields, field_options.as_ref())?;
            }
        } else {
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
            label_indexes.register_fields(&new_fields, field_options.as_ref())?;
        }

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

        let mut new_map = (*map).clone();
        new_map.insert(label.clone(), Arc::new(label_indexes));
        self.index.store(Arc::new(new_map));
        Ok(())
    }

    /// Drop index fields and return (dropped_count, remaining_count).
    /// Returns `None` if the label has no index.
    ///
    /// Caller must hold [`Self::write_lock`] — map mutations are published
    /// via clone-and-swap and would otherwise race with other schema ops.
    pub fn drop_index(
        &self,
        label: &Arc<String>,
        attrs: &[Arc<String>],
        index_type: &IndexType,
        total: u64,
    ) -> Option<(usize, usize)> {
        let map = self.index.load_full();
        let mut index = map.get(label)?.clone_for_update();
        let before = index.index_count();
        let mut removed = false;
        // Empty `attrs` means "drop all fields of this index_type"
        // (e.g. db.idx.fulltext.drop('L') drops every fulltext field
        // on label L without requiring callers to enumerate them).
        let target_attrs: Vec<Arc<String>> = if attrs.is_empty() {
            index
                .fields()
                .iter()
                .filter(|(_, fields)| fields.iter().any(|f| f.ty == *index_type))
                .map(|(attr, _)| attr.clone())
                .collect()
        } else {
            attrs.to_vec()
        };
        for attr in &target_attrs {
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
        }
        let after = index.index_count();
        let mut new_map = (*map).clone();
        new_map.insert(label.clone(), Arc::new(index));
        self.index.store(Arc::new(new_map));
        Some((before - after, after))
    }

    /// Remove the whole per-label index entry.
    ///
    /// Caller must hold [`Self::write_lock`] (see `drop_index_bg`).
    pub fn remove(
        &self,
        label: &Arc<String>,
    ) {
        let map = self.index.load_full();
        if !map.contains_key(label) {
            return;
        }
        let mut new_map = (*map).clone();
        new_map.remove(label);
        self.index.store(Arc::new(new_map));
    }

    #[must_use]
    pub fn has_field_for_label(
        &self,
        label: &Arc<String>,
        field: &Arc<String>,
        index_type: &IndexType,
    ) -> bool {
        if let Some(index) = self.index.load().get(label) {
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
        if let Some(index) = self.index.load().get(label)
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
        if let Some(index) = self.index.load().get(label)
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
        if let Some(index) = self.index.load().get(label) {
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
        if let Some(index) = self.index.load().get(label) {
            return index.query_edges(query);
        }
        super::EdgeTripleIter::empty()
    }

    pub fn fulltext_query(
        &self,
        label: &Arc<String>,
        query: &str,
    ) -> Result<ScoredIdIter, String> {
        if let Some(index) = self.index.load().get(label) {
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
        if let Some(index) = self.index.load().get(label) {
            return index.fulltext_query_edges(query);
        }
        Ok(super::ScoredEdgeTripleIter::empty())
    }

    /// Execute a KNN vector query against the per-label index and
    /// yield `(entity_id, distance)` pairs ordered by ascending
    /// distance. Returns an empty iterator if no index exists for
    /// `label`.
    ///
    /// Takes the vector by `Arc` so the underlying `f32` buffer can be
    /// pinned for the iterator's lifetime — see
    /// [`super::Index::vector_query`].
    pub fn vector_query(
        &self,
        label: &Arc<String>,
        field: &str,
        vector: Arc<thin_vec::ThinVec<f32>>,
        k: usize,
    ) -> Result<VectorScoredIdIter, String> {
        if let Some(index) = self.index.load().get(label) {
            return index.vector_query(field, vector, k);
        }
        Ok(VectorScoredIdIter::empty(vector))
    }

    /// Like [`vector_query`], but for *edge* indexes: yields
    /// `(src, dst, edge_id, distance)` tuples.
    pub fn vector_query_edges(
        &self,
        label: &Arc<String>,
        field: &str,
        vector: Arc<thin_vec::ThinVec<f32>>,
        k: usize,
    ) -> Result<VectorScoredEdgeTripleIter, String> {
        if let Some(index) = self.index.load().get(label) {
            return index.vector_query_edges(field, vector, k);
        }
        Ok(VectorScoredEdgeTripleIter::empty(vector))
    }

    /// Look up the similarity-function name (`"euclidean"`,
    /// `"cosine"`, or `"ip"`) for the vector field on the given label.
    /// Returns `None` if the label has no index, the attribute has no
    /// vector field, or the field is missing similarity metadata.
    /// Used by the runtime to compute distances when materializing
    /// KNN results — RediSearch's iterator-level score is not the
    /// distance, so the caller must compute it manually using the
    /// query and entity vectors plus this metric.
    #[must_use]
    pub fn get_vector_metric(
        &self,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<String> {
        let guard = self.index.load();
        let index = guard.get(label)?;
        let fields = index.get_fields(attr)?;
        for field in fields {
            if field.ty == super::IndexType::Vector
                && let Some(opts) = field.vector_options()
                && let Some(sim) = opts.similarity_function.as_ref()
            {
                return Some(sim.clone());
            }
        }
        None
    }

    /// Look up the configured dimension for the vector field on the
    /// given label, if any. Returns `None` if no such vector field
    /// exists on the index.
    #[must_use]
    pub fn get_vector_dimension(
        &self,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<u32> {
        let guard = self.index.load();
        let index = guard.get(label)?;
        let fields = index.get_fields(attr)?;
        for field in fields {
            if field.ty == super::IndexType::Vector
                && let Some(opts) = field.vector_options()
            {
                return Some(opts.dimension);
            }
        }
        None
    }

    /// Reserve one population ticket for the current generation of `label`.
    ///
    /// The returned ticket is generation-scoped: releasing it later decrements
    /// only that generation's pending counter, even if the label was recreated.
    #[must_use]
    pub fn acquire_population_ticket(
        &self,
        label: &Arc<String>,
    ) -> Option<PopulationTicket> {
        let index = self.index.load();
        let index = index.get(label)?;
        let generation_id = index.id();
        index.increment_pending_for_generation(generation_id);
        Some(PopulationTicket {
            label: label.clone(),
            generation_id,
        })
    }

    /// Capture a field snapshot and ticket for one label while holding the
    /// same index read lock, so the schema used to build documents matches
    /// the generation the ticket was taken from.
    #[must_use]
    pub fn acquire_population_snapshot(
        &self,
        label: &Arc<String>,
    ) -> Option<PopulationSnapshot> {
        let index = self.index.load();
        let index = index.get(label)?;
        let generation_id = index.id();
        let fields = index.fields().clone();
        index.increment_pending_for_generation(generation_id);
        Some(PopulationSnapshot {
            fields,
            ticket: PopulationTicket {
                label: label.clone(),
                generation_id,
            },
        })
    }

    /// Capture field snapshots and tickets for all currently indexed labels.
    /// Used by synchronous population so each label's field set is coupled to
    /// the ticket acquired for the same read-side snapshot.
    #[must_use]
    pub fn acquire_population_snapshots(&self) -> Vec<PopulationSnapshot> {
        self.index
            .load()
            .iter()
            .map(|(label, index)| {
                let generation_id = index.id();
                index.increment_pending_for_generation(generation_id);
                PopulationSnapshot {
                    fields: index.fields().clone(),
                    ticket: PopulationTicket {
                        label: label.clone(),
                        generation_id,
                    },
                }
            })
            .collect()
    }

    /// Release a previously-acquired population ticket.
    ///
    /// Saturating for that ticket's generation: if the counter was already 0,
    /// no decrement happens.
    pub fn release_population_ticket(
        &self,
        ticket: &PopulationTicket,
    ) {
        let index = self.index.load();
        if let Some(index) = index.get(ticket.label()) {
            index.try_decrement_pending_for_generation(ticket.generation_id());
        }
    }

    /// Returns true if `ticket` still targets the current generation.
    #[must_use]
    pub fn is_ticket_current(
        &self,
        ticket: &PopulationTicket,
    ) -> bool {
        self.index
            .load()
            .get(ticket.label())
            .is_some_and(|index| index.id() == ticket.generation_id())
    }

    /// Return pending count for the ticket's generation.
    #[must_use]
    pub fn ticket_pending_changes(
        &self,
        ticket: &PopulationTicket,
    ) -> i32 {
        self.index.load().get(ticket.label()).map_or(0, |index| {
            index.pending_count_for_generation(ticket.generation_id())
        })
    }

    #[must_use]
    pub fn enabled(
        &self,
        label: &Arc<String>,
    ) -> bool {
        if let Some(index) = self.index.load().get(label) {
            return index.pending_count() == 0;
        }
        false
    }

    pub fn commit(
        &self,
        add_docs: &mut HashMap<Arc<String>, Vec<Document>>,
        remove_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
    ) {
        let index = self.index.load();
        for (label, add_docs) in add_docs {
            let Some(index) = index.get(label) else {
                continue;
            };
            for mut doc in add_docs.drain(..) {
                index.add_document(&mut doc);
            }
        }
        for (label, remove_docs) in remove_docs {
            let Some(index) = index.get(label) else {
                continue;
            };
            for id in remove_docs.iter() {
                index.delete_document(id);
            }
        }
    }

    /// Edge-index variant of `commit`: adds documents built with
    /// `Document::new_edge`, deletes by the 24-byte `[src, dst, edge_id]`
    /// key. Callers pass the delete set as
    /// `relationship-type-name → { edge_id → (src, dst) }`; the
    /// type-id → name conversion is done upstream in
    /// `Graph::commit_edge_index` before this call.
    pub fn commit_edge(
        &self,
        add_docs: &mut HashMap<Arc<String>, Vec<Document>>,
        remove_docs: &mut HashMap<Arc<String>, std::collections::HashMap<u64, (u64, u64)>>,
    ) {
        let index = self.index.load();
        for (label, add_docs) in add_docs {
            let Some(index) = index.get(label) else {
                continue;
            };
            for mut doc in add_docs.drain(..) {
                index.add_document(&mut doc);
            }
        }
        for (label, edges) in remove_docs {
            let Some(index) = index.get(label) else {
                continue;
            };
            for (&edge_id, &(src, dst)) in edges.iter() {
                index.delete_edge_document(src, dst, edge_id);
            }
            edges.clear();
        }
    }

    #[must_use]
    pub fn get_fields(
        &self,
        label: &Arc<String>,
    ) -> HashMap<Arc<String>, Vec<Arc<Field>>> {
        self.index
            .load()
            .get(label)
            .map(|index| index.fields().clone())
            .unwrap_or_default()
    }

    /// Get fields for all labels.
    #[must_use]
    pub fn get_all_fields(&self) -> Vec<(Arc<String>, HashMap<Arc<String>, Vec<Arc<Field>>>)> {
        self.index
            .load()
            .iter()
            .filter(|(_, index)| !index.is_empty())
            .map(|(label, index)| (label.clone(), index.fields().clone()))
            .collect()
    }

    #[must_use]
    pub fn index_info(&self) -> Vec<IndexInfo> {
        // Lock-free snapshot: safe to call from a BGSAVE fork child even if
        // the parent's writer thread was mid-commit at fork time.
        let mut infos: Vec<IndexInfo> = self
            .index
            .load()
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
                    field_order: index.field_order().to_vec(),
                    language: index.language().cloned(),
                    stopwords: index.stopwords().cloned(),
                    entity_type: String::new(),
                }
            })
            .collect();
        infos.sort_by(|a, b| a.label.cmp(&b.label));
        infos
    }

    #[must_use]
    pub fn has_index(
        &self,
        label: &Arc<String>,
    ) -> bool {
        self.index.load().contains_key(label)
    }

    #[must_use]
    pub fn has_indexed_attr(
        &self,
        label: &Arc<String>,
        field: &Arc<String>,
    ) -> bool {
        if let Some(index) = self.index.load().get(label) {
            return index.contains_field(field);
        }
        false
    }

    pub fn update_progress(
        &self,
        label: &Arc<String>,
        progress: u64,
    ) {
        if let Some(index) = self.index.load().get(label) {
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
        self.graph.store(None);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Rebuild the RediSearch spec for `label` from scratch (drop + create +
    /// re-register fields), bumping the generation id so stale populate
    /// workers bail out.
    ///
    /// Caller must hold [`Self::write_lock`] — the new generation is
    /// published via clone-and-swap.
    pub fn recreate_index(
        &self,
        label: &Arc<String>,
    ) -> Result<(), String> {
        let map = self.index.load_full();
        if let Some(index) = map.get(label) {
            let mut index = index.clone_for_update();
            index.recreate_index(label)?;
            let mut new_map = (*map).clone();
            new_map.insert(label.clone(), Arc::new(index));
            self.index.store(Arc::new(new_map));
        }
        Ok(())
    }

    /// Only touches the `Indexer`'s own lock-free `graph` slot, so this takes
    /// `&self` deliberately: it must be callable while the caller holds only
    /// an immutable borrow of the enclosing `Graph` (see
    /// `Graph::set_indexer_graph`), so that background index population
    /// (which also just needs an immutable borrow of the committed graph)
    /// never races with this call and panics with "already mutably
    /// borrowed".
    pub fn set_graph(
        &self,
        graph: Arc<AtomicRefCell<Graph>>,
    ) {
        self.graph.store(Some(graph));
    }

    #[must_use]
    pub fn get_graph(&self) -> Option<Arc<AtomicRefCell<Graph>>> {
        self.graph.load_full()
    }
}
