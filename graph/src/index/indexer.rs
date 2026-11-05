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
//! Every graph *version* owns an `Indexer`; versions share only the
//! [`Shared`] part.
//!
//! ```text
//! Indexer (one per graph version)
//!    |
//!    +-- index: ArcSwap<HashMap<Label, Arc<Index>>>
//!    |      THIS VERSION's indexes, one Index per label; each Index owns a
//!    |      RSIndex handle and its field definitions. A fork starts with
//!    |      an Arc clone of the committed map; schema changes clone the
//!    |      map, modify the copy and `store()` it, touching only the fork.
//!    |      Committing the fork publishes the map with the graph;
//!    |      dropping it (rollback) drops the DDL with it.
//!    |
//!    +-- shared: Arc<Shared>   (the same for every version of the graph)
//!           write_lock   serializes background population batches with
//!                        write-path commit_index calls
//!           cancelled    stops background population
//!           graph        latest committed graph, for population
//! ```
//!
//! # Concurrency
//!
//! Read-side queries (`query`, `fulltext_query`, `is_label_indexed`, ...)
//! `load()` the version's map snapshot without locking. Schema mutations
//! (`create_index`, `drop_index`) never touch anything another version can
//! see: a changed label gets a new [`Index`] with its own RediSearch spec,
//! generation id and pending-ticket counters. What must happen only if the
//! version is published (starting population) is
//! found by [`Indexer::changes_since`] and done by the publisher. A retired spec
//! is freed by whoever drops the last reference to it (see `SpecHandle`).
//! Per-document mutations (`commit`, ...) call `&self` methods on the
//! internally-synchronized RediSearch spec. Background population works on
//! the map of the committed graph ([`IndexerHandle::get_graph`]) and uses
//! `write_lock` to avoid racing with per-transaction commit calls.

use std::{
    collections::{HashMap, HashSet},
    ffi::CString,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use arc_swap::ArcSwapOption;
use atomic_refcell::AtomicRefCell;
use parking_lot::Mutex;
use roaring::RoaringTreemap;

use super::Index;
pub use super::{
    Document, Field, IdIter, IndexInfo, IndexQuery, IndexResultsIter, IndexType, ScoredIdIter,
    TextIndexOptions, VectorIndexOptions, VectorScoredEdgeTripleIter, VectorScoredIdIter,
};
use crate::{graph::graph::Graph, runtime::value::Value};

#[derive(Clone)]
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
                        phonetic: opts.phonetic.clone(),
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

type IndexMap = HashMap<Arc<String>, Arc<Index>>;

/// What every version of one graph's `Indexer` shares.
struct Shared {
    /// Serializes background index population batches with write-path
    /// `commit_index` calls so they never run concurrently.
    write_lock: Arc<Mutex<()>>,
    cancelled: AtomicBool,
    /// Latest committed graph, shared with background index population.
    /// Updated by `MvccGraph::commit()` so background batches see fresh data.
    graph: ArcSwapOption<AtomicRefCell<Graph>>,
}

/// The indexes of one graph version. See the module docs.
///
/// `Clone` is a *fork*: the clone starts with this version's map (one `Arc`
/// clone) and changes it without affecting the original.
pub struct Indexer {
    /// This version's per-label indexes. Schema mutations (`&mut self`, on a
    /// version nobody else can see) clone the map, modify the copy and replace
    /// this pointer. A plain `Arc`, not an `ArcSwap`: forking a version creates
    /// one of these per indexer, and dropping an `ArcSwap` waits out every
    /// thread's readers.
    index: Arc<IndexMap>,
    shared: Arc<Shared>,
}

impl Default for Indexer {
    fn default() -> Self {
        Self {
            index: Arc::new(HashMap::new()),
            shared: Arc::new(Shared {
                write_lock: Arc::new(Mutex::new(())),
                cancelled: AtomicBool::new(false),
                graph: ArcSwapOption::empty(),
            }),
        }
    }
}

impl Clone for Indexer {
    fn clone(&self) -> Self {
        Self {
            index: Arc::clone(&self.index),
            shared: self.shared.clone(),
        }
    }
}

/// A reference to what all versions of one `Indexer` share, which holds no
/// version's map: background work keeps it, and reads the committed graph's map per
/// batch (from [`Self::get_graph`]) instead of pinning an old one (and with it an
/// old, possibly retired, RediSearch spec).
#[derive(Clone)]
pub struct IndexerHandle(Arc<Shared>);

impl IndexerHandle {
    /// The latest committed graph, `None` until the first commit.
    #[must_use]
    pub fn get_graph(&self) -> Option<Arc<AtomicRefCell<Graph>>> {
        self.0.graph.load_full()
    }

    /// Whether the graph is going away: no more population.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::Relaxed)
    }

    /// The serialization lock, see [`Indexer::write_lock`].
    #[must_use]
    pub fn write_lock(&self) -> Arc<Mutex<()>> {
        self.0.write_lock.clone()
    }
}

/// What publishing a version changed in its indexes, found by comparing its map
/// with the previous published one ([`Indexer::changes_since`]).
#[derive(Default)]
pub struct IndexChanges {
    /// Labels whose index the version created or rebuilt and that still owe
    /// their first population: the publisher starts it.
    pub owed: Vec<Arc<String>>,
}

impl IndexChanges {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.owed.is_empty()
    }
}

unsafe impl Send for Indexer {}
unsafe impl Sync for Indexer {}

impl Indexer {
    #[must_use]
    pub fn has_indices(&self) -> bool {
        !self.index.is_empty()
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.index.values().map(|index| index.memory_usage()).sum()
    }

    pub fn create_index(
        &mut self,
        index_type: &IndexType,
        label: &Arc<String>,
        attrs: &Vec<Arc<String>>,
        total: u64,
        options: Option<IndexOptions>,
        owes_population: bool,
    ) -> Result<(), String> {
        // No `write_lock`: this changes only this version's map, with an `Index`
        // and RediSearch spec nothing else can reach, so it cannot race with a
        // population batch (which works on the committed graph's map and its specs).
        // `owes_population`: the index is left UNDER CONSTRUCTION and whoever
        // publishes this version starts its population
        // ([`Self::changes_since`]); `false` for a caller that populates it
        // itself before the version is published (RDB load).
        let map = Arc::clone(&self.index);

        let (language, stopwords, field_options, vector_options) = match options {
            Some(IndexOptions::Text(text_opts)) => {
                let language = text_opts.language.clone();
                // RediSearch lowercases stopwords internally; mirror that in
                // the index metadata so db.indexes() and the RDB encoding
                // report what the index actually matches on.
                let stopwords = text_opts.stopwords.as_ref().map(|words| {
                    words
                        .iter()
                        .map(|w| Arc::new(w.to_lowercase()))
                        .collect::<Vec<_>>()
                });
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
        // attributes partially registered.
        let mut seen = HashSet::with_capacity(attrs.len());
        for attr in attrs {
            if !seen.insert(attr) {
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

        // Validation passed — materialize a private copy of the entry, with its own
        // generation and pending counters, to mutate and publish below. It never changes
        // the published version's spec while the version is unpublished: a version that is
        // rolled back must leave it as it was.
        //
        // A statement that only adds fields shares that spec (the fields go into it when
        // the version is published), so the fields it already has stay queryable while the
        // new ones populate. Otherwise the copy gets a fresh spec. That is needed when a vector
        // field is added (adding one to a live spec corrupts its HNSW state, see below), or
        // when the statement sets a language or stopword list (a spec has them from its
        // creation), and for a caller that populates synchronously, which no publish
        // follows.
        let share_spec = owes_population
            && *index_type != IndexType::Vector
            && language.is_none()
            && stopwords.is_none()
            && existing.is_some_and(Index::has_rs_index);
        let mut label_indexes =
            existing.map_or_else(Index::default, |e| e.fork_for_ddl(share_spec));

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
                    *index_type,
                    vopts.clone(),
                ))
            } else {
                Arc::new(Field::new(
                    CString::new(field_name.as_str()).map_err(|e| e.to_string())?,
                    *index_type,
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
        if share_spec {
            label_indexes.defer_fields(new_fields.clone(), field_options.clone());
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
            label_indexes.register_all_fields()?;
        }

        // Record the settings the spec above was built with. An explicit language or
        // stopword list reaches here only when the label has no fulltext field left
        // (the check at the top refuses an override otherwise), and the spec uses it, so
        // the metadata `db.indexes()` reports and every later rebuild starts from must
        // follow. Default to "english" for fulltext indexes when no language is
        // specified, matching RediSearch's default behavior.
        if let Some(lang) = language {
            label_indexes.set_language(Some(lang));
        } else if label_indexes.language().is_none() && *index_type == IndexType::Fulltext {
            label_indexes.set_language(Some(Arc::new(String::from("english"))));
        }
        if stopwords.is_some() {
            label_indexes.set_stopwords(stopwords);
        }

        label_indexes.set_progress(0, total);

        // The population is owed from here: the generation `fork_for_ddl` gave
        // this entry is new, so a job still running for the published one holds
        // a field snapshot without the new field and bails at its next batch
        // once this version is published (`is_ticket_current`).
        if owes_population {
            label_indexes.owe_population();
        }

        let mut new_map = (*map).clone();
        new_map.insert(label.clone(), Arc::new(label_indexes));
        self.index = Arc::new(new_map);
        Ok(())
    }

    /// Drop index fields and return (dropped_count, remaining_count).
    /// Returns `None` if the label has no index.
    ///
    /// Changes only this version's map. With fields left, the label gets a new
    /// `Index` with a fresh RediSearch spec holding them, which owes its
    /// population; with none left the label leaves the map. The published
    /// version's index is untouched either way.
    ///
    /// # Errors
    ///
    /// If RediSearch cannot build the spec for the fields left. The map is
    /// unchanged.
    pub fn drop_index(
        &mut self,
        label: &Arc<String>,
        attrs: &[Arc<String>],
        index_type: &IndexType,
        total: u64,
    ) -> Result<Option<(usize, usize)>, String> {
        let map = Arc::clone(&self.index);
        let Some(existing) = map.get(label) else {
            return Ok(None);
        };
        let mut index = existing.fork_for_ddl(false);
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
        if !removed {
            return Ok(Some((0, before)));
        }
        let after = index.index_count();
        let mut new_map = (*map).clone();
        if after == 0 {
            new_map.remove(label);
        } else {
            index.recreate_index(label)?;
            index.set_progress(0, total);
            index.owe_population();
            new_map.insert(label.clone(), Arc::new(index));
        }
        self.index = Arc::new(new_map);
        Ok(Some((before - after, after)))
    }

    /// What publishing this version, whose predecessor is `previous`, changed:
    /// the labels whose index it built and that await population. Empty (one pointer comparison) when this
    /// version's map is the predecessor's, as it is for any version that ran no
    /// index DDL.
    ///
    /// Takes each owed population ([`Index::take_population_owed`]), so each is
    /// reported once; the caller starts them.
    #[must_use]
    pub fn changes_since(
        &self,
        previous: &Self,
    ) -> IndexChanges {
        let new = &self.index;
        let old = &previous.index;
        if Arc::ptr_eq(new, old) {
            return IndexChanges::default();
        }
        let mut changes = IndexChanges::default();
        for (label, index) in new.iter() {
            if old.get(label).is_some_and(|o| Arc::ptr_eq(o, index)) {
                continue;
            }
            if index.take_population_owed() {
                changes.owed.push(label.clone());
            }
        }
        changes
    }

    /// Register the fields `label`'s index added to a spec it shares with the previous
    /// version ([`Index::defer_fields`]). For the publisher, before it starts the
    /// population that fills them.
    pub fn register_deferred_fields(
        &self,
        label: &Arc<String>,
    ) {
        if let Some(index) = self.index.get(label) {
            index.register_deferred_fields();
        }
    }

    /// The field snapshot and ticket of the population `label`'s index owes,
    /// which the DDL statement already took the ticket of
    /// ([`Index::owe_population`]): no new ticket is acquired.
    #[must_use]
    pub fn owed_population_snapshot(
        &self,
        label: &Arc<String>,
    ) -> Option<PopulationSnapshot> {
        let map = &self.index;
        let index = map.get(label)?;
        Some(PopulationSnapshot {
            fields: index.fields().clone(),
            ticket: PopulationTicket {
                label: label.clone(),
                generation_id: index.id(),
            },
        })
    }

    /// What background work keeps of this indexer: no version's map.
    #[must_use]
    pub fn handle(&self) -> IndexerHandle {
        IndexerHandle(self.shared.clone())
    }

    #[must_use]
    pub fn has_field_for_label(
        &self,
        label: &Arc<String>,
        field: &Arc<String>,
        index_type: &IndexType,
    ) -> bool {
        if let Some(index) = self.index.get(label) {
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
        if let Some(index) = self.index.get(label)
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
        if let Some(index) = self.index.get(label)
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
        if let Some(index) = self.index.get(label) {
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
        if let Some(index) = self.index.get(label) {
            return index.query_edges(query);
        }
        super::EdgeTripleIter::empty()
    }

    pub fn fulltext_query(
        &self,
        label: &Arc<String>,
        query: &str,
    ) -> Result<ScoredIdIter, String> {
        if let Some(index) = self.index.get(label) {
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
        if let Some(index) = self.index.get(label) {
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
        if let Some(index) = self.index.get(label) {
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
        if let Some(index) = self.index.get(label) {
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
        let guard = &self.index;
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
    ) -> Option<u64> {
        let guard = &self.index;
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
        let index = &self.index;
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
        let index = &self.index;
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
        let index = &self.index;
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
            .get(ticket.label())
            .is_some_and(|index| index.id() == ticket.generation_id())
    }

    /// Return pending count for the ticket's generation.
    #[must_use]
    pub fn ticket_pending_changes(
        &self,
        ticket: &PopulationTicket,
    ) -> i32 {
        self.index.get(ticket.label()).map_or(0, |index| {
            index.pending_count_for_generation(ticket.generation_id())
        })
    }

    #[must_use]
    pub fn enabled(
        &self,
        label: &Arc<String>,
    ) -> bool {
        if let Some(index) = self.index.get(label) {
            return index.pending_count() == 0;
        }
        false
    }

    pub fn commit(
        &self,
        add_docs: &mut HashMap<Arc<String>, Vec<Document>>,
        remove_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
    ) {
        let index = &self.index;
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
        let index = &self.index;
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
            .get(label)
            .map(|index| index.fields().clone())
            .unwrap_or_default()
    }

    /// Get fields for all labels.
    #[must_use]
    pub fn get_all_fields(&self) -> Vec<(Arc<String>, HashMap<Arc<String>, Vec<Arc<Field>>>)> {
        self.index
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
        self.index.contains_key(label)
    }

    #[must_use]
    pub fn has_indexed_attr(
        &self,
        label: &Arc<String>,
        field: &Arc<String>,
    ) -> bool {
        if let Some(index) = self.index.get(label) {
            return index.contains_field(field);
        }
        false
    }

    pub fn update_progress(
        &self,
        label: &Arc<String>,
        progress: u64,
    ) {
        if let Some(index) = self.index.get(label) {
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
        self.shared.write_lock.clone()
    }

    pub fn cancel(&self) {
        self.shared.cancelled.store(true, Ordering::Relaxed);
        // Clear the graph reference to break the circular Arc:
        // MvccGraph → Arc<Graph> → Indexer → Arc<Graph>
        self.shared.graph.store(None);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.shared.cancelled.load(Ordering::Relaxed)
    }

    /// Publishes the version this indexer belongs to: it is now the committed
    /// graph. Only touches the `Indexer`'s own lock-free slots, so this takes
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
        self.shared.graph.store(Some(graph));
    }

    #[must_use]
    pub fn get_graph(&self) -> Option<Arc<AtomicRefCell<Graph>>> {
        self.shared.graph.load_full()
    }
}

#[cfg(test)]
mod tests {
    //! The versioning of the map, without RediSearch (not linked into unit tests):
    //! entries are `Index::default()`s put straight into a version's map.

    use super::{Index, IndexMap, Indexer};
    use std::sync::Arc;

    fn put(
        indexer: &mut Indexer,
        label: &str,
        index: Index,
    ) -> Arc<Index> {
        let index = Arc::new(index);
        let mut map: IndexMap = (*indexer.index).clone();
        map.insert(Arc::new(label.to_string()), index.clone());
        indexer.index = Arc::new(map);
        index
    }

    #[test]
    fn a_fork_changes_only_its_own_map() {
        let mut committed = Indexer::default();
        put(&mut committed, "A", Index::default());
        let mut fork = committed.clone();
        put(&mut fork, "B", Index::default());
        fork.index = Arc::new(IndexMap::new());

        assert!(committed.has_index(&Arc::new("A".to_string())));
        assert!(!committed.has_index(&Arc::new("B".to_string())));
        assert!(!fork.has_indices());
    }

    #[test]
    fn a_fork_shares_the_committed_map_until_it_changes_it() {
        let mut committed = Indexer::default();
        put(&mut committed, "A", Index::default());
        let fork = committed.clone();
        let none = fork.changes_since(&committed);
        assert!(none.is_empty(), "no DDL, so nothing to do on publish");
    }

    #[test]
    fn an_owed_population_is_reported_once_and_only_for_a_new_index() {
        let mut committed = Indexer::default();
        let kept = put(&mut committed, "A", Index::default());
        kept.owe_population();
        let mut fork = committed.clone();
        let created = put(&mut fork, "B", Index::default());
        created.owe_population();
        assert_eq!(
            created.pending_count(),
            1,
            "UNDER CONSTRUCTION from creation"
        );

        let changes = fork.changes_since(&committed);
        assert_eq!(changes.owed, vec![Arc::new("B".to_string())]);
        assert!(
            fork.changes_since(&committed).owed.is_empty(),
            "taken: the publisher starts it once"
        );
        assert!(
            kept.take_population_owed(),
            "an index the fork did not touch keeps what it owes"
        );
    }

    #[test]
    fn a_ddl_copy_has_counters_of_its_own() {
        let published = Index::default();
        published.owe_population();
        assert_eq!(published.pending_count(), 1);

        let copy = published.fork_for_ddl(false);
        assert_ne!(copy.id(), published.id(), "a new generation");
        assert_eq!(copy.pending_count(), 0);
        copy.owe_population();
        copy.owe_population();
        assert_eq!(copy.pending_count(), 2);
        assert_eq!(
            published.pending_count(),
            1,
            "what the copy does to its counters never reaches the published index"
        );
        assert!(!copy.has_rs_index(), "it builds a spec of its own");
    }
}
