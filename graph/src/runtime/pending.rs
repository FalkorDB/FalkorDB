//! Deferred write operations for transactional semantics.
//!
//! This module provides [`Pending`], which batches write operations during
//! query execution. This enables:
//!
//! - Read-your-writes within a query (created nodes visible to later clauses)
//! - Atomic commit/rollback of all changes
//! - Efficient bulk updates to indexes
//!
//! ## Batched Operations
//!
//! - `created_nodes`: Nodes created in this query
//! - `deleted_nodes`: Nodes marked for deletion
//! - `created_rels_by_type`: Edges created in this query, grouped by type
//! - `deleted_relationships`: Edges marked for deletion
//! - `set_*_attrs`: Property updates by entity ID
//! - `set/remove_node_labels`: Label changes
//!
//! ## Commit Flow
//!
//! ```text
//! Query execution → accumulate in Pending → apply_all() → update Graph
//! ```
//!
//! On error or ROLLBACK, the Pending is simply dropped without applying.

use std::{cell::RefCell, sync::Arc};

use rustc_hash::{FxHashMap, FxHashSet};

use atomic_refcell::AtomicRefCell;
use roaring::RoaringTreemap;

use crate::graph::graph::{DeletedEdge, DeletedNodeLabel};

use crate::{
    entity_type::EntityType,
    graph::{
        constraint::{ConstraintStatus, ConstraintType},
        graph::{Graph, LabelId, NodeId, NodeOpError, RelationshipId},
    },
    index::{
        Index,
        indexer::{IndexQuery, IndexType},
    },
    runtime::{
        ordermap::OrderMap,
        orderset::OrderSet,
        runtime::QueryStatistics,
        value::{MAX_VALUE_NESTING, Value},
    },
};

/// Flatten a node_id → [label_ids] map into parallel (rows, cols) arrays.
fn flatten_label_map(map: &FxHashMap<u64, Vec<u64>>) -> (Vec<u64>, Vec<u64>) {
    let total: usize = map.values().map(std::vec::Vec::len).sum();
    let mut rows = Vec::with_capacity(total);
    let mut cols = Vec::with_capacity(total);
    for (&node_id, label_ids) in map {
        for &label_id in label_ids {
            rows.push(node_id);
            cols.push(label_id);
        }
    }
    (rows, cols)
}

const INVALID_PROPERTY_MSG: &str =
    "Property values can only be of primitive types or arrays of primitive types";

/// Whether a non-list value is a valid property: a primitive, or NULL when
/// `allow_null`. Lists are checked by [`validate_list_property`].
fn is_valid_property(
    value: &Value,
    allow_null: bool,
) -> bool {
    match value {
        Value::Null => allow_null,
        Value::Bool(_)
        | Value::Int(_)
        | Value::Float(_)
        | Value::String(_)
        | Value::Point(_)
        | Value::VecF32(_)
        | Value::Datetime(_)
        | Value::Date(_)
        | Value::Time(_)
        | Value::Duration(_) => true,
        _ => false,
    }
}

/// Checks a list property: elements are primitives or lists, nested at most
/// [`MAX_VALUE_NESTING`] deep. `depth` is the nesting of `items` itself (a
/// top-level list is depth 1).
///
/// Lists can be built arbitrarily deep at runtime (e.g. `reduce(a = 1, i IN
/// range(1, n) | [a])`), so this cannot be caught when the query is parsed.
/// The bound is what lets every stored value be saved and loaded back: the
/// RDB decoder refuses anything deeper. The walk is recursive, but stops at
/// the bound, so it cannot itself run too deep.
fn validate_list_property(
    items: &[Value],
    depth: usize,
) -> Result<(), String> {
    if depth > MAX_VALUE_NESTING {
        return Err(format!(
            "Property values can only be nested up to {MAX_VALUE_NESTING} levels deep"
        ));
    }
    items.iter().try_for_each(|v| match v {
        Value::List(inner) => validate_list_property(inner, depth + 1),
        v if is_valid_property(v, false) => Ok(()),
        _ => Err(INVALID_PROPERTY_MSG.into()),
    })
}

/// Checks a property value: a primitive, NULL, or a list of primitives.
fn validate_property(value: &Value) -> Result<(), String> {
    match value {
        Value::List(items) => validate_list_property(items, 1),
        v if is_valid_property(v, true) => Ok(()),
        _ => Err(INVALID_PROPERTY_MSG.into()),
    }
}

/// Validate that a value is a valid node property type.
fn validate_node_property(value: &Value) -> Result<(), String> {
    validate_property(value)
}

/// Validate that a value is a valid relationship property type.
fn validate_relationship_property(value: &Value) -> Result<(), String> {
    validate_property(value)
}

/// Binary-search a sorted `(attr_id, Value)` slice.
fn lookup_sorted(
    attrs: &[(u16, Value)],
    attr_id: u16,
) -> Option<&Value> {
    attrs
        .binary_search_by_key(&attr_id, |(k, _)| *k)
        .ok()
        .map(|pos| &attrs[pos].1)
}

/// How many staged attributes set a value, i.e. would count as `Properties set`.
fn count_set(attrs: &[(u16, Value)]) -> usize {
    attrs
        .iter()
        .filter(|(_, v)| !matches!(v, Value::Null))
        .count()
}

/// Whether the supporting index's exact match on `value` returns every entity
/// whose composite-key component equals it. Other values (large integers the
/// index stores rounded, NaN, the empty string, and any type it does not index)
/// are checked by a scan instead. A string containing NUL is fine: the tag
/// encoding escapes it to `_00` on both the write and the query path.
fn is_value_valid_for_index_lookup(value: &Value) -> bool {
    match value {
        Value::Int(i) => !Index::int_loses_f64_precision(*i),
        Value::Float(f) => !f.is_nan(),
        Value::Bool(_) => true,
        Value::String(s) => !s.is_empty(),
        _ => false,
    }
}

/// Whether a unique constraint is violated by this commit, given `members`: the
/// affected entities (a subset of `affected`) carrying the constrained label or
/// type.
///
/// Like C's `EnforceUniqueEntity`, each member's values are looked up in the
/// constraint's supporting index, so the cost is per member, not per entity in
/// the label. The index lags only this commit (each `Commit` publishes its
/// documents before the next runs), so a candidate it returns is re-verified
/// against the graph, and members are compared among themselves. Without an
/// operational index, or for a value it cannot match exactly, the label is
/// scanned once for the whole set instead.
///
/// `is_candidate` tells whether an entity the index or scan returned is still a
/// live member of the label or type.
#[allow(clippy::too_many_arguments)]
fn unique_violated<Id, L, S>(
    properties: &[Arc<String>],
    members: impl Iterator<Item = Id>,
    affected: &RoaringTreemap,
    attr: impl Fn(Id, &Arc<String>) -> Option<Value>,
    is_candidate: impl Fn(Id) -> bool,
    indexed: bool,
    lookup: impl Fn(IndexQuery<Value>) -> L,
    scan: impl FnOnce() -> S,
) -> bool
where
    Id: Copy + Into<u64>,
    L: Iterator<Item = Id>,
    S: Iterator<Item = Id>,
{
    let mut keys: FxHashSet<Vec<u8>> = FxHashSet::default();
    let mut lookups = Vec::new();
    let mut use_index = indexed;
    for id in members {
        // A NULL or absent constrained property: the entity does not participate.
        let Some(values) = properties
            .iter()
            .map(|prop| attr(id, prop).filter(|v| !matches!(v, Value::Null)))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let mut it = values.iter();
        let key = Graph::build_composite_key(properties, |_| it.next().cloned());
        if !keys.insert(key) {
            // Two entities written in this commit share a key. That is a
            // violation whatever the index or the rest of the label holds, so
            // there is nothing left to look up or scan.
            return true;
        }
        if use_index {
            if values.iter().all(is_value_valid_for_index_lookup) {
                lookups.push(values);
            } else {
                // One value the index cannot match exactly sends the whole set
                // to a single scan, so the lookups gathered so far are unused.
                use_index = false;
                lookups = Vec::new();
            }
        }
    }
    if keys.is_empty() {
        return false;
    }

    // Members are already compared among themselves above.
    let collides = |other: Id| {
        !affected.contains(other.into())
            && is_candidate(other)
            && keys.contains(&Graph::build_composite_key(properties, |prop| {
                attr(other, prop)
            }))
    };
    if use_index {
        lookups.into_iter().any(|values| {
            let mut eqs = properties
                .iter()
                .zip(values)
                .map(|(key, value)| IndexQuery::Equal {
                    key: key.clone(),
                    value,
                })
                .collect::<Vec<_>>();
            let query = if eqs.len() == 1 {
                eqs.pop().unwrap()
            } else {
                IndexQuery::And(eqs)
            };
            lookup(query).any(collides)
        })
    } else {
        scan().any(collides)
    }
}

/// A relationship whose id was handed out and then taken back, because the node
/// it hung off was cancelled in the same commit.
///
/// The endpoints and the type are kept even though the edge never survives:
/// `CreateEdge` has no form without them, and the record pair is what tells the
/// replica the id was used. See [`Pending::cancelled_relationships`].
#[derive(Debug, Clone)]
pub(crate) struct CancelledRelationship {
    pub(crate) id: u64,
    pub(crate) type_name: Arc<String>,
    pub(crate) src: u64,
    pub(crate) dst: u64,
}

/// Accumulated write operations for deferred application.
///
/// All mutations during query execution are collected here and applied
/// atomically at the end. This enables transactional semantics.
///
/// The fields are `pub(crate)` rather than hidden behind per-format view
/// structs. Those views were a projection — every field a reference to the same
/// type — so they documented the dependency surface and did nothing else, while
/// costing a struct and a constructor per wire format. What states that surface
/// now is `effects_v3_emit::digest`, whose return type says precisely what
/// replication makes of this.
pub struct Pending {
    /// Nodes created in this transaction, and what the allocator must not
    /// reissue: reserved, still destined for `create_nodes` at commit, and so
    /// not yet recorded anywhere the id space can see.
    ///
    /// Cancelled ids are deliberately *not* here — `IdSpace::cancel` records
    /// them and `IdSpace::reserve` excludes what it has recorded, so passing
    /// them again would count one id twice and leave a gap the batch never
    /// fills.
    pub(crate) created_nodes: RoaringTreemap,
    pub(crate) created_rels_by_type: FxHashMap<Arc<String>, Vec<(RelationshipId, NodeId, NodeId)>>,
    /// Reverse index: rel_id → type_name for O(1) existence/type lookups
    pub(crate) created_rel_types: FxHashMap<RelationshipId, Arc<String>>,
    /// The relationship ids this batch has reserved and not yet cancelled — the
    /// mirror of `created_nodes`.
    ///
    /// A set of its own because the created relationships are kept as a map
    /// keyed by id and a vector of cancellation records, neither of which can be
    /// differenced against the free set without being walked.
    pub(crate) taken_relationship_ids: RoaringTreemap,
    /// Nodes to be deleted
    pub(crate) deleted_nodes: RoaringTreemap,
    /// Relationships to be deleted
    pub(crate) deleted_relationships: RoaringTreemap,
    /// Endpoints for deleted relationships — populated by commit(), read by the
    /// effects emitter, which takes a `Pending` and never the other way round.
    pub(crate) deleted_endpoints: Vec<DeletedEdge>,
    /// `(node_id, label_id)` for every deleted node that carried a label —
    /// populated by commit(), used by the effects emitter.
    ///
    /// Captured here because `delete_nodes` clears the label matrices, so by
    /// the time effects are encoded the labels are unrecoverable; and a label
    /// set is what tells a replica which label-scoped indexes to clear. Stored
    /// as flat pairs rather than grouped: `delete_nodes` already built this
    /// exact vector, so keeping it costs a move, while grouping would charge
    /// every delete for a partitioning only a replicating server reads.
    pub(crate) deleted_node_labels: Vec<DeletedNodeLabel>,
    /// Nodes created and deleted inside one segment, which the graph never
    /// sees.
    ///
    /// `delete_pending_node` unwinds such a node out of `created_nodes` and
    /// `cancel_node_id` hands its id straight back to the recycle bin, so the
    /// master's id space acquires a hole that no create and no delete
    /// describes. Replicating nothing for it left the replica's id space one
    /// short, and its next allocation after a promotion landed on a live node.
    ///
    /// Kept here so the effects emitter can say what happened — a create and a
    /// delete, which is what C's payload carries for the same query — without
    /// the master having to do the work it deliberately skips.
    /// Ids of nodes created and then cancelled in this commit.
    ///
    /// Ids alone: the record pair they become carries no labels or attributes,
    /// because the node is deleted a record later and its content would be
    /// applied and immediately undone. [`Self::cancelled_relationships`] keeps
    /// more only because `CreateEdge` has no form without a type and endpoints.
    pub(crate) cancelled_nodes: RoaringTreemap,
    /// Relationships cascaded away by a cancelled node, in the same commit.
    ///
    /// Their ids were reserved and returned exactly as a cancelled node's is —
    /// `cancel_relationship_id` puts them in the recycle bin — so a buffer that
    /// says nothing about them leaves the replica's relationship id space with a
    /// hole, and `IdSpace::verify` refuses the whole payload.
    pub(crate) cancelled_relationships: Vec<CancelledRelationship>,
    /// Properties the cancelled nodes and relationships were created with.
    ///
    /// Cancelling unwinds their staged attributes, so commit never imports
    /// them and would not count them; C reports them as set, together with
    /// the create and the delete that `cancelled_*` count.
    pub(crate) cancelled_properties: usize,
    /// Property updates for newly created nodes (fast path: skip fjall).
    /// Values are attribute-id-resolved, sorted by id, unique.
    pub(crate) new_nodes_attrs: FxHashMap<u64, Vec<(u16, Value)>>,
    /// Property updates for existing nodes (full merge path)
    pub(crate) existing_nodes_attrs: FxHashMap<u64, Vec<(u16, Value)>>,
    /// Property updates for newly created relationships (fast path)
    pub(crate) new_relationships_attrs: FxHashMap<u64, Vec<(u16, Value)>>,
    /// Property updates for existing relationships (full merge path)
    pub(crate) existing_relationships_attrs: FxHashMap<u64, Vec<(u16, Value)>>,
    /// Labels to add: node_id → [label_ids]
    pub(crate) set_labels: FxHashMap<u64, Vec<u64>>,
    /// Labels to remove: node_id → [label_ids]
    pub(crate) remove_labels: FxHashMap<u64, Vec<u64>>,
    /// Index documents this `Commit` produced.
    pub(crate) index_docs: IndexDocs,
    /// Index documents accumulated across the query's commits, applied only
    /// once the whole query succeeds so a failed one never leaves stale
    /// entries in RediSearch.
    pub(crate) deferred_docs: IndexDocs,
    /// Union of every index document this query has published, accumulated across
    /// all of its `Commit`s, so a later failure can resync them against committed
    /// state (see [`Self::resync_published_indexes`]).
    ///
    /// Deliberately **not** reset by [`Self::end_segment`], which runs after every
    /// `Commit`: the undo has to cover the whole query, not just the last `Commit`.
    /// Per-query state — `Pending` belongs to one `Runtime`.
    published: IndexDocs,
    /// Schema baseline: number of labels when the current commit window started.
    pub(crate) schema_label_count: usize,
    /// Schema baseline: number of relationship types when the current commit window started.
    pub(crate) schema_rel_type_count: usize,
    /// Schema baseline: number of node attribute names when the current commit window started.
    pub(crate) schema_node_attr_count: usize,
    /// Schema baseline: number of relationship attribute names when the current commit window started.
    pub(crate) schema_rel_attr_count: usize,
}

pub use crate::index::index_docs::{EdgeDocRemovals, IndexDocs};

impl IndexDocs {
    /// True when nothing was produced, so a caller can skip the work of
    /// publishing an empty set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.node_adds.is_empty()
            && self.node_removes.is_empty()
            && self.edge_adds.is_empty()
            && self.edge_removes.is_empty()
    }

    /// Fold `other` in, leaving it empty.
    ///
    /// Replaces four hand-written loops that folded a commit's documents into
    /// the query's deferred set — one per map, each a chance to fold the wrong
    /// pair together.
    pub fn absorb(
        &mut self,
        other: &mut Self,
    ) {
        // The native index's changes are applied to the private version before a commit's
        // documents are folded for publishing; anything still here would be dropped.
        #[cfg(feature = "index-falkordb")]
        debug_assert!(
            other.native.is_empty(),
            "native index writes must be applied before the documents are folded"
        );
        for (slot, ids) in other.node_adds.drain() {
            *self.node_adds.entry(slot).or_default() |= ids;
        }
        for (slot, ids) in other.node_removes.drain() {
            *self.node_removes.entry(slot).or_default() |= ids;
        }
        for (slot, ids) in other.edge_adds.drain() {
            *self.edge_adds.entry(slot).or_default() |= ids;
        }
        for (slot, ids) in other.edge_removes.drain() {
            self.edge_removes.entry(slot).or_default().extend(ids);
        }
    }

    /// Fold `other` in, for accumulating everything one query published.
    fn merge(
        &mut self,
        other: &Self,
    ) {
        for (slot, ids) in &other.node_adds {
            *self.node_adds.entry(*slot).or_default() |= ids;
        }
        for (slot, ids) in &other.node_removes {
            *self.node_removes.entry(*slot).or_default() |= ids;
        }
        for (slot, ids) in &other.edge_adds {
            *self.edge_adds.entry(*slot).or_default() |= ids;
        }
        for (slot, ids) in &other.edge_removes {
            self.edge_removes.entry(*slot).or_default().extend(ids);
        }
    }

    /// Every id in this set per slot, regardless of direction — all the undo path
    /// needs, since it re-derives each document from committed state.
    fn ids_by_slot(
        &self
    ) -> (
        FxHashMap<u64, RoaringTreemap>,
        FxHashMap<u64, RoaringTreemap>,
    ) {
        let mut nodes: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        let mut edges: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        for docs in [&self.node_adds, &self.node_removes] {
            for (slot, ids) in docs {
                *nodes.entry(*slot).or_default() |= ids;
            }
        }
        for (slot, ids) in &self.edge_adds {
            *edges.entry(*slot).or_default() |= ids;
        }
        for (slot, ids) in &self.edge_removes {
            edges.entry(*slot).or_default().extend(ids.keys().copied());
        }
        (nodes, edges)
    }

    /// Write the batch to the shared RediSearch index.
    ///
    /// Only valid in writer mode: the index is not MVCC, so readers must be
    /// excluded (mirrors C, which updates index docs inside the graph write lock).
    pub fn commit(
        &mut self,
        g: &mut Graph,
    ) {
        // This is the query's own private version, which nothing else can borrow, so
        // graph-then-indexer is safe here — see `Pending::resync_published_indexes`
        // for the published version, where the order has to be the other way round.
        g.commit_index(&mut self.node_adds, &mut self.node_removes);
        g.commit_edge_index(&mut self.edge_adds, &mut self.edge_removes);
    }
}

impl Default for Pending {
    fn default() -> Self {
        Self::new()
    }
}

impl Pending {
    #[must_use]
    pub fn new() -> Self {
        Self {
            created_nodes: RoaringTreemap::new(),
            created_rels_by_type: FxHashMap::default(),
            created_rel_types: FxHashMap::default(),
            taken_relationship_ids: RoaringTreemap::new(),
            deleted_nodes: RoaringTreemap::new(),
            deleted_relationships: RoaringTreemap::new(),
            deleted_endpoints: Vec::new(),
            deleted_node_labels: Vec::new(),
            cancelled_nodes: RoaringTreemap::new(),
            cancelled_relationships: Vec::new(),
            cancelled_properties: 0,
            new_nodes_attrs: FxHashMap::default(),
            existing_nodes_attrs: FxHashMap::default(),
            new_relationships_attrs: FxHashMap::default(),
            existing_relationships_attrs: FxHashMap::default(),
            set_labels: FxHashMap::default(),
            remove_labels: FxHashMap::default(),
            index_docs: IndexDocs::default(),
            published: IndexDocs::default(),
            deferred_docs: IndexDocs::default(),
            schema_label_count: 0,
            schema_rel_type_count: 0,
            schema_node_attr_count: 0,
            schema_rel_attr_count: 0,
        }
    }

    /// Record the current dictionary sizes, so an effects emitter can tell
    /// which labels, types and attributes this query added.
    pub fn set_schema_baseline(
        &mut self,
        graph: &Graph,
    ) {
        self.schema_label_count = graph.get_labels().len();
        self.schema_rel_type_count = graph.get_types().len();
        self.schema_node_attr_count = graph.get_node_attribute_names().len();
        self.schema_rel_attr_count = graph.get_relationship_attribute_names().len();
    }

    pub fn created_nodes(
        &mut self,
        ids: &[NodeId],
    ) {
        for id in ids {
            self.created_nodes.insert((*id).into());
        }
    }

    /// Set all attributes for a node. `attrs` must be attribute-id-resolved,
    /// sorted by id, and unique (last-wins dedup done by the caller).
    pub fn set_node_attributes(
        &mut self,
        id: NodeId,
        attrs: Vec<(u16, Value)>,
    ) -> Result<(), String> {
        for (_, value) in &attrs {
            validate_node_property(value)?;
        }
        // Empty attribute maps from CREATE without `{...}` props would otherwise
        // create an empty pinned cache entry per node on commit; skip them.
        if attrs.is_empty() {
            return Ok(());
        }
        // Strict: the doc above requires unique ids too, and
        // `AttributeStore::insert_attrs_rows` asserts strictly ascending at the
        // other end, so a non-strict check here would pass a duplicate along to
        // an assert that rejects it.
        debug_assert!(attrs.windows(2).all(|w| w[0].0 < w[1].0));
        let is_new = self.created_nodes.contains(id.into());
        if is_new {
            self.new_nodes_attrs.insert(id.into(), attrs);
        } else {
            self.existing_nodes_attrs.insert(id.into(), attrs);
        }
        Ok(())
    }

    pub fn set_node_attribute(
        &mut self,
        id: NodeId,
        attr_id: u16,
        value: Value,
    ) -> Result<(), String> {
        validate_node_property(&value)?;
        let map = if self.created_nodes.contains(id.into()) {
            &mut self.new_nodes_attrs
        } else {
            &mut self.existing_nodes_attrs
        };
        let entry = map.entry(id.into()).or_default();
        match entry.binary_search_by_key(&attr_id, |(k, _)| *k) {
            Ok(pos) => entry[pos].1 = value,
            Err(pos) => entry.insert(pos, (attr_id, value)),
        }
        Ok(())
    }

    pub fn clear_node_attributes(
        &mut self,
        id: NodeId,
    ) {
        self.new_nodes_attrs.remove(&id.into());
        self.existing_nodes_attrs.remove(&id.into());
    }

    pub fn clear_relationship_attributes(
        &mut self,
        id: RelationshipId,
    ) {
        self.new_relationships_attrs.remove(&id.into());
        self.existing_relationships_attrs.remove(&id.into());
    }

    #[must_use]
    #[inline]
    pub fn has_node_attrs(&self) -> bool {
        !self.new_nodes_attrs.is_empty() || !self.existing_nodes_attrs.is_empty()
    }

    #[must_use]
    #[inline]
    pub fn has_relationship_attrs(&self) -> bool {
        !self.new_relationships_attrs.is_empty() || !self.existing_relationships_attrs.is_empty()
    }

    #[must_use]
    pub fn get_node_attribute(
        &self,
        id: NodeId,
        attr_id: u16,
    ) -> Option<&Value> {
        if !self.has_node_attrs() {
            return None;
        }
        self.new_nodes_attrs
            .get(&id.into())
            .and_then(|attrs| lookup_sorted(attrs, attr_id))
            .or_else(|| {
                self.existing_nodes_attrs
                    .get(&id.into())
                    .and_then(|attrs| lookup_sorted(attrs, attr_id))
            })
    }

    pub fn update_node_attrs(
        &self,
        id: NodeId,
        attrs: &mut OrderMap<Arc<String>, Value>,
        g: &Graph,
    ) {
        let added = self
            .new_nodes_attrs
            .get(&id.into())
            .or_else(|| self.existing_nodes_attrs.get(&id.into()));
        if let Some(added) = added {
            for (attr_id, value) in added {
                let Some(key) = g.node_attr_name(*attr_id) else {
                    continue;
                };
                if matches!(value, Value::Null) {
                    attrs.remove(&key);
                } else {
                    attrs.insert(key, value.clone());
                }
            }
        }
    }

    /// Stage label adds for `id`, cancelling any removal of those same labels
    /// staged earlier in this query. Mirrors [`Self::remove_node_labels`], which
    /// cancels earlier adds; together they keep `set_labels` and `remove_labels`
    /// disjoint per node, so the last clause to touch a label wins in either
    /// direction (`REMOVE n:L SET n:L` keeps `L`, `SET n:L REMOVE n:L` drops it).
    fn stage_node_labels(
        &mut self,
        raw_id: u64,
        labels: &OrderSet<LabelId>,
    ) {
        let entry = self.set_labels.entry(raw_id).or_default();
        for label in labels.iter() {
            entry.push(usize::from(*label) as u64);
        }
        if let Some(removed) = self.remove_labels.get_mut(&raw_id) {
            removed.retain(|&l| !labels.contains(&LabelId(l as usize)));
            if removed.is_empty() {
                self.remove_labels.remove(&raw_id);
            }
        }
    }

    pub fn set_node_labels(
        &mut self,
        id: NodeId,
        labels: &OrderSet<LabelId>,
    ) {
        self.stage_node_labels(id.into(), labels);
    }

    pub fn set_nodes_labels(
        &mut self,
        ids: &[NodeId],
        labels: &OrderSet<LabelId>,
    ) {
        for id in ids {
            self.stage_node_labels((*id).into(), labels);
        }
    }

    /// Stage label removals for `id`, cancelling any add of those same labels
    /// staged earlier in this query — the mirror image of
    /// [`Self::stage_node_labels`], keeping the two sets disjoint per node.
    pub fn remove_node_labels(
        &mut self,
        id: NodeId,
        labels: &[LabelId],
    ) {
        let raw_id: u64 = id.into();
        for label in labels {
            let label_id = usize::from(*label) as u64;
            // Remove from pending set labels
            if let Some(set) = self.set_labels.get_mut(&raw_id) {
                set.retain(|&l| l != label_id);
                if set.is_empty() {
                    self.set_labels.remove(&raw_id);
                }
            }
            self.remove_labels.entry(raw_id).or_default().push(label_id);
        }
    }

    /// What this query has staged about `label` on `id`: `Some(true)` if it was
    /// added, `Some(false)` if it was removed, `None` if this query says nothing
    /// about it and the committed label matrix is the answer.
    ///
    /// `set_labels` and `remove_labels` are disjoint per node (see
    /// [`Self::stage_node_labels`]), so at most one of the two branches below can
    /// match and the order they are consulted in carries no meaning.
    pub fn node_has_label(
        &self,
        id: NodeId,
        label: LabelId,
    ) -> Option<bool> {
        let raw_id: u64 = id.into();
        let label_id = usize::from(label) as u64;
        if self
            .remove_labels
            .get(&raw_id)
            .is_some_and(|removed| removed.contains(&label_id))
        {
            return Some(false);
        }
        if self
            .set_labels
            .get(&raw_id)
            .is_some_and(|set| set.contains(&label_id))
        {
            return Some(true);
        }
        None
    }

    /// Overlay this query's staged label changes onto `labels`.
    ///
    /// Adds are applied before removals, but the two sets are disjoint per node
    /// (see [`Self::stage_node_labels`]), so no label is touched by both passes
    /// and the order is immaterial.
    pub fn update_node_labels(
        &self,
        id: NodeId,
        labels: &mut OrderSet<LabelId>,
    ) {
        let raw_id: u64 = id.into();
        if let Some(set) = self.set_labels.get(&raw_id) {
            for &label_id in set {
                labels.insert(LabelId(label_id as usize));
            }
        }
        if let Some(removed) = self.remove_labels.get(&raw_id) {
            for &label_id in removed {
                labels.remove(&LabelId(label_id as usize));
            }
        }
    }

    pub fn deleted_node(
        &mut self,
        id: NodeId,
    ) {
        self.deleted_nodes.insert(id.into());
    }

    /// Delete a pending-created node: mark it deleted, collect its labels and attrs,
    /// and also mark any pending-created relationships connected to it for deletion.
    /// Returns (label_ids, attrs, connected_pending_rels) — the relationships carry
    /// their type and staged attrs so callers can snapshot them for later reads.
    ///
    /// # Errors
    ///
    /// [`NodeOpError`] if returning an id left the graph's id space
    /// contradicting itself.
    pub fn delete_pending_node(
        &mut self,
        id: NodeId,
        g: &mut Graph,
    ) -> Result<
        (
            OrderSet<LabelId>,
            Vec<(u16, Value)>,
            Vec<(
                RelationshipId,
                NodeId,
                NodeId,
                Arc<String>,
                Option<Vec<(u16, Value)>>,
            )>,
        ),
        NodeOpError,
    > {
        self.created_nodes.remove(id.into());
        // Collect pending labels
        let mut label_ids = OrderSet::default();
        self.update_node_labels(id, &mut label_ids);
        self.set_labels.remove(&id.into());

        // Collect pending attrs
        let attrs = self
            .new_nodes_attrs
            .remove(&id.into())
            .or_else(|| self.existing_nodes_attrs.remove(&id.into()))
            .unwrap_or_default();
        self.cancelled_properties += count_set(&attrs);

        let rels = self.remove_pending_relationships_for_node(id, g)?;

        // The one durable record that this id was ever handed out. Everything
        // above has just erased it from the structures the graph commits.
        self.cancelled_nodes.insert(id.into());
        // And the graph's half, paired here for the same reason.
        g.cancel_node_id(id)?;

        Ok((label_ids, attrs, rels))
    }

    /// Remove and return all pending-created relationships incident on the
    /// given node, along with their staged attributes. Also cleans up
    /// `new_relationships_attrs` and `deleted_relationships` entries for
    /// each removed relationship so that `commit()` has no stale state.
    ///
    /// # Errors
    ///
    /// [`NodeOpError`] if returning an id left the graph's id space
    /// contradicting itself.
    pub fn remove_pending_relationships_for_node(
        &mut self,
        id: NodeId,
        g: &mut Graph,
    ) -> Result<
        Vec<(
            RelationshipId,
            NodeId,
            NodeId,
            Arc<String>,
            Option<Vec<(u16, Value)>>,
        )>,
        NodeOpError,
    > {
        let mut rels = Vec::new();
        for (type_name, entries) in &self.created_rels_by_type {
            for &(rel_id, from, to) in entries {
                if from == id || to == id {
                    rels.push((rel_id, from, to, type_name.clone()));
                }
            }
        }

        let mut result = Vec::with_capacity(rels.len());
        for (rel_id, from, to, type_name) in rels {
            self.created_rel_types.remove(&rel_id);
            if let Some(entries) = self.created_rels_by_type.get_mut(&type_name) {
                entries.retain(|(rid, _, _)| *rid != rel_id);
                // And drop the group once it is empty. A type whose every edge
                // was cascaded away was never registered on the graph — nothing
                // of that type reached `create_relationships_bulk` — so leaving
                // the key behind hands the effects emitter a type name it
                // cannot resolve.
                if entries.is_empty() {
                    self.created_rels_by_type.remove(&type_name);
                }
            }
            let attrs = self.new_relationships_attrs.remove(&rel_id.into());
            self.cancelled_properties += attrs.as_deref().map_or(0, count_set);
            self.deleted_relationships.remove(rel_id.into());
            // Handed over to the id space, which records the cancellation as it
            // returns the id to the free set. Leaving it here too would have the
            // allocator count it twice.
            self.taken_relationship_ids.remove(rel_id.into());
            // The graph's half of the same event. Paired here so a caller cannot
            // unwind the pending side and leave the id unreturned.
            g.cancel_relationship_id(rel_id)?;
            // The one durable record that this id was ever handed out, the same
            // role `cancelled_nodes` plays for the node above.
            self.cancelled_relationships.push(CancelledRelationship {
                id: rel_id.into(),
                type_name: type_name.clone(),
                src: from.into(),
                dst: to.into(),
            });
            result.push((rel_id, from, to, type_name, attrs));
        }

        Ok(result)
    }

    pub fn created_relationship(
        &mut self,
        id: RelationshipId,
        from: NodeId,
        to: NodeId,
        type_name: Arc<String>,
    ) {
        self.created_rels_by_type
            .entry(type_name.clone())
            .or_default()
            .push((id, from, to));
        self.created_rel_types.insert(id, type_name);
        self.taken_relationship_ids.insert(id.into());
    }

    /// Set all attributes for a relationship. `attrs` must be
    /// attribute-id-resolved, sorted by id, and unique.
    pub fn set_relationship_attributes(
        &mut self,
        id: RelationshipId,
        attrs: Vec<(u16, Value)>,
    ) -> Result<(), String> {
        for (_, value) in &attrs {
            validate_relationship_property(value)?;
        }
        // Empty attribute maps from CREATE without `{...}` props would otherwise
        // create an empty pinned cache entry per relationship on commit; skip them.
        if attrs.is_empty() {
            return Ok(());
        }
        // Strict: the doc above requires unique ids too, and
        // `AttributeStore::insert_attrs_rows` asserts strictly ascending at the
        // other end, so a non-strict check here would pass a duplicate along to
        // an assert that rejects it.
        debug_assert!(attrs.windows(2).all(|w| w[0].0 < w[1].0));
        if self.created_rel_types.contains_key(&id) {
            self.new_relationships_attrs.insert(id.into(), attrs);
        } else {
            self.existing_relationships_attrs.insert(id.into(), attrs);
        }
        Ok(())
    }

    pub fn set_relationship_attribute(
        &mut self,
        id: RelationshipId,
        attr_id: u16,
        value: Value,
    ) -> Result<(), String> {
        validate_relationship_property(&value)?;
        let map = if self.created_rel_types.contains_key(&id) {
            &mut self.new_relationships_attrs
        } else {
            &mut self.existing_relationships_attrs
        };
        let entry = map.entry(id.into()).or_default();
        match entry.binary_search_by_key(&attr_id, |(k, _)| *k) {
            Ok(pos) => entry[pos].1 = value,
            Err(pos) => entry.insert(pos, (attr_id, value)),
        }
        Ok(())
    }

    #[must_use]
    pub fn get_relationship_attribute(
        &self,
        id: RelationshipId,
        attr_id: u16,
    ) -> Option<&Value> {
        if !self.has_relationship_attrs() {
            return None;
        }
        self.new_relationships_attrs
            .get(&id.into())
            .and_then(|attrs| lookup_sorted(attrs, attr_id))
            .or_else(|| {
                self.existing_relationships_attrs
                    .get(&id.into())
                    .and_then(|attrs| lookup_sorted(attrs, attr_id))
            })
    }

    pub fn update_relationship_attrs(
        &self,
        id: RelationshipId,
        attrs: &mut OrderMap<Arc<String>, Value>,
        g: &Graph,
    ) {
        let added = self
            .new_relationships_attrs
            .get(&id.into())
            .or_else(|| self.existing_relationships_attrs.get(&id.into()));
        if let Some(added) = added {
            for (attr_id, value) in added {
                let Some(key) = g.rel_attr_name(*attr_id) else {
                    continue;
                };
                if matches!(value, Value::Null) {
                    attrs.remove(&key);
                } else {
                    attrs.insert(key, value.clone());
                }
            }
        }
    }

    pub fn deleted_relationship(
        &mut self,
        id: RelationshipId,
    ) {
        self.deleted_relationships.insert(id.into());
    }

    pub fn deleted_relationships_bulk(
        &mut self,
        rels: &[RelationshipId],
    ) {
        for &id in rels {
            self.deleted_relationships.insert(id.into());
        }
    }

    #[must_use]
    pub fn get_relationship_type(
        &self,
        id: RelationshipId,
    ) -> Option<Arc<String>> {
        self.created_rel_types.get(&id).cloned()
    }

    #[must_use]
    pub fn is_node_created(
        &self,
        id: NodeId,
    ) -> bool {
        self.created_nodes.contains(id.into())
    }

    /// Returns pending-created node IDs that have ALL of the given labels.
    /// When `label_ids` is empty, returns all pending-created nodes.
    #[must_use]
    pub fn get_pending_nodes_with_labels(
        &self,
        label_ids: &[LabelId],
    ) -> RoaringTreemap {
        if label_ids.is_empty() {
            return self.created_nodes.clone();
        }
        let label_ids_u64: Vec<u64> = label_ids.iter().map(|l| usize::from(*l) as u64).collect();
        let mut result = RoaringTreemap::new();
        for (&node_id, node_labels) in &self.set_labels {
            if label_ids_u64.iter().all(|lid| node_labels.contains(lid)) {
                result.insert(node_id);
            }
        }
        result
    }

    /// Returns existing (non-created) node IDs that have pending label REMOVEs
    /// for ANY of the given label_ids.
    #[must_use]
    pub fn nodes_with_pending_label_removes(
        &self,
        label_ids: &[LabelId],
    ) -> RoaringTreemap {
        if label_ids.is_empty() {
            return RoaringTreemap::new();
        }
        let label_ids_u64: Vec<u64> = label_ids.iter().map(|l| usize::from(*l) as u64).collect();
        let mut result = RoaringTreemap::new();
        for (&node_id, removed_labels) in &self.remove_labels {
            if label_ids_u64.iter().any(|lid| removed_labels.contains(lid)) {
                result.insert(node_id);
            }
        }
        result
    }

    #[must_use]
    pub fn is_relationship_created(
        &self,
        id: RelationshipId,
    ) -> bool {
        self.created_rel_types.contains_key(&id)
    }

    /// Returns (src, dst) for a pending-created relationship, or None if not found.
    #[must_use]
    pub fn get_created_relationship_endpoints(
        &self,
        id: RelationshipId,
    ) -> Option<(NodeId, NodeId)> {
        let type_name = self.created_rel_types.get(&id)?;
        self.created_rels_by_type
            .get(type_name)?
            .iter()
            .find(|(rid, _, _)| *rid == id)
            .map(|&(_, from, to)| (from, to))
    }

    #[must_use]
    pub fn has_deleted_nodes(&self) -> bool {
        !self.deleted_nodes.is_empty()
    }

    #[must_use]
    pub fn has_deleted_relationships(&self) -> bool {
        !self.deleted_relationships.is_empty()
    }

    #[must_use]
    pub fn has_created_relationships(&self) -> bool {
        !self.created_rel_types.is_empty()
    }

    #[must_use]
    pub fn is_node_deleted(
        &self,
        id: NodeId,
    ) -> bool {
        self.deleted_nodes.contains(id.into())
    }

    /// Returns a clone of the pending-deleted nodes bitmap.
    #[must_use]
    pub fn deleted_nodes(&self) -> RoaringTreemap {
        self.deleted_nodes.clone()
    }

    #[must_use]
    pub fn is_relationship_deleted(
        &self,
        id: RelationshipId,
    ) -> bool {
        self.deleted_relationships.contains(id.into())
    }

    /// Count pending-created relationships whose destination is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    #[must_use]
    pub fn pending_indegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
    ) -> usize {
        if types.is_empty() {
            self.created_rels_by_type
                .values()
                .flat_map(|v| v.iter())
                .filter(|(_, _, to)| *to == node_id)
                .count()
        } else {
            types
                .iter()
                .filter_map(|t| self.created_rels_by_type.get(t))
                .flat_map(|v| v.iter())
                .filter(|(_, _, to)| *to == node_id)
                .count()
        }
    }

    /// Count pending-created relationships whose source is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    #[must_use]
    pub fn pending_outdegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
    ) -> usize {
        if types.is_empty() {
            self.created_rels_by_type
                .values()
                .flat_map(|v| v.iter())
                .filter(|(_, from, _)| *from == node_id)
                .count()
        } else {
            types
                .iter()
                .filter_map(|t| self.created_rels_by_type.get(t))
                .flat_map(|v| v.iter())
                .filter(|(_, from, _)| *from == node_id)
                .count()
        }
    }

    /// Count pending-deleted relationships whose destination is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    /// Requires access to the graph to resolve relationship type IDs.
    #[must_use]
    pub fn pending_deleted_indegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
        g: &Graph,
    ) -> usize {
        self.deleted_relationships
            .iter()
            .filter(|rel_id| {
                let (_from, to) = g.get_relationship_endpoints(RelationshipId::from(*rel_id));
                to == node_id
                    && (types.is_empty()
                        || g.get_type(g.get_relationship_type_id(RelationshipId::from(*rel_id)))
                            .is_some_and(|t| types.contains(&t)))
            })
            .count()
    }

    /// Count pending-deleted relationships whose source is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    /// Requires access to the graph to resolve relationship type IDs.
    #[must_use]
    pub fn pending_deleted_outdegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
        g: &Graph,
    ) -> usize {
        self.deleted_relationships
            .iter()
            .filter(|rel_id| {
                let (from, _to) = g.get_relationship_endpoints(RelationshipId::from(*rel_id));
                from == node_id
                    && (types.is_empty()
                        || g.get_type(g.get_relationship_type_id(RelationshipId::from(*rel_id)))
                            .is_some_and(|t| types.contains(&t)))
            })
            .count()
    }

    pub fn commit(
        &mut self,
        g: &mut Graph,
        stats: &RefCell<QueryStatistics>,
    ) -> Result<(), String> {
        // Entities created and deleted within this segment never reach the
        // graph, but the query did create and delete them: count both, as C
        // does, along with the properties they were created with.
        if !self.cancelled_nodes.is_empty() || !self.cancelled_relationships.is_empty() {
            let cancelled_nodes = self.cancelled_nodes.len();
            let mut s = stats.borrow_mut();
            s.nodes_created += cancelled_nodes;
            s.nodes_deleted += cancelled_nodes;
            s.relationships_created += self.cancelled_relationships.len();
            s.relationships_deleted += self.cancelled_relationships.len();
            s.properties_set += self.cancelled_properties;
        }
        if !self.created_nodes.is_empty() {
            stats.borrow_mut().nodes_created += self.created_nodes.len();
            g.create_nodes(&self.created_nodes)
                .map_err(|e| e.to_string())?;
        }
        if !self.created_rel_types.is_empty() {
            stats.borrow_mut().relationships_created += self.created_rel_types.len();
            for (type_name, rel_ids) in &self.created_rels_by_type {
                let mut srcs = Vec::with_capacity(rel_ids.len());
                let mut dsts = Vec::with_capacity(rel_ids.len());
                let mut ids = Vec::with_capacity(rel_ids.len());
                for &(rel_id, from, to) in rel_ids {
                    srcs.push(from.into());
                    dsts.push(to.into());
                    ids.push(rel_id.into());
                }
                g.create_relationships_bulk(type_name, &srcs, &dsts, &ids)
                    .map_err(|e| e.to_string())?;
            }
        }
        if !self.set_labels.is_empty() {
            let (rows, cols) = flatten_label_map(&self.set_labels);
            // Pairs for nodes created in this transaction cannot already be
            // committed (fresh ids; a reclaimed id's stale entries carry dm
            // tombstones), so a pure-create batch takes the unchecked insert.
            let all_new = self
                .set_labels
                .keys()
                .all(|id| self.created_nodes.contains(*id));
            g.set_nodes_labels_bulk(&rows, &cols, &mut self.index_docs, all_new);
        }
        if !self.remove_labels.is_empty() {
            let (rows, cols) = flatten_label_map(&self.remove_labels);
            stats.borrow_mut().labels_removed += rows.len();
            g.remove_nodes_labels(&rows, &cols, &mut self.index_docs);
        }
        if !self.new_nodes_attrs.is_empty() || !self.existing_nodes_attrs.is_empty() {
            if !self.new_nodes_attrs.is_empty() {
                let nset = g.import_node_attrs(
                    &self.new_nodes_attrs,
                    &self.set_labels,
                    &mut self.index_docs,
                );
                stats.borrow_mut().properties_set += nset;
            }
            if !self.existing_nodes_attrs.is_empty() {
                let (nremoved, nset) =
                    g.set_nodes_attributes(&self.existing_nodes_attrs, &mut self.index_docs)?;
                let mut s = stats.borrow_mut();
                s.properties_set += nset;
                s.properties_removed += nremoved;
            }
        }

        if !self.new_relationships_attrs.is_empty() || !self.existing_relationships_attrs.is_empty()
        {
            if !self.new_relationships_attrs.is_empty() {
                let nset = g
                    .import_relationship_attrs(&self.new_relationships_attrs, &mut self.index_docs);
                stats.borrow_mut().properties_set += nset;
            }
            if !self.existing_relationships_attrs.is_empty() {
                let (nremoved, nset) = g.set_relationships_attributes(
                    &self.existing_relationships_attrs,
                    &mut self.index_docs,
                )?;
                let mut s = stats.borrow_mut();
                s.properties_set += nset;
                s.properties_removed += nremoved;
            }
        }
        if !self.deleted_nodes.is_empty() {
            stats.borrow_mut().nodes_deleted += self.deleted_nodes.len();
            self.deleted_node_labels = g
                .delete_nodes(&self.deleted_nodes, &mut self.index_docs)
                .map_err(|e| e.to_string())?;
        }
        // Take relationship deletions BEFORE implicit edge processing
        // so we can pass them to delete_implicit_edges for dedup.
        let explicit_rels = std::mem::take(&mut self.deleted_relationships);

        // Bulk cascade-delete edges for implicitly deleted nodes.
        // This must run after delete_nodes so that node matrices are already
        // cleaned up, and before delete_relationships so explicit edges are
        // still tracked separately.
        if !self.deleted_nodes.is_empty() {
            let implicit_edges =
                g.delete_implicit_edges(&self.deleted_nodes, &explicit_rels, &mut self.index_docs)?;
            let count = implicit_edges.len();
            stats.borrow_mut().relationships_deleted += count;
            // Record in deleted_relationships so effects buffer can serialize them
            for DeletedEdge {
                id: rel_id,
                type_id,
                src: from,
                dst: to,
            } in implicit_edges
            {
                self.deleted_relationships.insert(u64::from(rel_id));
                self.deleted_endpoints.push(DeletedEdge {
                    id: rel_id,
                    type_id,
                    src: from,
                    dst: to,
                });
            }
        }
        if !explicit_rels.is_empty() {
            let endpoints = g
                .delete_relationships(&explicit_rels, &mut self.index_docs)
                .map_err(|e| e.to_string())?;
            // Use the actually-removed relationships (delete_relationships skips
            // stale/missing ids) for stats and effects/constraint bookkeeping.
            stats.borrow_mut().relationships_deleted += endpoints.len();
            self.deleted_relationships
                .extend(endpoints.iter().map(|e| u64::from(e.id)));
            self.deleted_endpoints.extend(endpoints);
        }
        // The native index, once for the whole `Commit`. Before the constraint checks, which is
        // where each write call used to leave it, so nothing that runs after the writes sees the
        // index in a different state than before. SETs run before DELETEs above, so one batch can
        // hold an entry's add and then its remove; `apply` keeps only the last.
        g.apply_index_writes(&mut self.index_docs);

        // Enforce constraints before accumulating index operations.
        // The constraint checks read the attribute store, which already has
        // the mutations from this transaction (writes apply immediately to
        // the private MVCC graph).
        self.enforce_constraints(g)?;

        // Index operations are deferred — they will be applied only after
        // the full query succeeds to avoid stale RediSearch entries on
        // rollback.

        // Accumulate this commit's documents into the query's deferred set.
        self.deferred_docs.absorb(&mut self.index_docs);

        Ok(())
    }

    /// Enforce graph constraints on all entities affected by this transaction.
    fn enforce_constraints(
        &self,
        g: &Graph,
    ) -> Result<(), String> {
        let constraints = g.constraints();
        if constraints.is_empty() {
            return Ok(());
        }

        // Collect affected node IDs and their labels
        // Affected = created + had attributes modified + had labels added
        let mut affected_node_ids = RoaringTreemap::new();
        affected_node_ids |= &self.created_nodes;
        for &id in self.new_nodes_attrs.keys() {
            affected_node_ids.insert(id);
        }
        for &id in self.existing_nodes_attrs.keys() {
            affected_node_ids.insert(id);
        }
        for &id in self.set_labels.keys() {
            affected_node_ids.insert(id);
        }

        // Collect affected edge IDs
        let mut affected_edge_ids = RoaringTreemap::new();
        for rels in self.created_rels_by_type.values() {
            for &(rel_id, _, _) in rels {
                affected_edge_ids.insert(rel_id.into());
            }
        }
        for &id in self.new_relationships_attrs.keys() {
            affected_edge_ids.insert(id);
        }
        for &id in self.existing_relationships_attrs.keys() {
            affected_edge_ids.insert(id);
        }

        // Remove deleted entities from affected sets
        for id in &self.deleted_nodes {
            affected_node_ids.remove(id);
        }
        for rel_id in &self.deleted_relationships {
            affected_edge_ids.remove(rel_id);
        }

        // Check each OPERATIONAL constraint
        for constraint in constraints {
            if constraint.status != ConstraintStatus::Operational {
                continue;
            }

            match constraint.entity_type {
                EntityType::Node => {
                    self.check_node_constraint(g, constraint, &affected_node_ids)?;
                }
                EntityType::Relationship => {
                    self.check_edge_constraint(g, constraint, &affected_edge_ids)?;
                }
            }
        }

        Ok(())
    }

    /// Label check for constraint enforcement that answers from this
    /// transaction's own bookkeeping first. Reading the label matrix here
    /// would force a pending-tuple materialization of its delta on every
    /// commit (`O(|delta|)` per write query, quadratic between folds) —
    /// measured as the dominant cost of small repeated creates. Mirrors
    /// [`Self::update_node_labels`] semantics; `set_labels` and `remove_labels`
    /// are disjoint per node (see [`Self::stage_node_labels`]), so the two
    /// branches below cannot both match.
    fn constraint_node_has_label(
        &self,
        g: &Graph,
        node_id: u64,
        label_id: u64,
    ) -> bool {
        if let Some(removed) = self.remove_labels.get(&node_id)
            && removed.contains(&label_id)
        {
            return false;
        }
        if let Some(set) = self.set_labels.get(&node_id)
            && set.contains(&label_id)
        {
            return true;
        }
        if self.created_nodes.contains(node_id) {
            // Labels of nodes created this transaction live entirely in
            // set_labels — no need to touch the graph.
            return false;
        }
        g.node_has_label_id(node_id.into(), LabelId(label_id as usize))
    }

    fn check_node_constraint(
        &self,
        g: &Graph,
        constraint: &crate::graph::constraint::Constraint,
        affected_node_ids: &RoaringTreemap,
    ) -> Result<(), String> {
        let label = &constraint.label;
        let Some(label_id) = g.get_label_id(label) else {
            // Label doesn't exist yet — no node can carry it.
            return Ok(());
        };
        let label_id = usize::from(label_id) as u64;
        let members = affected_node_ids
            .iter()
            .filter(|&node_id| self.constraint_node_has_label(g, node_id, label_id));

        match constraint.ct {
            ConstraintType::Mandatory => {
                for node_id in members {
                    for prop in &constraint.properties {
                        let has_prop = g
                            .get_node_attribute(node_id.into(), prop)
                            .is_some_and(|val| !matches!(val, Value::Null));
                        if !has_prop {
                            return Err(format!(
                                "mandatory constraint violation: node with label {label} missing property {prop}"
                            ));
                        }
                    }
                }
            }
            ConstraintType::Unique => {
                let violated = unique_violated(
                    &constraint.properties,
                    members.map(NodeId::from),
                    affected_node_ids,
                    |node_id: NodeId, prop| g.get_node_attribute(node_id, prop),
                    |node_id| {
                        let node_id = u64::from(node_id);
                        !self.deleted_nodes.contains(node_id)
                            && self.constraint_node_has_label(g, node_id, label_id)
                    },
                    constraint
                        .properties
                        .iter()
                        .all(|prop| g.is_indexed(label, prop, &IndexType::Range)),
                    |query| g.get_indexed_nodes(label, query),
                    || {
                        g.get_label_matrix(label).into_iter().flat_map(|lm| {
                            lm.iter(0, u64::MAX)
                                .map(|(node_id, _)| NodeId::from(node_id))
                        })
                    },
                );
                if violated {
                    return Err(format!(
                        "unique constraint violation on node of type {label}"
                    ));
                }
            }
        }
        Ok(())
    }

    fn check_edge_constraint(
        &self,
        g: &Graph,
        constraint: &crate::graph::constraint::Constraint,
        affected_edge_ids: &RoaringTreemap,
    ) -> Result<(), String> {
        let type_name = &constraint.label;
        let members = affected_edge_ids.iter().filter(|&edge_id| {
            // Edges created this transaction resolve their type from
            // Pending's reverse index — no relationship-matrix read, which
            // would materialize the delta's pending tuples on every commit.
            // An edge's type never changes, so created under a different
            // type means not a member.
            match self.created_rel_types.get(&edge_id.into()) {
                Some(created_type) => created_type.as_str() == type_name.as_str(),
                None => g.edge_has_type(edge_id.into(), type_name),
            }
        });

        match constraint.ct {
            ConstraintType::Mandatory => {
                for edge_id in members {
                    for prop in &constraint.properties {
                        let has_prop = g
                            .get_relationship_attribute(edge_id.into(), prop)
                            .is_some_and(|val| !matches!(val, Value::Null));
                        if !has_prop {
                            return Err(format!(
                                "mandatory constraint violation: edge with relationship-type {type_name} missing property {prop}"
                            ));
                        }
                    }
                }
            }
            ConstraintType::Unique => {
                let violated = unique_violated(
                    &constraint.properties,
                    members.map(RelationshipId::from),
                    affected_edge_ids,
                    |edge_id: RelationshipId, prop| g.get_relationship_attribute(edge_id, prop),
                    // Both the index and the tensor hold edges of this type
                    // only, and a type never changes.
                    |edge_id| !self.deleted_relationships.contains(edge_id.into()),
                    constraint
                        .properties
                        .iter()
                        .all(|prop| g.is_edge_indexed(type_name, prop, &IndexType::Range)),
                    |query| {
                        g.get_indexed_edges(type_name, query)
                            .map(|(_, _, edge_id)| edge_id)
                    },
                    || {
                        g.get_relationship_matrix(type_name)
                            .into_iter()
                            .flat_map(|t| {
                                t.iter(0, u64::MAX, false)
                                    .map(|(_, _, edge_id)| RelationshipId::from(edge_id))
                            })
                    },
                );
                if violated {
                    return Err(format!(
                        "unique constraint violation, on edge of relationship-type {type_name}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Take the accumulated index document changes, leaving pending empty.
    pub fn take_deferred_indexes(&mut self) -> IndexDocs {
        std::mem::take(&mut self.deferred_docs)
    }

    /// Undo the index documents earlier `Commit`s published, after this query
    /// failed, by re-synchronising them against committed state.
    ///
    /// Mirrors C's undo log (`_UndoLog_Rollback_Update_Entity`): a failed *create*
    /// loses its document, a failed *update* is re-indexed from the entity's
    /// previous values — deleting it instead would drop a live entity out of the
    /// index. No undo log is needed for that, because MVCC still holds the previous
    /// state: `committed` *is* the old value, and re-adding rewrites the document.
    ///
    /// `private` is the version this query built and is about to discard; a removed
    /// edge's document key comes from there, since committed state no longer has the
    /// edge at all.
    ///
    /// Writer mode only, and before the write lock is released.
    pub fn resync_published_indexes(
        &mut self,
        committed: &AtomicRefCell<Graph>,
        private: &Graph,
    ) {
        let (nodes, edges) = std::mem::take(&mut self.published).ids_by_slot();
        if nodes.is_empty() && edges.is_empty() {
            return;
        }

        let mut node_adds: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        let mut node_removes: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        let mut edge_adds: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        let mut edge_removes: FxHashMap<u64, FxHashMap<u64, (u64, u64)>> = FxHashMap::default();
        {
            let g = committed.borrow();
            for (slot, ids) in &nodes {
                for id in ids {
                    if g.node_has_label_id(NodeId::from(id), LabelId(*slot as usize)) {
                        // Still live: rewrite the document from committed values.
                        node_adds.entry(*slot).or_default().insert(id);
                    } else {
                        // Never existed outside the discarded version.
                        node_removes.entry(*slot).or_default().insert(id);
                    }
                }
            }
            let p = private;
            for (slot, ids) in &edges {
                for id in ids {
                    if g.endpoints_for_edge(id).is_some() {
                        edge_adds.entry(*slot).or_default().insert(id);
                    } else if let Some(endpoints) = p.endpoints_for_edge(id) {
                        // The doc key is `(src, dst, edge_id)` as written, and only the
                        // discarded version still knows the endpoints.
                        edge_removes.entry(*slot).or_default().insert(id, endpoints);
                    }
                }
            }
        }

        // Indexer locks *before* the graph borrow — the order `populate_index_batch`
        // uses, and the only one that keeps this mutable borrow of the published
        // version from colliding with a populate batch reading it (`AtomicRefCell`
        // reports that collision by panicking). Concurrent *readers* are excluded
        // instead by the write lock this query holds.
        let (node_lock, edge_lock) = committed.borrow().index_locks();
        let node_guard = node_lock.lock();
        let edge_guard = edge_lock.lock();
        let mut g = committed.borrow_mut();
        g.commit_index_locked(&node_guard, &mut node_adds, &mut node_removes);
        g.commit_edge_index_locked(&edge_guard, &mut edge_adds, &mut edge_removes);
    }

    /// Write this `Commit`'s index documents to RediSearch. Writer mode only.
    pub fn commit_deferred_indexes(
        &mut self,
        g: &mut Graph,
    ) {
        let mut deferred = self.take_deferred_indexes();
        // Fold the batch into the published log *before* writing it, because `commit`
        // drains the batch. Removals matter as much as additions: a rolled-back
        // DELETE must get its document back, or the entity silently disappears from
        // the index.
        self.published.merge(&deferred);
        deferred.commit(g);
    }

    /// End the segment: discard its accumulated mutations and roll the graph's
    /// id batches over to where the commit just left the boundary.
    ///
    /// One call because they are one event, and because splitting them was
    /// silent: a batch left unopened does not fail, it just stops reclaiming ids
    /// across the boundary, so `CREATE (n) DELETE n WITH 1 AS x CREATE (m)`
    /// allocates id 1 where it should allocate 0 — measured, with the whole flow
    /// suite passing. Against a C primary that is a divergence in id-reuse
    /// order, which costs a full resync per delete-then-create cycle.
    ///
    /// Fused, forgetting this also forgets to discard the segment's mutations,
    /// so the next commit re-creates live ids and the query dies with `node 0 is
    /// already live below the boundary 0`. That covers every segment that
    /// created something which survived, and **not** one whose only content was
    /// a cancellation — nothing to re-create, so nothing collides. Both
    /// measured. That last case cannot be closed from here: a batch that never
    /// rolled over is indistinguishable from a longer one, and
    /// [`IdSpace::verify`] passes on either. Catching it needs a notion of "a
    /// segment happened" that the id space does not have.
    ///
    /// Runs after every `Commit`, so it must NOT touch `published` — a later failure
    /// has to undo the documents *all* of this query's `Commit`s published, not just
    /// the last one's.
    ///
    /// # Errors
    ///
    /// [`NodeOpError::IdSpace`] if an id space already contradicts itself;
    /// opening a batch over it would re-anchor on the bad value and hide it.
    pub fn end_segment(
        &mut self,
        g: &mut Graph,
    ) -> Result<(), NodeOpError> {
        // Dropping millions of per-entity Vec allocations is O(n) frees and
        // stalls the serialized write thread; move large maps to a background
        // thread and let it pay the deallocation cost.
        const OFFLOAD_THRESHOLD: usize = 4096;
        let big_entries = self.new_nodes_attrs.len()
            + self.existing_nodes_attrs.len()
            + self.new_relationships_attrs.len()
            + self.existing_relationships_attrs.len()
            + self.set_labels.len()
            + self.remove_labels.len()
            + self
                .created_rels_by_type
                .values()
                .map(Vec::len)
                .sum::<usize>();
        if big_entries >= OFFLOAD_THRESHOLD {
            let maps = (
                std::mem::take(&mut self.new_nodes_attrs),
                std::mem::take(&mut self.existing_nodes_attrs),
                std::mem::take(&mut self.new_relationships_attrs),
                std::mem::take(&mut self.existing_relationships_attrs),
                std::mem::take(&mut self.set_labels),
                std::mem::take(&mut self.remove_labels),
                std::mem::take(&mut self.created_rels_by_type),
            );
            std::thread::spawn(move || drop(maps));
        } else {
            self.new_nodes_attrs.clear();
            self.existing_nodes_attrs.clear();
            self.new_relationships_attrs.clear();
            self.existing_relationships_attrs.clear();
            self.set_labels.clear();
            self.remove_labels.clear();
            self.created_rels_by_type.clear();
        }
        self.created_nodes.clear();
        self.created_rel_types.clear();

        self.deleted_nodes.clear();
        self.deleted_relationships.clear();
        self.deleted_endpoints.clear();
        self.deleted_node_labels.clear();
        self.cancelled_nodes.clear();
        self.taken_relationship_ids.clear();
        self.cancelled_relationships.clear();
        self.cancelled_properties = 0;
        self.index_docs.node_adds.clear();
        self.index_docs.node_removes.clear();
        self.index_docs.edge_adds.clear();
        self.index_docs.edge_removes.clear();
        #[cfg(feature = "index-falkordb")]
        self.index_docs.native.clear();

        // Close this segment's batch and open the next at the boundary the
        // commit just left. Closing is where the batch's shape is owed —
        // `taken` must fill the range from the entry boundary upward — and it is
        // asked here rather than earlier because it is legitimately false
        // partway through: a segment creates, cancels and deletes in whatever
        // order the query says.
        //
        // This is the one check on this path that can fail a user's query rather
        // than a replica's buffer. It fires only on an id space this engine has
        // already corrupted, and failing is the cheaper outcome: the query holds
        // a private MVCC version, so the damage is discarded rather than
        // published and then replicated.
        g.roll_id_batches()
    }

    /// Returns the number of effects (operations) tracked in this Pending.
    #[must_use]
    pub fn effects_count(&self) -> u64 {
        self.created_nodes.len()
            + self.created_rel_types.len() as u64
            + self.deleted_nodes.len()
            + self.deleted_relationships.len()
            + self.new_nodes_attrs.len() as u64
            + self.existing_nodes_attrs.len() as u64
            + self.new_relationships_attrs.len() as u64
            + self.existing_relationships_attrs.len() as u64
            + self
                .set_labels
                .values()
                .map(|v| v.len() as u64)
                .sum::<u64>()
            + self
                .remove_labels
                .values()
                .map(|v| v.len() as u64)
                .sum::<u64>()
    }
}

#[cfg(test)]
mod label_effect_tests {
    use super::*;
    use crate::effects::v3::emit::for_each_record;
    use crate::effects::v3::{IdList, Record};
    use crate::graph::graphblas::test_init::ensure_init;

    /// The one node every scenario below stages labels on.
    const NODE: u64 = 0;

    /// A graph whose only labels are `names`, and a `Pending` that has already
    /// taken its schema baseline — so the emitter announces no schema, and the
    /// label records are the whole of what it produces.
    fn fixture(names: &[&str]) -> (AtomicRefCell<Graph>, Vec<LabelId>, Pending) {
        ensure_init();
        let g = AtomicRefCell::new(Graph::new(16, 16, 1, 0, "label_effects"));
        let labels: Vec<LabelId> = {
            let mut graph = g.borrow_mut();
            names.iter().map(|n| graph.get_label_id_mut(n)).collect()
        };
        let mut pending = Pending::new();
        pending.set_schema_baseline(&g.borrow());
        (g, labels, pending)
    }

    /// Every record the emitter produces, in emission order.
    ///
    /// The assertions below compare this whole vec against an expected one
    /// rather than searching it for a record, because a search can only show
    /// that something is present. Whole-vec equality pins an absence just as
    /// tightly, which is the half that matters here.
    fn records(
        pending: &Pending,
        g: &AtomicRefCell<Graph>,
    ) -> Vec<Record> {
        let mut out = Vec::new();
        for_each_record(pending, &g.borrow(), |r| out.push(r));
        out
    }

    /// A label record naming [`NODE`] alone, shaped as `digest_labels` builds
    /// it: the label ids sorted and deduped.
    fn labelled(
        add: bool,
        labels: &[LabelId],
    ) -> Record {
        let ids: IdList = [NODE].into_iter().collect();
        let mut shape: Vec<u32> = labels.iter().map(|l| usize::from(*l) as u32).collect();
        shape.sort_unstable();
        shape.dedup();
        if add {
            Record::SetLabels { ids, labels: shape }
        } else {
            Record::RemoveLabels { ids, labels: shape }
        }
    }

    /// Positive control for [`cancelled_add_emits_no_set_labels`]: an
    /// uncancelled `SET n:L` does produce a `SetLabels`. Without this, an
    /// emitter that had stopped producing label records altogether would
    /// satisfy the cancellation tests vacuously.
    #[test]
    fn plain_set_emits_set_labels() {
        let (g, labels, mut pending) = fixture(&["L"]);
        let l = labels[0];

        pending.set_node_labels(NODE.into(), &[l].into_iter().collect());

        assert_eq!(records(&pending, &g), vec![labelled(true, &[l])]);
    }

    /// Positive control for [`cancelled_removal_emits_no_remove_labels`]: an
    /// uncancelled `REMOVE n:L` does produce a `RemoveLabels`.
    #[test]
    fn plain_remove_emits_remove_labels() {
        let (g, labels, mut pending) = fixture(&["L"]);
        let l = labels[0];

        pending.remove_node_labels(NODE.into(), &[l]);

        assert_eq!(records(&pending, &g), vec![labelled(false, &[l])]);
    }

    /// `SET n:L REMOVE n:L`: the removal cancels the staged add, and the
    /// emptied `set_labels` entry goes with it.
    ///
    /// Two layers keep the cancelled add off the wire, and this checks both.
    /// The map assertion is the one this fix owns: dropping the emptied entry
    /// is what stops a zero-label record existing to be emitted. The record
    /// assertion is the outcome, which `digest_labels` also defends by
    /// suppressing an empty label set — so it holds even if the cleanup here
    /// regresses, and only the map assertion would catch that.
    #[test]
    fn cancelled_add_emits_no_set_labels() {
        let (g, labels, mut pending) = fixture(&["L"]);
        let l = labels[0];

        pending.set_node_labels(NODE.into(), &[l].into_iter().collect());
        pending.remove_node_labels(NODE.into(), &[l]);

        assert!(
            !pending.set_labels.contains_key(&NODE),
            "the emptied set_labels entry must be dropped, not left empty"
        );
        assert_eq!(
            records(&pending, &g),
            vec![labelled(false, &[l])],
            "the cancelled add must leave no SetLabels behind"
        );
        assert_eq!(pending.node_has_label(NODE.into(), l), Some(false));
    }

    /// `REMOVE n:L SET n:L`, the mirror direction and the #2777 report: the add
    /// cancels the staged removal, the label survives, and the emptied
    /// `remove_labels` entry is not left behind either.
    #[test]
    fn cancelled_removal_emits_no_remove_labels() {
        let (g, labels, mut pending) = fixture(&["L"]);
        let l = labels[0];

        pending.remove_node_labels(NODE.into(), &[l]);
        pending.set_node_labels(NODE.into(), &[l].into_iter().collect());

        assert!(
            !pending.remove_labels.contains_key(&NODE),
            "the emptied remove_labels entry must be dropped, not left empty"
        );
        assert_eq!(
            records(&pending, &g),
            vec![labelled(true, &[l])],
            "the cancelled removal must leave no RemoveLabels behind"
        );
        assert_eq!(
            pending.node_has_label(NODE.into(), l),
            Some(true),
            "the re-added label must survive the cancelled removal"
        );
    }

    /// Cancellation is per label, not per node: `SET n:L:M REMOVE n:L` drops
    /// only `L` from the add, and both records still go out.
    #[test]
    fn cancelling_one_label_of_several_keeps_the_rest() {
        let (g, labels, mut pending) = fixture(&["L", "M"]);
        let (l, m) = (labels[0], labels[1]);

        pending.set_node_labels(NODE.into(), &[l, m].into_iter().collect());
        pending.remove_node_labels(NODE.into(), &[l]);

        assert_eq!(pending.node_has_label(NODE.into(), m), Some(true));
        assert_eq!(
            records(&pending, &g),
            vec![labelled(true, &[m]), labelled(false, &[l])]
        );
    }
}

#[cfg(test)]
mod property_nesting_tests {
    use super::{INVALID_PROPERTY_MSG, validate_node_property, validate_relationship_property};
    use crate::runtime::value::{MAX_VALUE_NESTING, Value};
    use std::sync::Arc;
    use thin_vec::thin_vec;

    fn nested(depth: usize) -> Value {
        let mut v = Value::Int(1);
        for _ in 0..depth {
            v = Value::List(Arc::new(thin_vec![v]));
        }
        v
    }

    /// Validation is recursive, so it runs on the query pool's stack size.
    fn on_worker_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn a_property_at_the_cap_is_valid() {
        on_worker_stack(|| {
            let v = nested(MAX_VALUE_NESTING);
            assert_eq!(validate_node_property(&v), Ok(()));
            assert_eq!(validate_relationship_property(&v), Ok(()));
        });
    }

    #[test]
    fn a_property_past_the_cap_is_refused() {
        // Every stored value has to load back, and the RDB decoder refuses
        // anything deeper than the cap.
        on_worker_stack(|| {
            let v = nested(MAX_VALUE_NESTING + 1);
            let expected = Err(format!(
                "Property values can only be nested up to {MAX_VALUE_NESTING} levels deep"
            ));
            assert_eq!(validate_node_property(&v), expected);
            assert_eq!(validate_relationship_property(&v), expected);
        });
    }

    #[test]
    fn non_primitive_elements_keep_their_error() {
        let v = Value::List(Arc::new(thin_vec![Value::Map(Arc::default())]));
        assert_eq!(validate_node_property(&v), Err(INVALID_PROPERTY_MSG.into()));
        assert_eq!(
            validate_node_property(&Value::List(Arc::new(thin_vec![Value::Null]))),
            Err(INVALID_PROPERTY_MSG.into())
        );
        assert_eq!(validate_node_property(&Value::Null), Ok(()));
    }
}
