//! Keeping CCH indexes correct after writes (#3214).
//!
//! CCH maintenance rides the native index's unit-of-work carrier. Each graph write method that
//! can change what a CCH index answers calls one of the recording functions below, which read
//! the graph and the held hierarchies and record the write into the unit's change set,
//! [`IndexDocs::cch`](crate::index::index_docs::IndexDocs::cch), next to the native index's
//! tuples and RediSearch's documents. The owner of the unit flushes it once its graph writes
//! are all in ([`Graph::apply_cch_writes`]), on the private version: a Cypher query at the end
//! of each `Commit` segment, a replicated effects buffer at the end of the buffer,
//! `GRAPH.BULK` once every token is in, before it escalates. Each held index the unit changed
//! gets a new `Arc<Cch>`; the rest keep their `Arc` untouched.
//!
//! # The change set
//!
//! Per held index, a [`CchChanges`]:
//! - **updated pairs**: the undirected node pairs `{u, v}` whose cheapest edge may have
//!   changed: an edge of an indexed type created, deleted, or its weight attribute set or
//!   removed. A pair the hierarchy already has an arc for is absorbed by a scoped
//!   recustomization, whatever happened to it: a weight change reseeds the arc, a deleted
//!   edge (even the last of its pair) reseeds it with `+inf`, a new edge between two nodes
//!   the hierarchy already joins (an original edge or a fill-in arc) seeds it.
//! - **structural**: something the hierarchy cannot represent, so it is rebuilt: a new edge
//!   between nodes it has no arc for, an endpoint at or past its node count (its capacity:
//!   the graph's matrix dimension when it was built) included. Deletions never are, and
//!   neither is a created node, within the capacity or past it: it has no edge yet, so no
//!   route changes, and a query naming it finds no path either way. The edge that later
//!   joins it is what is structural.
//!
//! Classification costs one lookup per written edge, against the held hierarchy itself
//! ([`Topology::arc_slot`](super::order::Topology::arc_slot): two rank reads and a binary
//! search in one rank's upward row), not against the graph: the hierarchy is what has to
//! absorb the pair, and it has more arcs than the graph has pairs (fill-in, and the arcs of
//! deleted edges), so it absorbs every pair a graph lookup would and more. It needs no
//! GraphBLAS call, and stays right inside a unit, where the graph already holds the unit's
//! own earlier writes.
//!
//! The pair set is capped at [`PAIR_CAP`] distinct pairs per index; past it only "many" is
//! kept and the flush runs a full customization (still not a rebuild).
//!
//! # The flush
//!
//! Per changed index ([`flush`]), while it is built (one still `Building` keeps the edge ids
//! written, for the build's commit; see [`CchChanges`]):
//! - structural: the entry is set back to `Building` under a new generation, and the commit
//!   hands it to the index builder, which rebuilds it off the lock: no hierarchy is built in
//!   the write path. Until the build is adopted this version holds no hierarchy for it; older
//!   versions keep theirs;
//! - past the cap: a full customization that shares the topology, the weights sized to the
//!   hierarchy ([`CchInput::weights_for`]) and [`Cch::customized`], so a node capacity grown
//!   since the build is no reason to rebuild;
//! - otherwise: [`Cch::recustomize_seeded`] on a copy ([`Cch::new_version`], which shares the
//!   metric's chunks and copies only those it writes), seeded pair by pair from the graph
//!   ([`PairWeights`]); no weight matrix is built. Its `n / 8` valve hands over to the full
//!   customization, and its `NeedsRebuild` (which the classification makes unreachable) to a
//!   rebuild in the background.
//!
//! Nothing is kept on the version and nothing in `Pending` beyond the carrier it already
//! holds: a failed unit drops its carrier, and a failed flush fails the unit, whose private
//! version (with any hierarchy already updated into it) is dropped by the existing rollback.
//!
//! Recording is gated on the graph holding at least one CCH index, so a graph without one
//! never fills a change set and the flush is a single `is_empty()` check.
//!
//! The write methods that set relationship attributes also refuse, before writing, a weight a
//! held index cannot take ([`check_edge_weights`]), the way a constraint refuses a value. An
//! index never sees an invalid weight through a write: what can still fail a flush is the
//! graph's size, a resource, or a bug.
//!

use std::fmt;
use std::sync::Arc;

use roaring::RoaringTreemap;
use rustc_hash::{FxHashMap, FxHashSet};

use super::input::{PairWeights, is_valid_weight};
use super::{Cch, CchError, CchIndex, CchIndexDef, CchInput, Hierarchy, Recustomized};
use crate::graph::graph::{Graph, RelationshipId, TypeId};
use crate::index::falkordb::versioned::IndexEntry;
use crate::runtime::value::Value;

/// Distinct updated pairs kept per index and unit of work; past it the index only records
/// "many" and the flush runs a full customization instead of a scoped one.
///
/// A cost line, measured on California (1.25 M nodes, 5.93 M arcs; in process, release,
/// median of 3): one changed pair recomputes a cone of a handful to tens of thousands of arcs,
/// and the scoped path costs about 1.3 us per recomputed arc (two graph lookups seed each)
/// against about 0.09 us per arc for the full sweep. k random pairs cost 0.41 s at k = 512
/// and 0.51 s at k = 1,024, where the full path (weight matrix and customize) costs about
/// 0.5 s; at 4,096 pairs the scoped path is 0.95 s. So up to the cap a unit is scoped, and
/// past it the full path's fixed cost is the cheaper bound. Pairs that cluster share their
/// cones and stay cheaper. The sets hold at most 1,024 pairs per index (about 35 KiB with
/// the hash tables' slack). (The scoped path's own valve, `n / 8` distinct arcs, is about
/// 190 k on California, where `n` is the node capacity, 1.54 M, so it never decides there.)
const PAIR_CAP: usize = 1 << 10;

/// What a unit of work changed for one held index. See the module doc.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub(crate) struct CchChanges {
    /// The hierarchy cannot represent the change: it is rebuilt. Once set, pairs are no
    /// longer recorded.
    pub(crate) structural: bool,
    /// More than [`PAIR_CAP`] distinct pairs were updated; `pairs` was dropped.
    overflow: bool,
    /// Updated undirected pairs, `(min, max)`, self-loops excluded.
    pairs: FxHashSet<(u64, u64)>,
    /// For an index still `Building`, which has no hierarchy to classify against: the ids of
    /// the edges written, which the flush hands to the entry
    /// ([`VersionedIndex::writer_record`](crate::index::falkordb::versioned::VersionedIndex::writer_record))
    /// for the build's commit to catch up on.
    edges: RoaringTreemap,
}

impl CchChanges {
    /// The updated pairs, `(min, max)`, in no particular order; empty past the cap or once
    /// structural.
    pub(crate) fn pairs(&self) -> impl ExactSizeIterator<Item = (u64, u64)> + '_ {
        self.pairs.iter().copied()
    }

    fn set_structural(&mut self) {
        self.structural = true;
        self.pairs = FxHashSet::default();
    }

    fn push(
        &mut self,
        u: u64,
        v: u64,
    ) {
        if self.structural || self.overflow || u == v {
            return;
        }
        if self.pairs.insert((u.min(v), u.max(v))) && self.pairs.len() > PAIR_CAP {
            self.overflow = true;
            self.pairs = FxHashSet::default();
        }
    }

    /// An edge `u - v` that exists after this write (created, or its weight written):
    /// absorbable when `hierarchy` already joins the two, structural otherwise.
    fn classify(
        &mut self,
        hierarchy: &Cch,
        u: u64,
        v: u64,
    ) {
        if self.structural || u == v {
            return;
        }
        if hierarchy.joins(u, v) {
            self.push(u, v);
        } else {
            self.set_structural();
        }
    }
}

/// A unit of work's CCH change set, staged in its
/// [`IndexDocs::cch`](crate::index::index_docs::IndexDocs::cch) until the owner flushes it:
/// one [`CchChanges`] per held index it changed.
///
/// Empty for every graph with no CCH index, so the maintenance hook costs one `is_empty()`
/// check on the existing write path. `Default` allocates nothing.
#[derive(Clone, Default, Debug)]
pub(crate) struct CchDirty {
    /// Keyed by definition, not position: a graph holds a handful of indexes at most.
    indexes: Vec<(CchIndexDef, CchChanges)>,
}

impl CchDirty {
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    /// Whether the index for `def` may answer differently after these changes.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn touches(
        &self,
        def: &CchIndexDef,
    ) -> bool {
        self.get(def).is_some()
    }

    /// What changed for the index over `def`, if anything.
    #[must_use]
    pub(crate) fn get(
        &self,
        def: &CchIndexDef,
    ) -> Option<&CchChanges> {
        self.indexes.iter().find(|(d, _)| d == def).map(|(_, c)| c)
    }

    fn entry(
        &mut self,
        def: &CchIndexDef,
    ) -> &mut CchChanges {
        let at = match self.indexes.iter().position(|(d, _)| d == def) {
            Some(at) => at,
            None => {
                self.indexes.push((def.clone(), CchChanges::default()));
                self.indexes.len() - 1
            }
        };
        &mut self.indexes[at].1
    }
}

// ---- What each graph write records ----
//
// Called by the graph's write methods while what they read (held indexes, edge types and
// endpoints) is still in the graph. Each is one `is_empty()` on a graph with no CCH index.

/// Edges of `type_id` created, as `(id, src, dst)`: each classified against every held index
/// spanning the type.
pub(crate) fn edges_created(
    g: &Graph,
    dirty: &mut CchDirty,
    type_id: TypeId,
    edges: impl Iterator<Item = (u64, u64, u64)> + Clone,
) {
    if edges.clone().next().is_none() {
        return;
    }
    for entry in spanning(g, type_id) {
        let changes = dirty.entry(&entry.index.def);
        for (id, u, v) in edges.clone() {
            record(changes, entry, id, u, v, true);
        }
    }
}

/// Edges of `type_id` deleted, as `(id, src, dst)`, explicitly or by a node deletion's
/// cascade: updated pairs, never structural (the arc stays, reseeded with `+inf` if it was the
/// pair's last edge).
pub(crate) fn edges_deleted(
    g: &Graph,
    dirty: &mut CchDirty,
    type_id: TypeId,
    edges: impl Iterator<Item = (u64, u64, u64)> + Clone,
) {
    if edges.clone().next().is_none() {
        return;
    }
    for entry in spanning(g, type_id) {
        let changes = dirty.entry(&entry.index.def);
        for (id, u, v) in edges.clone() {
            record(changes, entry, id, u, v, false);
        }
    }
}

/// Attributes set or removed on existing relationships: the pair of every edge whose written
/// attributes include the weight of a held index spanning its type, classified (an existing
/// edge's pair has an arc, so in practice this records the pair). `type_id` is the edges'
/// shared type when the caller knows it; `None` looks it up, once per edge that writes a
/// weight.
pub(crate) fn edge_attrs_set(
    g: &Graph,
    dirty: &mut CchDirty,
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
    type_id: Option<TypeId>,
) {
    let held = &g.falkordb_index.cch_indexes;
    if held.is_empty() {
        return;
    }
    for (&id, written) in attrs {
        if !written
            .iter()
            .any(|(attribute, _)| held.keys().any(|def| def.weight_attr == *attribute))
        {
            continue;
        }
        let Some((u, v)) = g.endpoints_for_edge(id) else {
            continue;
        };
        let edge_type =
            type_id.unwrap_or_else(|| g.get_relationship_type_id(RelationshipId::from(id)));
        for entry in spanning(g, edge_type) {
            if written
                .iter()
                .any(|(attribute, _)| *attribute == entry.index.def.weight_attr)
            {
                record(dirty.entry(&entry.index.def), entry, id, u, v, true);
            }
        }
    }
}

/// Record one written edge `id`, `u - v`, for one index, as its state takes it: a `Building`
/// entry keeps the id, a built one classifies the pair (`exists`: the edge is there after the
/// write, created or its weight set) or records it as updated (deleted), and one whose build
/// failed keeps nothing.
fn record(
    changes: &mut CchChanges,
    entry: &IndexEntry<CchIndex>,
    id: u64,
    u: u64,
    v: u64,
    exists: bool,
) {
    if entry.is_building() {
        changes.edges.insert(id);
    } else if let Some(hierarchy) = entry.index.cch() {
        if exists {
            changes.classify(hierarchy, u, v);
        } else {
            changes.push(u, v);
        }
    }
}

/// The held entries whose definition spans `type_id`.
fn spanning(
    g: &Graph,
    type_id: TypeId,
) -> impl Iterator<Item = &IndexEntry<CchIndex>> {
    g.falkordb_index
        .cch_indexes
        .values()
        .filter(move |entry| entry.index.def.rel_types().contains(&type_id))
}

/// Refuse attribute writes that would give an edge of an indexed type an invalid weight, like a
/// constraint: the unit of work fails before anything is written, and rolls back like any
/// failing write. "Invalid" is [`is_valid_weight`]'s rule: a negative number (`-inf` included)
/// or NaN; `+inf` is legal (a closed edge), and so is removing the weight or writing a
/// non-numeric value (it weighs 1.0).
///
/// Called by every graph method that writes relationship attributes, before the write, so one
/// check covers Cypher, replicated effects and `GRAPH.BULK`. `type_id` is the edges' shared type
/// when the caller knows it. Costs one `is_none()` on a graph with no CCH index; with one, a
/// compare per written attribute against the held weight attributes, and the edge's type is
/// looked up only for an invalid value of a weight attribute, so a valid write never reads it.
///
/// It reads the held index definitions, never their hierarchies, so it holds for every index
/// the version holds, whatever state its hierarchy is in.
///
/// # Errors
/// The first offending write, naming the edge's type, the attribute, the value and the index.
pub(crate) fn check_edge_weights<'a>(
    g: &Graph,
    attrs: impl IntoIterator<Item = (&'a u64, &'a Vec<(u16, Value)>)>,
    type_id: Option<TypeId>,
) -> Result<(), String> {
    if g.cch_indexes().next().is_none() {
        return Ok(());
    }
    for (&edge_id, written) in attrs {
        for (attribute, value) in written {
            let is_weight = g.cch_indexes().any(|def| def.weight_attr == *attribute);
            if !is_weight || is_valid_weight(value) {
                continue;
            }
            let edge_type = match type_id {
                Some(edge_type) => edge_type,
                // An edge this graph does not hold is not ours to refuse: the write itself
                // reports it.
                None if g.endpoints_for_edge(edge_id).is_none() => continue,
                None => g.get_relationship_type_id(RelationshipId::from(edge_id)),
            };
            let Some(def) = g
                .cch_indexes()
                .find(|def| def.weight_attr == *attribute && def.rel_types().contains(&edge_type))
            else {
                continue;
            };
            let value = match value {
                Value::Float(weight) => weight.to_string(),
                Value::Int(integer) => integer.to_string(),
                _ => unreachable!("only a number can be an invalid weight"),
            };
            return Err(format!(
                "CCH weight constraint violation: edge of relationship-type {} cannot have \
                 {} = {value}; CCH index {} needs weights that are >= 0 (+inf marks a closed \
                 edge)",
                type_name(g, edge_type),
                attribute_name(g, *attribute),
                index_name(g, def)
            ));
        }
    }
    Ok(())
}

/// A held index met a weight the write-time check should have refused. The write that caused
/// it fails, and its rollback leaves the committed indexes as they were. Every other reason an
/// index cannot be updated hands it to the builder instead ([`flush`]).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CchFlushError {
    def: CchIndexDef,
    /// The index as a user would write it, `(:A|B {w})`, resolved when the error was made.
    name: String,
    cause: CchError,
}

impl fmt::Display for CchFlushError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let cause: &dyn fmt::Display = match &self.cause {
            // `check_edge_weights` refuses every write of one and an index is never created over
            // one, so reaching it here is a write path that skipped the check.
            CchError::InvalidWeight => {
                &"internal error: an invalid weight got past the write-time check"
            }
            CchError::UnknownRelType => &"unknown relationship type",
            CchError::UnknownAttribute => &"unknown weight attribute",
            CchError::NeedsRebuild => &self.cause,
            CchError::Internal(msg) => msg,
            CchError::Ordering(error) => error,
            CchError::AlreadyExists | CchError::NotFound | CchError::Cancelled => {
                &"unexpected index state"
            }
        };
        write!(
            f,
            "CCH index {} could not be updated after this write: {cause}",
            self.name
        )
    }
}

/// How [`flush`] brought one index up to date.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Update {
    /// The hierarchy cannot take the change: the entry is set back to `Building` under a new
    /// generation, and the commit hands it to the index builder. Nothing is built in the write.
    Rebuild,
    /// The old topology, every arc recomputed from the graph's weights
    /// ([`CchInput::weights_for`]).
    Full,
    /// The old topology, only the arcs the updated pairs reach recomputed, seeded from the
    /// graph pair by pair; no [`CchInput`] built.
    Scoped,
}

/// Bring every held CCH index that `dirty` changed up to date with this version of the
/// graph, on the private version the writes went into: what changes is published together
/// with the graph. [`Graph::apply_cch_writes`] hands it what a unit of work staged.
///
/// Per entry:
/// - `Building`: the edge ids written join its written set
///   ([`VersionedIndex::writer_record`](crate::index::falkordb::versioned::VersionedIndex::writer_record));
///   the build's commit catches up on them;
/// - built: a scoped or full recustomization into its own new `Arc<Cch>` (the committed one is
///   never written); a change the hierarchy cannot take sets the entry back to `Building`
///   under a new generation, for the index builder ([`Update::Rebuild`]), so no hierarchy is
///   built in the write path;
/// - failed: nothing, until it is dropped and created again.
///
/// An index that cannot be updated in place for any other reason (a resource limit, a bug in
/// the update) is not the write's fault: it is handed to the builder as for
/// [`Update::Rebuild`], and the write goes on. The builder builds it from scratch, or holds it
/// `FAILED` with the reason.
///
/// Every update is computed before any entry is changed, so on error none is.
///
/// # Errors
/// An invalid weight, which the write-time check ([`check_edge_weights`]) refuses before
/// anything is written, so here it is a bug: it fails the unit, and the error says so.
pub(crate) fn flush(
    g: &mut Graph,
    dirty: &CchDirty,
) -> Result<(), CchFlushError> {
    #[cfg(test)]
    tests::FLUSHES.with(|c| c.set(c.get() + 1));
    if g.falkordb_index.cch_indexes.is_empty() || dirty.is_empty() {
        return Ok(());
    }
    let mut written = Vec::new();
    let mut updated = Vec::new();
    let mut rebuilt = Vec::new();
    let mut warnings = Vec::new();
    for (def, entry) in &g.falkordb_index.cch_indexes {
        let Some(changes) = dirty.get(def) else {
            continue;
        };
        if entry.is_building() {
            if !changes.edges.is_empty() {
                written.push((def.clone(), changes.edges.clone()));
            }
            continue;
        }
        let Some(hierarchy) = entry.index.cch() else {
            continue;
        };
        let (cch, how) = match update(g, def, hierarchy, changes) {
            Ok(done) => done,
            Err(CchError::InvalidWeight) => {
                return Err(CchFlushError {
                    def: def.clone(),
                    name: index_name(g, def),
                    cause: CchError::InvalidWeight,
                });
            }
            // Not the write's fault: the builder builds the index again, or records why not.
            // Said out loud, since only a bug or a resource failure gets here.
            Err(cause) => {
                warnings.push(format!(
                    "CCH index {} could not be updated ({cause}); rebuilding it in the background",
                    index_name(g, def)
                ));
                (None, Update::Rebuild)
            }
        };
        #[cfg(test)]
        tests::UPDATES.with(|u| u.borrow_mut().push(how));
        match cch {
            Some(cch) => updated.push((def.clone(), Arc::new(cch))),
            None => rebuilt.push(def.clone()),
        }
        let _ = how;
    }
    // Every update is computed, so none of them fails the flush from here on.
    for warning in warnings {
        g.falkordb_index.cch_warnings.push(warning);
    }
    for (def, edges) in written {
        let entry = g.falkordb_index.cch_indexes.get_mut(&def).expect("held");
        entry.writer_record(edges, |_| None, &mut ());
    }
    for (def, cch) in updated {
        let entry = g.falkordb_index.cch_indexes.get_mut(&def).expect("held");
        entry.index.hierarchy = Hierarchy::Built(cch);
    }
    for def in rebuilt {
        g.falkordb_index.rebuild_cch_in_background(&def);
    }
    Ok(())
}

/// The updated hierarchy for the index over `def`, held as `hierarchy`, after `changes`, and
/// which update produced it; `None` for [`Update::Rebuild`], which only the builder does.
/// Reads the graph and the committed hierarchy; writes neither.
fn update(
    g: &Graph,
    def: &CchIndexDef,
    hierarchy: &Cch,
    changes: &CchChanges,
) -> Result<(Option<Cch>, Update), CchError> {
    #[cfg(test)]
    if let Some(cause) = tests::FAIL_UPDATE.take() {
        return Err(cause);
    }
    if changes.structural {
        return Ok((None, Update::Rebuild));
    }
    if !changes.overflow {
        let pairs: Vec<(u64, u64)> = changes.pairs().collect();
        let mut cch = hierarchy.new_version();
        let weights = PairWeights::new(g, def)?;
        match cch.recustomize_seeded(&pairs, &|u, v| weights.get(u, v))? {
            Recustomized::Scoped { .. } => return Ok((Some(cch), Update::Scoped)),
            // Nothing was written. `NeedsRebuild` cannot follow from a classified change
            // set; if it does anyway, the full path below finds the misfit and rebuilds.
            Recustomized::NeedsRebuild => {
                #[cfg(test)]
                tests::MISCLASSIFIED.with(|c| c.set(c.get() + 1));
            }
            Recustomized::NeedsFull => {}
        }
    }
    // Sized to the hierarchy, not the graph: nodes created since it was built have no
    // indexed edge (one would be structural), so a grown node capacity is no reason to
    // rebuild.
    let node_count = u64::try_from(hierarchy.node_count())
        .map_err(|_| CchError::Internal("a hierarchy's node count does not fit a u64"))?;
    let Some(weights) = CchInput::weights_for(g, def, node_count)? else {
        // An indexed edge past the hierarchy's nodes that the change set did not report.
        return Ok((None, Update::Rebuild));
    };
    match hierarchy.customized(&weights) {
        Ok(cch) => Ok((Some(cch), Update::Full)),
        // A shape the topology cannot take: a new adjacency the change set did not report.
        // Only a rebuild can.
        Err(CchError::NeedsRebuild) => Ok((None, Update::Rebuild)),
        Err(e) => Err(e),
    }
}

/// Take edges written between two versions into a hierarchy that holds `before`'s edges or
/// later ones: for a build's commit (`before` its version, `after` the published one) and for a
/// writer's rebase onto an adoption (`before` the published version, `after` the writer's
/// fork). Each id in `written` is classified, as a write to a built index is, by its pair in
/// `before` (an edge deleted since) and in `after` (one created or reweighted since), if the
/// edge is of an indexed type there; the update then runs over `after`. `None`: the hierarchy
/// cannot take them, and only a build can ([`Update::Rebuild`]).
///
/// # Errors
/// What [`flush`]'s update reports: a resource, a bug.
pub(crate) fn absorb(
    hierarchy: &Cch,
    def: &CchIndexDef,
    written: &RoaringTreemap,
    before: &Graph,
    after: &Graph,
) -> Result<Option<Cch>, CchError> {
    let mut changes = CchChanges::default();
    for id in written {
        for version in [before, after] {
            let Some((u, v)) = version.endpoints_for_edge(id) else {
                continue;
            };
            let edge_type = version.get_relationship_type_id(RelationshipId::from(id));
            if def.rel_types().contains(&edge_type) {
                changes.classify(hierarchy, u, v);
            }
        }
        if changes.structural {
            return Ok(None);
        }
    }
    let (cch, how) = update(after, def, hierarchy, &changes)?;
    #[cfg(test)]
    tests::UPDATES.with(|u| u.borrow_mut().push(how));
    let _ = how;
    Ok(cch)
}

/// The relationship type's name, or `#id` if it no longer resolves.
fn type_name(
    g: &Graph,
    type_id: TypeId,
) -> String {
    g.get_type(type_id)
        .map_or_else(|| format!("#{}", type_id.0), |name| name.to_string())
}

/// The relationship attribute's name, or `#id` if it no longer resolves.
fn attribute_name(
    g: &Graph,
    attribute: u16,
) -> String {
    g.rel_attr_name(attribute)
        .map_or_else(|| format!("#{attribute}"), |name| name.to_string())
}

/// `(:A|B {w})` for `def`; ids that no longer resolve print as `#id`.
fn index_name(
    g: &Graph,
    def: &CchIndexDef,
) -> String {
    let types = def
        .rel_types()
        .iter()
        .map(|&type_id| type_name(g, type_id))
        .collect::<Vec<_>>()
        .join("|");
    format!("(:{types} {{{}}})", attribute_name(g, def.weight_attr))
}

/// Every update [`flush`] made on this thread so far, and how many scoped attempts met a pair
/// the change set had not marked structural, for tests elsewhere in the crate.
#[cfg(test)]
pub(crate) fn updates_made() -> (Vec<Update>, usize) {
    (
        tests::UPDATES.with(|u| u.borrow().clone()),
        tests::MISCLASSIFIED.with(std::cell::Cell::get),
    )
}

#[cfg(test)]
mod tests {
    //! Through real Cypher: a `Runtime` over an `MvccGraph` write slot, the way the module
    //! runs a write query (minus the locks), so `Commit`, `Pending` and the per-`Commit`
    //! flush are the production ones. The oracle is the test kit over a mirror edge list.

    use std::cell::{Cell, RefCell};
    use std::collections::HashMap;
    use std::fmt::Write as _;
    use std::sync::Arc;

    use orx_tree::{Bfs, NodeRef};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::super::testkit::{Edit, EditGraph, INDEXED, RealEdge, entries, random_edit_batch};
    use super::super::{Cch, CchError, CchIndexDef};
    use super::Update;
    use crate::graph::graph::Graph;
    use crate::graph::graphblas::matrix::Matrix;
    use crate::graph::graphblas::test_init;
    use crate::graph::mvcc_graph::MvccGraph;
    use crate::index::index_docs::IndexDocs;
    use crate::locks::WriteEscalation;
    use crate::planner::IR;
    use crate::runtime::functions::{init_functions, init_udf_functions};
    use crate::runtime::runtime::Runtime;
    use crate::runtime::value::Value;
    use crate::testkit::{check_index_against_oracle, random_sparse, random_zero_heavy, weights};

    thread_local! {
        /// Calls to [`flush`](super::flush) on this thread.
        pub(super) static FLUSHES: Cell<usize> = const { Cell::new(0) };
        /// Every update [`flush`](super::flush) made on this thread, in order.
        pub(super) static UPDATES: RefCell<Vec<Update>> = const { RefCell::new(Vec::new()) };
        /// Scoped attempts on this thread that found a pair the change set had not marked
        /// structural but the hierarchy cannot absorb.
        pub(super) static MISCLASSIFIED: Cell<usize> = const { Cell::new(0) };
        /// An error the next [`update`](super::update) on this thread returns, as a resource
        /// limit or a bug would.
        pub(super) static FAIL_UPDATE: Cell<Option<CchError>> = const { Cell::new(None) };
    }

    /// The updates made since `start` (an earlier [`updates_len`]).
    fn updates_since(start: usize) -> Vec<Update> {
        UPDATES.with(|u| u.borrow()[start..].to_vec())
    }

    fn updates_len() -> usize {
        UPDATES.with(|u| u.borrow().len())
    }

    fn flushes() -> usize {
        FLUSHES.with(Cell::get)
    }

    /// The host side of escalation; a unit test holds no locks.
    struct NoLocks;

    impl WriteEscalation for NoLocks {
        fn upgrade_to_write(
            &self,
            _fork: &mut Graph,
        ) -> Result<(), String> {
            Ok(())
        }
    }

    struct Db {
        mvcc: Arc<MvccGraph>,
        /// The CCH warnings the last successful write took from its version, as the host does
        /// before publishing it.
        warnings: Vec<String>,
    }

    impl Db {
        fn new(name: &str) -> Self {
            test_init::ensure_init();
            // Idempotent: whichever test gets here first registers them.
            let _ = init_functions();
            init_udf_functions();
            Self {
                mvcc: MvccGraph::from_graph(Graph::new(16, 16, 1, 0, name)),
                warnings: Vec::new(),
            }
        }

        fn committed(&self) -> Arc<Graph> {
            self.mvcc.read()
        }

        /// One write query, committed on success and rolled back on error, as
        /// `execute_query_write` does.
        fn write(
            &mut self,
            query: &str,
        ) -> Result<(), String> {
            self.rows(query).map(|_| ())
        }

        /// [`Self::write`], returning every row it returns.
        fn rows(
            &mut self,
            query: &str,
        ) -> Result<Vec<Vec<Value>>, String> {
            let plan = self.committed().get_plan(query)?;
            let runtime = Runtime::new(
                self.mvcc.fork(),
                plan.parameters.clone(),
                Arc::clone(&plan.plan),
                false,
                String::new(),
                -1,
                false,
                None,
                0,
                None,
                &NoLocks,
            );
            // An error drops the fork with the runtime, discarding the write.
            let summary = runtime.query()?;
            let mut rows = Vec::new();
            for batch in &summary.result {
                for row in batch.active_indices() {
                    rows.push(
                        runtime
                            .return_names
                            .iter()
                            .map(|var| batch.column(var.id).get(row))
                            .collect(),
                    );
                }
            }
            let mut warnings = Vec::new();
            runtime
                .publish(|mut fork| {
                    warnings = fork.take_cch_warnings();
                    self.mvcc.commit(fork)
                })
                .expect("the version validates");
            self.warnings = warnings;
            self.build_in_background();
            Ok(rows)
        }

        /// What the index builder does once a write set an index back to `Building`: build it
        /// from the published version and adopt it. Here it runs right after the write, with
        /// no write in between (the builder's own tests interleave them).
        fn build_in_background(&self) {
            let published = self.committed();
            let building: Vec<CchIndexDef> = published
                .cch_indexes()
                .filter(|def| {
                    published
                        .falkordb_index
                        .cch_entry(def)
                        .is_some_and(|entry| entry.is_building())
                })
                .cloned()
                .collect();
            if building.is_empty() {
                return;
            }
            let mut fork = self.mvcc.fork();
            for def in &building {
                assert!(fork.build_held_cch_index(def));
            }
            self.mvcc.commit(fork).expect("the version validates");
        }

        fn ok(
            &mut self,
            query: &str,
        ) {
            if let Err(error) = self.write(query) {
                panic!("{query}: {error}");
            }
        }

        /// Every edge of `relationship_type` as `[src, dst, w]`, in that order.
        fn stored(
            &mut self,
            relationship_type: &str,
        ) -> Vec<Vec<Value>> {
            self.rows(&format!(
                "MATCH (a)-[r:{relationship_type}]->(b) RETURN id(a), id(b), r.w \
                 ORDER BY id(a), id(b)"
            ))
            .expect("a read")
        }

        /// Create and commit the index over `types` (`T{i}` each) and `attribute`.
        fn create_index(
            &mut self,
            types: &[u16],
            attribute: &str,
        ) -> CchIndexDef {
            let mut g = self.mvcc.fork();
            let ids = types
                .iter()
                .map(|t| g.get_type_id(&format!("T{t}")).expect("type exists"))
                .collect();
            let attribute = g.get_or_create_rel_attr_id(&Arc::new(String::from(attribute)));
            let def = CchIndexDef::new(ids, attribute);
            g.build_cch_index_now(def.clone()).expect("create");
            self.mvcc.commit(g).expect("the version validates");
            def
        }

        /// Drop the index over `def` and build it again over the graph as it is now: what a
        /// write left behind is what a fresh index reads.
        fn rebuild_index(
            &mut self,
            def: &CchIndexDef,
        ) {
            let mut g = self.mvcc.fork();
            g.drop_cch_index(def).expect("held");
            g.build_cch_index_now(def.clone())
                .expect("the stored weights are indexable");
            self.mvcc.commit(g).expect("the version validates");
        }

        fn cch(
            &self,
            def: &CchIndexDef,
        ) -> Arc<Cch> {
            Arc::clone(
                self.committed()
                    .cch_index(def)
                    .expect("held")
                    .cch()
                    .expect("built"),
            )
        }
    }

    fn literal(weight: f64) -> String {
        format!("{weight:?}")
    }

    /// A graph of `n` nodes (ids `0..n`) and one relationship per entry of `edges`, weight
    /// attribute `w` (none for a `None` weight), through Cypher: one query per type, in order
    /// of appearance.
    fn load(
        db: &mut Db,
        n: usize,
        edges: &[RealEdge],
    ) {
        db.ok(&format!("UNWIND range(1, {n}) AS i CREATE (:N)"));
        let mut types: Vec<&str> = Vec::new();
        for edge in edges {
            if !types.contains(&edge.3) {
                types.push(edge.3);
            }
        }
        for relationship_type in types {
            let mut rows = String::new();
            for edge in edges.iter().filter(|edge| edge.3 == relationship_type) {
                let separator = if rows.is_empty() { "" } else { "," };
                let weight = edge.2.map_or_else(|| String::from("null"), literal);
                write!(rows, "{separator}[{}, {}, {weight}]", edge.0, edge.1).unwrap();
            }
            db.ok(&format!(
                "UNWIND [{rows}] AS e MATCH (a), (b) WHERE id(a) = e[0] AND id(b) = e[1] \
                 CREATE (a)-[:{relationship_type} {{w: e[2]}}]->(b)"
            ));
        }
    }

    fn pair(
        u: u64,
        v: u64,
    ) -> String {
        format!("MATCH (a)-[r:T0]->(b) WHERE id(a) = {u} AND id(b) = {v}")
    }

    fn ends(
        u: u64,
        v: u64,
    ) -> String {
        format!("MATCH (a), (b) WHERE id(a) = {u} AND id(b) = {v}")
    }

    /// `[src, dst, w]`, as [`Db::stored`] reads an edge.
    fn edge(
        u: i64,
        v: i64,
        weight: Value,
    ) -> Vec<Value> {
        vec![Value::Int(u), Value::Int(v), weight]
    }

    /// Every edge of `g` on `T0`, parallel ones included.
    fn on_t0(g: &EditGraph) -> Vec<RealEdge> {
        g.pairs
            .iter()
            .flat_map(|(&(u, v), weights)| {
                weights
                    .iter()
                    .map(move |&weight| (u, v, Some(weight), INDEXED[0]))
            })
            .collect()
    }

    /// The Cypher for one kit edit on type `T0`.
    fn cypher(edit: &Edit) -> String {
        match *edit {
            Edit::SetWeight { u, v, w, .. } => {
                format!("{} SET r.w = {}", pair(u, v), literal(w))
            }
            Edit::RemoveCheapest { u, v } => {
                format!("{} WITH r ORDER BY r.w LIMIT 1 DELETE r", pair(u, v))
            }
            Edit::AddEdge { u, v, w, .. } => {
                format!("{} CREATE (a)-[:T0 {{w: {}}}]->(b)", ends(u, v), literal(w))
            }
            Edit::AddNode => String::from("CREATE (:N)"),
        }
    }

    fn check(
        db: &Db,
        def: &CchIndexDef,
        oracle: &Matrix<f64>,
    ) {
        let cch = db.cch(def);
        check_index_against_oracle(|s, t| cch.query(s, t), oracle);
    }

    /// The write-time check, through every Cypher shape that writes a weight: each is refused
    /// with the constraint error, the committed index is the very same `Arc` and the graph is
    /// as before.
    #[test]
    fn every_cypher_write_of_an_invalid_weight_is_refused_at_the_write() {
        let edges = [(0, 1, Some(2.0), INDEXED[0]), (1, 2, Some(2.0), INDEXED[0])];
        let mut db = Db::new("cch_weight_check_cypher");
        load(&mut db, 3, &edges);
        let def = db.create_index(&[0], "w");
        let before = db.cch(&def);
        let p = pair(0, 1);
        let queries = [
            format!("{p} SET r.w = -1"),
            format!("{p} SET r.w = -0.5"),
            format!("{p} SET r.w = 0.0 / 0.0"),
            format!("{p} SET r.w = -1.0 / 0.0"),
            format!("{p} SET r = {{w: -1}}"),
            format!("{p} SET r += {{w: -1, other: 1}}"),
            format!("{} CREATE (a)-[:T0 {{w: -3}}]->(b)", ends(0, 2)),
            format!("{} MERGE (a)-[:T0 {{w: -3}}]->(b)", ends(0, 2)),
            format!(
                "{} MERGE (a)-[r:T0 {{k: 1}}]->(b) ON CREATE SET r.w = -4",
                ends(0, 2)
            ),
            format!("{} MERGE (a)-[r:T0]->(b) ON MATCH SET r.w = -5", ends(0, 1)),
            // The first `Commit` is valid, the second refused. And the other way round: a
            // later `Commit` that would repair it never runs, as with a constraint.
            format!("{p} SET r.w = 1 WITH r SET r.w = -1"),
            format!("{p} SET r.w = -1 WITH r SET r.w = 1"),
        ];
        for query in &queries {
            let error = db.write(query).expect_err(query);
            assert!(
                error.contains("CCH weight constraint violation"),
                "{query}: {error}"
            );
            assert!(error.contains("CCH index (:T0 {w})"), "{query}: {error}");
            assert!(Arc::ptr_eq(&before, &db.cch(&def)), "{query}");
        }
        assert_eq!(
            db.stored("T0"),
            vec![edge(0, 1, Value::Float(2.0)), edge(1, 2, Value::Float(2.0))]
        );
    }

    /// What the check lets through: removing the weight or writing a non-numeric one (both
    /// weigh 1.0), a valid value that overwrites an invalid one inside one `SET`, an invalid
    /// value on another type or another attribute, and anything on a graph with no index. An
    /// index built over the result answers like the oracle.
    #[test]
    fn removed_non_numeric_and_unindexed_weights_stay_legal() {
        let edges = [
            (0, 1, Some(2.0), INDEXED[0]),
            (1, 2, Some(2.0), INDEXED[0]),
            (2, 3, Some(2.0), INDEXED[1]),
        ];
        let mut db = Db::new("cch_weight_check_legal");
        load(&mut db, 4, &edges);
        let def = db.create_index(&[0], "w");
        for query in [
            format!("{} SET r.w = null", pair(0, 1)),
            format!("{} REMOVE r.w", pair(1, 2)),
            format!("{} SET r.w = 'fast'", pair(1, 2)),
            format!("{} SET r.w = -1, r.w = 4", pair(0, 1)),
            format!("{} SET r.other = -1", pair(0, 1)),
            String::from("MATCH ()-[r:T1]->() SET r.w = -1"),
            format!("{} CREATE (a)-[:T1 {{w: -2}}]->(b)", ends(3, 0)),
        ] {
            db.ok(&query);
        }
        assert_eq!(
            db.stored("T0"),
            vec![
                edge(0, 1, Value::Int(4)),
                edge(1, 2, Value::String(Arc::new(String::from("fast"))))
            ]
        );
        assert_eq!(
            db.stored("T1"),
            vec![edge(2, 3, Value::Int(-1)), edge(3, 0, Value::Int(-2))]
        );
        db.rebuild_index(&def);
        check(&db, &def, &weights(4, &[(0, 1, 4.0), (1, 2, 1.0)]));

        let mut none = Db::new("cch_weight_check_no_index");
        load(&mut none, 4, &edges);
        none.ok(&format!("{} SET r.w = -1", pair(0, 1)));
    }

    /// `+inf` is a closed edge and `-0.0` weighs 0: both are legal through every Cypher shape
    /// that writes a weight, and an index built over them answers like the oracle, which reads
    /// `+inf` as no edge. The refusal of a negative weight says what is legal.
    #[test]
    fn a_closed_edge_and_negative_zero_are_legal_writes() {
        let edges = [
            (0, 1, Some(2.0), INDEXED[0]),
            (1, 2, Some(2.0), INDEXED[0]),
            (2, 3, Some(2.0), INDEXED[0]),
        ];
        let mut db = Db::new("cch_weight_check_closed");
        load(&mut db, 4, &edges);
        let def = db.create_index(&[0], "w");
        for query in [
            format!("{} SET r.w = 1.0 / 0.0", pair(0, 1)),
            format!("{} SET r = {{w: -0.0}}", pair(1, 2)),
            format!("{} SET r += {{w: 1.0 / 0.0}}", pair(2, 3)),
            format!("{} CREATE (a)-[:T0 {{w: 1.0 / 0.0}}]->(b)", ends(0, 2)),
            format!(
                "{} MERGE (a)-[r:T0 {{k: 1}}]->(b) ON CREATE SET r.w = 1.0 / 0.0",
                ends(0, 3)
            ),
            format!(
                "{} MERGE (a)-[r:T0]->(b) ON MATCH SET r.w = -0.0",
                ends(1, 2)
            ),
        ] {
            db.ok(&query);
        }
        db.rebuild_index(&def);
        let inf = f64::INFINITY;
        let oracle = weights(
            4,
            &[
                (0, 1, inf),
                (1, 2, 0.0),
                (2, 3, inf),
                (0, 2, inf),
                (0, 3, inf),
            ],
        );
        check(&db, &def, &oracle);
        assert_eq!(db.cch(&def).query(1, 2), Some((0.0, vec![1, 2])));
        assert_eq!(db.cch(&def).query(0, 3), None);

        let error = db
            .write(&format!("{} SET r.w = -1", pair(1, 2)))
            .expect_err("a negative weight is refused");
        assert!(
            error.contains(
                "CCH weight constraint violation: edge of relationship-type T0 cannot have \
                 w = -1; CCH index (:T0 {w}) needs weights that are >= 0 (+inf marks a \
                 closed edge)"
            ),
            "{error}"
        );
    }

    /// With several indexes, the error names the one whose rule the write breaks: the index
    /// spanning the edge's type whose weight the attribute is. An attribute that is the weight
    /// of one index only is free on the types the others span.
    #[test]
    fn the_error_names_the_index_whose_weight_is_refused() {
        let edges = [(0, 1, Some(2.0), INDEXED[0]), (1, 2, Some(2.0), INDEXED[1])];
        let mut db = Db::new("cch_weight_check_names");
        load(&mut db, 3, &edges);
        db.create_index(&[0, 1], "w");
        db.create_index(&[1], "length");
        for (query, says) in [
            (
                "MATCH ()-[r:T1]->() SET r.length = -1",
                "edge of relationship-type T1 cannot have length = -1; CCH index (:T1 {length})",
            ),
            (
                "MATCH ()-[r:T0]->() SET r.w = -2.5",
                "edge of relationship-type T0 cannot have w = -2.5; CCH index (:T0|T1 {w})",
            ),
            (
                "MATCH ()-[r:T1]->() SET r.w = -2",
                "edge of relationship-type T1 cannot have w = -2; CCH index (:T0|T1 {w})",
            ),
        ] {
            let error = db.write(query).expect_err(query);
            assert!(error.contains(says), "{query}: {error}");
        }
        db.ok("MATCH ()-[r:T0]->() SET r.length = -1");
    }

    /// `GRAPH.BULK`'s attribute import ([`Graph::import_relationship_attrs_resolved`]) runs
    /// the check too, before it drains anything.
    #[test]
    fn a_bulk_import_of_an_invalid_weight_is_refused() {
        let edges = [(0, 1, Some(2.0), INDEXED[0]), (1, 2, Some(2.0), INDEXED[1])];
        let mut db = Db::new("cch_weight_check_bulk");
        load(&mut db, 3, &edges);
        db.create_index(&[0], "w");
        let edge_of = |db: &mut Db, relationship_type: &str| {
            let rows = db
                .rows(&format!(
                    "MATCH ()-[r:{relationship_type}]->() RETURN id(r)"
                ))
                .unwrap();
            let Value::Int(id) = rows[0][0] else {
                panic!("an id")
            };
            u64::try_from(id).expect("an edge id")
        };
        let (t0_edge, t1_edge) = (edge_of(&mut db, "T0"), edge_of(&mut db, "T1"));
        // A fork that is never committed: dropping it at the end discards it.
        let mut g = db.mvcc.fork();
        let w = g.get_or_create_rel_attr_id(&Arc::new(String::from("w")));
        let (t0, t1) = (g.get_type_id("T0").unwrap(), g.get_type_id("T1").unwrap());
        let mut data = vec![(t0_edge, vec![(w, Value::Float(-1.0))])];
        let error = g
            .import_relationship_attrs_resolved(&mut data, t0, &mut IndexDocs::default())
            .expect_err("refused");
        assert!(error.contains("CCH weight constraint violation"), "{error}");
        assert_eq!(data.len(), 1, "refused before the import drained it");
        let mut data = vec![(t0_edge, vec![(w, Value::Float(f64::INFINITY))])];
        g.import_relationship_attrs_resolved(&mut data, t0, &mut IndexDocs::default())
            .expect("+inf is a closed edge, legal");
        let mut data = vec![(t1_edge, vec![(w, Value::Float(-1.0))])];
        g.import_relationship_attrs_resolved(&mut data, t1, &mut IndexDocs::default())
            .expect("T1 is not indexed");
    }

    /// The correctness bar: after every write query, every pair answers like Dijkstra on
    /// the current graph. Weight up/down, a pair's cheapest edge removed (revealing a parallel
    /// one, if any), edge added between connected and unconnected nodes (possibly parallel),
    /// node added; random graphs, zero-weight-heavy ones included.
    #[test]
    fn every_write_leaves_the_index_answering_like_dijkstra() {
        for seed in 0..12u64 {
            let base = EditGraph::of(&if seed % 2 == 0 {
                random_sparse(12, 26, 9, seed)
            } else {
                random_zero_heavy(12, 26, 9, seed)
            });
            let mut db = Db::new(&format!("cch_maint_random_{seed}"));
            load(&mut db, base.n, &on_t0(&base));
            let def = db.create_index(&[0], "w");
            check(&db, &def, &base.oracle());

            let mut now = base;
            for batch in 0..3 {
                let (after, edits) = random_edit_batch(&now, seed * 31 + batch);
                for edit in &edits {
                    db.ok(&cypher(edit));
                }
                check(&db, &def, &after.oracle());
                now = after;
            }
        }
    }

    /// Writes that cannot change what the index answers leave its hierarchy alone: the
    /// very same `Arc` before and after.
    #[test]
    fn writes_that_touch_nothing_indexed_do_not_rebuild() {
        let edges = [
            (0, 1, Some(2.0), INDEXED[0]),
            (1, 2, Some(3.0), INDEXED[1]),
            (2, 3, Some(4.0), INDEXED[0]),
        ];
        let mut db = Db::new("cch_maint_unrelated");
        load(&mut db, 4, &edges);
        let def = db.create_index(&[0], "w");
        let before = db.cch(&def);

        for q in [
            // another type: created, weight changed, deleted
            "MATCH (a), (b) WHERE id(a) = 0 AND id(b) = 3 CREATE (a)-[:T1 {w: 0.5}]->(b)",
            "MATCH ()-[r:T1]->() SET r.w = 0.25",
            "MATCH ()-[r:T1]->() DELETE r",
            // the indexed type, another attribute
            "MATCH ()-[r:T0]->() SET r.other = 7",
            "MATCH ()-[r:T0]->() REMOVE r.other",
            // node properties and labels
            "MATCH (n) WHERE id(n) = 1 SET n.p = 1, n:L",
            "MATCH (n:L) REMOVE n:L",
            // nodes with no edge yet: no route changes
            "UNWIND range(1, 3) AS i CREATE (:N)",
            // a brand-new type
            "MATCH (a), (b) WHERE id(a) = 0 AND id(b) = 1 CREATE (a)-[:Fresh {w: 0.0}]->(b)",
        ] {
            db.ok(q);
            assert!(Arc::ptr_eq(&before, &db.cch(&def)), "rebuilt after: {q}");
        }

        // The nodes created above answer no path, as they have none, inside the hierarchy
        // or past it.
        for new in 4..7 {
            assert_eq!(db.cch(&def).query(0, new), None, "node {new}");
        }

        db.ok("MATCH ()-[r:T0]->() SET r.w = 1.5");
        assert!(
            !Arc::ptr_eq(&before, &db.cch(&def)),
            "a weight change on the indexed type does rebuild"
        );
    }

    /// No CCH index: nothing is staged, so the hook never reaches `flush` at all.
    #[test]
    fn a_graph_without_a_cch_index_never_reaches_the_flush() {
        let mut db = Db::new("cch_maint_none");
        let start = flushes();
        db.ok("UNWIND range(1, 5) AS i CREATE (:N)");
        db.ok("MATCH (a), (b) WHERE id(a) = 0 AND id(b) = 1 CREATE (a)-[:T0 {w: 1.0}]->(b)");
        db.ok("MATCH ()-[r]->() SET r.w = 2.0");
        db.ok("MATCH (n) WHERE id(n) = 0 DETACH DELETE n");
        assert_eq!(flushes(), start);
    }

    /// Several `Commit`s in one query: each `Commit` that touched an index flushes what it
    /// staged, so the index after the query has seen every `Commit`'s changes (the first adds
    /// an edge, the second removes another).
    #[test]
    fn a_query_with_several_commits_flushes_each_and_sees_them_all() {
        let edges = [
            (0, 1, Some(1.0), INDEXED[0]),
            (1, 2, Some(1.0), INDEXED[0]),
            (2, 3, Some(1.0), INDEXED[0]),
        ];
        let mut db = Db::new("cch_maint_commits");
        load(&mut db, 4, &edges);
        let def = db.create_index(&[0], "w");

        let q = "MATCH (a), (b) WHERE id(a) = 0 AND id(b) = 3 CREATE (a)-[:T0 {w: 5.0}]->(b) \
                 WITH a MATCH (x)-[r:T0]->(y) WHERE id(x) = 1 AND id(y) = 2 DELETE r \
                 WITH a MATCH (c), (d) WHERE id(c) = 3 AND id(d) = 2 \
                 CREATE (c)-[:T0 {w: 0.5}]->(d)";
        let plan = db.committed().get_plan(q).expect("plan").plan;
        let commits = plan
            .root()
            .walk::<Bfs>()
            .filter(|n| matches!(n, IR::Commit))
            .count();
        assert!(
            commits >= 3,
            "the query must run several Commits, has {commits}"
        );

        let start = flushes();
        db.ok(q);
        assert_eq!(flushes() - start, 3, "one flush per Commit that touched T0");

        let oracle = weights(4, &[(0, 1, 1.0), (2, 3, 1.0), (0, 3, 5.0), (3, 2, 0.5)]);
        check(&db, &def, &oracle);
    }

    /// Node deletion cascades to incident edges inside `Pending::commit`; the rebuild must
    /// see the graph after the cascade. Also attribute removal (missing = 1.0) and a new
    /// node joined by an edge in the same query.
    #[test]
    fn deletes_cascades_and_removed_weights_are_seen() {
        let edges = [
            (0, 1, 4.0),
            (1, 2, 4.0),
            (0, 2, 9.0),
            (2, 3, 2.0),
            (3, 4, 2.0),
        ]
        .map(|(u, v, w)| (u, v, Some(w), INDEXED[0]));
        let mut db = Db::new("cch_maint_cascade");
        load(&mut db, 5, &edges);
        let def = db.create_index(&[0], "w");

        db.ok("MATCH (n) WHERE id(n) = 1 DETACH DELETE n");
        let mut oracle = weights(5, &[(0, 2, 9.0), (2, 3, 2.0), (3, 4, 2.0)]);
        check(&db, &def, &oracle);

        db.ok(&format!("{} REMOVE r.w", pair(0, 2)));
        oracle.set(0, 2, 1.0);
        check(&db, &def, &oracle);

        // Node id 1 is free again; whichever id the new node gets, it must be reachable
        // from 4 at weight 0.0 and reach 0 at weight 3.0.
        db.ok(
            "MATCH (a), (b) WHERE id(a) = 4 AND id(b) = 0 CREATE (a)-[:T0 {w: 0.0}]->(n:N)\
             -[:T0 {w: 3.0}]->(b)",
        );
        let g = db.committed();
        let new_id = (0..g.node_cap())
            .find(|&id| {
                db.cch(&def)
                    .query(4, id)
                    .is_some_and(|(w, p)| w == 0.0 && p.len() == 2)
            })
            .expect("the new node is reachable from 4");
        let (w, _) = db.cch(&def).query(new_id, 0).expect("and reaches 0");
        assert_eq!(w, 3.0);
    }

    /// Failure after a `Commit` that the flush would have acted on: the earlier `Commit`'s
    /// change is rolled back, and so is any rebuilt hierarchy.
    #[test]
    fn a_query_failing_after_a_commit_keeps_the_committed_index() {
        let edges = [(0, 1, Some(2.0), INDEXED[0]), (1, 2, Some(2.0), INDEXED[0])];
        let oracle = weights(3, &[(0, 1, 2.0), (1, 2, 2.0)]);
        let mut db = Db::new("cch_maint_late_failure");
        load(&mut db, 3, &edges);
        let def = db.create_index(&[0], "w");
        let before = db.cch(&def);
        let start = flushes();

        let err = db.write(&format!(
            "{} SET r.w = 0.5 WITH r RETURN toInteger('x') + 1 / 0",
            pair(0, 1)
        ));
        assert!(err.is_err(), "the query must fail after its Commit");
        assert_eq!(
            flushes() - start,
            1,
            "the Commit flushed into the private version before the query failed"
        );
        assert!(Arc::ptr_eq(&before, &db.cch(&def)));
        check(&db, &def, &oracle);
    }

    /// Two relationship types plus parallel edges: an index over `T0` alone is rebuilt by
    /// `T0` writes only, and the cheapest parallel edge keeps winning.
    #[test]
    fn multi_type_and_parallel_edges_stay_correct() {
        let (t0, t1) = (INDEXED[0], INDEXED[1]);
        let edges = [
            (0, 1, Some(5.0), t0),
            (0, 1, Some(3.0), t0),
            (0, 1, Some(1.0), t1),
            (1, 2, Some(2.0), t0),
            (2, 3, Some(2.0), t1),
            (2, 3, Some(6.0), t0),
        ];
        let mut db = Db::new("cch_maint_multi");
        load(&mut db, 4, &edges);
        let both = db.create_index(&[0, 1], "w");
        let zero = db.create_index(&[0], "w");
        check(
            &db,
            &both,
            &weights(4, &[(0, 1, 1.0), (1, 2, 2.0), (2, 3, 2.0)]),
        );
        let only_t0 = weights(4, &[(0, 1, 3.0), (1, 2, 2.0), (2, 3, 6.0)]);
        check(&db, &zero, &only_t0);

        let zero_before = db.cch(&zero);
        db.ok("MATCH ()-[r:T1]->() SET r.w = r.w + 10.0");
        assert!(Arc::ptr_eq(&zero_before, &db.cch(&zero)));
        check(&db, &both, &only_t0);

        db.ok("MATCH ()-[r:T0]->() WHERE r.w = 3.0 DELETE r");
        check(
            &db,
            &both,
            &weights(4, &[(0, 1, 5.0), (1, 2, 2.0), (2, 3, 6.0)]),
        );
        check(
            &db,
            &zero,
            &weights(4, &[(0, 1, 5.0), (1, 2, 2.0), (2, 3, 6.0)]),
        );
    }

    // ---- The change set and the scoped flush ----

    /// After a write: the held index answers like Dijkstra on `oracle`, and its metric is
    /// bit-identical (weights and mids) to a full customization of its own topology from the
    /// graph as it now stands, i.e. what a full flush would have produced.
    fn check_exact(
        db: &Db,
        def: &CchIndexDef,
        oracle: &Matrix<f64>,
        what: &str,
    ) {
        let cch = db.cch(def);
        let g = db.committed();
        let input = super::super::CchInput::build(&g, def).expect("input");
        let snap = |c: &Cch| {
            (
                c.metric.upward_weight.iter().collect::<Vec<_>>(),
                c.metric.downward_weight.iter().collect::<Vec<_>>(),
                c.metric.upward_middle.iter().collect::<Vec<_>>(),
                c.metric.downward_middle.iter().collect::<Vec<_>>(),
            )
        };
        if input.n > cch.node_count() {
            // Nodes created past the hierarchy record nothing until an indexed edge joins
            // one, so there is no full customization of this shape to compare with; what
            // makes skipping them safe is that no indexed edge touches one.
            for (id, (s, t)) in all_t0_edges(&g, def) {
                assert!(
                    (s as usize) < cch.node_count() && (t as usize) < cch.node_count(),
                    "{what}: edge {id} ({s}, {t}) lies past the hierarchy"
                );
            }
        } else {
            let full = cch
                .customized(&input.weights)
                .unwrap_or_else(|e| panic!("{what}: the held topology must fit the graph: {e}"));
            assert!(
                snap(&cch) == snap(&full),
                "{what}: metric != full customize"
            );
        }
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_index_against_oracle(|s, t| cch.query(s, t), oracle);
        }))
        .is_err()
        {
            // Each pair's cheapest edge, as the oracle holds it.
            let mut cheapest = HashMap::new();
            for (id, pair) in all_t0_edges(&g, def) {
                let w = oracle_weight_of(&g, id);
                cheapest
                    .entry(pair)
                    .and_modify(|c: &mut f64| *c = c.min(w))
                    .or_insert(w);
            }
            let mut have: Vec<(u64, u64, String)> = cheapest
                .into_iter()
                .map(|((s, t), w)| (s, t, format!("{w:?}")))
                .collect();
            let mut want: Vec<(u64, u64, String)> = entries(oracle)
                .into_iter()
                .map(|(s, t, w)| (s, t, format!("{w:?}")))
                .collect();
            have.sort();
            want.sort();
            let only_graph: Vec<_> = have.iter().filter(|e| !want.contains(e)).collect();
            let only_mirror: Vec<_> = want.iter().filter(|e| !have.contains(e)).collect();
            panic!(
                "{what}: the index disagrees with the oracle; graph only {only_graph:?}, \
                 mirror only {only_mirror:?}"
            );
        }
    }

    /// The mirror of a graph under random writes: its `T0` edges (every parallel edge of a
    /// pair; the other types are never indexed) and the live node ids.
    struct Mirror {
        g: EditGraph,
        alive: Vec<u64>,
    }

    impl Mirror {
        fn pair(
            &self,
            rng: &mut SmallRng,
        ) -> Option<(u64, u64)> {
            let pairs: Vec<_> = self.g.pairs.keys().copied().collect();
            (!pairs.is_empty()).then(|| pairs[rng.random_range(0..pairs.len())])
        }
    }

    /// Every `(u, v)` edge of `g` at weight `w`, as `SET r.w` writes it.
    fn set_pair(
        g: &mut EditGraph,
        u: u64,
        v: u64,
        w: f64,
    ) {
        g.pairs
            .get_mut(&(u, v))
            .expect("the pair has an edge")
            .iter_mut()
            .for_each(|x| *x = w);
    }

    /// One more `(u, v)` edge in `g`, parallel to its others.
    fn add_edge(
        g: &mut EditGraph,
        u: u64,
        v: u64,
        w: f64,
    ) {
        g.pairs.entry((u, v)).or_default().push(w);
    }

    /// One random write, as the Cypher of one `Commit` (no `RETURN`), applied to the mirror.
    /// Node creation is done by the caller, which needs the id back.
    fn random_write(
        rng: &mut SmallRng,
        m: &mut Mirror,
    ) -> Option<String> {
        let ends = |u: u64, v: u64| format!("MATCH (a), (b) WHERE id(a) = {u} AND id(b) = {v}");
        let w = |rng: &mut SmallRng| (rng.random_range(0..12)) as f64;
        Some(match rng.random_range(0..10) {
            // a weight up or down, every parallel T0 edge of the pair
            0 | 1 => {
                let (u, v) = m.pair(rng)?;
                let x = w(rng);
                set_pair(&mut m.g, u, v, x);
                format!("{} SET r.w = {}", pair(u, v), literal(x))
            }
            // close (+inf), or reopen a closed pair
            2 => {
                let (u, v) = m.pair(rng)?;
                let closed = m.g.pairs[&(u, v)].iter().any(|w| w.is_infinite());
                let x = if closed { w(rng) } else { f64::INFINITY };
                set_pair(&mut m.g, u, v, x);
                let lit = if x.is_infinite() {
                    String::from("1.0 / 0.0")
                } else {
                    literal(x)
                };
                format!("{} SET r.w = {lit}", pair(u, v))
            }
            // weight removed: weighs 1.0
            3 => {
                let (u, v) = m.pair(rng)?;
                set_pair(&mut m.g, u, v, 1.0);
                format!("{} REMOVE r.w", pair(u, v))
            }
            // every T0 edge of a pair deleted, often the pair's last
            4 => {
                let (u, v) = m.pair(rng)?;
                m.g.pairs.remove(&(u, v));
                format!("{} DELETE r", pair(u, v))
            }
            // a new T0 edge between two live nodes: parallel, reversed, joined or not
            5 | 6 => {
                let (u, v) = if rng.random_range(0..2) == 0 {
                    let (u, v) = m.pair(rng)?;
                    if rng.random_range(0..2) == 0 {
                        (u, v)
                    } else {
                        (v, u)
                    }
                } else {
                    let u = m.alive[rng.random_range(0..m.alive.len())];
                    let v = m.alive[rng.random_range(0..m.alive.len())];
                    if u == v {
                        return None;
                    }
                    (u, v)
                };
                let x = if rng.random_range(0..6) == 0 {
                    f64::INFINITY
                } else {
                    w(rng)
                };
                add_edge(&mut m.g, u, v, x);
                let lit = if x.is_infinite() {
                    String::from("1.0 / 0.0")
                } else {
                    literal(x)
                };
                format!("{} CREATE (a)-[:T0 {{w: {lit}}}]->(b)", ends(u, v))
            }
            // a node deleted with its edges (the cascade)
            7 => {
                if m.alive.len() < 4 {
                    return None;
                }
                let k = m.alive.swap_remove(rng.random_range(0..m.alive.len()));
                m.g.pairs.retain(|&(u, v), _| u != k && v != k);
                format!("MATCH (n) WHERE id(n) = {k} DETACH DELETE n")
            }
            // noise: another type, another attribute
            8 => {
                let u = m.alive[rng.random_range(0..m.alive.len())];
                let v = m.alive[rng.random_range(0..m.alive.len())];
                format!("{} CREATE (a)-[:T1 {{w: 0.0}}]->(b)", ends(u, v))
            }
            _ => {
                let (u, v) = m.pair(rng)?;
                format!("{} SET r.other = {}", pair(u, v), rng.random_range(0..5))
            }
        })
    }

    /// The correctness bar for the change set: random write sequences through Cypher (weight
    /// edits, closes and reopens with `+inf`, removed weights, deletes including a pair's last
    /// edge, new edges between joined and unjoined nodes, node deletes with their cascade,
    /// new nodes within and past the hierarchy's capacity, unrelated writes), several per
    /// query across `Commit`s a third of the time. After every query the index answers like
    /// Dijkstra and its metric is bit-identical to a full customization; a scoped attempt
    /// never meets a pair the classification missed; and the scoped path is what most
    /// queries take.
    #[test]
    fn random_writes_keep_the_index_exact_through_the_change_set() {
        let (mut scoped, mut rebuilt, mut full) = (0, 0, 0);
        let misclassified = MISCLASSIFIED.with(Cell::get);
        for seed in 1..=8u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let base = EditGraph::of(&random_sparse(40, 110, 9, seed));
            let mut db = Db::new(&format!("cch_cs_random_{seed}"));
            let mut edges = on_t0(&base);
            edges.push((0, 1, Some(1.0), INDEXED[1]));
            load(&mut db, base.n, &edges);
            let def = db.create_index(&[0], "w");
            let mut m = Mirror {
                g: base,
                alive: (0..40).collect(),
            };
            check_exact(&db, &def, &m.g.oracle(), "after create");

            for step in 0..40 {
                let start = updates_len();
                let what = if rng.random_range(0..9) == 0 {
                    // a node, sometimes enough of them to pass the hierarchy's capacity
                    let count = if rng.random_range(0..3) == 0 { 30 } else { 1 };
                    let rows = db
                        .rows(&format!(
                            "UNWIND range(1, {count}) AS i CREATE (n:N) RETURN id(n)"
                        ))
                        .unwrap();
                    for row in rows {
                        let Value::Int(id) = row[0] else {
                            panic!("an id")
                        };
                        let id = id as u64;
                        m.alive.push(id);
                        m.g.n = m.g.n.max(id as usize + 1);
                    }
                    format!("seed {seed} step {step}: {count} node(s)")
                } else {
                    let k = if rng.random_range(0..3) == 0 {
                        2 + rng.random_range(0..2)
                    } else {
                        1
                    };
                    let qs: Vec<String> = (0..k)
                        .filter_map(|_| random_write(&mut rng, &mut m))
                        .collect();
                    if qs.is_empty() {
                        continue;
                    }
                    // `WITH DISTINCT 1`, not `WITH count(*)`: on this engine an aggregation
                    // after a relationship pattern makes a later `MATCH (a), (b) WHERE id(a)
                    // = .. AND id(b) = ..` find nothing (independent of CCH; reported).
                    let q = qs.join(" WITH DISTINCT 1 AS c ");
                    db.ok(&q);
                    format!("seed {seed} step {step}: {q}")
                };
                check_exact(&db, &def, &m.g.oracle(), &what);
                for u in updates_since(start) {
                    match u {
                        Update::Scoped => scoped += 1,
                        Update::Rebuild => rebuilt += 1,
                        Update::Full => full += 1,
                    }
                }
            }
        }
        assert_eq!(
            MISCLASSIFIED.with(Cell::get),
            misclassified,
            "a pair the hierarchy cannot absorb was not marked structural"
        );
        eprintln!("updates: {scoped} scoped, {full} full, {rebuilt} rebuilds");
        assert!(
            scoped > 3 * rebuilt,
            "{scoped} scoped against {rebuilt} rebuilds"
        );
        assert!(rebuilt > 0, "structural writes must occur and rebuild");
    }

    /// Each kind of write takes the update it should: weight edits, closes, removed weights,
    /// deletes (a pair's last edge too) and a new edge between joined nodes are scoped; a
    /// node created within the capacity or past it changes nothing; an edge between unjoined
    /// nodes rebuilds, an edge to a node past the capacity included; more pairs than the cap
    /// in one unit run a full customization.
    #[test]
    fn each_write_kind_takes_its_update() {
        let mut oracle = EditGraph::of(&random_sparse(120, 300, 9, 77));
        let mut db = Db::new("cch_cs_kinds");
        load(&mut db, oracle.n, &on_t0(&oracle));
        let def = db.create_index(&[0], "w");
        let topo_of = |db: &Db| db.cch(&def);
        let (u, v) = *oracle.pairs.keys().next().expect("an edge");

        let expect = |db: &mut Db, q: &str, want: &[Update], oracle: &EditGraph| {
            let start = updates_len();
            db.ok(q);
            assert_eq!(updates_since(start), want, "{q}");
            check_exact(db, &def, &oracle.oracle(), q);
        };

        set_pair(&mut oracle, u, v, 3.0);
        expect(
            &mut db,
            &format!("{} SET r.w = 3.0", pair(u, v)),
            &[Update::Scoped],
            &oracle,
        );
        set_pair(&mut oracle, u, v, f64::INFINITY);
        expect(
            &mut db,
            &format!("{} SET r.w = 1.0 / 0.0", pair(u, v)),
            &[Update::Scoped],
            &oracle,
        );
        oracle.pairs.remove(&(u, v));
        expect(
            &mut db,
            &format!("{} DELETE r", pair(u, v)),
            &[Update::Scoped],
            &oracle,
        );
        // The pair lost its last edge but keeps its arc: an edge back is absorbed, either way.
        add_edge(&mut oracle, v, u, 2.0);
        expect(
            &mut db,
            &format!(
                "MATCH (a), (b) WHERE id(a) = {v} AND id(b) = {u} CREATE (a)-[:T0 {{w: 2.0}}]->(b)"
            ),
            &[Update::Scoped],
            &oracle,
        );
        // A fill-in arc absorbs a brand-new pair.
        let cch = topo_of(&db);
        let topo = cch.topology();
        let edge_pairs: std::collections::HashSet<(u64, u64)> = oracle
            .pairs
            .keys()
            .flat_map(|&(a, b)| [(a, b), (b, a)])
            .collect();
        let fill = topo
            .ranks()
            .flat_map(|lo| topo.up(lo).iter().map(move |&hi| (lo, hi)))
            .map(|(lo, hi)| (u64::from(topo.node_of(lo)), u64::from(topo.node_of(hi))))
            .find(|&(a, b)| a < 120 && b < 120 && !edge_pairs.contains(&(a, b)))
            .expect("a fill-in arc between live nodes");
        add_edge(&mut oracle, fill.0, fill.1, 4.0);
        expect(
            &mut db,
            &format!(
                "MATCH (a), (b) WHERE id(a) = {} AND id(b) = {} CREATE (a)-[:T0 {{w: 4.0}}]->(b)",
                fill.0, fill.1
            ),
            &[Update::Scoped],
            &oracle,
        );
        // A pair with no arc rebuilds.
        let cch = topo_of(&db);
        let (a, b) = (0..120u64)
            .flat_map(|a| (0..120u64).map(move |b| (a, b)))
            .find(|&(a, b)| a != b && !cch.joins(a, b))
            .expect("an unjoined pair");
        add_edge(&mut oracle, a, b, 1.0);
        expect(
            &mut db,
            &format!(
                "MATCH (a), (b) WHERE id(a) = {a} AND id(b) = {b} CREATE (a)-[:T0 {{w: 1.0}}]->(b)"
            ),
            &[Update::Rebuild],
            &oracle,
        );
        // A node within the capacity or past it: nothing at all. The indexed edge that
        // joins one past it is what rebuilds.
        let cap = db.cch(&def).node_count();
        let before = db.cch(&def);
        expect(&mut db, "CREATE (:N)", &[], &oracle);
        assert!(Arc::ptr_eq(&before, &db.cch(&def)));
        let fill_to = cap - 120;
        expect(
            &mut db,
            &format!("UNWIND range(1, {fill_to}) AS i CREATE (:N)"),
            &[],
            &oracle,
        );
        assert!(
            Arc::ptr_eq(&before, &db.cch(&def)),
            "nodes past the capacity change nothing"
        );
        let past = db
            .rows("MATCH (n:N) RETURN max(id(n))")
            .expect("the new nodes")
            .into_iter()
            .next()
            .and_then(|row| {
                if let Value::Int(id) = row[0] {
                    Some(id as u64)
                } else {
                    None
                }
            })
            .expect("an id");
        assert!(
            past as usize >= cap,
            "node {past} lies past the capacity {cap}"
        );
        oracle.n = oracle.n.max(past as usize + 1);
        add_edge(&mut oracle, 0, past, 1.0);
        expect(
            &mut db,
            &format!(
                "MATCH (a), (b) WHERE id(a) = 0 AND id(b) = {past} \
                 CREATE (a)-[:T0 {{w: 1.0}}]->(b)"
            ),
            &[Update::Rebuild],
            &oracle,
        );
        assert!(
            db.cch(&def).node_count() > cap,
            "the rebuild took the new nodes in"
        );
    }

    /// Past [`PAIR_CAP`](super::PAIR_CAP) distinct pairs in one unit the set keeps only
    /// "many", not structural.
    #[test]
    fn the_pair_cap_turns_a_set_into_many() {
        let mut c = super::CchChanges::default();
        for k in 0..super::PAIR_CAP as u64 {
            c.push(k, k + 1);
            c.push(k + 1, k); // the same undirected pair
        }
        assert_eq!(c.pairs().len(), super::PAIR_CAP);
        assert!(!c.overflow);
        c.push(0, 0); // a self-loop is no pair
        assert!(!c.overflow);
        c.push(u64::MAX - 1, u64::MAX);
        assert!(c.overflow && !c.structural);
        assert_eq!(c.pairs().len(), 0);
        c.set_structural();
        assert!(c.structural);
    }

    /// Many pairs at once: every edge reweighted in one query (scoped or, past the
    /// hierarchy's own `n / 8` valve, full), and a unit whose pairs passed the cap (full).
    /// Neither rebuilds, and both land on the metric a full flush gives. The valve itself is
    /// tested on `recustomize_seeded` (`partial.rs`).
    #[test]
    fn many_pairs_take_a_full_customization() {
        let mut oracle = EditGraph::of(&random_sparse(100, 280, 9, 5));
        let mut db = Db::new("cch_cs_full");
        load(&mut db, oracle.n, &on_t0(&oracle));
        let def = db.create_index(&[0], "w");

        // Every edge reweighted in one query: scoped below the hierarchy's `n / 8` valve,
        // full past it (`n` is the node capacity, which the graph rounds up).
        let start = updates_len();
        db.ok("MATCH ()-[r:T0]->() SET r.w = r.w + 2.0");
        oracle.pairs.values_mut().flatten().for_each(|w| *w += 2.0);
        let cch = db.cch(&def);
        let arcs: std::collections::HashSet<(u64, u64)> = oracle
            .pairs
            .keys()
            .map(|&(a, b)| (a.min(b), a.max(b)))
            .collect();
        let want = if arcs.len() > cch.node_count() / 8 {
            Update::Full
        } else {
            Update::Scoped
        };
        assert_eq!(updates_since(start), [want]);
        check_exact(&db, &def, &oracle.oracle(), "every edge");

        // A change set that overflowed its cap, flushed by hand over one weight write.
        let mut next = db.mvcc.fork();
        {
            let g = &mut next;
            let w = g.get_or_create_rel_attr_id(&Arc::new(String::from("w")));
            let rows = g
                .relationship_tensors()
                .get(def.rel_types()[0].0)
                .expect("T0")
                .iter(0, u64::MAX, false)
                .map(|(_, _, id)| id)
                .take(1)
                .collect::<Vec<_>>();
            let mut attrs = rustc_hash::FxHashMap::default();
            attrs.insert(rows[0], vec![(w, Value::Float(0.5))]);
            g.set_relationships_attributes(&attrs, &mut IndexDocs::default())
                .expect("set");
            let mut dirty = super::CchDirty::default();
            dirty.entry(&def).overflow = true;
            let start = updates_len();
            super::flush(g, &dirty).expect("flush");
            assert_eq!(updates_since(start), [Update::Full]);
        }
        db.mvcc.commit(next).expect("the version validates");
        // The oracle read back from the graph: every live edge at the weight it reads as.
        let mut rebuilt = EditGraph {
            n: oracle.n,
            ..EditGraph::default()
        };
        let g = db.committed();
        for (id, (s, t)) in all_t0_edges(&g, &def) {
            add_edge(&mut rebuilt, s, t, oracle_weight_of(&g, id));
        }
        check_exact(&db, &def, &rebuilt.oracle(), "overflow");
    }

    /// An index that cannot be updated for a reason that is not the write's (a resource limit,
    /// a bug in the update) does not fail the write: it is handed to the builder, which builds
    /// it again from the published graph. Only an invalid weight, which the write-time check
    /// should have refused, still fails the write.
    #[test]
    fn an_index_that_cannot_be_updated_is_rebuilt_and_the_write_goes_on() {
        let edges = [(0, 1, Some(2.0), INDEXED[0]), (1, 2, Some(3.0), INDEXED[0])];
        let mut db = Db::new("cch_update_error");
        load(&mut db, 3, &edges);
        let def = db.create_index(&[0], "w");

        let start = updates_len();
        FAIL_UPDATE.set(Some(CchError::Internal("injected")));
        db.ok(&format!("{} SET r.w = 5.0", pair(0, 1)));
        assert_eq!(updates_since(start), [Update::Rebuild]);
        // The host is told, with the index and the cause.
        let [warning] = db.warnings.as_slice() else {
            panic!("one warning: {:?}", db.warnings);
        };
        assert!(
            warning.contains("CCH index (:") && warning.contains("injected"),
            "{warning}"
        );
        // The harness ran the builder's part: the index is built again, with the write.
        check(&db, &def, &weights(3, &[(0, 1, 5.0), (1, 2, 3.0)]));
        // A write that updates the index warns of nothing, and the published version holds none.
        db.ok(&format!("{} SET r.w = 5.0", pair(1, 2)));
        assert_eq!(db.warnings, Vec::<String>::new());
        assert!(db.committed().falkordb_index.cch_warnings.0.is_empty());

        FAIL_UPDATE.set(Some(CchError::InvalidWeight));
        let error = db
            .write(&format!("{} SET r.w = 6.0", pair(0, 1)))
            .expect_err("an invalid weight at the flush fails the write");
        assert!(error.contains("got past the write-time check"), "{error}");
        // The failed write's version is dropped, warnings and all.
        assert!(db.committed().falkordb_index.cch_warnings.0.is_empty());
        check(&db, &def, &weights(3, &[(0, 1, 5.0), (1, 2, 5.0)]));
    }

    /// Nodes created with no indexed edge grow the graph's node capacity past the hierarchy's,
    /// and are no reason to rebuild: a change set past the cap still customizes the held
    /// topology (`Update::Full`), over weights sized to the hierarchy. An indexed edge past
    /// the hierarchy's nodes is lost by that sizing, and the weights say so (`None`), so only
    /// a rebuild takes it.
    #[test]
    fn a_grown_node_capacity_still_takes_a_full_customization() {
        let edges = [
            (0, 1, Some(2.0), INDEXED[0]),
            (1, 2, Some(3.0), INDEXED[0]),
            (2, 0, Some(4.0), INDEXED[0]),
        ];
        let mut db = Db::new("cch_grown_capacity");
        load(&mut db, 3, &edges);
        let def = db.create_index(&[0], "w");
        let hierarchy_nodes = u64::try_from(db.cch(&def).node_count()).expect("fits");
        while db.committed().node_cap() <= hierarchy_nodes {
            db.ok("UNWIND range(1, 64) AS i CREATE (:N)");
        }
        let built = db.cch(&def);
        assert_eq!(
            built.node_count(),
            db.cch(&def).node_count(),
            "no rebuild yet"
        );

        // A change set that overflowed its cap, flushed by hand over one weight write.
        let mut next = db.mvcc.fork();
        {
            let g = &mut next;
            let w = g.get_or_create_rel_attr_id(&Arc::new(String::from("w")));
            let first = all_t0_edges(g, &def)[0].0;
            let mut attrs = rustc_hash::FxHashMap::default();
            attrs.insert(first, vec![(w, Value::Float(0.5))]);
            g.set_relationships_attributes(&attrs, &mut IndexDocs::default())
                .expect("set");
            let mut dirty = super::CchDirty::default();
            dirty.entry(&def).overflow = true;
            let start = updates_len();
            super::flush(g, &dirty).expect("flush");
            assert_eq!(updates_since(start), [Update::Full]);
        }
        db.mvcc.commit(next).expect("the version validates");
        assert_eq!(db.cch(&def).node_count(), built.node_count());
        let mut oracle = EditGraph {
            n: built.node_count(),
            ..EditGraph::default()
        };
        let g = db.committed();
        for (id, (s, t)) in all_t0_edges(&g, &def) {
            add_edge(&mut oracle, s, t, oracle_weight_of(&g, id));
        }
        check_exact(&db, &def, &oracle.oracle(), "grown capacity");

        // An indexed edge to a node past the hierarchy: sized to the hierarchy, the weights
        // lose it, and say so.
        assert!(
            super::super::CchInput::weights_for(&g, &def, hierarchy_nodes)
                .expect("weights")
                .is_some()
        );
        let past = db.rows("MATCH (n:N) RETURN max(id(n))").expect("a read")[0][0].clone();
        let Value::Int(past) = past else {
            panic!("an id: {past:?}")
        };
        assert!(u64::try_from(past).expect("an id") >= hierarchy_nodes);
        db.ok(&format!(
            "MATCH (a), (b) WHERE id(a) = 0 AND id(b) = {past} CREATE (a)-[:T0 {{w: 1.0}}]->(b)"
        ));
        assert!(
            super::super::CchInput::weights_for(&db.committed(), &def, hierarchy_nodes)
                .expect("weights")
                .is_none()
        );
    }

    /// The weight an edge reads as, by the index's rule.
    fn oracle_weight_of(
        g: &Graph,
        id: u64,
    ) -> f64 {
        let attr = g
            .get_rel_attr_id(&Arc::new(String::from("w")))
            .expect("the weight attribute exists");
        super::super::input::edge_weight(g.relationship_attrs(), attr, id)
    }

    /// Every live `T0` edge as `(id, (src, dst))`.
    fn all_t0_edges(
        g: &Graph,
        def: &CchIndexDef,
    ) -> Vec<(u64, (u64, u64))> {
        g.relationship_tensors()[def.rel_types()[0].0]
            .iter(0, u64::MAX, false)
            .map(|(s, t, id)| (id, (s, t)))
            .collect()
    }
}
