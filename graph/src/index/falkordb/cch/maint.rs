//! Keeping CCH indexes correct after writes (#3214).
//!
//! CCH maintenance rides the native index's unit-of-work carrier. Each graph write method that
//! can change what a CCH index answers calls one of the functions below, which read the graph and
//! return the [`CchDirty`] that write makes (they hold no state, like
//! [`graph_writes`](super::super::graph_writes)); the method stages it in the unit's
//! [`IndexDocs::cch`](crate::index::index_docs::IndexDocs::cch), next to the native index's
//! tuples and RediSearch's documents. The owner of the unit flushes it once its graph writes
//! are all in ([`Graph::apply_cch_writes`]), on the
//! private version: a Cypher query at the end of each `Commit` segment, a replicated effects
//! buffer at the end of the buffer, `GRAPH.BULK` once every token is in, before it escalates.
//! Each held index the unit could have affected is rebuilt from scratch into a new `Arc<Cch>`;
//! the rest keep their `Arc` untouched.
//!
//! Nothing is kept on the version and nothing in `Pending` beyond the carrier it already holds: a
//! failed unit drops its carrier, and a failed flush fails the unit, whose private version (with
//! any hierarchy already rebuilt into it) is dropped by the existing rollback.
//!
//! Where the marks are kept is behind three seams, so moving them is a small change: the rules
//! ([`nodes_changed`], [`rel_types_changed`], [`edge_attrs_set`], which return a `CchDirty` and
//! keep nothing), the staging (one `docs.cch.extend(..)` in each graph write method), and the
//! flush ([`flush`], which the three owners reach through [`Graph::apply_cch_writes`]).
//!
//! Staging is gated on the graph holding at least one CCH index, so a graph without one never
//! fills a `CchDirty` and the flush is a single `is_empty()` check.
//!
//! What counts as touching an index (mirrors C's `graph_hub.c` marks), one rule set for every
//! path because every path writes through the same graph methods:
//! - any node created or deleted: every index (C's `GraphContext_CCHMarkAllRebuild`); this
//!   also covers the incident edges a node deletion cascades to;
//! - a relationship created or explicitly deleted: every index spanning its type;
//! - an attribute set or removed on an existing relationship: every index spanning its type
//!   whose weight attribute it is.
//!
//! The write methods that set relationship attributes also refuse, before writing, a weight a
//! held index cannot take ([`check_edge_weights`]), the way a constraint refuses a value. An
//! index never sees an invalid weight through a write: what can still fail a flush is the graph's
//! size, a resource, or a bug.
//!
//! A replay ([`IndexDocs::replay`](crate::index::index_docs::IndexDocs::replay): a replica's
//! effects, the AOF) is not checked, as it is not checked against a constraint: the primary
//! validated the write before emitting it, and refusing it here would fail the buffer (a full
//! resync on a replica, `exit(1)` while loading the AOF).
//!
//! CCH DDL is not replicated yet, while every RDB carries the definitions (the AOF preamble and
//! a replica's full sync included), so a replaying server can hold an index its primary has
//! dropped, and receive a weight that index cannot take. The flush gives up on that index
//! ([`flush`]). Replicating CCH DDL removes that case. C's effects v3 sends a CCH index as an
//! ordinary index record whose `field_type` carries bit `0x20`, its relationship types in the
//! record's schema list; the v3 reader refuses that bit today (#3219).
//!
//! A later change replaces "rebuild" with a choice between rebuilding, recustomizing and a
//! scoped recustomization; the staging and the flush points stay.

use std::fmt;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use super::input::is_valid_weight;
use super::order::log_warning;
use super::{Cch, CchError, CchIndexDef, CchInput};
use crate::graph::graph::{Graph, RelationshipId, TypeId};
use crate::index::index_docs::IndexDocs;
use crate::runtime::value::Value;

/// What a unit of work's writes touched that the graph's CCH indexes might care about, staged in
/// its [`IndexDocs::cch`](crate::index::index_docs::IndexDocs::cch) until the owner flushes it.
///
/// Empty for every graph with no CCH index, so the maintenance hook costs one `is_empty()`
/// check on the existing write path. `Default` allocates nothing.
#[derive(Clone, Default, Debug)]
pub(crate) struct CchDirty {
    /// A node was created or deleted, or the caller asked for everything ([`Self::all`]).
    all: bool,
    /// Types of relationships created or explicitly deleted.
    rel_types: FxHashSet<TypeId>,
    /// `(type, attribute)` of attributes set or removed on existing relationships.
    attrs: FxHashSet<(TypeId, u16)>,
}

impl CchDirty {
    /// Every held index is dirty.
    #[must_use]
    pub(crate) fn all() -> Self {
        Self {
            all: true,
            ..Self::default()
        }
    }

    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        !self.all && self.rel_types.is_empty() && self.attrs.is_empty()
    }

    /// Stage one write's marks after everything already staged.
    pub(crate) fn extend(
        &mut self,
        other: Self,
    ) {
        if other.is_empty() {
            return;
        }
        self.all |= other.all;
        self.rel_types.extend(other.rel_types);
        self.attrs.extend(other.attrs);
    }

    /// Whether the index for `def` may answer differently after these changes.
    #[must_use]
    pub(crate) fn touches(
        &self,
        def: &CchIndexDef,
    ) -> bool {
        self.all
            || def
                .rel_types()
                .iter()
                .any(|t| self.rel_types.contains(t) || self.attrs.contains(&(*t, def.weight_attr)))
    }
}

// ---- What each graph write stages ----
//
// Called by the graph's write methods while what they read (held indexes, edge types) is still
// in the graph.

/// Nodes created or deleted: every index. A deleted node's cascaded edges need no mark of their
/// own.
pub(crate) fn nodes_changed(g: &Graph) -> CchDirty {
    if g.falkordb_index.cch_indexes.is_empty() {
        return CchDirty::default();
    }
    CchDirty::all()
}

/// Relationships of each type in `types` created or deleted.
pub(crate) fn rel_types_changed(
    g: &Graph,
    types: impl IntoIterator<Item = TypeId>,
) -> CchDirty {
    if g.falkordb_index.cch_indexes.is_empty() {
        return CchDirty::default();
    }
    CchDirty {
        rel_types: types.into_iter().collect(),
        ..CchDirty::default()
    }
}

/// Attributes set or removed on existing relationships: `(type, attr)` for every written
/// attribute that is a held index's weight. `type_id` is the edges' shared type when the caller
/// knows it; `None` looks it up, once per edge that writes a weight.
pub(crate) fn edge_attrs_set(
    g: &Graph,
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
    type_id: Option<TypeId>,
) -> CchDirty {
    let mut dirty = CchDirty::default();
    if g.falkordb_index.cch_indexes.is_empty() {
        return dirty;
    }
    let weight_attrs = weight_attrs(g);
    for (&id, written) in attrs {
        let mut weights = written
            .iter()
            .map(|(attr, _)| *attr)
            .filter(|attr| weight_attrs.contains(attr))
            .peekable();
        if weights.peek().is_none() {
            continue;
        }
        let t = match type_id {
            Some(t) => t,
            None => {
                if g.endpoints_for_edge(id).is_none() {
                    continue;
                }
                g.get_relationship_type_id(RelationshipId::from(id))
            }
        };
        for attr in weights {
            dirty.attrs.insert((t, attr));
        }
    }
    dirty
}

/// Refuse attribute writes that would give an edge of an indexed type an invalid weight, like a
/// constraint: the unit of work fails before anything is written, and rolls back like any
/// failing write. "Invalid" is [`is_valid_weight`]'s rule: a negative number (`-inf` included)
/// or NaN; `+inf` is legal (a closed edge), and so is removing the weight or writing a
/// non-numeric value (it weighs 1.0).
///
/// Called by every graph method that writes relationship attributes, before the write, so one
/// check covers Cypher and `GRAPH.BULK`. `type_id` is the edges' shared type
/// when the caller knows it. Costs one `is_empty()` on a graph with no CCH index; with one, a
/// compare per written attribute against the held weight attributes, and the edge's type is
/// looked up only for an invalid value of a weight attribute, so a valid write never reads it.
///
/// It reads the held index definitions, never their hierarchies, so it holds for every index
/// the version holds, whatever state its hierarchy is in.
///
/// A replay (`docs.replay`) is not checked, as a constraint is not: the primary already
/// validated it. See the module doc.
///
/// # Errors
/// The first offending write, naming the edge's type, the attribute, the value and the index.
pub(crate) fn check_edge_weights<'a>(
    g: &Graph,
    docs: &IndexDocs,
    attrs: impl IntoIterator<Item = (&'a u64, &'a Vec<(u16, Value)>)>,
    type_id: Option<TypeId>,
) -> Result<(), String> {
    let held = &g.falkordb_index.cch_indexes;
    if held.is_empty() || docs.replay {
        return Ok(());
    }
    for (&id, written) in attrs {
        for (attr, value) in written {
            if !held.keys().any(|def| def.weight_attr == *attr) || is_valid_weight(value) {
                continue;
            }
            let t = match type_id {
                Some(t) => t,
                None => {
                    if g.endpoints_for_edge(id).is_none() {
                        continue;
                    }
                    g.get_relationship_type_id(RelationshipId::from(id))
                }
            };
            let Some(def) = held
                .keys()
                .find(|def| def.weight_attr == *attr && def.rel_types().contains(&t))
            else {
                continue;
            };
            let type_name = g
                .get_type(t)
                .map_or_else(|| format!("#{}", t.0), |n| n.to_string());
            let attr_name = g
                .rel_attr_name(*attr)
                .map_or_else(|| format!("#{attr}"), |n| n.to_string());
            let value = match value {
                Value::Float(f) => f.to_string(),
                Value::Int(k) => k.to_string(),
                _ => unreachable!("only a number can be an invalid weight"),
            };
            return Err(format!(
                "CCH weight constraint violation: edge of relationship-type {type_name} cannot \
                 have {attr_name} = {value}; CCH index {} needs weights that are >= 0 \
                 (+inf marks a closed edge)",
                index_name(g, def)
            ));
        }
    }
    Ok(())
}

/// A held index could not be brought up to date. The write that caused it fails, and its
/// rollback leaves the committed indexes as they were.
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
        let cause = match &self.cause {
            // `check_edge_weights` refuses every write of one, and an index is never created
            // or loaded over one, so reaching it here is a write path that skipped the check.
            CchError::InvalidWeight => {
                "internal error: an invalid weight got past the write-time check"
            }
            CchError::UnknownRelType => "unknown relationship type",
            CchError::UnknownAttribute => "unknown weight attribute",
            CchError::Internal(msg) => msg,
            CchError::AlreadyExists | CchError::NotFound => "unexpected index state",
        };
        write!(
            f,
            "CCH index {} could not be updated after this write: {cause}",
            self.name
        )
    }
}

/// Bring every held CCH index that `dirty` touches up to date with this version of the graph,
/// on the private version the writes went into: the new hierarchies are published together
/// with the graph. [`Graph::apply_cch_writes`] hands it what a unit of work staged.
///
/// All touched indexes are built before any is replaced, so on error none is. Every entry is
/// `Ready` until the background build, which will route a building entry's changes through
/// [`VersionedIndex::writer_record`](crate::index::falkordb::versioned::VersionedIndex::writer_record)
/// instead.
///
/// A replay (`replay`, [`IndexDocs::replay`]) is never failed by a weight: its writes were
/// not checked ([`check_edge_weights`]), and an index that cannot take one is dropped from
/// this version, with a warning, rather than fail the buffer. Its primary accepted the weight,
/// so it held no such index: dropping it is the nearest this server can come to the
/// primary's state until CCH DDL is replicated. (There is no failed state for an index to
/// stay in yet.)
///
/// # Errors
/// The first index that fails to build, with the reason (an ordering backend failure, ...).
/// An invalid weight outside a replay is refused at the write ([`check_edge_weights`]), so
/// here it is a bug: it still fails the unit, and is logged.
pub(crate) fn flush(
    g: &mut Graph,
    dirty: &CchDirty,
    replay: bool,
) -> Result<(), CchFlushError> {
    #[cfg(test)]
    tests::FLUSHES.with(|c| c.set(c.get() + 1));
    if g.falkordb_index.cch_indexes.is_empty() || dirty.is_empty() {
        return Ok(());
    }
    let mut rebuilt = Vec::new();
    let mut dropped = Vec::new();
    for (def, entry) in &g.falkordb_index.cch_indexes {
        debug_assert!(!entry.is_building(), "CCH is built synchronously");
        let idx = &entry.index;
        if !dirty.touches(&idx.def) {
            continue;
        }
        match CchInput::build(g, &idx.def).and_then(|input| Cch::build(&input)) {
            Ok(cch) => rebuilt.push((def.clone(), Arc::new(cch))),
            Err(CchError::InvalidWeight) if replay => {
                log_warning(&format!(
                    "CCH index {} is dropped: a replicated write set a weight it cannot \
                     take, which its primary accepted, so the primary holds no such index \
                     (CCH DDL is not replicated)",
                    index_name(g, &idx.def)
                ));
                dropped.push(def.clone());
            }
            Err(cause) => {
                let e = CchFlushError {
                    def: idx.def.clone(),
                    name: index_name(g, &idx.def),
                    cause,
                };
                if e.cause == CchError::InvalidWeight {
                    log_warning(&format!("CCH: {e}"));
                }
                return Err(e);
            }
        }
    }
    for (def, cch) in rebuilt {
        let entry = g.falkordb_index.cch_indexes.get_mut(&def).expect("held");
        entry.index.cch = cch;
    }
    for def in dropped {
        g.falkordb_index
            .drop_cch(&def)
            .expect("the dropped index is held");
    }
    Ok(())
}

/// The weight attribute of each held index, for staging which attribute writes matter.
fn weight_attrs(g: &Graph) -> Vec<u16> {
    g.falkordb_index
        .cch_indexes
        .keys()
        .map(|def| def.weight_attr)
        .collect()
}

/// `(:A|B {w})` for `def`; ids that no longer resolve print as `#id`.
pub(crate) fn index_name(
    g: &Graph,
    def: &CchIndexDef,
) -> String {
    let types = def
        .rel_types()
        .iter()
        .map(|&t| {
            g.get_type(t)
                .map_or_else(|| format!("#{}", t.0), |n| n.to_string())
        })
        .collect::<Vec<_>>()
        .join("|");
    let attr = g
        .rel_attr_name(def.weight_attr)
        .map_or_else(|| format!("#{}", def.weight_attr), |n| n.to_string());
    format!("(:{types} {{{attr}}})")
}

/// Calls to [`flush`] on this thread so far, for tests elsewhere in the crate.
#[cfg(test)]
pub(crate) fn flush_count() -> usize {
    tests::FLUSHES.with(|c| c.get())
}

#[cfg(test)]
mod tests {
    //! Through real Cypher: a `Runtime` over an `MvccGraph` write slot, the way the module
    //! runs a write query (minus the locks), so `Commit`, `Pending` and the per-`Commit`
    //! flush are the production ones. The oracle is the test kit over a mirror edge list.

    use std::cell::Cell;
    use std::collections::HashMap;
    use std::fmt::Write as _;
    use std::sync::Arc;

    use atomic_refcell::AtomicRefCell;
    use orx_tree::{Bfs, NodeRef};

    use super::super::testkit::{
        Edit, EditGraph, INDEXED, RealEdge, check_index_against_oracle, random_edit_batch,
        random_sparse, random_zero_heavy, weights,
    };
    use super::super::{Cch, CchIndexDef};
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

    thread_local! {
        /// Calls to [`flush`](super::flush) on this thread.
        pub(super) static FLUSHES: Cell<usize> = const { Cell::new(0) };
    }

    fn flushes() -> usize {
        super::flush_count()
    }

    /// The host side of escalation; a unit test holds no locks.
    struct NoLocks;

    impl WriteEscalation for NoLocks {
        fn upgrade_to_write(&self) -> Result<(), String> {
            Ok(())
        }
    }

    struct Db {
        mvcc: MvccGraph,
    }

    impl Db {
        fn new(name: &str) -> Self {
            test_init::ensure_init();
            // Idempotent: whichever test gets here first registers them.
            let _ = init_functions();
            init_udf_functions();
            Self {
                mvcc: MvccGraph::from_graph(Graph::new(16, 16, 1, 0, name)),
            }
        }

        fn committed(&self) -> Arc<AtomicRefCell<Graph>> {
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

        /// [`Self::write`], returning the first column of every row.
        fn rows(
            &mut self,
            query: &str,
        ) -> Result<Vec<Value>, String> {
            let plan = self.committed().borrow().get_plan(query)?.plan;
            let next = self.mvcc.fork();
            let outcome = {
                let runtime = Runtime::new(
                    next.clone(),
                    HashMap::new(),
                    true,
                    plan,
                    false,
                    String::new(),
                    -1,
                    false,
                    None,
                    0,
                    None,
                    &NoLocks,
                );
                runtime.query().map(|summary| {
                    let mut rows = Vec::new();
                    if let Some(var) = runtime.return_names.first() {
                        for batch in &summary.result {
                            for row in batch.active_indices() {
                                rows.push(batch.column(var.id).get(row));
                            }
                        }
                    }
                    rows
                })
            };
            match outcome {
                Ok(_) => self.mvcc.commit(next).expect("the version validates"),
                // Dropping the fork discards the write.
                Err(_) => drop(next),
            }
            outcome
        }

        fn ok(
            &mut self,
            query: &str,
        ) {
            if let Err(e) = self.write(query) {
                panic!("{query}: {e}");
            }
        }

        fn create_index(
            &mut self,
            types: &[u16],
        ) -> CchIndexDef {
            let next = self.mvcc.fork();
            let def = {
                let mut g = next.borrow_mut();
                let ids = types
                    .iter()
                    .map(|t| g.get_type_id(&format!("T{t}")).expect("type exists"))
                    .collect();
                let attr = g.get_or_create_rel_attr_id(&Arc::new(String::from("w")));
                let def = CchIndexDef::new(ids, attr);
                g.create_cch_index(def.clone()).expect("create");
                def
            };
            self.mvcc.commit(next).expect("the version validates");
            def
        }

        fn cch(
            &self,
            def: &CchIndexDef,
        ) -> Arc<Cch> {
            Arc::clone(&self.committed().borrow().cch_index(def).expect("held").cch)
        }
    }

    fn lit(w: f64) -> String {
        format!("{w:?}")
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
        for e in edges {
            if !types.contains(&e.3) {
                types.push(e.3);
            }
        }
        for ty in types {
            let mut rows = String::new();
            for e in edges.iter().filter(|e| e.3 == ty) {
                let sep = if rows.is_empty() { "" } else { "," };
                let w = e.2.map_or_else(|| String::from("null"), lit);
                write!(rows, "{sep}[{}, {}, {w}]", e.0, e.1).unwrap();
            }
            db.ok(&format!(
                "UNWIND [{rows}] AS e MATCH (a), (b) WHERE id(a) = e[0] AND id(b) = e[1] \
                 CREATE (a)-[:{ty} {{w: e[2]}}]->(b)"
            ));
        }
    }

    /// Every edge of `g` on `T0`, parallel ones included.
    fn on_t0(g: &EditGraph) -> Vec<RealEdge> {
        g.pairs
            .iter()
            .flat_map(|(&(u, v), ws)| ws.iter().map(move |&w| (u, v, Some(w), INDEXED[0])))
            .collect()
    }

    fn pair(
        u: u64,
        v: u64,
    ) -> String {
        format!("MATCH (a)-[r:T0]->(b) WHERE id(a) = {u} AND id(b) = {v}")
    }

    /// The Cypher for one kit edit on type `T0`.
    fn cypher(edit: &Edit) -> String {
        match *edit {
            Edit::SetWeight { u, v, w, .. } => format!("{} SET r.w = {}", pair(u, v), lit(w)),
            Edit::RemoveCheapest { u, v } => {
                format!("{} WITH r ORDER BY r.w LIMIT 1 DELETE r", pair(u, v))
            }
            Edit::AddEdge { u, v, w, .. } => format!(
                "MATCH (a), (b) WHERE id(a) = {u} AND id(b) = {v} CREATE (a)-[:T0 {{w: {}}}]->(b)",
                lit(w)
            ),
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
            let def = db.create_index(&[0]);
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
        let def = db.create_index(&[0]);
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
            // a brand-new type
            "MATCH (a), (b) WHERE id(a) = 0 AND id(b) = 1 CREATE (a)-[:Fresh {w: 0.0}]->(b)",
        ] {
            db.ok(q);
            assert!(Arc::ptr_eq(&before, &db.cch(&def)), "rebuilt after: {q}");
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
        let def = db.create_index(&[0]);

        let q = "MATCH (a), (b) WHERE id(a) = 0 AND id(b) = 3 CREATE (a)-[:T0 {w: 5.0}]->(b) \
                 WITH a MATCH (x)-[r:T0]->(y) WHERE id(x) = 1 AND id(y) = 2 DELETE r \
                 WITH a MATCH (c), (d) WHERE id(c) = 3 AND id(d) = 2 \
                 CREATE (c)-[:T0 {w: 0.5}]->(d)";
        let plan = db.committed().borrow().get_plan(q).expect("plan").plan;
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
        let def = db.create_index(&[0]);

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
        let new_id = (0..g.borrow().node_cap())
            .find(|&id| {
                db.cch(&def)
                    .query(4, id)
                    .is_some_and(|(w, p)| w == 0.0 && p.len() == 2)
            })
            .expect("the new node is reachable from 4");
        let (w, _) = db.cch(&def).query(new_id, 0).expect("and reaches 0");
        assert_eq!(w, 3.0);
    }

    /// The write-time check, through every Cypher shape that writes a weight: each is refused
    /// with the constraint error, the committed index is the very same `Arc` and the graph is
    /// as before. A refusal in the query's only `Commit` never reaches the flush.
    #[test]
    fn every_cypher_write_of_an_invalid_weight_is_refused_at_the_write() {
        let edges = [(0, 1, Some(2.0), INDEXED[0]), (1, 2, Some(2.0), INDEXED[0])];
        let mut db = Db::new("cch_wcheck_cypher");
        load(&mut db, 3, &edges);
        let def = db.create_index(&[0]);
        let before = db.cch(&def);
        let p = pair(0, 1);
        let ends = "MATCH (a), (b) WHERE id(a) = 0 AND id(b) = 2";
        let one_commit = [
            format!("{p} SET r.w = -1"),
            format!("{p} SET r.w = -0.5"),
            format!("{p} SET r.w = 0.0 / 0.0"),
            format!("{p} SET r.w = -1.0 / 0.0"),
            format!("{p} SET r = {{w: -1}}"),
            format!("{p} SET r += {{w: -1, other: 1}}"),
            format!("{ends} CREATE (a)-[:T0 {{w: -3}}]->(b)"),
            format!("{ends} MERGE (a)-[:T0 {{w: -3}}]->(b)"),
            format!("{ends} MERGE (a)-[r:T0 {{k: 1}}]->(b) ON CREATE SET r.w = -4"),
            String::from(
                "MATCH (a), (b) WHERE id(a) = 0 AND id(b) = 1 \
                 MERGE (a)-[r:T0]->(b) ON MATCH SET r.w = -5",
            ),
        ];
        // The first `Commit` is valid and flushes; the second is refused. And the other way
        // round: a later `Commit` that would repair it never runs, as with a constraint.
        let two_commits = [
            format!("{p} SET r.w = 1 WITH r SET r.w = -1"),
            format!("{p} SET r.w = -1 WITH r SET r.w = 1"),
        ];
        for q in one_commit.iter().chain(&two_commits) {
            let start = flushes();
            let err = db.write(q).expect_err(q);
            assert!(
                err.contains("CCH weight constraint violation"),
                "{q}: {err}"
            );
            assert!(err.contains("CCH index (:T0 {w})"), "{q}: {err}");
            if one_commit.contains(q) {
                assert_eq!(flushes(), start, "{q}: refused before the flush");
            }
            assert!(Arc::ptr_eq(&before, &db.cch(&def)), "{q}");
        }
        assert_eq!(
            db.rows("MATCH ()-[r:T0]->() RETURN count(r)").unwrap(),
            vec![Value::Int(2)]
        );
        check(&db, &def, &weights(3, &[(0, 1, 2.0), (1, 2, 2.0)]));
    }

    /// What the check lets through: removing the weight or writing a non-numeric one (both
    /// weigh 1.0), a valid value that overwrites an invalid one inside one `SET`, an invalid
    /// value on another type or another attribute, and anything on a graph with no index.
    #[test]
    fn removed_non_numeric_and_unindexed_weights_stay_legal() {
        let edges = [
            (0, 1, Some(2.0), INDEXED[0]),
            (1, 2, Some(2.0), INDEXED[0]),
            (2, 3, Some(2.0), INDEXED[1]),
        ];
        let mut db = Db::new("cch_wcheck_legal");
        load(&mut db, 4, &edges);
        let def = db.create_index(&[0]);
        for q in [
            format!("{} SET r.w = null", pair(0, 1)),
            format!("{} REMOVE r.w", pair(1, 2)),
            format!("{} SET r.w = 'fast'", pair(1, 2)),
            format!("{} SET r.w = -1, r.w = 4", pair(0, 1)),
            format!("{} SET r.other = -1", pair(0, 1)),
            String::from("MATCH ()-[r:T1]->() SET r.w = -1"),
            String::from(
                "MATCH (a), (b) WHERE id(a) = 3 AND id(b) = 0 CREATE (a)-[:T1 {w: -2}]->(b)",
            ),
        ] {
            db.ok(&q);
        }
        check(&db, &def, &weights(4, &[(0, 1, 4.0), (1, 2, 1.0)]));

        let mut none = Db::new("cch_wcheck_no_index");
        load(&mut none, 4, &edges);
        none.ok(&format!("{} SET r.w = -1", pair(0, 1)));
    }

    /// `+inf` is a closed edge and `-0.0` weighs 0: both are legal through every Cypher shape
    /// that writes a weight, and the index answers like the oracle, which reads `+inf` as no
    /// edge. The refusal of a negative weight says what is legal.
    #[test]
    fn a_closed_edge_and_negative_zero_are_legal_writes() {
        let edges = [
            (0, 1, Some(2.0), INDEXED[0]),
            (1, 2, Some(2.0), INDEXED[0]),
            (2, 3, Some(2.0), INDEXED[0]),
        ];
        let mut db = Db::new("cch_wcheck_closed");
        load(&mut db, 4, &edges);
        let def = db.create_index(&[0]);
        let ends = |u: u64, v: u64| format!("MATCH (a), (b) WHERE id(a) = {u} AND id(b) = {v}");
        for q in [
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
            db.ok(&q);
        }
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

        let err = db
            .write(&format!("{} SET r.w = -1", pair(1, 2)))
            .expect_err("a negative weight is refused");
        assert!(
            err.contains(
                "CCH weight constraint violation: edge of relationship-type T0 cannot have \
                 w = -1; CCH index (:T0 {w}) needs weights that are >= 0 (+inf marks a \
                 closed edge)"
            ),
            "{err}"
        );
    }

    /// `GRAPH.BULK`'s attribute import ([`Graph::import_relationship_attrs_resolved`]) runs
    /// the check too, before it drains anything.
    #[test]
    fn a_bulk_import_of_an_invalid_weight_is_refused() {
        let edges = [(0, 1, Some(2.0), INDEXED[0]), (1, 2, Some(2.0), INDEXED[1])];
        let mut db = Db::new("cch_wcheck_bulk");
        load(&mut db, 3, &edges);
        db.create_index(&[0]);
        let ids = |db: &mut Db, t: &str| {
            let rows = db
                .rows(&format!("MATCH ()-[r:{t}]->() RETURN id(r)"))
                .unwrap();
            let Value::Int(id) = rows[0] else {
                panic!("an id")
            };
            id as u64
        };
        let (t0_edge, t1_edge) = (ids(&mut db, "T0"), ids(&mut db, "T1"));
        let next = db.mvcc.fork();
        {
            let mut g = next.borrow_mut();
            let w = g.get_or_create_rel_attr_id(&Arc::new(String::from("w")));
            let (t0, t1) = (g.get_type_id("T0").unwrap(), g.get_type_id("T1").unwrap());
            let mut data = vec![(t0_edge, vec![(w, Value::Float(-1.0))])];
            let err = g
                .import_relationship_attrs_resolved(&mut data, t0, &mut IndexDocs::default())
                .expect_err("refused");
            assert!(err.contains("CCH weight constraint violation"), "{err}");
            assert_eq!(data.len(), 1, "refused before the import drained it");
            let mut data = vec![(t0_edge, vec![(w, Value::Float(f64::INFINITY))])];
            g.import_relationship_attrs_resolved(&mut data, t0, &mut IndexDocs::default())
                .expect("+inf is a closed edge, legal");
            let mut data = vec![(t1_edge, vec![(w, Value::Float(-1.0))])];
            g.import_relationship_attrs_resolved(&mut data, t1, &mut IndexDocs::default())
                .expect("T1 is not indexed");
        }
        // Dropping the fork discards it.
        drop(next);
    }

    /// Failure after a `Commit` that the flush would have acted on: the earlier `Commit`'s
    /// change is rolled back, and so is any rebuilt hierarchy.
    #[test]
    fn a_query_failing_after_a_commit_keeps_the_committed_index() {
        let edges = [(0, 1, Some(2.0), INDEXED[0]), (1, 2, Some(2.0), INDEXED[0])];
        let oracle = weights(3, &[(0, 1, 2.0), (1, 2, 2.0)]);
        let mut db = Db::new("cch_maint_late_failure");
        load(&mut db, 3, &edges);
        let def = db.create_index(&[0]);
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
        let both = db.create_index(&[0, 1]);
        let zero = db.create_index(&[0]);
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

    /// 0 -> 1 -> 2 at 1.0 each and a direct 0 -> 2 at 5.0, type `T0`; `0 -> 1` again at 1.0
    /// as type `T1`. Indexes over `T0` and over `T1`.
    fn triangle(db: &mut Db) -> (CchIndexDef, CchIndexDef) {
        let edges = [
            (0, 1, Some(1.0), INDEXED[0]),
            (1, 2, Some(1.0), INDEXED[0]),
            (0, 2, Some(5.0), INDEXED[0]),
            (0, 1, Some(1.0), INDEXED[1]),
        ];
        load(db, 3, &edges);
        (db.create_index(&[0]), db.create_index(&[1]))
    }

    /// `db.idx.cch.query` from node 0 to node 2 over `T0`, as the rest of a query.
    const QUERY_0_2: &str = "WITH 1 AS one MATCH (s), (t) WHERE id(s) = 0 AND id(t) = 2 \
        CALL db.idx.cch.query({sourceNode: s, targetNode: t, relTypes: ['T0'], weightProp: 'w'}) \
        YIELD pathWeight RETURN pathWeight";

    /// A procedure reading an index the same write query has touched sees that query's
    /// writes: the `Commit` the `WITH` plans rebuilt it, once, and nothing is rebuilt at the end.
    /// With the cheap edge deleted earlier in the query, the procedure answers without it
    /// rather than failing the query on a hierarchy that still holds it.
    #[test]
    fn a_procedure_sees_the_writes_earlier_in_its_query() {
        let mut db = Db::new("cch_maint_inquery_set");
        triangle(&mut db);
        let start = flushes();
        let rows = db.rows(&format!("{} SET r.w = 10.0 {QUERY_0_2}", pair(0, 1)));
        assert_eq!(
            rows,
            Ok(vec![Value::Float(5.0)]),
            "read the pre-write hierarchy"
        );
        assert_eq!(
            flushes() - start,
            1,
            "one rebuild, by the Commit before the read"
        );

        let rows = db.rows(&format!("{} DELETE r {QUERY_0_2}", pair(0, 1)));
        assert_eq!(rows, Ok(vec![Value::Float(5.0)]), "after a delete");
    }

    /// The procedure adds no rebuild of its own: the count is one per `Commit` that touched an
    /// index, with or without the procedure, none for a query that touched nothing indexed, and
    /// a read of one index does not rebuild another.
    #[test]
    fn the_procedure_adds_no_rebuild_of_its_own() {
        let mut db = Db::new("cch_maint_inquery_count");
        let (zero, one) = triangle(&mut db);

        let start = flushes();
        db.ok(&format!("{} SET r.w = 2.0", pair(0, 1)));
        assert_eq!(
            flushes() - start,
            1,
            "a write without the procedure: its one Commit"
        );

        let start = flushes();
        let rows = db.rows(&format!(
            "MATCH (n:N) WHERE id(n) = 0 SET n.x = 1 {QUERY_0_2}"
        ));
        assert_eq!(rows, Ok(vec![Value::Float(3.0)]));
        assert_eq!(
            flushes() - start,
            0,
            "a node property touches no index: no flush"
        );

        let start = flushes();
        let rows = db.rows(&QUERY_0_2.replacen("WITH 1 AS one ", "", 1));
        assert_eq!(rows, Ok(vec![Value::Float(3.0)]));
        assert_eq!(
            flushes() - start,
            0,
            "a query writing nothing does not flush"
        );

        let (zero_before, one_before) = (db.cch(&zero), db.cch(&one));
        let start = flushes();
        let q = format!(
            "MATCH ()-[r:T1]->() SET r.w = 7.0 {} WITH pathWeight \
             MATCH ()-[q:T1]->() SET q.w = 8.0 RETURN pathWeight",
            QUERY_0_2.trim_end_matches(" RETURN pathWeight")
        );
        assert_eq!(db.rows(&q), Ok(vec![Value::Float(3.0)]));
        assert_eq!(
            flushes() - start,
            2,
            "T1 is written before and after the read: one rebuild per Commit"
        );
        assert!(
            Arc::ptr_eq(&zero_before, &db.cch(&zero)),
            "T0 was never touched"
        );
        assert!(!Arc::ptr_eq(&one_before, &db.cch(&one)), "T1 was rebuilt");
    }
}
