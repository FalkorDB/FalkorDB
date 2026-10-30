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
//! A later change replaces "rebuild" with a choice between rebuilding, recustomizing and a
//! scoped recustomization; the staging and the flush points stay.

use std::fmt;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use super::input::is_valid_weight;
use super::order::log_warning;
use super::{Cch, CchError, CchIndexDef, CchInput};
use crate::graph::graph::{Graph, RelationshipId, TypeId};
use crate::runtime::value::Value;

/// What a unit of work's writes touched that the graph's CCH indexes might care about, staged in
/// its [`IndexDocs::cch`](crate::index::index_docs::IndexDocs::cch) until the owner flushes it.
///
/// Empty for every graph with no CCH index, so the maintenance hook costs one `is_empty()`
/// check on the existing write path. `Default` allocates nothing.
#[derive(Clone, Default, Debug)]
pub struct CchDirty {
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
    pub fn all() -> Self {
        Self {
            all: true,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.all && self.rel_types.is_empty() && self.attrs.is_empty()
    }

    /// Stage one write's marks after everything already staged.
    pub fn extend(
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
    pub fn touches(
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
/// check covers Cypher, replicated effects and `GRAPH.BULK`. `type_id` is the edges' shared type
/// when the caller knows it. Costs one `is_empty()` on a graph with no CCH index; with one, a
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
    let held = &g.falkordb_index.cch_indexes;
    if held.is_empty() {
        return Ok(());
    }
    for (&id, written) in attrs {
        for (attr, value) in written {
            if !held.iter().any(|i| i.index.def.weight_attr == *attr) || is_valid_weight(value) {
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
            let Some(idx) = held
                .iter()
                .find(|i| i.index.def.weight_attr == *attr && i.index.def.rel_types().contains(&t))
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
                index_name(g, &idx.index.def)
            ));
        }
    }
    Ok(())
}

/// A held index could not be brought up to date. The write that caused it fails, and its
/// rollback leaves the committed indexes as they were.
#[derive(Debug, PartialEq, Eq)]
pub struct CchFlushError {
    pub def: CchIndexDef,
    /// The index as a user would write it, `(:A|B {w})`, resolved when the error was made.
    pub name: String,
    pub cause: CchError,
}

impl fmt::Display for CchFlushError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let cause = match &self.cause {
            // `check_edge_weights` refuses every write of one and an index is never created over
            // one, so reaching it here is a write path that skipped the check.
            CchError::InvalidWeight => {
                "internal error: an invalid weight got past the write-time check"
            }
            CchError::GraphTooLarge => "graph is too large for a CCH index",
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
/// # Errors
/// The first index that fails to build, with the reason (a graph past the size ceiling, ...).
/// An invalid weight is refused at the write ([`check_edge_weights`]), so here it is a bug: it
/// still fails the unit, and is logged.
pub(crate) fn flush(
    g: &mut Graph,
    dirty: &CchDirty,
) -> Result<(), CchFlushError> {
    #[cfg(test)]
    tests::FLUSHES.with(|c| c.set(c.get() + 1));
    if g.falkordb_index.cch_indexes.is_empty() || dirty.is_empty() {
        return Ok(());
    }
    let mut rebuilt = Vec::new();
    for (at, entry) in g.falkordb_index.cch_indexes.iter().enumerate() {
        debug_assert!(!entry.is_building(), "CCH is built synchronously");
        let idx = &entry.index;
        if !dirty.touches(&idx.def) {
            continue;
        }
        let cch = CchInput::build(g, &idx.def)
            .and_then(|input| Cch::build(&input))
            .map_err(|cause| {
                let e = CchFlushError {
                    def: idx.def.clone(),
                    name: index_name(g, &idx.def),
                    cause,
                };
                if e.cause == CchError::InvalidWeight {
                    log_warning(&format!("CCH: {e}"));
                }
                e
            })?;
        rebuilt.push((at, Arc::new(cch)));
    }
    for (at, cch) in rebuilt {
        g.falkordb_index.cch_indexes[at].index.cch = cch;
    }
    Ok(())
}

/// The weight attribute of each held index, for staging which attribute writes matter.
fn weight_attrs(g: &Graph) -> Vec<u16> {
    g.falkordb_index
        .cch_indexes
        .iter()
        .map(|i| i.index.def.weight_attr)
        .collect()
}

/// `(:A|B {w})` for `def`; ids that no longer resolve print as `#id`.
fn index_name(
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
        Edit, TypedEdgeList, check_index_against_oracle, random_edit_batch, random_sparse,
        random_zero_heavy,
    };
    use super::super::{Cch, CchIndexDef};
    use crate::graph::graph::Graph;
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
            let next = self.mvcc.write().expect("the write slot is free");
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
                Err(_) => self.mvcc.rollback(),
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
            let next = self.mvcc.write().expect("the write slot is free");
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

    /// A graph of `edges.n` nodes (ids `0..n`) and one relationship per entry, type `T<t>`,
    /// weight attribute `w`.
    fn load(
        db: &mut Db,
        edges: &TypedEdgeList,
        types: u16,
    ) {
        db.ok(&format!("UNWIND range(1, {}) AS i CREATE (:N)", edges.n));
        for t in 0..types {
            let mut rows = String::new();
            for e in edges.edges.iter().filter(|e| e.3 == t) {
                let sep = if rows.is_empty() { "" } else { "," };
                write!(rows, "{sep}[{}, {}, {}]", e.0, e.1, lit(e.2)).unwrap();
            }
            if !rows.is_empty() {
                db.ok(&format!(
                    "UNWIND [{rows}] AS e MATCH (a), (b) WHERE id(a) = e[0] AND id(b) = e[1] \
                     CREATE (a)-[:T{t} {{w: e[2]}}]->(b)"
                ));
            }
        }
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
            Edit::RemoveEdge { u, v } => format!("{} DELETE r", pair(u, v)),
            Edit::AddEdge { u, v, w, .. } => format!(
                "MATCH (a), (b) WHERE id(a) = {u} AND id(b) = {v} CREATE (a)-[:T0 {{w: {}}}]->(b)",
                lit(w)
            ),
            Edit::AddNode => String::from("CREATE (:N)"),
        }
    }

    fn single_type(edges: &super::super::testkit::EdgeList) -> TypedEdgeList {
        let mut typed = TypedEdgeList::new(edges.n);
        for &(u, v, w) in &edges.edges {
            typed.push(u, v, w, 0);
        }
        typed
    }

    fn check(
        db: &Db,
        def: &CchIndexDef,
        oracle: &super::super::testkit::EdgeList,
    ) {
        let cch = db.cch(def);
        check_index_against_oracle(|s, t| cch.query(s, t), oracle);
    }

    /// The correctness bar: after every write query, every pair answers like Dijkstra on
    /// the current graph. Weight up/down, edge removed, edge added between connected and
    /// unconnected nodes, node added; random graphs, zero-weight-heavy ones included.
    #[test]
    fn every_write_leaves_the_index_answering_like_dijkstra() {
        for seed in 0..12u64 {
            let base = if seed % 2 == 0 {
                random_sparse(12, 26, 9, seed)
            } else {
                random_zero_heavy(12, 26, 9, seed)
            };
            let mut db = Db::new(&format!("cch_maint_random_{seed}"));
            load(&mut db, &single_type(&base), 1);
            let def = db.create_index(&[0]);
            check(&db, &def, &base);

            let mut now = base;
            for batch in 0..3 {
                let (after, edits) = random_edit_batch(&now, seed * 31 + batch);
                for edit in &edits {
                    db.ok(&cypher(edit));
                }
                check(&db, &def, &after);
                now = after;
            }
        }
    }

    /// Writes that cannot change what the index answers leave its hierarchy alone: the
    /// very same `Arc` before and after.
    #[test]
    fn writes_that_touch_nothing_indexed_do_not_rebuild() {
        let mut edges = TypedEdgeList::new(4);
        edges.push(0, 1, 2.0, 0);
        edges.push(1, 2, 3.0, 1);
        edges.push(2, 3, 4.0, 0);
        let mut db = Db::new("cch_maint_unrelated");
        load(&mut db, &edges, 2);
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
        let mut edges = TypedEdgeList::new(4);
        edges.push(0, 1, 1.0, 0);
        edges.push(1, 2, 1.0, 0);
        edges.push(2, 3, 1.0, 0);
        let mut db = Db::new("cch_maint_commits");
        load(&mut db, &edges, 1);
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

        let mut oracle = super::super::testkit::EdgeList::new(4);
        oracle.push(0, 1, 1.0);
        oracle.push(2, 3, 1.0);
        oracle.push(0, 3, 5.0);
        oracle.push(3, 2, 0.5);
        check(&db, &def, &oracle);
    }

    /// Node deletion cascades to incident edges inside `Pending::commit`; the rebuild must
    /// see the graph after the cascade. Also attribute removal (missing = 1.0) and a new
    /// node joined by an edge in the same query.
    #[test]
    fn deletes_cascades_and_removed_weights_are_seen() {
        let mut edges = TypedEdgeList::new(5);
        for (u, v, w) in [
            (0, 1, 4.0),
            (1, 2, 4.0),
            (0, 2, 9.0),
            (2, 3, 2.0),
            (3, 4, 2.0),
        ] {
            edges.push(u, v, w, 0);
        }
        let mut db = Db::new("cch_maint_cascade");
        load(&mut db, &edges, 1);
        let def = db.create_index(&[0]);

        db.ok("MATCH (n) WHERE id(n) = 1 DETACH DELETE n");
        let mut oracle = super::super::testkit::EdgeList::new(5);
        for (u, v, w) in [(0, 2, 9.0), (2, 3, 2.0), (3, 4, 2.0)] {
            oracle.push(u, v, w);
        }
        check(&db, &def, &oracle);

        db.ok(&format!("{} REMOVE r.w", pair(0, 2)));
        oracle.edges[0].2 = 1.0;
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

    /// A rebuild that fails fails the query, names the index and the cause, and its
    /// rollback leaves the committed index (the very same `Arc`) and the graph untouched.
    #[test]
    fn an_invalid_weight_fails_the_write_and_keeps_the_committed_index() {
        let mut edges = TypedEdgeList::new(3);
        edges.push(0, 1, 2.0, 0);
        edges.push(1, 2, 2.0, 0);
        let mut db = Db::new("cch_maint_invalid");
        load(&mut db, &edges, 1);
        let def = db.create_index(&[0]);
        let before = db.cch(&def);

        let start = flushes();
        let err = db
            .write(&format!("{} SET r.w = -1.0", pair(0, 1)))
            .expect_err("a negative weight on an indexed type is refused");
        assert!(err.contains("CCH weight constraint violation"), "{err}");
        assert!(err.contains("CCH index (:T0 {w})"), "{err}");
        assert_eq!(flushes(), start, "refused at the write, before the flush");
        assert!(err.contains("needs weights that are >= 0"), "{err}");

        assert!(Arc::ptr_eq(&before, &db.cch(&def)));
        check(&db, &def, &edges.select(&[0]));
        // The write was rolled back with it: a later valid write sees weight 2.0.
        db.ok(&format!("{} SET r.other = 1", pair(1, 2)));
        check(&db, &def, &edges.select(&[0]));
    }

    /// The write-time check, through every Cypher shape that writes a weight: each is refused
    /// with the constraint error, the committed index is the very same `Arc` and the graph is
    /// as before. A refusal in the query's only `Commit` never reaches the flush.
    #[test]
    fn every_cypher_write_of_an_invalid_weight_is_refused_at_the_write() {
        let mut edges = TypedEdgeList::new(3);
        edges.push(0, 1, 2.0, 0);
        edges.push(1, 2, 2.0, 0);
        let mut db = Db::new("cch_wcheck_cypher");
        load(&mut db, &edges, 1);
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
        check(&db, &def, &edges.select(&[0]));
    }

    /// What the check lets through: removing the weight or writing a non-numeric one (both
    /// weigh 1.0), a valid value that overwrites an invalid one inside one `SET`, an invalid
    /// value on another type or another attribute, and anything on a graph with no index.
    #[test]
    fn removed_non_numeric_and_unindexed_weights_stay_legal() {
        let mut edges = TypedEdgeList::new(4);
        edges.push(0, 1, 2.0, 0);
        edges.push(1, 2, 2.0, 0);
        edges.push(2, 3, 2.0, 1);
        let mut db = Db::new("cch_wcheck_legal");
        load(&mut db, &edges, 2);
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
        let mut oracle = super::super::testkit::EdgeList::new(4);
        oracle.push(0, 1, 4.0);
        oracle.push(1, 2, 1.0);
        check(&db, &def, &oracle);

        let mut none = Db::new("cch_wcheck_no_index");
        load(&mut none, &edges, 2);
        none.ok(&format!("{} SET r.w = -1", pair(0, 1)));
    }

    /// `+inf` is a closed edge and `-0.0` weighs 0: both are legal through every Cypher shape
    /// that writes a weight, and the index answers like the oracle, which reads `+inf` as no
    /// edge. The refusal of a negative weight says what is legal.
    #[test]
    fn a_closed_edge_and_negative_zero_are_legal_writes() {
        let mut edges = TypedEdgeList::new(4);
        edges.push(0, 1, 2.0, 0);
        edges.push(1, 2, 2.0, 0);
        edges.push(2, 3, 2.0, 0);
        let mut db = Db::new("cch_wcheck_closed");
        load(&mut db, &edges, 1);
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
        let mut oracle = super::super::testkit::EdgeList::new(4);
        for (u, v, w) in [
            (0, 1, inf),
            (1, 2, 0.0),
            (2, 3, inf),
            (0, 2, inf),
            (0, 3, inf),
        ] {
            oracle.push(u, v, w);
        }
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
        let mut edges = TypedEdgeList::new(3);
        edges.push(0, 1, 2.0, 0);
        edges.push(1, 2, 2.0, 1);
        let mut db = Db::new("cch_wcheck_bulk");
        load(&mut db, &edges, 2);
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
        let next = db.mvcc.write().expect("the write slot is free");
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
        db.mvcc.rollback();
    }

    /// Failure after a `Commit` that the flush would have acted on: the earlier `Commit`'s
    /// change is rolled back, and so is any rebuilt hierarchy.
    #[test]
    fn a_query_failing_after_a_commit_keeps_the_committed_index() {
        let mut edges = TypedEdgeList::new(3);
        edges.push(0, 1, 2.0, 0);
        edges.push(1, 2, 2.0, 0);
        let mut db = Db::new("cch_maint_late_failure");
        load(&mut db, &edges, 1);
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
        check(&db, &def, &edges.select(&[0]));
    }

    /// `+inf` (`1.0 / 0.0` in Cypher) is a closed edge: a write that sets it is legal, and
    /// the rebuilt index answers like the oracle, which reads it as no edge — a detour, a
    /// finite parallel edge, or no path; reopening gives back the first answers.
    #[test]
    fn closing_and_reopening_an_edge_through_cypher_answers_like_dijkstra() {
        let mut edges = TypedEdgeList::new(4);
        for (u, v, w) in [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 2, 10.0),
            (2, 3, 1.0),
            (1, 3, 7.0),
        ] {
            edges.push(u, v, w, 0);
        }
        let mut db = Db::new("cch_maint_closed");
        load(&mut db, &edges, 1);
        let def = db.create_index(&[0]);
        check(&db, &def, &edges.select(&[0]));

        // Close 0 -> 1: legal, and routes go 0 -> 2 directly.
        db.ok(&format!("{} SET r.w = 1.0 / 0.0", pair(0, 1)));
        edges.edges[0].2 = f64::INFINITY;
        check(&db, &def, &edges.select(&[0]));
        assert_eq!(db.cch(&def).query(0, 3).map(|(w, _)| w), Some(11.0));

        // A finite edge parallel to the closed one wins the pair.
        db.ok("MATCH (a), (b) WHERE id(a) = 0 AND id(b) = 1 CREATE (a)-[:T0 {w: 2.0}]->(b)");
        edges.push(0, 1, 2.0, 0);
        check(&db, &def, &edges.select(&[0]));
        assert_eq!(db.cch(&def).query(0, 3).map(|(w, _)| w), Some(4.0));

        // Close everything into 3: no path.
        db.ok("MATCH ()-[r:T0]->(b) WHERE id(b) = 3 SET r.w = 1.0 / 0.0");
        for e in edges.edges.iter_mut().filter(|e| e.1 == 3) {
            e.2 = f64::INFINITY;
        }
        check(&db, &def, &edges.select(&[0]));
        assert_eq!(db.cch(&def).query(0, 3), None);

        // Reopen.
        db.ok("MATCH ()-[r:T0]->(b) WHERE id(b) = 3 SET r.w = 1.0");
        db.ok(&format!("{} AND r.w > 1.0e300 SET r.w = 1.0", pair(0, 1)));
        for e in &mut edges.edges {
            if e.1 == 3 || e.2.is_infinite() {
                e.2 = 1.0;
            }
        }
        check(&db, &def, &edges.select(&[0]));
        assert_eq!(db.cch(&def).query(0, 3).map(|(w, _)| w), Some(2.0));
    }

    /// Two relationship types plus parallel edges: an index over `T0` alone is rebuilt by
    /// `T0` writes only, and the cheapest parallel edge keeps winning.
    #[test]
    fn multi_type_and_parallel_edges_stay_correct() {
        let mut edges = TypedEdgeList::new(4);
        for (u, v, w, t) in [
            (0, 1, 5.0, 0),
            (0, 1, 3.0, 0),
            (0, 1, 1.0, 1),
            (1, 2, 2.0, 0),
            (2, 3, 2.0, 1),
            (2, 3, 6.0, 0),
        ] {
            edges.push(u, v, w, t);
        }
        let mut db = Db::new("cch_maint_multi");
        load(&mut db, &edges, 2);
        let both = db.create_index(&[0, 1]);
        let zero = db.create_index(&[0]);
        check(&db, &both, &edges.select(&[0, 1]));
        check(&db, &zero, &edges.select(&[0]));

        let zero_before = db.cch(&zero);
        db.ok("MATCH ()-[r:T1]->() SET r.w = r.w + 10.0");
        for e in edges.edges.iter_mut().filter(|e| e.3 == 1) {
            e.2 += 10.0;
        }
        assert!(Arc::ptr_eq(&zero_before, &db.cch(&zero)));
        check(&db, &both, &edges.select(&[0, 1]));

        db.ok("MATCH ()-[r:T0]->() WHERE r.w = 3.0 DELETE r");
        edges.edges.retain(|e| !(e.3 == 0 && e.2 == 3.0));
        check(&db, &both, &edges.select(&[0, 1]));
        check(&db, &zero, &edges.select(&[0]));
    }
}
