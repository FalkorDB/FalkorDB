//! Keeping CCH indexes correct after write queries (#3214).
//!
//! A write query records what it touched in a [`CchDirty`] (held by `Pending`, accumulated
//! across all of the query's `Commit`s), and once the whole query has succeeded the runtime
//! hands it to [`Graph::cch_flush`] on the query's private MVCC version. Each held index that
//! the query could have affected is rebuilt from scratch into a new `Arc<Cch>`; the rest keep
//! their `Arc` untouched.
//!
//! Recording is gated on the graph holding at least one CCH index, so a graph without one
//! never fills a `CchDirty` and the flush is a single `is_empty()` check.
//!
//! What counts as touching an index (mirrors C's `graph_hub.c` marks):
//! - any node created or deleted: every index (C's `GraphContext_CCHMarkAllRebuild`); this
//!   also covers the incident edges a node deletion cascades to;
//! - a relationship created or explicitly deleted: every index spanning its type;
//! - an attribute set or removed on an existing relationship: every index spanning its type
//!   whose weight attribute it is.
//!
//! Effects replay fills a `CchDirty` by the same rules, from the records themselves
//! (`effects::v3::apply`), and flushes it once per buffer.
//!
//! A later change replaces "rebuild" with a choice between rebuilding, recustomizing and a
//! scoped recustomization; the recording and the hook stay.

use std::fmt;
use std::sync::Arc;

use rustc_hash::FxHashSet;

use super::{Cch, CchError, CchIndexDef, CchInput};
use crate::graph::graph::{Graph, TypeId};

/// What a write query touched, accumulated in `Pending` across `Commit`s (survives `end_segment()`).
///
/// Empty for every graph with no CCH index, so the maintenance hook costs one `is_empty()`
/// check on the existing write path. `Default` allocates nothing.
#[derive(Default, Debug)]
pub struct CchDirty {
    /// A node was created or deleted, or the caller asked for everything ([`Self::all`]).
    all: bool,
    /// Types of relationships created or explicitly deleted.
    rel_types: FxHashSet<TypeId>,
    /// `(type, attribute)` of attributes set or removed on existing relationships.
    attrs: FxHashSet<(TypeId, u16)>,
}

impl CchDirty {
    /// Every held index is dirty. For writers that do not go through `Pending` and cannot say
    /// what they touched (bulk insert).
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

    /// Nodes were created or deleted.
    pub(crate) fn nodes_changed(&mut self) {
        self.all = true;
    }

    /// A relationship of `rel_type` was created or deleted.
    pub(crate) fn rel_type_changed(
        &mut self,
        rel_type: TypeId,
    ) {
        self.rel_types.insert(rel_type);
    }

    /// Attribute `attr` was set or removed on an existing relationship of `rel_type`.
    pub(crate) fn attr_changed(
        &mut self,
        rel_type: TypeId,
        attr: u16,
    ) {
        self.attrs.insert((rel_type, attr));
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
            CchError::InvalidWeight => "edge weights must be finite and >= 0",
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

impl Graph {
    /// Bring every held CCH index that `dirty` touches up to date with this version of the
    /// graph. Called once per successful write query, after its last `Commit`, on the
    /// query's private version: the new hierarchies are published together with the graph.
    ///
    /// All touched indexes are built before any is replaced, so on error none is.
    ///
    /// # Errors
    /// The first index that fails to build, with the reason (a weight that is negative or
    /// not finite, a graph past the size ceiling, ...).
    pub fn cch_flush(
        &mut self,
        dirty: CchDirty,
    ) -> Result<(), CchFlushError> {
        #[cfg(test)]
        tests::FLUSHES.with(|c| c.set(c.get() + 1));
        if self.cch_indexes.is_empty() || dirty.is_empty() {
            return Ok(());
        }
        let mut rebuilt = Vec::new();
        for (at, idx) in self.cch_indexes.iter().enumerate() {
            if !dirty.touches(&idx.def) {
                continue;
            }
            let cch = CchInput::build(self, &idx.def)
                .and_then(|input| Cch::build(&input))
                .map_err(|cause| CchFlushError {
                    def: idx.def.clone(),
                    name: self.cch_index_name(&idx.def),
                    cause,
                })?;
            rebuilt.push((at, Arc::new(cch)));
        }
        for (at, cch) in rebuilt {
            self.cch_indexes[at].cch = cch;
        }
        Ok(())
    }

    /// The weight attribute of each held index, for recording which attribute writes matter.
    pub(crate) fn cch_weight_attrs(&self) -> Vec<u16> {
        self.cch_indexes.iter().map(|i| i.def.weight_attr).collect()
    }

    /// `(:A|B {w})` for `def`; ids that no longer resolve print as `#id`.
    pub(crate) fn cch_index_name(
        &self,
        def: &CchIndexDef,
    ) -> String {
        let types = def
            .rel_types()
            .iter()
            .map(|&t| {
                self.get_type(t)
                    .map_or_else(|| format!("#{}", t.0), |n| n.to_string())
            })
            .collect::<Vec<_>>()
            .join("|");
        let attr = self
            .rel_attr_name(def.weight_attr)
            .map_or_else(|| format!("#{}", def.weight_attr), |n| n.to_string());
        format!("(:{types} {{{attr}}})")
    }
}

#[cfg(test)]
mod tests {
    //! Through real Cypher: a `Runtime` over an `MvccGraph` write slot, the way the module
    //! runs a write query (minus the locks), so `Commit`, `Pending` and the end-of-query
    //! hook are the production ones. The oracle is the test kit over a mirror edge list.

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
    use crate::locks::WriteEscalation;
    use crate::planner::IR;
    use crate::runtime::functions::{init_functions, init_udf_functions};
    use crate::runtime::runtime::Runtime;
    use crate::runtime::value::Value;

    thread_local! {
        /// Calls to [`Graph::cch_flush`] on this thread.
        pub(super) static FLUSHES: Cell<usize> = const { Cell::new(0) };
    }

    fn flushes() -> usize {
        FLUSHES.with(Cell::get)
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
        u: u32,
        v: u32,
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

    /// No CCH index: nothing is recorded, so the hook never reaches `cch_flush` at all.
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

    /// Several `Commit`s in one query: the flush runs once, after the last, and sees every
    /// `Commit`'s changes (the first adds an edge, the second removes another).
    #[test]
    fn a_query_with_several_commits_flushes_once_and_sees_them_all() {
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
        assert_eq!(flushes() - start, 1, "one flush for the whole query");

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
        let new_id = (0..g.borrow().node_cap() as u32)
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

        let err = db
            .write(&format!("{} SET r.w = -1.0", pair(0, 1)))
            .expect_err("a negative weight on an indexed type is refused");
        assert!(err.contains("CCH index (:T0 {w})"), "{err}");
        assert!(err.contains("finite and >= 0"), "{err}");

        assert!(Arc::ptr_eq(&before, &db.cch(&def)));
        check(&db, &def, &edges.select(&[0]));
        // The write was rolled back with it: a later valid write sees weight 2.0.
        db.ok(&format!("{} SET r.other = 1", pair(1, 2)));
        check(&db, &def, &edges.select(&[0]));
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
        assert_eq!(flushes(), start, "a failed query does not flush");
        assert!(Arc::ptr_eq(&before, &db.cch(&def)));
        check(&db, &def, &edges.select(&[0]));
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

    /// 0 -> 1 -> 2 at 1.0 each and a direct 0 -> 2 at 5.0, type `T0`; `0 -> 1` again at 1.0
    /// as type `T1`. Indexes over `T0` and over `T1`.
    fn triangle(db: &mut Db) -> (CchIndexDef, CchIndexDef) {
        let mut edges = TypedEdgeList::new(3);
        edges.push(0, 1, 1.0, 0);
        edges.push(1, 2, 1.0, 0);
        edges.push(0, 2, 5.0, 0);
        edges.push(0, 1, 1.0, 1);
        load(db, &edges, 2);
        (db.create_index(&[0]), db.create_index(&[1]))
    }

    /// `db.idx.cch.query` from node 0 to node 2 over `T0`, as the rest of a query.
    const QUERY_0_2: &str = "WITH 1 AS one MATCH (s), (t) WHERE id(s) = 0 AND id(t) = 2 \
        CALL db.idx.cch.query({sourceNode: s, targetNode: t, relTypes: ['T0'], weightProp: 'w'}) \
        YIELD pathWeight RETURN pathWeight";

    /// A procedure reading an index the same write query has touched sees that query's
    /// writes, and the query still flushes once, not once more at its end.
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
            "flushed before the read only, not again at the end"
        );
    }

    /// Same with the cheap edge deleted earlier in the query: the procedure answers without
    /// it rather than failing the query on a hierarchy that still holds it.
    #[test]
    fn a_procedure_after_a_delete_in_its_query() {
        let mut db = Db::new("cch_maint_inquery_delete");
        triangle(&mut db);
        let rows = db.rows(&format!("{} DELETE r {QUERY_0_2}", pair(0, 1)));
        assert_eq!(rows, Ok(vec![Value::Float(5.0)]));
    }

    /// The in-query flush happens only when the procedure reads an index the query touched:
    /// a query without the procedure, a procedure in a query that touched nothing, and a
    /// procedure reading an untouched index (with a touched one written again after the read)
    /// each flush exactly as often as without the procedure.
    #[test]
    fn only_a_read_of_a_touched_index_flushes_early() {
        let mut db = Db::new("cch_maint_inquery_count");
        let (zero, one) = triangle(&mut db);

        let start = flushes();
        db.ok(&format!("{} SET r.w = 2.0", pair(0, 1)));
        assert_eq!(
            flushes() - start,
            1,
            "a write without the procedure: the end flush"
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
            1,
            "reading T0 must not flush the T1 change early (T1 is written again after the read)"
        );
        assert!(
            Arc::ptr_eq(&zero_before, &db.cch(&zero)),
            "T0 was never touched"
        );
        assert!(
            !Arc::ptr_eq(&one_before, &db.cch(&one)),
            "T1 was rebuilt, once"
        );
    }
}
