//! Keeping CCH indexes correct under writes (#3214).
//!
//! For now this is the write-time weight check, [`check_edge_weights`]: the graph methods that
//! write relationship attributes refuse, before writing, a weight a held index cannot take,
//! the way a constraint refuses a value. Every path writes relationship attributes through
//! those methods (Cypher, replicated effects and `GRAPH.BULK`), so one check covers them all.
//!
//! The check reads the held index definitions, never their hierarchies, so it holds for every
//! index the version holds, whatever state its hierarchy is in. It does not bring a hierarchy
//! up to date with the write: that is the maintenance change (#3214).

use super::CchIndexDef;
use super::input::is_valid_weight;
use crate::graph::graph::{Graph, RelationshipId, TypeId};
use crate::runtime::value::Value;

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

#[cfg(test)]
mod tests {
    //! Through real Cypher: a `Runtime` over an `MvccGraph` write slot, the way the module
    //! runs a write query (minus the locks), so `Commit` and `Pending` are the production
    //! ones. Nothing brings a held hierarchy up to date with a write yet (#3214), so what a
    //! legal write leaves behind is checked by building the index again over it, against the
    //! test kit's oracle.

    use std::fmt::Write as _;
    use std::sync::Arc;

    use super::super::testkit::{INDEXED, RealEdge};
    use super::super::{Cch, CchIndexDef};
    use crate::graph::graph::Graph;
    use crate::graph::graphblas::matrix::Matrix;
    use crate::graph::graphblas::test_init;
    use crate::graph::mvcc_graph::MvccGraph;
    use crate::index::index_docs::IndexDocs;
    use crate::locks::WriteEscalation;
    use crate::runtime::functions::{init_functions, init_udf_functions};
    use crate::runtime::runtime::Runtime;
    use crate::runtime::value::Value;
    use crate::testkit::{check_index_against_oracle, weights};

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
            runtime
                .publish(|fork| self.mvcc.commit(fork))
                .expect("the version validates");
            Ok(rows)
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
            g.create_cch_index(def.clone()).expect("create");
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
            g.create_cch_index(def.clone())
                .expect("the stored weights are indexable");
            self.mvcc.commit(g).expect("the version validates");
        }

        fn cch(
            &self,
            def: &CchIndexDef,
        ) -> Arc<Cch> {
            Arc::clone(&self.committed().cch_index(def).expect("held").cch)
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
}
