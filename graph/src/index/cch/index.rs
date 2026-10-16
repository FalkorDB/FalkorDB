//! The `Graph` API over [`CchIndex`] (#3212).
//!
//! The signatures were fixed up front so the DDL, maintenance and procedure changes could be
//! written against them.
//!
//! MVCC: indexes live on a [`Graph`] version as `Arc<Cch>`. `new_version` clones the `Arc`,
//! and every write here — create, drop, rebuild — replaces or removes an entry on the
//! *new* version only, building a fresh `Cch` rather than mutating a shared one. A reader
//! holding an older version therefore keeps its hierarchy, and rolling a write back is
//! simply dropping the new version.
//!
//! Every write is all-or-nothing: an error leaves this version's indexes as they were.

use std::sync::Arc;

use super::{Cch, CchError, CchIndex, CchIndexDef, CchInput};
use crate::graph::graph::Graph;

impl Graph {
    /// Build a CCH index for `def` and hold it on this version of the graph. Synchronous:
    /// unlike the RediSearch indexes there is no background population pass.
    ///
    /// # Errors
    /// [`CchError::AlreadyExists`] if an index with the same definition is already held,
    /// [`CchError::UnknownRelType`] / [`CchError::UnknownAttribute`] for an unresolvable
    /// definition, and whatever [`Cch::build`] reports.
    pub fn create_cch_index(
        &mut self,
        def: CchIndexDef,
    ) -> Result<(), CchError> {
        if self.cch_index(&def).is_some() {
            return Err(CchError::AlreadyExists);
        }
        let cch = Cch::build(&CchInput::build(self, &def)?)?;
        self.cch_indexes.push(CchIndex {
            def,
            cch: Arc::new(cch),
        });
        Ok(())
    }

    /// Stop holding the index for `def` on this version. Earlier versions that still hold
    /// it keep their `Arc`, so a reader mid-query is unaffected.
    ///
    /// # Errors
    /// [`CchError::NotFound`] if no index with this definition is held.
    pub fn drop_cch_index(
        &mut self,
        def: &CchIndexDef,
    ) -> Result<(), CchError> {
        let at = self
            .cch_indexes
            .iter()
            .position(|idx| &idx.def == def)
            .ok_or(CchError::NotFound)?;
        self.cch_indexes.swap_remove(at);
        Ok(())
    }

    /// Build the CCH indexes an RDB load decoded, after every other index. Only definitions
    /// are persisted, so each hierarchy is built here from the loaded graph.
    ///
    /// A definition that cannot be built (a weight this engine refuses but a C dump may hold,
    /// a graph past the size ceiling, a type or attribute the dump does not have, a duplicate
    /// definition) is dropped rather than failing the load: the index is derived data and the
    /// graph must load. Returns one message per dropped definition, for the caller to log.
    pub fn load_cch_indexes(
        &mut self,
        defs: Vec<CchIndexDef>,
    ) -> Vec<String> {
        defs.into_iter()
            .filter_map(|def| {
                let name = self.cch_index_name(&def);
                self.create_cch_index(def)
                    .err()
                    .map(|e| format!("CCH index {name} was not rebuilt and is dropped: {e}"))
            })
            .collect()
    }

    /// Every CCH index definition held by this version, in no guaranteed order.
    pub fn cch_indexes(&self) -> impl Iterator<Item = &CchIndexDef> {
        self.cch_indexes.iter().map(|idx| &idx.def)
    }

    /// The index held for exactly `def`, if any. A linear scan: a graph holds a handful of
    /// CCH indexes at most, and the definition is the whole key.
    #[must_use]
    pub fn cch_index(
        &self,
        def: &CchIndexDef,
    ) -> Option<&CchIndex> {
        self.cch_indexes.iter().find(|idx| &idx.def == def)
    }

    /// Rebuild every CCH index from the current graph. Used by RDB load (the hierarchy is not
    /// persisted, only the definition) and by maintenance.
    ///
    /// Every definition is re-resolved, so one whose relationship type or attribute no
    /// longer exists fails here rather than being rebuilt over nothing. All indexes are
    /// built before any is replaced: on error, none is.
    ///
    /// # Errors
    /// Whatever [`Self::create_cch_index`] would report for any held definition, other
    /// than [`CchError::AlreadyExists`].
    pub fn rebuild_cch_indexes(&mut self) -> Result<(), CchError> {
        let rebuilt = self
            .cch_indexes
            .iter()
            .map(|idx| Ok(Arc::new(Cch::build(&CchInput::build(self, &idx.def)?)?)))
            .collect::<Result<Vec<_>, CchError>>()?;
        for (idx, cch) in self.cch_indexes.iter_mut().zip(rebuilt) {
            idx.cch = cch;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::graph::graph::TypeId;
    use crate::graph::{graph::Graph, graphblas::test_init};
    use crate::index::cch::testkit::{grid, to_input};
    use crate::index::cch::{Cch, CchError, CchIndex, CchIndexDef};

    /// Checked before the input is built, so it needs no graph data and no graph input.
    #[test]
    fn creating_a_held_index_again_is_refused() {
        let mut g = graph("cch_exists");
        let a = CchIndexDef::new(vec![TypeId(2), TypeId(1)], 0);
        hold(&mut g, &a);
        let same_set = CchIndexDef::new(vec![TypeId(1), TypeId(2), TypeId(1)], 0);
        assert_eq!(g.create_cch_index(same_set), Err(CchError::AlreadyExists));
        assert_eq!(
            g.cch_indexes().count(),
            1,
            "the refused create changed nothing"
        );
    }

    /// A rebuild with nothing held builds nothing, so it needs no input either.
    #[test]
    fn rebuilding_with_no_index_is_a_no_op() {
        let mut g = graph("cch_rebuild_empty");
        assert_eq!(g.rebuild_cch_indexes(), Ok(()));
        assert_eq!(g.cch_indexes().count(), 0);
    }

    /// Pushes straight onto the field instead of calling `create_cch_index`, so these tests
    /// need no graph input.
    fn hold(
        g: &mut Graph,
        def: &CchIndexDef,
    ) -> Arc<Cch> {
        let cch = Arc::new(Cch::build(&to_input(&grid(2, 2))).expect("build"));
        g.cch_indexes.push(CchIndex {
            def: def.clone(),
            cch: Arc::clone(&cch),
        });
        cch
    }

    fn graph(name: &str) -> Graph {
        test_init::ensure_init();
        Graph::new(16, 16, 1, 0, name)
    }

    #[test]
    fn lookup_and_listing_see_exactly_what_is_held() {
        let mut g = graph("cch_lookup");
        let a = CchIndexDef::new(vec![TypeId(1)], 0);
        let b = CchIndexDef::new(vec![TypeId(1), TypeId(2)], 0);
        assert_eq!(g.cch_indexes().count(), 0);
        assert!(g.cch_index(&a).is_none());

        let held = hold(&mut g, &a);
        assert!(Arc::ptr_eq(&g.cch_index(&a).expect("held").cch, &held));
        assert!(
            g.cch_index(&b).is_none(),
            "a different rel set is a different index"
        );
        assert_eq!(g.cch_indexes().collect::<Vec<_>>(), vec![&a]);
    }

    #[test]
    fn drop_removes_only_the_named_index() {
        let mut g = graph("cch_drop");
        let a = CchIndexDef::new(vec![TypeId(1)], 0);
        let b = CchIndexDef::new(vec![TypeId(1)], 1);
        hold(&mut g, &a);
        hold(&mut g, &b);

        assert_eq!(g.drop_cch_index(&a), Ok(()));
        assert!(g.cch_index(&a).is_none());
        assert!(g.cch_index(&b).is_some());
        assert_eq!(
            g.drop_cch_index(&a),
            Err(CchError::NotFound),
            "already dropped"
        );
    }

    /// MVCC: a version taken before a drop keeps the hierarchy; the drop only affects the
    /// version it ran on. Rollback is free for the same reason.
    #[test]
    fn a_drop_on_a_new_version_leaves_the_old_one_holding_the_index() {
        let mut old = graph("cch_mvcc_drop");
        let a = CchIndexDef::new(vec![TypeId(3)], 0);
        let held = hold(&mut old, &a);

        let mut next = old.new_version();
        assert_eq!(next.drop_cch_index(&a), Ok(()));
        assert!(next.cch_index(&a).is_none());
        let kept = old.cch_index(&a).expect("old version still holds it");
        assert!(Arc::ptr_eq(&kept.cch, &held));
    }

    // ---- end to end: a real Graph, built without Cypher -------------------------------

    use roaring::RoaringTreemap;
    use rustc_hash::FxHashMap;

    use crate::index::cch::testkit::{
        EdgeList, TypedEdgeList, assert_path_valid, dijkstra, random_multi_type,
    };
    use crate::index::index_docs::IndexDocs;
    use crate::runtime::value::Value;

    /// Weight attribute name used by every graph below.
    const W: &str = "w";

    /// A `Graph` holding exactly `edges`: one relationship type `T<t>` per kit type tag,
    /// one relationship per entry (so parallel and multi-type edges stay distinct
    /// relationships), and weight attribute `w`. Edges of weight exactly 1.0 get *no*
    /// attribute at all, so the contract's "missing attribute counts as 1.0" rule is
    /// exercised on every graph rather than in one special test.
    fn real_graph(
        edges: &TypedEdgeList,
        types: u16,
        name: &str,
    ) -> (Graph, u16) {
        let mut g = graph(name);
        // A fresh graph's id spaces open their first batch themselves.
        let ids: RoaringTreemap = g
            .node_id_space()
            .reserve(edges.n, &RoaringTreemap::new())
            .expect("reserved")
            .into_iter()
            .collect();
        assert_eq!(ids.len(), edges.n as u64, "a fresh graph hands out 0..n");
        g.create_nodes(&ids).expect("creatable");
        let attr = g.get_or_create_rel_attr_id(&Arc::new(String::from(W)));

        let mut weights: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
        for t in 0..types {
            let mine: Vec<_> = edges.edges.iter().filter(|e| e.3 == t).collect();
            // One batch per type, as one query each would be.
            g.roll_id_batches().expect("a consistent space");
            let rel_ids: Vec<u64> = g
                .relationship_id_space()
                .reserve(mine.len(), &RoaringTreemap::new())
                .expect("reserved");
            let srcs: Vec<u64> = mine.iter().map(|e| u64::from(e.0)).collect();
            let dsts: Vec<u64> = mine.iter().map(|e| u64::from(e.1)).collect();
            g.create_relationships_bulk(&Arc::new(format!("T{t}")), &srcs, &dsts, &rel_ids)
                .expect("creatable");
            for (e, &id) in mine.iter().zip(&rel_ids) {
                if e.2 != 1.0 {
                    weights.insert(id, vec![(attr, Value::Float(e.2))]);
                }
            }
        }
        g.import_relationship_attrs(&weights, &mut IndexDocs::default());
        (g, attr)
    }

    fn def_for(
        g: &Graph,
        types: &[u16],
        attr: u16,
    ) -> CchIndexDef {
        let ids = types
            .iter()
            .map(|t| g.get_type_id(&format!("T{t}")).expect("type exists"))
            .collect();
        CchIndexDef::new(ids, attr)
    }

    /// Every ordered pair of real nodes: weight against the kit's Dijkstra, nodes against
    /// its path checker. Node ids past `oracle.n` are capacity slots and are skipped.
    fn assert_queries_match(
        cch: &Cch,
        oracle: &EdgeList,
    ) {
        assert!(
            cch.node_count() >= oracle.n,
            "the hierarchy spans every node"
        );
        for s in 0..oracle.n as u32 {
            let truth = dijkstra(oracle, s);
            for t in 0..oracle.n as u32 {
                match cch.query(s, t) {
                    Some((w, nodes)) => {
                        assert_eq!(w, truth[t as usize], "{s}->{t} weight");
                        assert_path_valid(oracle, s, t, w, &nodes);
                    }
                    None => assert!(truth[t as usize].is_infinite(), "{s}->{t} missed"),
                }
            }
        }
    }

    /// The whole stack: `Graph` -> input -> Phase 1/2 -> query. Parallel edges and
    /// three relationship types, of which the index covers two: an edge of the third type
    /// must not be usable, and between two nodes the cheapest edge over the covered
    /// types wins.
    #[test]
    fn a_created_index_answers_like_dijkstra_on_the_real_graph() {
        for seed in 0..4u64 {
            let edges = random_multi_type(24, 90, 6, 3, seed);
            let (mut g, attr) = real_graph(&edges, 3, &format!("cch_e2e_{seed}"));
            let def = def_for(&g, &[0, 2], attr);
            g.create_cch_index(def.clone()).expect("create");

            let idx = g.cch_index(&def).expect("held after create");
            assert_queries_match(&idx.cch, &edges.select(&[0, 2]));
        }
    }

    /// Same, with every pair of nodes joined by several parallel edges of one type, so the
    /// "cheapest parallel edge" rule is load-bearing.
    #[test]
    fn parallel_edges_of_one_type_resolve_to_the_cheapest() {
        let mut edges = TypedEdgeList::new(6);
        for u in 0..5u32 {
            for (k, w) in [7.0, 2.0, 5.0].into_iter().enumerate() {
                edges.push(u, u + 1, w + u as f64, (k % 2) as u16);
            }
        }
        let (mut g, attr) = real_graph(&edges, 2, "cch_e2e_parallel");
        let def = def_for(&g, &[0, 1], attr);
        g.create_cch_index(def.clone()).expect("create");
        assert_queries_match(&g.cch_index(&def).unwrap().cch, &edges.select(&[0, 1]));
    }

    /// A bad weight is refused even when a cheaper parallel edge, or a cheaper edge of
    /// another indexed type, would hide it from a "cheapest per pair" merge. Nothing is
    /// held afterwards.
    #[test]
    fn a_nan_on_a_losing_parallel_edge_fails_create_and_holds_nothing() {
        for (losing_type, name) in [(0u16, "cch_nan_same_type"), (1, "cch_nan_other_type")] {
            let mut edges = TypedEdgeList::new(3);
            edges.push(0, 1, 2.0, 0);
            edges.push(1, 2, 2.0, 0);
            edges.push(0, 1, f64::NAN, losing_type); // parallel to the 2.0 edge
            let (mut g, attr) = real_graph(&edges, 2, name);
            let def = def_for(&g, &[0, 1], attr);

            assert_eq!(
                g.create_cch_index(def.clone()),
                Err(CchError::InvalidWeight),
                "{name}"
            );
            assert!(g.cch_index(&def).is_none(), "{name}: nothing held");
            assert_eq!(g.cch_indexes().count(), 0, "{name}");
        }
    }

    #[test]
    fn an_unresolvable_definition_is_refused_at_create() {
        let edges = random_multi_type(5, 8, 3, 1, 1);
        let (mut g, attr) = real_graph(&edges, 1, "cch_e2e_unknown");
        assert_eq!(
            g.create_cch_index(CchIndexDef::new(vec![TypeId(42)], attr)),
            Err(CchError::UnknownRelType)
        );
        let real = g.get_type_id("T0").unwrap();
        assert_eq!(
            g.create_cch_index(CchIndexDef::new(vec![real], 999)),
            Err(CchError::UnknownAttribute)
        );
        assert_eq!(g.cch_indexes().count(), 0, "a refused create holds nothing");
    }

    /// MVCC across create: the version a reader holds does not grow an index because a
    /// writer created one on the next version, and throwing the next version away (the
    /// rollback path) leaves the reader's version exactly as it was.
    #[test]
    fn a_create_on_a_new_version_is_invisible_to_the_old_one() {
        let edges = random_multi_type(10, 25, 4, 1, 3);
        let (reader, attr) = real_graph(&edges, 1, "cch_mvcc_create");
        let def = def_for(&reader, &[0], attr);

        let mut writer = reader.new_version();
        writer.create_cch_index(def.clone()).expect("create");
        assert!(writer.cch_index(&def).is_some());
        assert!(
            reader.cch_index(&def).is_none(),
            "reader's version is untouched"
        );

        drop(writer); // rollback
        assert_eq!(reader.cch_indexes().count(), 0);
    }

    /// MVCC across rebuild: the writer gets a fresh hierarchy; the reader keeps the very
    /// same `Arc`, which still answers.
    #[test]
    fn a_rebuild_on_a_new_version_leaves_the_reader_its_own_hierarchy() {
        let edges = random_multi_type(12, 30, 5, 1, 4);
        let (mut reader, attr) = real_graph(&edges, 1, "cch_mvcc_rebuild");
        let def = def_for(&reader, &[0], attr);
        reader.create_cch_index(def.clone()).expect("create");
        let before = Arc::clone(&reader.cch_index(&def).unwrap().cch);

        let mut writer = reader.new_version();
        writer.rebuild_cch_indexes().expect("rebuild");
        let after = &writer.cch_index(&def).unwrap().cch;

        assert!(
            !Arc::ptr_eq(&before, after),
            "a rebuild builds a new hierarchy"
        );
        assert!(Arc::ptr_eq(&before, &reader.cch_index(&def).unwrap().cch));
        assert_queries_match(&reader.cch_index(&def).unwrap().cch, &edges.select(&[0]));
        assert_queries_match(after, &edges.select(&[0]));
    }

    /// A rebuild re-resolves every definition, and is all-or-nothing: one held index whose
    /// type no longer resolves fails the rebuild and leaves the good one untouched.
    #[test]
    fn a_rebuild_that_fails_on_one_definition_replaces_none() {
        let edges = random_multi_type(8, 20, 4, 1, 5);
        let (mut g, attr) = real_graph(&edges, 1, "cch_rebuild_partial");
        let good = def_for(&g, &[0], attr);
        g.create_cch_index(good.clone()).expect("create");
        let kept = Arc::clone(&g.cch_index(&good).unwrap().cch);
        hold(&mut g, &CchIndexDef::new(vec![TypeId(77)], attr));

        assert_eq!(g.rebuild_cch_indexes(), Err(CchError::UnknownRelType));
        assert!(Arc::ptr_eq(&kept, &g.cch_index(&good).unwrap().cch));
    }

    /// RDB load: a definition that cannot be built is dropped with a message, and the graph
    /// and every other index load as usual.
    #[test]
    fn a_load_drops_only_the_definitions_it_cannot_build() {
        let mut edges = TypedEdgeList::new(3);
        edges.push(0, 1, 2.0, 0);
        edges.push(1, 2, 3.0, 0);
        edges.push(0, 2, -1.0, 1); // C accepts it; this engine does not
        let (mut g, attr) = real_graph(&edges, 2, "cch_load_drops");
        let good = def_for(&g, &[0], attr);
        let bad = def_for(&g, &[0, 1], attr);
        let unknown = CchIndexDef::new(vec![TypeId(9)], attr);

        let dropped = g.load_cch_indexes(vec![bad.clone(), good.clone(), unknown, good.clone()]);
        assert_eq!(dropped.len(), 3, "{dropped:?}");
        assert!(dropped[0].contains("(:T0|T1 {w})") && dropped[0].contains("negative"));
        assert!(dropped[1].contains("(:#9 {w})"), "{dropped:?}");
        assert!(dropped[2].contains("already exists"), "{dropped:?}");
        assert_eq!(g.cch_indexes().collect::<Vec<_>>(), vec![&good]);
        assert_eq!(g.cch_index(&good).unwrap().cch.query(0, 2).unwrap().0, 5.0);
        assert_eq!(g.relationship_count(), 3, "the graph itself loaded");
    }
}
