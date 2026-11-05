//! `db.idx.cch.query` and the CCH rows of `db.indexes()`: the procedure side of the CCH index.
//!
//! Registered in `runtime::functions::procedures` only in a build with the `index-falkordb`
//! feature. Split out from `register` so the argument parsing and validation can be unit
//! tested directly against a hand-built `Graph`, without needing a `Runtime`.

use std::sync::Arc;

use thin_vec::{ThinVec, thin_vec};

use super::{CchIndexDef, input::cheapest_edge};
use crate::{
    graph::graph::{Graph, NodeId},
    runtime::{ordermap::OrderMap, value::Value},
};

/// Validated `db.idx.cch.query({sourceNode, targetNode, relTypes, weightProp})`
/// config. `rel_types` and `weight_prop` are checked to exist in the graph's
/// schema/attribute tables; `cch_query_rows` resolves them to a `CchIndexDef`.
#[derive(Debug)]
pub(crate) struct CchQueryConfig {
    source: NodeId,
    target: NodeId,
    rel_types: Vec<Arc<String>>,
    weight_prop: Arc<String>,
}

/// Parses and validates the single map argument to `db.idx.cch.query`,
/// matching C's `_read_config` (`proc_cch_idx_query.c`) error for error:
/// sourceNode/targetNode presence and type, relTypes shape/non-emptiness and
/// existence, weightProp type and existence.
pub(crate) fn parse_cch_query_config(
    g: &Graph,
    args: &[Value],
) -> Result<CchQueryConfig, String> {
    let Some(Value::Map(map)) = args.first() else {
        return Err(String::from(
            "db.idx.cch.query expects a single map argument",
        ));
    };

    let source_val = map.get_str("sourceNode");
    let target_val = map.get_str("targetNode");
    let source_present = matches!(source_val, Some(v) if !matches!(v, Value::Null));
    let target_present = matches!(target_val, Some(v) if !matches!(v, Value::Null));
    if !source_present || !target_present {
        return Err(String::from(
            "db.idx.cch.query requires sourceNode and targetNode",
        ));
    }
    let (source, target) = match (source_val.unwrap(), target_val.unwrap()) {
        (Value::Node(s), Value::Node(t)) => (*s, *t),
        _ => {
            return Err(String::from(
                "db.idx.cch.query, sourceNode/targetNode must be nodes",
            ));
        }
    };

    let rel_types = match map.get_str("relTypes") {
        Some(Value::List(list)) => {
            let mut types = Vec::with_capacity(list.len());
            for v in list.iter() {
                match v {
                    Value::String(s) => types.push(s.clone()),
                    _ => {
                        return Err(String::from(
                            "db.idx.cch.query, relTypes must be an array of strings",
                        ));
                    }
                }
            }
            types
        }
        _ => {
            return Err(String::from(
                "db.idx.cch.query, relTypes must be an array of strings",
            ));
        }
    };
    if rel_types.is_empty() {
        return Err(String::from("db.idx.cch.query, relTypes must not be empty"));
    }
    for t in &rel_types {
        if g.get_relationship_matrix(t).is_none() {
            return Err(format!("db.idx.cch.query, unknown relationship type: {t}"));
        }
    }

    let weight_prop = match map.get_str("weightProp") {
        Some(Value::String(s)) => s.clone(),
        _ => return Err(String::from("db.idx.cch.query requires string weightProp")),
    };
    if g.get_relationship_attribute_id(&weight_prop).is_none() {
        return Err(format!(
            "db.idx.cch.query, unknown attribute: {weight_prop}"
        ));
    }

    Ok(CchQueryConfig {
        source,
        target,
        rel_types,
        weight_prop,
    })
}

/// Resolves a CCH hierarchy path -- already-unpacked repo `NodeId`s, in
/// order -- into a `Value::Path`, choosing the cheapest edge among
/// `rel_types` at each hop by `weight_attr` (a graph-wide property-key id,
/// as returned by `Graph::get_relationship_attribute_id`). A
/// missing/non-numeric attribute defaults to weight 1.0, the same rule
/// `CchInput::weights` uses. Mirrors C's `_best_road_edge`
/// (`proc_cch_idx_query.c`).
///
/// Each hop is [`cheapest_edge`] over the pair's own edges
/// ([`Graph::get_src_dest_relationships`]), not a scan of `from`'s outgoing
/// edges. Ties go to the earlier type in `rel_types`, then to the lowest edge
/// id.
///
/// Unpacking rank space back into repo `NodeId`s is `Cch::query` plus the
/// index's node permutation, not this helper's job. This only
/// turns an already-resolved node sequence into the `path` yield.
///
/// # Errors
/// If a consecutive pair in `node_ids` has no realizing edge among
/// `rel_types`: the hierarchy and the live graph have gone out of sync.
pub(crate) fn cch_path_from_node_ids(
    g: &Graph,
    node_ids: &[NodeId],
    rel_types: &[Arc<String>],
    weight_attr: u16,
) -> Result<Value, String> {
    let Some(&first) = node_ids.first() else {
        return Err(String::from("db.idx.cch.query: empty hierarchy path"));
    };

    let mut elems: ThinVec<Value> = ThinVec::with_capacity(node_ids.len() * 2);
    elems.push(Value::Node(first));

    for pair in node_ids.windows(2) {
        let (from, to) = (pair[0], pair[1]);
        let best = cheapest_edge(g, rel_types, weight_attr, from, to);
        let Some((eid, _)) = best else {
            return Err(format!(
                "db.idx.cch.query: hierarchy inconsistent with the graph, no edge among the \
                 indexed types from node {} to node {}",
                u64::from(from),
                u64::from(to)
            ));
        };
        elems.push(Value::Relationship(eid));
        elems.push(Value::Node(to));
    }

    Ok(Value::Path(Arc::new(elems)))
}

/// One `db.indexes()` row for a CCH index, shaped like C's `_EmitCCHIndex`: the relationship types
/// joined by `,` as the label, the weight attribute as the only property, `{weight: ['CCH']}` as
/// the types. The RediSearch-specific columns are filled by the caller.
pub(crate) struct CchIndexRow {
    pub(crate) label: Value,
    pub(crate) properties: Value,
    pub(crate) types: Value,
}

pub(crate) fn cch_index_rows(g: &Graph) -> Vec<CchIndexRow> {
    let type_names = g.get_types();
    let attr_names = g.get_relationship_attribute_names();
    g.cch_indexes()
        .filter_map(|def| {
            let rels: Option<Vec<&str>> = def
                .rel_types()
                .iter()
                .map(|t| type_names.get(t.0).map(|n| n.as_str()))
                .collect();
            let weight = attr_names.get(usize::from(def.weight_attr))?.clone();
            let mut types = OrderMap::default();
            types.insert(
                weight.clone(),
                Value::List(Arc::new(thin_vec![Value::String(Arc::new(String::from(
                    "CCH"
                )))])),
            );
            Some(CchIndexRow {
                label: Value::String(Arc::new(rels?.join(","))),
                properties: Value::List(Arc::new(thin_vec![Value::String(weight)])),
                types: Value::Map(Arc::new(types)),
            })
        })
        .collect()
}

/// Looks up the index named by `config` and answers the query: no row when the nodes are
/// unconnected, one `(pathWeight, path)` row otherwise. Mirrors C's `Proc_CCHIdxQueryInvoke`.
///
/// The hierarchy is the one this graph version holds. Inside a write query it already has the
/// query's earlier writes: a `CALL` after an update clause needs a `WITH`, which plans a
/// `Commit`, and each `Commit` rebuilds the indexes it touched.
pub(crate) fn cch_query_rows(
    g: &Graph,
    config: &CchQueryConfig,
) -> Result<Vec<(Value, Value)>, String> {
    let index = cch_query_def(g, config).and_then(|def| g.cch_index(&def));
    let Some(index) = index else {
        return Err(String::from(
            "db.idx.cch.query: no CCH index over these relationship types and weight \
             attribute; create one with CREATE CCH INDEX FOR ()-[e:<types>]->() ON (e.<weight>)",
        ));
    };

    let (src, dst) = (u64::from(config.source), u64::from(config.target));
    let Some((weight, nodes)) = index.cch.query(src, dst) else {
        return Ok(vec![]);
    };
    let nodes: Vec<NodeId> = nodes.into_iter().map(NodeId::from).collect();
    let path = cch_path_from_node_ids(g, &nodes, &config.rel_types, attr_index(g, config)?)?;
    Ok(vec![(Value::Float(weight), path)])
}

/// The index definition `config` names, if its types and weight attribute all exist.
fn cch_query_def(
    g: &Graph,
    config: &CchQueryConfig,
) -> Option<CchIndexDef> {
    let types: Option<Vec<_>> = config.rel_types.iter().map(|t| g.get_type_id(t)).collect();
    let attr = g
        .get_relationship_attribute_id(&config.weight_prop)
        .and_then(|a| u16::try_from(a).ok());
    types
        .zip(attr)
        .map(|(types, attr)| CchIndexDef::new(types, attr))
}

fn attr_index(
    g: &Graph,
    config: &CchQueryConfig,
) -> Result<u16, String> {
    g.get_relationship_attribute_id(&config.weight_prop)
        .and_then(|a| u16::try_from(a).ok())
        .ok_or_else(|| {
            format!(
                "db.idx.cch.query, unknown attribute: {}",
                config.weight_prop
            )
        })
}

#[cfg(test)]
mod cch_query_tests {
    use super::{
        CchIndexDef, CchQueryConfig, NodeId, cch_index_rows, cch_path_from_node_ids,
        cch_query_rows, parse_cch_query_config,
    };
    use crate::graph::{
        graph::{Graph, RelationshipId},
        graphblas::test_init,
    };
    use crate::index::index_docs::IndexDocs;
    use crate::runtime::ordermap::OrderMap;
    use crate::runtime::value::Value;
    use roaring::RoaringTreemap;
    use rustc_hash::FxHashMap;
    use std::sync::Arc;
    use thin_vec::thin_vec;

    fn s(v: &str) -> Arc<String> {
        Arc::new(String::from(v))
    }

    /// Reserve `n` node ids the way one query's `CreateOp` does: open the
    /// graph's batch, then ask the space for ids this query does not hold yet.
    fn reserve_nodes(
        g: &mut Graph,
        n: usize,
    ) -> Vec<u64> {
        g.roll_id_batches().expect("a consistent space");
        g.node_id_space()
            .reserve(n, &RoaringTreemap::new())
            .expect("reserved")
    }

    fn reserve_relationships(
        g: &Graph,
        n: usize,
    ) -> Vec<u64> {
        g.relationship_id_space()
            .reserve(n, &RoaringTreemap::new())
            .expect("reserved")
    }

    fn map_arg(pairs: Vec<(&str, Value)>) -> Vec<Value> {
        let mut m = OrderMap::default();
        for (k, v) in pairs {
            m.insert(s(k), v);
        }
        vec![Value::Map(Arc::new(m))]
    }

    /// A graph with a `ROAD` schema (and `t` attribute) but no `w` attribute,
    /// matching test_cch.py's test03_query_validation fixture.
    fn fixture_graph() -> Graph {
        test_init::ensure_init();
        let mut g = Graph::new(16, 16, 1, 0, "cch_query_validation_test");
        let node_ids = reserve_nodes(&mut g, 2);
        g.create_nodes(
            &node_ids.iter().copied().collect(),
            &mut IndexDocs::default(),
        )
        .unwrap();

        g.get_type_id_mut("ROAD");
        let rel_ids = reserve_relationships(&g, 1);
        g.create_relationships_bulk(
            &s("ROAD"),
            &[node_ids[0]],
            &[node_ids[1]],
            &[rel_ids[0]],
            &mut IndexDocs::default(),
        )
        .unwrap();
        let t_attr = g.get_or_create_rel_attr_id(&s("t"));
        let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
        attrs.insert(rel_ids[0], vec![(t_attr, Value::Int(2))]);
        let mut index_add_edge_docs = IndexDocs::default();
        g.set_relationships_attributes(&attrs, &mut index_add_edge_docs)
            .unwrap();
        g
    }

    /// Every malformed argument is refused with C's message: a missing or null end, an end
    /// that is not a node, an unknown or empty relationship type list, an unknown weight
    /// attribute, and an argument that is not a map.
    #[test]
    fn parse_errors_match_c() {
        let g = fixture_graph();
        let road = || Value::List(Arc::new(thin_vec![Value::String(s("ROAD"))]));
        let args = |src: Value, dst: Value, types: Value, w: &str| {
            map_arg(vec![
                ("sourceNode", src),
                ("targetNode", dst),
                ("relTypes", types),
                ("weightProp", Value::String(s(w))),
            ])
        };
        let n0 = || Value::Node(0u64.into());
        let n1 = || Value::Node(1u64.into());
        for (case, argv, needle) in [
            (
                "no targetNode",
                map_arg(vec![("sourceNode", n0())]),
                "sourceNode and targetNode",
            ),
            (
                "a null sourceNode",
                args(Value::Null, n1(), road(), "w"),
                "sourceNode and targetNode",
            ),
            (
                "a null targetNode",
                args(n0(), Value::Null, road(), "w"),
                "sourceNode and targetNode",
            ),
            (
                "an end that is not a node",
                args(Value::Int(1), n1(), road(), "w"),
                "must be nodes",
            ),
            (
                "an unknown relationship type",
                args(
                    n0(),
                    n1(),
                    Value::List(Arc::new(thin_vec![Value::String(s("NOPE"))])),
                    "w",
                ),
                "unknown relationship type",
            ),
            (
                "no relationship type",
                args(n0(), n1(), Value::List(Arc::new(thin_vec![])), "w"),
                "must not be empty",
            ),
            (
                "an unknown attribute",
                args(n0(), n1(), road(), "nope"),
                "unknown attribute",
            ),
            ("not a map", vec![Value::Int(1)], "single map argument"),
        ] {
            let err = parse_cch_query_config(&g, &argv).expect_err(case);
            assert!(err.contains(needle), "{case}: {err}");
        }
    }

    // ---- cch_path_from_node_ids --------------------------------------------

    /// Builds n0 -[ROAD w=10]-> n1, n0 -[FERRY w=2]-> n1 (parallel, cheaper),
    /// n1 -[ROAD]-> n2 (no `w` set, defaults to 1.0).
    fn path_fixture() -> (Graph, Vec<NodeId>, u16, Vec<RelationshipId>) {
        test_init::ensure_init();
        let mut g = Graph::new(16, 16, 1, 0, "cch_path_helper_test");
        let node_ids = reserve_nodes(&mut g, 3);
        g.create_nodes(
            &node_ids.iter().copied().collect(),
            &mut IndexDocs::default(),
        )
        .unwrap();

        g.get_type_id_mut("ROAD");
        g.get_type_id_mut("FERRY");

        let rel_ids = reserve_relationships(&g, 3);
        g.create_relationships_bulk(
            &s("ROAD"),
            &[node_ids[0]],
            &[node_ids[1]],
            &[rel_ids[0]],
            &mut IndexDocs::default(),
        )
        .unwrap();
        g.create_relationships_bulk(
            &s("FERRY"),
            &[node_ids[0]],
            &[node_ids[1]],
            &[rel_ids[1]],
            &mut IndexDocs::default(),
        )
        .unwrap();
        g.create_relationships_bulk(
            &s("ROAD"),
            &[node_ids[1]],
            &[node_ids[2]],
            &[rel_ids[2]],
            &mut IndexDocs::default(),
        )
        .unwrap();

        let w_attr = g.get_or_create_rel_attr_id(&s("w"));
        let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
        attrs.insert(rel_ids[0], vec![(w_attr, Value::Float(10.0))]);
        attrs.insert(rel_ids[1], vec![(w_attr, Value::Float(2.0))]);
        // rel_ids[2] gets no `w` attribute at all -> defaults to 1.0.
        let mut index_add_edge_docs = IndexDocs::default();
        g.set_relationships_attributes(&attrs, &mut index_add_edge_docs)
            .unwrap();

        let node_ids = node_ids.into_iter().map(NodeId::from).collect();
        let rel_ids = rel_ids.into_iter().map(RelationshipId::from).collect();
        (g, node_ids, w_attr, rel_ids)
    }

    #[test]
    fn errors_when_no_realizing_edge_exists() {
        let (g, node_ids, w_attr, _rel_ids) = path_fixture();
        // no edge at all from n2 back to n0
        let rel_types = vec![s("ROAD"), s("FERRY")];
        let err = cch_path_from_node_ids(&g, &[node_ids[2], node_ids[0]], &rel_types, w_attr)
            .unwrap_err();
        assert!(err.contains("hierarchy inconsistent"), "{err}");
    }

    // ---- the real index: create, list, query -----------------------------------

    fn indexed_fixture() -> (Graph, Vec<NodeId>, Vec<RelationshipId>) {
        let (mut g, nodes, w_attr, rels) = path_fixture();
        let types = vec![
            g.get_type_id("ROAD").unwrap(),
            g.get_type_id("FERRY").unwrap(),
        ];
        g.create_cch_index(CchIndexDef::new(types, w_attr))
            .expect("build");
        (g, nodes, rels)
    }

    fn config(
        g: &Graph,
        src: NodeId,
        dst: NodeId,
        types: &[&str],
        w: &str,
    ) -> Result<CchQueryConfig, String> {
        let args = map_arg(vec![
            ("sourceNode", Value::Node(src)),
            ("targetNode", Value::Node(dst)),
            (
                "relTypes",
                Value::List(Arc::new(
                    types.iter().map(|t| Value::String(s(t))).collect(),
                )),
            ),
            ("weightProp", Value::String(s(w))),
        ]);
        parse_cch_query_config(g, &args)
    }

    #[test]
    fn query_returns_the_cheapest_path_through_the_real_index() {
        let (g, n, rels) = indexed_fixture();
        let cfg = config(&g, n[0], n[2], &["ROAD", "FERRY"], "w").unwrap();
        let rows = cch_query_rows(&g, &cfg).unwrap();
        assert_eq!(rows.len(), 1);
        // FERRY (2) beats the parallel ROAD (10); the second hop defaults to 1
        assert_eq!(rows[0].0, Value::Float(3.0));
        let Value::Path(p) = &rows[0].1 else {
            panic!("path")
        };
        assert_eq!(p[1], Value::Relationship(rels[1]));
        assert_eq!(p[3], Value::Relationship(rels[2]));
    }

    #[test]
    fn query_same_node_is_weight_zero_and_unreachable_is_no_row() {
        let (g, n, _) = indexed_fixture();
        let same = cch_query_rows(
            &g,
            &config(&g, n[1], n[1], &["ROAD", "FERRY"], "w").unwrap(),
        )
        .unwrap();
        assert_eq!(same[0].0, Value::Float(0.0));
        let Value::Path(p) = &same[0].1 else {
            panic!("path")
        };
        assert_eq!(&p[..], &[Value::Node(n[1])], "the path is just that node");
        let back = cch_query_rows(
            &g,
            &config(&g, n[2], n[0], &["ROAD", "FERRY"], "w").unwrap(),
        )
        .unwrap();
        assert!(back.is_empty(), "no path n2 -> n0");
    }

    #[test]
    fn query_without_a_matching_index_says_so() {
        let (g, n, _) = indexed_fixture();
        // a different relationship set is a different index
        let cfg = config(&g, n[0], n[1], &["ROAD"], "w").unwrap();
        let err = cch_query_rows(&g, &cfg).unwrap_err();
        assert!(err.contains("no CCH index"), "{err}");
    }

    #[test]
    fn db_indexes_lists_the_cch_index_with_its_types_and_weight() {
        let (g, _, _) = indexed_fixture();
        let rows = cch_index_rows(&g);
        assert_eq!(rows.len(), 1);
        // rel types sorted by id (ROAD was registered first), joined by ','
        assert_eq!(rows[0].label, Value::String(s("ROAD,FERRY")));
        assert_eq!(
            rows[0].properties,
            Value::List(Arc::new(thin_vec![Value::String(s("w"))]))
        );
        let Value::Map(m) = &rows[0].types else {
            panic!("types")
        };
        assert_eq!(
            m.get_str("w"),
            Some(&Value::List(Arc::new(thin_vec![Value::String(s("CCH"))])))
        );
    }
}
