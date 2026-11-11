//! `CREATE`/`DROP CCH INDEX` as effects, run as queries: what each statement puts
//! on the wire, and that a replica applying it ends up holding what the primary
//! holds.

use std::cell::RefCell;
use std::sync::Arc;

use crate::effects::ReplicationSink;
use crate::effects::payload::take_effects_buffer;
use crate::effects::v3::{AttrRef, Record, SchemaRef, open_payload};
use crate::graph::attribute_store::MAX_ATTRIBUTES;
use crate::graph::graph::{Graph, TypeId};
use crate::graph::graphblas::test_init::ensure_init;
use crate::index::falkordb::cch::CchIndexDef;
use crate::locks::WriteEscalation;
use crate::runtime::functions::init_functions;
use crate::runtime::runtime::Runtime;

/// The escalation a query run alone is given: it is the only writer already.
struct AlreadyWriter;

impl WriteEscalation for AlreadyWriter {
    fn upgrade_to_write(
        &self,
        _fork: &mut Graph,
    ) -> Result<(), String> {
        Ok(())
    }
}

/// Keeps the payload a query replicates.
#[derive(Default)]
struct Captured(RefCell<Vec<u8>>);

impl ReplicationSink for Captured {
    fn replicate(
        &self,
        _key: &[u8],
        payload: &[u8],
    ) {
        *self.0.borrow_mut() = payload.to_vec();
    }
}

/// A primary: a published version, and the write queries run on forks of it.
struct Primary(Arc<Graph>);

impl Primary {
    fn new() -> Self {
        ensure_init();
        let _ = init_functions();
        Self(Arc::new(Graph::new(64, 64, 0, 0, "cch_effects")))
    }

    /// Run the write query `query` and publish it, returning the payload it
    /// replicates (empty when it produced none), or its error.
    fn write(
        &mut self,
        query: &str,
    ) -> Result<Vec<u8>, String> {
        let plan = self.0.get_plan(query)?;
        let runtime = Runtime::new(
            self.0.new_version(),
            plan.parameters.clone(),
            Arc::clone(&plan.plan),
            false,
            String::new(),
            -1,
            false,
            None,
            0,
            None,
            &AlreadyWriter,
        );
        runtime.query()?;
        let captured = Captured::default();
        if let Some(buf) = take_effects_buffer(&runtime) {
            buf.replicate(&captured, b"cch_effects");
        }
        let mut published = None;
        runtime
            .publish(|fork| {
                let fork = Arc::new(fork);
                published = Some(Arc::clone(&fork));
                Ok::<_, ()>(fork)
            })
            .unwrap();
        self.0 = published.unwrap();
        Ok(captured.0.into_inner())
    }
}

fn records(payload: &[u8]) -> Vec<Record> {
    open_payload(payload)
        .unwrap()
        .records()
        .collect::<Result<_, _>>()
        .unwrap()
}

fn schema_ref(
    id: u32,
    name: &str,
) -> SchemaRef<String> {
    SchemaRef {
        id,
        name: name.to_owned(),
    }
}

fn weight(
    id: u16,
    name: &str,
) -> AttrRef<String> {
    AttrRef {
        id,
        name: name.to_owned(),
    }
}

/// `CREATE CCH INDEX` over nothing creates the types and the attribute, as C
/// does, and announces them ahead of the index — each new type, the attribute,
/// then one index record naming every type, sorted by id with the repeat kept.
#[test]
fn create_registers_missing_schema_and_announces_it_first() {
    let mut primary = Primary::new();
    let payload = primary
        .write("CREATE CCH INDEX FOR ()-[e:ROAD|FERRY|ROAD]->() ON (e.time)")
        .unwrap();
    assert_eq!(
        records(&payload),
        vec![
            Record::AddRelType {
                id: 0,
                name: "ROAD".to_owned(),
            },
            Record::AddRelType {
                id: 1,
                name: "FERRY".to_owned(),
            },
            Record::AddAttribute {
                id: 0,
                name: "time".to_owned(),
            },
            Record::cch_index(
                true,
                vec![
                    schema_ref(0, "ROAD"),
                    schema_ref(0, "ROAD"),
                    schema_ref(1, "FERRY"),
                ],
                weight(0, "time"),
            ),
        ]
    );
    let index = CchIndexDef::new(vec![TypeId(0), TypeId(1)], 0);
    assert!(primary.0.cch_index(&index).is_some());
}

/// Over existing schema only the index record goes out, and a drop names the
/// same index.
#[test]
fn create_and_drop_over_existing_schema_emit_only_the_index_record() {
    let mut primary = Primary::new();
    primary
        .write("CREATE (:N)-[:A {w: 1.0}]->(:N)-[:B {w: 2.0}]->(:N)")
        .unwrap();
    let types = || vec![schema_ref(0, "A"), schema_ref(1, "B")];

    let created = primary
        .write("CREATE CCH INDEX FOR ()-[e:B|A]->() ON (e.w)")
        .unwrap();
    assert_eq!(
        records(&created),
        vec![Record::cch_index(true, types(), weight(0, "w"))]
    );

    let dropped = primary
        .write("DROP CCH INDEX FOR ()-[e:B|A]->() ON (e.w)")
        .unwrap();
    assert_eq!(
        records(&dropped),
        vec![Record::cch_index(false, types(), weight(0, "w"))]
    );
    assert_eq!(primary.0.cch_indexes().count(), 0);
}

/// A create that registers a new type and then fails — the attribute
/// dictionary is full, so the weight cannot be registered — leaves the type
/// behind in a fork that is discarded: the published graph has no new type.
#[test]
fn a_create_failing_after_it_registered_a_type_leaves_no_type_behind() {
    let mut primary = Primary::new();
    {
        let graph = Arc::get_mut(&mut primary.0).unwrap();
        for i in 0..MAX_ATTRIBUTES {
            graph.add_rel_attribute_name(&format!("a{i}"));
        }
    }
    let err = primary
        .write("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        .unwrap_err();
    assert!(err.contains("attribute dictionary is full"), "{err}");
    assert_eq!(primary.0.get_types().len(), 0);
    assert_eq!(primary.0.cch_indexes().count(), 0);
}
