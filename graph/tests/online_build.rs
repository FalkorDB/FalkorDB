//! The online index build pool, end to end, in a process of its own.
//!
//! The pool and its queue are one per process, and every commit that brings in a `Building` column
//! queues its graph. Among the graph crate's unit tests, other tests' commits would queue graphs of
//! their own and start or stop the pool under this test. This file is a test binary of its own, so
//! the only commits and the only pool are its own.
#![cfg(feature = "index-falkordb")]

use std::ffi::c_void;
use std::sync::{Arc, Weak};
use std::thread;
use std::time::{Duration, Instant};

use graph::entity_type::EntityType;
use graph::graph::graph::Graph;
use graph::graph::graphblas::{
    GrB_GLOBAL, GrB_Global_set_INT32, GrB_Info, GxB_JIT_Control, GxB_Option_Field, matrix,
};
use graph::graph::mvcc_graph::MvccGraph;
use graph::index::falkordb::graph_writes;
use graph::index::falkordb::online_build::{self, BuildHost};
use graph::index::index_docs::IndexDocs;
use graph::index::{IndexQuery, IndexType};
use graph::runtime::value::Value;
use parking_lot::RwLock;
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;

unsafe extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn calloc(
        n: usize,
        size: usize,
    ) -> *mut c_void;
    fn realloc(
        p: *mut c_void,
        size: usize,
    ) -> *mut c_void;
    fn free(p: *mut c_void);
}

/// GraphBLAS as the crate's unit tests set it up: the C allocator, and the JIT off.
fn init_graphblas() {
    matrix::init(Some(malloc), Some(calloc), Some(realloc), Some(free)).expect("GraphBLAS init");
    let info = unsafe {
        GrB_Global_set_INT32(
            GrB_GLOBAL,
            GxB_JIT_Control::GxB_JIT_OFF as i32,
            GxB_Option_Field::GxB_JIT_C_CONTROL as _,
        )
    };
    assert_eq!(info, GrB_Info::GrB_SUCCESS, "JIT_OFF failed");
}

/// One graph: a plain lock around its coordinator.
struct Host {
    mvcc: RwLock<MvccGraph>,
}

/// How the pool reaches a [`Host`]: weakly, as the Redis host does, so the graph's own
/// registration does not keep it alive.
struct HostRef(Weak<Host>);

impl BuildHost for HostRef {
    fn read(
        &self,
        f: &mut dyn FnMut(&MvccGraph),
    ) -> bool {
        let Some(host) = self.0.upgrade() else {
            return false;
        };
        f(&host.mvcc.read());
        true
    }

    fn publish(
        &self,
        f: &mut dyn FnMut(&mut MvccGraph),
    ) -> bool {
        let Some(host) = self.0.upgrade() else {
            return false;
        };
        f(&mut host.mvcc.write());
        true
    }
}

/// More nodes than one build step reads (10,000), so each build takes several turns.
const NODES: usize = 25_000;

impl Host {
    /// `NODES` `:Person` nodes with `v = 1..=NODES` and `w = v`, committed before any index exists,
    /// and registered with the build pool.
    fn people() -> Arc<Self> {
        let host = Arc::new(Self {
            mvcc: RwLock::new(MvccGraph::new(64, 64, 0, "people")),
        });
        host.mvcc
            .read()
            .set_build_host(Arc::new(HostRef(Arc::downgrade(&host))));
        host.write(|g| {
            let label_id = g.get_label_id_mut("Person");
            let ids = g
                .node_id_space()
                .reserve(NODES, &RoaringTreemap::new())
                .expect("reserved");
            g.create_nodes(&ids.iter().copied().collect())
                .expect("freshly reserved ids are creatable");
            let mut docs = IndexDocs::default();
            g.set_nodes_labels_bulk(&ids, &vec![label_id.0 as u64; NODES], &mut docs, false);
            let v = g.get_or_create_node_attr_id(&arc("v"));
            let w = g.get_or_create_node_attr_id(&arc("w"));
            let values: FxHashMap<u64, Vec<(u16, Value)>> = ids
                .iter()
                .zip(1..)
                .map(|(&id, x)| (id, vec![(v, Value::Int(x)), (w, Value::Int(x))]))
                .collect();
            g.set_nodes_attributes(&values, &mut docs)
                .expect("attributes set");
            g.apply_index_writes(&mut docs);
        });
        host
    }

    /// One committed write, the way a client's would be.
    fn write(
        &self,
        f: impl FnOnce(&mut Graph),
    ) {
        let mut mvcc = self.mvcc.write();
        let version = mvcc.write().expect("the slot is free");
        f(&mut version.borrow_mut());
        mvcc.commit(version).expect("commits");
    }

    fn create_index(
        &self,
        attr: &str,
    ) {
        self.write(|g| {
            graph_writes::create_index(
                g,
                &IndexType::Range,
                EntityType::Node,
                &arc("Person"),
                &[arc(attr)],
            );
        });
    }

    /// How many nodes the index answers for `attr >= min`, or `None` while the column is still
    /// `Building` — it answers no index query until it is `Ready`.
    fn indexed_from(
        &self,
        attr: &str,
        min: i64,
    ) -> Option<usize> {
        let published = self.mvcc.read().read();
        let g = published.borrow();
        g.query_index_numeric_nodes(
            &arc("Person"),
            &IndexQuery::Range {
                key: arc(attr),
                min: Some(Value::Int(min)),
                max: None,
                include_min: true,
                include_max: true,
            },
        )
        .map(Iterator::count)
    }
}

fn arc(s: &str) -> Arc<String> {
    Arc::new(s.to_string())
}

fn wait_until(
    what: &str,
    mut done: impl FnMut() -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(1));
    }
}

/// A `CREATE INDEX` commit queues its graph, once however many commits follow while it builds; the
/// pool's workers build every queued graph to `Ready`, including writes made while the graph was
/// queued; and a `CREATE INDEX` on an idle pool is queued and built in turn.
#[test]
fn commits_queue_graphs_and_the_pool_builds_them() {
    init_graphblas();
    let a = Host::people();
    let b = Host::people();
    assert_eq!(online_build::pending(), 0);

    // Before the pool starts, nothing runs, so the queue can be read exactly.
    a.create_index("v");
    assert_eq!(
        online_build::pending(),
        1,
        "the CREATE INDEX queued graph a"
    );
    // A write while a is queued and still `Building` does not queue it again. It moves the last
    // node's v past every other value, which the build must pick up.
    a.write(|g| {
        let v = g.get_or_create_node_attr_id(&arc("v"));
        let last = g
            .get_label_matrix(&arc("Person"))
            .expect("the label exists")
            .iter(0, u64::MAX)
            .map(|(n, _)| n)
            .last()
            .expect("there are nodes");
        let mut values: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
        values.insert(last, vec![(v, Value::Int(1_000_000))]);
        let mut docs = IndexDocs::default();
        g.set_nodes_attributes(&values, &mut docs).expect("set");
        g.apply_index_writes(&mut docs);
    });
    assert_eq!(online_build::pending(), 1, "graph a is queued once");
    b.create_index("v");
    assert_eq!(online_build::pending(), 2, "and graph b once");
    assert_eq!(a.indexed_from("v", 0), None, "Building until built");

    // Two workers build both graphs.
    online_build::start(2);
    wait_until("both graphs to be built", || online_build::pending() == 0);
    assert_eq!(a.indexed_from("v", 0), Some(NODES));
    assert_eq!(
        a.indexed_from("v", 1_000_000),
        Some(1),
        "the write made while a was queued is in its index"
    );
    assert_eq!(b.indexed_from("v", 0), Some(NODES));

    // The pool is idle: a CREATE INDEX now is queued and built in turn.
    a.create_index("w");
    wait_until("the second index to be built", || {
        a.indexed_from("w", 0) == Some(NODES)
    });
    wait_until("the queue to drain", || online_build::pending() == 0);

    online_build::stop();
}
