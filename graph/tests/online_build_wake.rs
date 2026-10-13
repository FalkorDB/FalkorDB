//! The online index builder, woken by a commit, in a process of its own.
//!
//! The builder is one thread per process, and every commit that brings in a `Building` column wakes
//! it. Among the graph crate's unit tests, other tests' commits wake it too, so a test there cannot
//! tell whether *its* commit did. This file is a test binary of its own: the only commits are its
//! own, so the builder wakes only if they wake it.
#![cfg(feature = "index-falkordb")]

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use graph::entity_type::EntityType;
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

/// A host for one graph: a plain lock around its coordinator.
struct Host {
    mvcc: RwLock<MvccGraph>,
}

impl BuildHost for Host {
    fn id(&self) -> usize {
        std::ptr::from_ref(self) as usize
    }

    fn read(
        &self,
        f: &mut dyn FnMut(&MvccGraph),
    ) -> bool {
        f(&self.mvcc.read());
        true
    }

    fn publish(
        &self,
        f: &mut dyn FnMut(&mut MvccGraph),
    ) -> bool {
        f(&mut self.mvcc.write());
        true
    }
}

/// More nodes than one build step reads (10,000), so the build takes several turns and commits
/// twice while a column is still `Building`.
const NODES: usize = 25_000;

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

/// With the builder running and done with the scan it makes at start, the commit of a `CREATE
/// INDEX` wakes it, and it builds the column to `Ready` with nobody running a turn. It wakes
/// exactly once: its own commits, two of which still carry the `Building` column, do not wake it
/// again, so it asks the host for its graphs twice in all — at start and on the wake.
#[test]
fn a_commit_that_brings_in_a_building_column_wakes_the_builder() {
    init_graphblas();
    let label = Arc::new("Person".to_string());
    let attr = Arc::new("v".to_string());
    let host = Arc::new(Host {
        mvcc: RwLock::new(MvccGraph::new(64, 64, 0, "wake")),
    });

    // `:Person` nodes with `v = 1..=NODES`, committed before any index exists.
    {
        let mut mvcc = host.mvcc.write();
        let version = mvcc.write().expect("the slot starts free");
        {
            let mut g = version.borrow_mut();
            let label_id = g.get_label_id_mut("Person");
            let ids = g
                .node_id_space()
                .reserve(NODES, &RoaringTreemap::new())
                .expect("reserved");
            g.create_nodes(&ids.iter().copied().collect())
                .expect("freshly reserved ids are creatable");
            let mut docs = IndexDocs::default();
            g.set_nodes_labels_bulk(&ids, &vec![label_id.0 as u64; NODES], &mut docs, false);
            let attr_id = g.get_or_create_node_attr_id(&attr);
            let values: FxHashMap<u64, Vec<(u16, Value)>> = ids
                .iter()
                .zip(1..)
                .map(|(&id, v)| (id, vec![(attr_id, Value::Int(v))]))
                .collect();
            g.set_nodes_attributes(&values, &mut docs)
                .expect("attributes set");
            g.apply_index_writes(&mut docs);
        }
        mvcc.commit(version).expect("commits");
    }

    // Start the builder, and let its scan at start find nothing to build.
    let scans = Arc::new(AtomicUsize::new(0));
    {
        let scans = Arc::clone(&scans);
        let host: Arc<dyn BuildHost> = host.clone();
        online_build::start(move || {
            scans.fetch_add(1, Ordering::SeqCst);
            vec![Arc::clone(&host)]
        });
    }
    wait_until("the builder's scan at start", || {
        scans.load(Ordering::SeqCst) >= 1
    });

    // CREATE INDEX, committed: the only thing that can wake the builder now.
    {
        let mut mvcc = host.mvcc.write();
        let version = mvcc.write().expect("free");
        graph_writes::create_index(
            &mut version.borrow_mut(),
            &IndexType::Range,
            EntityType::Node,
            &label,
            std::slice::from_ref(&attr),
        );
        mvcc.commit(version).expect("commits");
    }

    // A `Building` column answers no index query; a `Ready` one answers every one.
    let all = IndexQuery::Range {
        key: attr.clone(),
        min: None,
        max: None,
        include_min: true,
        include_max: true,
    };
    let indexed = || {
        let published = host.mvcc.read().read();
        let g = published.borrow();
        g.query_index_numeric_nodes(&label, &all)
            .map(Iterator::count)
    };
    wait_until("the builder to finish the column", || indexed().is_some());
    assert_eq!(indexed(), Some(NODES));
    assert_eq!(
        scans.load(Ordering::SeqCst),
        2,
        "woken once, by the CREATE INDEX commit"
    );

    online_build::stop();
}
