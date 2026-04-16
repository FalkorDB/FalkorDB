//! Redis native type declaration for graph storage and UDF persistence.
//!
//! Registers `GRAPH_TYPE` -- a Redis module type named `"graphdata"` --
//! along with RDB and lifecycle callbacks that Redis invokes automatically.
//!
//! ## Callbacks
//!
//! ```text
//! Redis event               Callback             Purpose
//! -------------------------+--------------------+------------------------------
//! Key deleted/expired      | graph_free()       | Drop Arc<RwLock<ThreadedGraph>>
//! RDB save (before RDB)    | graph_aux_save()   | Serialize UDF libraries
//! RDB load (aux payload)   | graph_aux_load()   | Deserialize + register UDFs
//! RDB save (per-key)       | graph_rdb_save()   | Encode graph to RDB stream
//! RDB load (per-key)       | graph_rdb_load()   | Decode graph from RDB stream
//! ```
//!
//! ## UDF persistence
//!
//! User-defined function (UDF) libraries are persisted through the auxiliary
//! RDB callbacks (`graph_aux_save` / `graph_aux_load`), which run once per
//! RDB cycle rather than per key. On load, existing UDFs are flushed and
//! replaced with the snapshot's contents, then each function is re-registered
//! with the runtime function table.
//!
//! ## Value lifecycle
//! ```text
//! set_value(GRAPH_TYPE, Arc<RwLock<ThreadedGraph>>)
//!              |
//!              +--> key survives Redis operations
//!              |
//!              +--> on key delete/overwrite/expire:
//!                        Redis invokes `free` callback -> graph_free()
//! ```

use crate::graph_core::{ThreadedGraph, graph_free};
use crate::serializers::decoder::rdb_load_graph;
use crate::serializers::encoder::rdb_save_graph;
use graph::graph::mvcc_graph::MvccGraph;
use graph::runtime::functions::{GraphFn, register_udf};
use graph::udf::get_udf_repo;
use parking_lot::RwLock;
use redis_module::raw::{load_string_buffer, load_unsigned, save_string, save_unsigned};
use redis_module::{
    REDISMODULE_TYPE_METHOD_VERSION, RedisModuleIO, RedisModuleTypeMethods, native_types::RedisType,
};
use std::sync::Arc;
use std::{os::raw::c_void, ptr::null_mut};

/// Decode a graph from the RDB stream.
///
/// Called by Redis for each key of type `GRAPH_TYPE` during RDB load.
/// Returns a heap-allocated `Arc<RwLock<ThreadedGraph>>` that Redis
/// will associate with the key, or null on failure.
#[unsafe(no_mangle)]
unsafe extern "C" fn graph_rdb_load(
    rdb: *mut RedisModuleIO,
    _encver: i32,
) -> *mut c_void {
    // Default cache size for the query plan cache.
    // During RDB load we don't have a Context to read the module config,
    // so use the default value (25).
    let cache_size = 25;

    match rdb_load_graph(rdb, cache_size) {
        Ok(Some(graph)) => {
            let mvcc = MvccGraph::from_graph(graph);
            let tg = Arc::new(RwLock::new(ThreadedGraph::from_mvcc(mvcc)));
            Box::into_raw(Box::new(tg)).cast::<c_void>()
        }
        Ok(None) => {
            // Multi-key graph: data accumulated in DECODE_STATE.
            // Return null for now; the graph will be finalized later.
            null_mut()
        }
        Err(e) => {
            eprintln!("FalkorDB: RDB load error: {e}");
            null_mut()
        }
    }
}

/// Encode a graph into the RDB stream.
///
/// Called by Redis for each key of type `GRAPH_TYPE` during RDB save
/// (BGSAVE, SAVE, or replication). The `value` pointer is the
/// `Arc<RwLock<ThreadedGraph>>` that was stored via `set_value`.
#[unsafe(no_mangle)]
unsafe extern "C" fn graph_rdb_save(
    rdb: *mut RedisModuleIO,
    value: *mut c_void,
) {
    let tg = &*(value.cast::<Arc<RwLock<ThreadedGraph>>>());
    let guard = tg.read();
    let g_arc = guard.graph.read();
    let g = g_arc.borrow();
    rdb_save_graph(rdb, &g);
}

/// Save UDF libraries to RDB.
#[unsafe(no_mangle)]
unsafe extern "C" fn graph_aux_save(
    rdb: *mut RedisModuleIO,
    _when: i32,
) {
    let repo = get_udf_repo();
    let libs = repo.serialize();
    save_unsigned(rdb, libs.len() as u64);
    for (name, code) in &libs {
        save_string(rdb, name);
        save_string(rdb, code);
    }
}

/// Load UDF libraries from RDB.
#[unsafe(no_mangle)]
unsafe extern "C" fn graph_aux_load(
    rdb: *mut RedisModuleIO,
    _encver: i32,
    _when: i32,
) -> i32 {
    let Ok(count) = load_unsigned(rdb) else {
        return 1; // REDISMODULE_ERR
    };

    let repo = get_udf_repo();
    let mut libs = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let name = match load_string_buffer(rdb) {
            Ok(buf) => String::from_utf8_lossy(buf.as_ref()).to_string(),
            Err(_) => return 1,
        };
        let code = match load_string_buffer(rdb) {
            Ok(buf) => String::from_utf8_lossy(buf.as_ref()).to_string(),
            Err(_) => return 1,
        };
        libs.push((name, code));
    }

    // Validate all libraries, then atomically swap the repo contents.
    // On failure the live repo and function table remain unchanged.
    repo.deserialize(&libs).map_or(1, |loaded_libs| {
        // Re-register bridge functions for the new set of libraries.
        graph::runtime::functions::flush_udfs();
        for lib in &loaded_libs {
            for qname in &lib.function_names {
                let graph_fn = Arc::new(GraphFn::new_udf(qname));
                register_udf(qname, graph_fn);
            }
        }
        0 // REDISMODULE_OK
    })
}

pub static GRAPH_TYPE: RedisType = RedisType::new(
    "graphdata",
    19,
    RedisModuleTypeMethods {
        version: REDISMODULE_TYPE_METHOD_VERSION as u64,
        rdb_load: Some(graph_rdb_load),
        rdb_save: Some(graph_rdb_save),
        aof_rewrite: None,
        free: Some(graph_free),

        mem_usage: None,
        digest: None,

        aux_load: Some(graph_aux_load),
        aux_save: None,
        aux_save2: Some(graph_aux_save),
        aux_save_triggers: 1, // REDISMODULE_AUX_BEFORE_RDB

        free_effort: None,
        unlink: None,
        copy: None,
        defrag: None,

        copy2: None,
        free_effort2: None,
        mem_usage2: None,
        unlink2: None,
    },
);
