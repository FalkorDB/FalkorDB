//! Redis native type declaration for graph storage and UDF persistence.
//!
//! Registers `GRAPH_TYPE` -- a Redis module type named `"graphdata"` --
//! and `GRAPHMETA_TYPE` -- a Redis module type named `"graphmeta"` --
//! along with RDB and lifecycle callbacks that Redis invokes automatically.
//!
//! `GRAPHMETA_TYPE` is needed to load C FalkorDB RDB files, which use
//! `"graphmeta"` for virtual keys and AUX data. Rust's own virtual keys
//! use `"graphdata"` so that C FalkorDB can also load them.
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

use crate::config::CONFIGURATION_VKEY_MAX_ENTITY_COUNT;
use crate::graph_core::{ThreadedGraph, graph_free};
use crate::serializers;
use crate::serializers::encoder::build_multi_key_payloads;
use crate::serializers::{DECODE_STATE, VKEY_STATE};
use graph::graph::mvcc_graph::MvccGraph;
use graph::runtime::functions::{GraphFn, register_udf};
use graph::udf::get_udf_repo;
use parking_lot::RwLock;
use redis_module::logging::{log_notice, log_warning};
use redis_module::raw::{
    self, RedisModuleCtx, RedisModuleIO, load_string_buffer, load_unsigned, save_string,
    save_unsigned,
};
use redis_module::{
    REDISMODULE_TYPE_METHOD_VERSION, RedisModuleTypeMethods, native_types::RedisType,
};
use std::ffi::CString;
use std::sync::Arc;
use std::{os::raw::c_void, ptr::null_mut};

/// Default cache size used when loading from RDB (no Redis context available).
const DEFAULT_CACHE_SIZE: usize = 25;

// ---------------------------------------------------------------------------
// graphdata rdb_load / rdb_save
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
unsafe extern "C" fn graph_rdb_load(
    rdb: *mut RedisModuleIO,
    _encver: i32,
) -> *mut c_void {
    // Get the key name for looking up finalized graphs.
    let key_name = unsafe {
        let rm_key_name = raw::RedisModule_GetKeyNameFromIO.unwrap()(rdb);
        if rm_key_name.is_null() {
            "<unknown>".to_string()
        } else {
            let mut len: usize = 0;
            let ptr = raw::RedisModule_StringPtrLen.unwrap()(rm_key_name, &raw mut len);
            String::from_utf8_lossy(std::slice::from_raw_parts(ptr.cast(), len)).to_string()
        }
    };

    match serializers::decoder::rdb_load_graph(rdb, DEFAULT_CACHE_SIZE) {
        Ok(Some(graph)) => {
            // Single-key load (key_count == 1) -- graph is fully loaded.
            let mvcc = MvccGraph::from_graph(graph);
            let graph_arc = mvcc.read();
            graph_arc.borrow_mut().set_indexer_graph(graph_arc.clone());
            let tg = ThreadedGraph::from_mvcc(mvcc);
            let arc = Arc::new(RwLock::new(tg));
            crate::graph_core::register_graph(key_name, arc.clone());
            let boxed: Box<Arc<RwLock<ThreadedGraph>>> = Box::new(arc);
            Box::into_raw(boxed).cast()
        }
        Ok(None) => {
            // Multi-key load (key_count > 1) -- data stored in DECODE_STATE.
            // Check if all keys have already been loaded (inline finalization),
            // in which case we can return the real graph directly.
            {
                let mut decode_state = DECODE_STATE.lock();
                if let Some(graph) = decode_state.finalized.remove(&key_name) {
                    let mvcc = MvccGraph::from_graph(graph);
                    let graph_arc = mvcc.read();
                    graph_arc.borrow_mut().set_indexer_graph(graph_arc.clone());
                    drop(graph_arc);
                    // If a placeholder Arc was already registered for this
                    // key (main key loaded in middle of stream), mutate it
                    // in place rather than re-registering a fresh Arc --
                    // that would displace the placeholder and leak any
                    // WriteMessages already routed through it.
                    let arc = if let Some(ph) = decode_state.placeholders.remove(&key_name) {
                        ph.write().graph = mvcc;
                        ph
                    } else {
                        let tg = ThreadedGraph::from_mvcc(mvcc);
                        let arc = Arc::new(RwLock::new(tg));
                        crate::graph_core::register_graph(key_name.clone(), arc.clone());
                        arc
                    };
                    let boxed: Box<Arc<RwLock<ThreadedGraph>>> = Box::new(arc);
                    return Box::into_raw(boxed).cast();
                }
            }

            // Graph not yet finalized - more keys still need to load.
            // Return a placeholder that will be replaced later.
            let tg = ThreadedGraph::new(DEFAULT_CACHE_SIZE, "__placeholder__");
            let arc = Arc::new(RwLock::new(tg));

            // Store an Arc clone keyed by graph name for later finalization.
            {
                let mut decode_state = DECODE_STATE.lock();
                decode_state
                    .placeholders
                    .insert(key_name.clone(), arc.clone());
            }

            crate::graph_core::register_graph(key_name, arc.clone());

            // Hand ownership of a Box<Arc<...>> to Redis.
            let boxed: Box<Arc<RwLock<ThreadedGraph>>> = Box::new(arc);
            Box::into_raw(boxed).cast()
        }
        Err(e) => {
            eprintln!("graph rdb_load error: {e}");
            null_mut()
        }
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn graph_rdb_save(
    rdb: *mut RedisModuleIO,
    value: *mut c_void,
) {
    unsafe {
        // Get the key name to determine if this is a main key or virtual key.
        let rm_key_name = raw::RedisModule_GetKeyNameFromIO.unwrap()(rdb);
        let key_name = if rm_key_name.is_null() {
            String::new()
        } else {
            let mut len: usize = 0;
            let ptr = raw::RedisModule_StringPtrLen.unwrap()(rm_key_name, &raw mut len);
            String::from_utf8_lossy(std::slice::from_raw_parts(ptr.cast(), len)).to_string()
        };

        let vkey_state = VKEY_STATE.lock();

        // Check if this is a virtual key with assigned payloads (SYNC SAVE).
        if let Some((graph_name, _key_idx, payloads)) = vkey_state.vkey_map.get(&key_name) {
            let key_count = 1 + vkey_state
                .graph_vkeys
                .get(graph_name)
                .map_or(0, std::vec::Vec::len) as u64;
            // Look up the real graph by name from GRAPH_REGISTRY.
            let registry = crate::graph_core::GRAPH_REGISTRY.lock();
            if let Some(real_graph_arc) = registry.get(graph_name) {
                let tg: &ThreadedGraph = &*real_graph_arc.data_ptr();
                let g = tg.graph.read();
                let graph = g.borrow();
                serializers::encoder::rdb_save_graph_key(rdb, &graph, payloads, key_count);
                return;
            }
        }
        drop(vkey_state);

        // Direct encoding: use data_ptr() to bypass parking_lot RwLock which
        // deadlocks in the BGSAVE fork child when the write loop holds the write lock.
        let graph_arc = &*(value.cast::<Arc<RwLock<ThreadedGraph>>>());
        let tg: &ThreadedGraph = &*graph_arc.data_ptr();
        let g = tg.graph.read();
        let graph = g.borrow();
        serializers::encoder::rdb_save_graph(rdb, &graph);
    }
}

// ---------------------------------------------------------------------------
// aux_save / aux_load
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
unsafe extern "C" fn graph_aux_save(
    rdb: *mut RedisModuleIO,
    when: i32,
) {
    if when == raw::Aux::Before as i32 {
        // BEFORE_RDB: Save UDF libraries.
        let repo = get_udf_repo();
        let libs = repo.serialize();
        save_unsigned(rdb, libs.len() as u64);
        for (name, code) in &libs {
            save_string(rdb, name);
            save_string(rdb, code);
        }
    } else {
        // AFTER_RDB: Write placeholder so aux_load(AFTER_RDB) has something to read.
        save_unsigned(rdb, 0);
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn graph_aux_load(
    rdb: *mut RedisModuleIO,
    _encver: i32,
    when: i32,
) -> i32 {
    if when == raw::Aux::Before as i32 {
        // BEFORE_RDB: Load UDFs.
        let Ok(count) = load_unsigned(rdb) else {
            return 1;
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

        repo.deserialize(&libs).map_or(1, |loaded_libs| {
            graph::runtime::functions::flush_udfs();
            for lib in &loaded_libs {
                for qname in &lib.function_names {
                    let graph_fn = Arc::new(GraphFn::new_udf(qname));
                    register_udf(qname, graph_fn);
                }
            }
            0
        })
    } else {
        // AFTER_RDB: Read placeholder, finalize pending multi-key graphs.
        let _ = load_unsigned(rdb);
        finalize_pending_graphs();
        0
    }
}

// ---------------------------------------------------------------------------
// Persistence event handler -- creates/deletes virtual keys
// ---------------------------------------------------------------------------

/// pthread_atfork prepare handler: materialize all pending GraphBLAS operations
/// before fork so the child process doesn't encounter held internal locks.
///
/// Only runs on the main thread (BGSAVE path). For non-main-thread forks
/// (RediSearch's ForkGC), we return immediately — mirroring the C port's
/// `_ForkPrepare`, which also opts out of graph-side synchronization for
/// ForkGC forks. That avoids a three-way deadlock between the writer's
/// RediSearch FFI calls (which take RediSearch's internal RWLock) and
/// ForkGC (which holds that RWLock across `fork()`).
///
/// On the main thread, BGSAVE is invoked from the Redis command loop which
/// already holds the GIL. Writers cannot be mid-mutation against the graph
/// at the moment BGSAVE forks because every writer must briefly take the
/// GIL during its commit phase, and the main thread holds it now.
///
/// # Safety
/// Called by libc before fork. Accesses graphs via data_ptr() (bypassing RwLock).
pub unsafe extern "C" fn pre_fork_prepare() {
    if !graph::thread_id::is_main_thread() {
        return;
    }
    let registry = crate::graph_core::GRAPH_REGISTRY.lock();
    for graph_arc in registry.values() {
        let tg: &ThreadedGraph = unsafe { &*graph_arc.data_ptr() };
        let g = tg.graph.read();
        let graph = g.borrow();
        graph.wait_all();
    }
}

/// Called by Redis persistence events. Creates virtual keys before RDB save,
/// deletes them after save completes or fails.
///
/// # Safety
/// Called by Redis internals with a valid module context.
pub unsafe extern "C" fn on_persistence(
    ctx: *mut RedisModuleCtx,
    _eid: redis_module::RedisModuleEvent,
    subevent: u64,
    _data: *mut c_void,
) {
    unsafe {
        match subevent {
            raw::REDISMODULE_SUBEVENT_PERSISTENCE_SYNC_RDB_START => {
                create_virtual_keys(ctx);
            }
            raw::REDISMODULE_SUBEVENT_PERSISTENCE_RDB_START => {
                // BGSAVE fork child: skip create_virtual_keys entirely.
                // The fork child must avoid Rust heap allocations because
                // glibc malloc arena locks may be held by parent threads
                // that no longer exist in the child, causing deadlock.
            }
            raw::REDISMODULE_SUBEVENT_PERSISTENCE_ENDED
            | raw::REDISMODULE_SUBEVENT_PERSISTENCE_FAILED => {
                delete_virtual_keys(ctx);
            }
            #[allow(clippy::match_same_arms)]
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Virtual key management helpers
// ---------------------------------------------------------------------------

pub unsafe fn create_virtual_keys(ctx: *mut RedisModuleCtx) {
    unsafe {
        // Delete stale graphmeta keys (from C FalkorDB RDB loads).
        delete_stale_graphmeta_keys(ctx);

        // Single graphdata scan: collect real graphs and delete stale virtual keys.
        let graphs = scan_and_clean_graphdata_keys(ctx);

        let mut vkey_state = VKEY_STATE.lock();
        vkey_state.clear();

        let context = redis_module::Context::new(ctx);
        let vkey_max = *CONFIGURATION_VKEY_MAX_ENTITY_COUNT.lock(&context);

        for (graph_name, graph_ref) in &graphs {
            // SAFETY: In the BGSAVE fork child, this process is single-threaded.
            // Threads that held the parking_lot RwLock at fork time are gone,
            // so lock acquisition would deadlock. We bypass the lock entirely
            // via data_ptr(). This is safe because no concurrent access exists
            // in the single-threaded fork child.
            // For synchronous SAVE (main thread), query threads only access
            // the graph through MvccGraph's committed version (Arc clone),
            // so reading ThreadedGraph fields here is also safe.
            let tg: &ThreadedGraph = &*graph_ref.data_ptr();
            let g = tg.graph.read();
            let graph = g.borrow();

            let multi_payloads = build_multi_key_payloads(&graph, vkey_max as u64);
            let key_count = multi_payloads.len();

            if key_count <= 1 {
                // Single-key graph: no virtual keys needed.
                // graph_rdb_save will encode directly.
                continue;
            }

            let virtual_key_count = key_count - 1;
            let mut vkey_names = Vec::with_capacity(virtual_key_count);

            // Store key 0's payloads under the graph name.
            vkey_state.vkey_map.insert(
                graph_name.clone(),
                (graph_name.clone(), 0, multi_payloads[0].clone()),
            );

            // Create virtual keys for keys 1..N.
            for (i, payloads) in multi_payloads.iter().enumerate().skip(1) {
                let uuid = uuid_v4();
                let vkey_name = if graph_name.contains('{') {
                    format!("{graph_name}_{uuid}")
                } else {
                    format!("{{{graph_name}}}{graph_name}_{uuid}")
                };

                vkey_state
                    .vkey_map
                    .insert(vkey_name.clone(), (graph_name.clone(), i, payloads.clone()));

                // Create the Redis key.
                let rm_str = raw::RedisModule_CreateString.unwrap()(
                    ctx,
                    vkey_name.as_ptr().cast(),
                    vkey_name.len(),
                );
                let key =
                    raw::RedisModule_OpenKey.unwrap()(ctx, rm_str, raw::KeyMode::WRITE.bits());
                // Must pass a non-null value; Redis skips keys with null values during RDB save.
                // Create a placeholder ThreadedGraph so graph_free can handle it.
                let tg_placeholder = ThreadedGraph::new(DEFAULT_CACHE_SIZE, "__vkey_placeholder__");
                let boxed: Box<Arc<RwLock<ThreadedGraph>>> =
                    Box::new(Arc::new(RwLock::new(tg_placeholder)));
                let value = Box::into_raw(boxed).cast();
                raw::RedisModule_ModuleTypeSetValue.unwrap()(
                    key,
                    *GRAPH_TYPE.raw_type.borrow(),
                    value,
                );
                raw::RedisModule_CloseKey.unwrap()(key);
                raw::RedisModule_FreeString.unwrap()(ctx, rm_str);

                vkey_names.push(vkey_name);
            }

            vkey_state
                .graph_vkeys
                .insert(graph_name.clone(), vkey_names);

            log_notice(format!(
                "Created {virtual_key_count} virtual keys for graph {graph_name}"
            ));
        }
    }
}

pub unsafe fn delete_virtual_keys(ctx: *mut RedisModuleCtx) {
    unsafe {
        let mut vkey_state = VKEY_STATE.lock();

        for (graph_name, vkey_names) in &vkey_state.graph_vkeys {
            let count = vkey_names.len();
            for vkey_name in vkey_names {
                let rm_str = raw::RedisModule_CreateString.unwrap()(
                    ctx,
                    vkey_name.as_ptr().cast(),
                    vkey_name.len(),
                );
                let key =
                    raw::RedisModule_OpenKey.unwrap()(ctx, rm_str, raw::KeyMode::WRITE.bits());
                raw::RedisModule_DeleteKey.unwrap()(key);
                raw::RedisModule_CloseKey.unwrap()(key);
                raw::RedisModule_FreeString.unwrap()(ctx, rm_str);
            }
            log_notice(format!(
                "Deleted {count} virtual keys for graph {graph_name}"
            ));
        }

        vkey_state.clear();
    }
}

/// Single-pass scan of graphdata keys: collects real graphs and deletes stale
/// virtual/placeholder keys in one traversal (instead of scanning twice).
unsafe fn scan_and_clean_graphdata_keys(
    ctx: *mut RedisModuleCtx
) -> Vec<(String, Arc<RwLock<ThreadedGraph>>)> {
    unsafe {
        let mut result = Vec::new();
        let mut stale_keys = Vec::new();

        let scan_cmd = CString::new("SCAN").unwrap();
        let type_arg = CString::new("TYPE").unwrap();
        let graphdata_arg = CString::new("graphdata").unwrap();
        let fmt = CString::new("ccc").unwrap();

        let mut cursor_val = CString::new("0").unwrap();

        loop {
            let reply = raw::RedisModule_Call.unwrap()(
                ctx,
                scan_cmd.as_ptr(),
                fmt.as_ptr(),
                cursor_val.as_ptr(),
                type_arg.as_ptr(),
                graphdata_arg.as_ptr(),
            );
            if reply.is_null() {
                break;
            }

            let reply_type = raw::call_reply_type(reply);
            if reply_type != raw::ReplyType::Array {
                raw::free_call_reply(reply);
                break;
            }

            let len = raw::call_reply_length(reply);
            if len < 2 {
                raw::free_call_reply(reply);
                break;
            }

            // Get new cursor.
            let cursor_reply = raw::call_reply_array_element(reply, 0);
            let mut cursor_len: usize = 0;
            let cursor_ptr =
                raw::RedisModule_CallReplyStringPtr.unwrap()(cursor_reply, &raw mut cursor_len);
            let new_cursor = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                cursor_ptr.cast(),
                cursor_len,
            ));
            let done = new_cursor == "0";

            // Get keys array.
            let arr_reply = raw::call_reply_array_element(reply, 1);
            let arr_len = raw::call_reply_length(arr_reply);

            for i in 0..arr_len {
                let elem = raw::call_reply_array_element(arr_reply, i);
                let mut key_len: usize = 0;
                let kptr = raw::RedisModule_CallReplyStringPtr.unwrap()(elem, &raw mut key_len);
                // Redis key names are binary-safe, so a graphdata-typed key may
                // have a non-UTF-8 name (e.g. `GRAPH.QUERY "\xff" ...`). Building a
                // `str` over non-UTF-8 bytes via `from_utf8_unchecked` is UB. Skip
                // any non-UTF-8 key rather than decoding it lossily: this name is
                // fed back into `RedisModule_CreateString` for the open/delete
                // below, and a lossy name no longer round-trips to the same key
                // (and could even alias a different one). A non-UTF-8 graph name
                // is not round-trippable here and is already handled only lossily
                // elsewhere (`graph_rdb_save`/`graph_rdb_load`), so skipping it
                // does not regress any normally (UTF-8) named graph.
                let key_bytes = std::slice::from_raw_parts(kptr.cast::<u8>(), key_len);
                let Ok(key_name) = std::str::from_utf8(key_bytes) else {
                    log_warning(format!(
                        "Skipping graphdata key with non-UTF-8 name during scan (lossy display: {})",
                        String::from_utf8_lossy(key_bytes)
                    ));
                    continue;
                };
                let key_name = key_name.to_string();

                let rm_str = raw::RedisModule_CreateString.unwrap()(
                    ctx,
                    key_name.as_ptr().cast(),
                    key_name.len(),
                );
                let key = raw::RedisModule_OpenKey.unwrap()(ctx, rm_str, raw::KeyMode::READ.bits());
                let value = raw::RedisModule_ModuleTypeGetValue.unwrap()(key);

                if !value.is_null() {
                    let graph_arc_ref = &*(value.cast::<Arc<RwLock<ThreadedGraph>>>());
                    // SAFETY: In the BGSAVE fork child, threads that held the
                    // parking_lot RwLock at fork time are gone. Lock acquisition
                    // would deadlock. We bypass the lock via data_ptr() since the
                    // fork child is single-threaded. The graph name is immutable
                    // so reading it without locking is safe even on the main thread.
                    let tg: &ThreadedGraph = &*graph_arc_ref.data_ptr();
                    let g = tg.graph.read();
                    let name = g.borrow().name().to_string();
                    if name.starts_with("__placeholder") || name.starts_with("__vkey_placeholder") {
                        // Stale virtual key — mark for deletion.
                        stale_keys.push(key_name);
                    } else {
                        // Real graph — collect it.
                        drop(g);
                        result.push((key_name, graph_arc_ref.clone()));
                    }
                }

                raw::RedisModule_CloseKey.unwrap()(key);
                raw::RedisModule_FreeString.unwrap()(ctx, rm_str);
            }

            cursor_val = CString::new(new_cursor).unwrap();
            raw::free_call_reply(reply);

            if done {
                break;
            }
        }

        // Delete stale virtual keys.
        for key_name in &stale_keys {
            let rm_str = raw::RedisModule_CreateString.unwrap()(
                ctx,
                key_name.as_ptr().cast(),
                key_name.len(),
            );
            let key = raw::RedisModule_OpenKey.unwrap()(ctx, rm_str, raw::KeyMode::WRITE.bits());
            raw::RedisModule_DeleteKey.unwrap()(key);
            raw::RedisModule_CloseKey.unwrap()(key);
            raw::RedisModule_FreeString.unwrap()(ctx, rm_str);
        }

        result
    }
}

/// Delete stale graphmeta keys (from C FalkorDB RDB loads).
unsafe fn delete_stale_graphmeta_keys(ctx: *mut RedisModuleCtx) {
    unsafe {
        let scan_cmd = CString::new("SCAN").unwrap();
        let type_arg = CString::new("TYPE").unwrap();
        let fmt = CString::new("ccc").unwrap();

        let mut keys_to_delete = Vec::new();
        let graphmeta_arg = CString::new("graphmeta").unwrap();
        scan_keys_by_type(
            ctx,
            &scan_cmd,
            &type_arg,
            &graphmeta_arg,
            &fmt,
            &mut keys_to_delete,
        );

        for key_name in &keys_to_delete {
            let rm_str = raw::RedisModule_CreateString.unwrap()(
                ctx,
                key_name.as_ptr().cast(),
                key_name.len(),
            );
            let key = raw::RedisModule_OpenKey.unwrap()(ctx, rm_str, raw::KeyMode::WRITE.bits());
            raw::RedisModule_DeleteKey.unwrap()(key);
            raw::RedisModule_CloseKey.unwrap()(key);
            raw::RedisModule_FreeString.unwrap()(ctx, rm_str);
        }

        if !keys_to_delete.is_empty() {}
    }
}

/// Delete any stale virtual keys left in the keyspace from a previous RDB load.
/// Public entry point used by the debug command.
pub unsafe fn delete_stale_virtual_keys(ctx: *mut RedisModuleCtx) {
    unsafe {
        delete_stale_graphmeta_keys(ctx);
        // scan_and_clean_graphdata_keys deletes stale graphdata keys as a side effect.
        let _ = scan_and_clean_graphdata_keys(ctx);
    }
}

unsafe fn scan_keys_by_type(
    ctx: *mut RedisModuleCtx,
    scan_cmd: &CString,
    type_arg: &CString,
    type_name: &CString,
    fmt: &CString,
    out: &mut Vec<String>,
) {
    unsafe {
        let mut cursor_val = CString::new("0").unwrap();

        loop {
            let reply = raw::RedisModule_Call.unwrap()(
                ctx,
                scan_cmd.as_ptr(),
                fmt.as_ptr(),
                cursor_val.as_ptr(),
                type_arg.as_ptr(),
                type_name.as_ptr(),
            );
            if reply.is_null() {
                break;
            }

            let reply_type = raw::call_reply_type(reply);
            if reply_type != raw::ReplyType::Array {
                raw::free_call_reply(reply);
                break;
            }

            let len = raw::call_reply_length(reply);
            if len < 2 {
                raw::free_call_reply(reply);
                break;
            }

            let cursor_reply = raw::call_reply_array_element(reply, 0);
            let mut cursor_len: usize = 0;
            let cursor_ptr =
                raw::RedisModule_CallReplyStringPtr.unwrap()(cursor_reply, &raw mut cursor_len);
            let new_cursor = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                cursor_ptr.cast(),
                cursor_len,
            ));
            let done = new_cursor == "0";

            let arr_reply = raw::call_reply_array_element(reply, 1);
            let arr_len = raw::call_reply_length(arr_reply);

            for i in 0..arr_len {
                let elem = raw::call_reply_array_element(arr_reply, i);
                let mut name_len: usize = 0;
                let kptr = raw::RedisModule_CallReplyStringPtr.unwrap()(elem, &raw mut name_len);
                // Key names are binary-safe; skip any non-UTF-8 name rather than
                // decoding it lossily. `out` names are round-tripped back through
                // `RedisModule_CreateString`/`OpenKey` for deletion, so a lossy name
                // would target the wrong key; and `from_utf8_unchecked` on non-UTF-8
                // bytes is UB (see scan_and_clean_graphdata_keys above).
                let key_bytes = std::slice::from_raw_parts(kptr.cast::<u8>(), name_len);
                let Ok(key_name) = std::str::from_utf8(key_bytes) else {
                    log_warning(format!(
                        "Skipping key with non-UTF-8 name during type scan (lossy display: {})",
                        String::from_utf8_lossy(key_bytes)
                    ));
                    continue;
                };
                out.push(key_name.to_string());
            }

            cursor_val = CString::new(new_cursor).unwrap();
            raw::free_call_reply(reply);

            if done {
                break;
            }
        }
    }
}

/// Finalize any pending multi-key graph loads from DECODE_STATE.
///
/// This handles two scenarios:
/// 1. Graphs already finalized inline (stored in decode_state.finalized)
/// 2. Graphs with keys_remaining == 0 that haven't been finalized yet
///
/// In both cases, the placeholder ThreadedGraph's inner MvccGraph is replaced
/// using the raw pointer stored during graph_rdb_load.
pub fn finalize_pending_graphs() {
    let mut decode_state = DECODE_STATE.lock();

    // First, handle graphs that were already finalized inline during rdb_load_graph.
    let finalized_names: Vec<String> = decode_state.finalized.keys().cloned().collect();
    for graph_name in &finalized_names {
        if let Some(graph) = decode_state.finalized.remove(graph_name) {
            let placeholder = decode_state.placeholders.remove(graph_name);
            install_graph(graph_name, graph, placeholder);
        }
    }

    // Then, handle graphs with keys_remaining == 0 (finalized via the old path).
    let pending_names: Vec<String> = decode_state
        .pending
        .iter()
        .filter(|(_, pg)| pg.keys_remaining == 0)
        .map(|(name, _)| name.clone())
        .collect();

    for graph_name in &pending_names {
        let pg = decode_state.pending.remove(graph_name).unwrap();
        let placeholder = decode_state.placeholders.remove(graph_name);

        match serializers::decoder::finalize_pending_graph(pg) {
            Ok(graph) => {
                install_graph(graph_name, graph, placeholder);
            }
            Err(e) => {
                eprintln!("FalkorDB: failed to finalize graph {graph_name}: {e}");
            }
        }
    }

    // Only clear if all pending graphs have been finalized.
    if decode_state.pending.is_empty() && decode_state.finalized.is_empty() {
        decode_state.placeholders.clear();
    }
}

/// Install a finalized Graph into the placeholder ThreadedGraph.
fn install_graph(
    graph_name: &str,
    graph: graph::graph::graph::Graph,
    placeholder: Option<Arc<RwLock<ThreadedGraph>>>,
) {
    let mvcc = MvccGraph::from_graph(graph);
    let graph_arc = mvcc.read();
    graph_arc.borrow_mut().set_indexer_graph(graph_arc.clone());
    drop(graph_arc);

    if let Some(ph) = placeholder {
        let mut placeholder_tg = ph.write();
        // Replace ONLY the inner MvccGraph. Preserving the existing sender,
        // receiver, write_loop, and slow_log keeps any WriteMessages that
        // were already enqueued against the placeholder reachable by the
        // write loop — otherwise blocked clients in `waiting` state would
        // never be replied to.
        placeholder_tg.graph = mvcc;
    } else {
        eprintln!(
            "FalkorDB: WARNING - no placeholder pointer for graph '{graph_name}', graph data will be lost"
        );
    }
}

/// Generate a simple UUID v4 string.
fn uuid_v4() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let a = (t as u64) ^ seq;
    let b = a
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (a >> 32) as u32,
        (a >> 16) as u16,
        (a & 0xFFF) as u16,
        (0x8000 | (b & 0x3FFF)) as u16,
        b & 0xFFFF_FFFF_FFFF
    )
}

// ---------------------------------------------------------------------------
// Type statics
// ---------------------------------------------------------------------------

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
        aux_save_triggers: 3, // REDISMODULE_AUX_BEFORE_RDB | REDISMODULE_AUX_AFTER_RDB

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

// ---------------------------------------------------------------------------
// graphmeta -- kept for loading C FalkorDB RDB files.
//
// C FalkorDB uses "graphmeta" for virtual keys and emits graphmeta AUX data.
// We register this type with rdb_load + aux_load so Rust can consume C's RDB
// stream. We intentionally omit aux_save so that Rust never emits graphmeta
// AUX data (which C can't load since it doesn't register "graphmeta" either).
// ---------------------------------------------------------------------------

/// Load a C FalkorDB graphmeta virtual key.
#[unsafe(no_mangle)]
unsafe extern "C" fn graphmeta_rdb_load(
    rdb: *mut RedisModuleIO,
    _encver: i32,
) -> *mut c_void {
    match serializers::decoder::rdb_load_graph(rdb, DEFAULT_CACHE_SIZE) {
        Ok(_) => {
            // Return a non-null dummy value. Redis needs non-null for successful load.
            Box::into_raw(Box::new(0u8)).cast()
        }
        Err(e) => {
            eprintln!("graphmeta rdb_load error: {e}");
            null_mut()
        }
    }
}

/// Save callback for graphmeta keys left over from a C RDB load.
/// These should be cleaned up before save by `delete_stale_virtual_keys`,
/// but this is kept as a safety net.
#[allow(clippy::missing_const_for_fn)]
#[unsafe(no_mangle)]
unsafe extern "C" fn graphmeta_rdb_save(
    _rdb: *mut RedisModuleIO,
    _value: *mut c_void,
) {
    // Stale graphmeta keys should have been deleted before save.
    // If we get here, write nothing — the key will be empty.
}

/// Free callback for graphmeta keys. These hold a dummy u8 value.
#[unsafe(no_mangle)]
unsafe extern "C" fn graphmeta_free(value: *mut c_void) {
    if !value.is_null() {
        unsafe {
            drop(Box::from_raw(value.cast::<u8>()));
        }
    }
}

/// Consume C FalkorDB's graphmeta AUX data during RDB load.
#[unsafe(no_mangle)]
unsafe extern "C" fn graphmeta_aux_load(
    rdb: *mut RedisModuleIO,
    _encver: i32,
    when: i32,
) -> i32 {
    let _ = load_unsigned(rdb);
    if when == raw::Aux::After as i32 {
        finalize_pending_graphs();
    }
    0
}

pub static GRAPHMETA_TYPE: RedisType = RedisType::new(
    "graphmeta",
    19,
    RedisModuleTypeMethods {
        version: REDISMODULE_TYPE_METHOD_VERSION as u64,
        rdb_load: Some(graphmeta_rdb_load),
        rdb_save: Some(graphmeta_rdb_save),
        aof_rewrite: None,
        free: Some(graphmeta_free),

        mem_usage: None,
        digest: None,

        // aux_load only — consume C's graphmeta AUX data but never emit it.
        aux_load: Some(graphmeta_aux_load),
        aux_save: None,
        aux_save2: None,
        aux_save_triggers: 3,

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
