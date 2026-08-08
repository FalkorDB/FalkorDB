//! # FalkorDB Redis Module
//!
//! This crate is the Redis-facing integration layer for FalkorDB. It registers
//! Redis commands, initializes runtime dependencies, and delegates query work to
//! the `graph` crate through focused internal modules.
//!
//! ## High-level flow
//! ```text
//! Redis command
//!     |
//!     v
//! commands/* handler  --->  graph_core::query_mut (async dispatch)
//!     |                              |
//!     |                              +--> read path (concurrent MVCC snapshots)
//!     |                              +--> write path (serialized queue)
//!     v
//! reply::* (Redis protocol serialization)
//! ```
//!
//! ## Module responsibilities
//! - `commands/`: command entrypoints and argument parsing.
//! - `graph_core`: query execution/concurrency primitives.
//! - `reply`: compact + verbose output formatting.
//! - `redis_type`: native Redis value type (`graphdata`).
//! - `module_init`: startup wiring (RediSearch, GraphBLAS, functions).
//! - `config`: runtime configuration state.

#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::non_std_lazy_statics)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::significant_drop_tightening)]
// Dependency version duplicates are outside our control.
#![allow(clippy::multiple_crate_versions)]
// Pedantic style lints with low signal in this codebase.
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

mod allocator;
mod commands;
mod config;
mod graph_core;
mod module_init;
mod query_session;
mod redis_type;
mod reply;
mod serializers;
mod slow_log;
mod telemetry;

use allocator::ThreadCountingAllocator;
use commands::{
    graph_bulk_insert, graph_config, graph_constraint, graph_copy, graph_debug, graph_delete,
    graph_effect, graph_explain, graph_info, graph_list, graph_memory, graph_profile, graph_query,
    graph_record, graph_restore, graph_ro_query, graph_slowlog, graph_udf,
};
use config::{
    CONFIGURATION_CACHE_SIZE, CONFIGURATION_CMD_INFO, CONFIGURATION_DELAY_INDEXING,
    CONFIGURATION_IMPORT_FOLDER, CONFIGURATION_INDEX_WORKER_THREADS, CONFIGURATION_JS_HEAP_SIZE,
    CONFIGURATION_JS_STACK_SIZE, CONFIGURATION_NODE_CREATION_BUFFER, CONFIGURATION_TEMP_FOLDER,
    CONFIGURATION_THREAD_COUNT, CONFIGURATION_VKEY_MAX_ENTITY_COUNT,
};
use module_init::graph_init;
use redis_module::{configuration::ConfigurationFlags, redis_module};
use redis_type::{GRAPH_TYPE, GRAPHMETA_TYPE};

redis_module! {
    name: "graph",
    version: env!("FALKORDB_VERSION_INT").parse::<i32>().unwrap(),
    allocator: (ThreadCountingAllocator, ThreadCountingAllocator),
    data_types: [GRAPH_TYPE, GRAPHMETA_TYPE],
    init: graph_init,
    commands: [
        ["graph.DELETE", graph_delete, "write deny-script", 1, 1, 1, ""],
        ["graph.COPY", graph_copy, "write deny-oom deny-script", 1, 2, 1, ""],
        ["graph.RESTORE", graph_restore, "write deny-script", 1, 1, 1, ""],
        ["graph.QUERY", graph_query, "write deny-oom deny-script blocking", 1, 1, 1, ""],
        ["graph.RO_QUERY", graph_ro_query, "readonly deny-script blocking", 1, 1, 1, ""],
        ["graph.EXPLAIN", graph_explain, "write deny-oom deny-script blocking", 1, 1, 1, ""],
        ["graph.PROFILE", graph_profile, "write deny-oom deny-script blocking", 1, 1, 1, ""],
        ["graph.LIST", graph_list, "readonly deny-script allow-busy", 0, 0, 0, ""],
        ["graph.RECORD", graph_record, "write deny-oom deny-script blocking", 1, 1, 1, ""],
        ["graph.MEMORY", graph_memory, "readonly deny-script blocking", 2, 2, 1, ""],
        ["graph.CONFIG", graph_config, "readonly deny-script allow-busy", 0, 0, 0, ""],
        ["graph.UDF", graph_udf, "write deny-oom deny-script", 0, 0, 0, ""],
        ["graph.DEBUG", graph_debug, "write deny-script", 0, 0, 0, ""],
        ["graph.EFFECT", graph_effect, "write deny-script", 1, 1, 1, ""],
        ["graph.CONSTRAINT", graph_constraint, "write deny-oom deny-script", 2, 2, 1, ""],
        ["graph.BULK", graph_bulk_insert, "write deny-oom deny-script", 1, 1, 1, ""],
        ["graph.SLOWLOG", graph_slowlog, "readonly deny-script allow-busy", 1, 1, 1, ""],
        ["graph.INFO", graph_info, "readonly deny-script allow-busy", 1, 1, 1, ""],
    ],
    configurations: [
        i64: [
            ["CACHE_SIZE", &*CONFIGURATION_CACHE_SIZE, 25, 0, 1000, ConfigurationFlags::IMMUTABLE, None],
            ["THREAD_COUNT", &*CONFIGURATION_THREAD_COUNT, 0, 0, 1024, ConfigurationFlags::IMMUTABLE, None],
            ["INDEX_WORKER_THREADS", &*CONFIGURATION_INDEX_WORKER_THREADS, 0, 0, 1024, ConfigurationFlags::IMMUTABLE, None],
            ["NODE_CREATION_BUFFER", &*CONFIGURATION_NODE_CREATION_BUFFER, graph::graph::graph::DEFAULT_NODE_CREATION_BUFFER as i64, 0, 1_073_741_824, ConfigurationFlags::IMMUTABLE, None],
            ["VKEY_MAX_ENTITY_COUNT", &*CONFIGURATION_VKEY_MAX_ENTITY_COUNT, 100_000, 1, 1_073_741_824, ConfigurationFlags::DEFAULT, None],
            ["JS_HEAP_SIZE", &*CONFIGURATION_JS_HEAP_SIZE, 268_435_456, 0, 4_294_967_296, ConfigurationFlags::DEFAULT, None],
            ["JS_STACK_SIZE", &*CONFIGURATION_JS_STACK_SIZE, 1_048_576, 0, 4_294_967_296, ConfigurationFlags::DEFAULT, None],
        ],
        string: [
            ["IMPORT_FOLDER", &*CONFIGURATION_IMPORT_FOLDER, "/var/lib/FalkorDB/import/", ConfigurationFlags::IMMUTABLE, None],
            ["TEMP_FOLDER", &*CONFIGURATION_TEMP_FOLDER, "/tmp", ConfigurationFlags::IMMUTABLE, None],
        ],
        bool: [
            ["CMD_INFO", &*CONFIGURATION_CMD_INFO, true, ConfigurationFlags::IMMUTABLE, None],
            ["DELAY_INDEXING", &*CONFIGURATION_DELAY_INDEXING, false, ConfigurationFlags::IMMUTABLE, None],
        ],
        enum: [],
        module_args_as_configuration: true,
    ]
}
