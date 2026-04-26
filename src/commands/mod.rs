//! Redis command handler module index.
//!
//! Re-exports all command entrypoints registered by `redis_module!` and
//! provides shared command-level constants.
//!
//! ## Why this module exists
//! It keeps command registration in `lib.rs` clean while allowing each command
//! to evolve independently in its own file.
//!
//! ```text
//! redis_module! command table
//!          |
//!          v
//!      commands::graph_* re-exports
//!          |
//!          v
//!   commands/<command>.rs implementation
//! ```

use redis_module::{RedisError, RedisResult};

pub mod bulk_insert;
pub mod config_cmd;
pub mod debug;
pub mod delete;
pub mod effect;
pub mod explain;
pub mod list;
pub mod memory;
pub mod profile;
pub mod query;
pub mod record;
pub mod ro_query;
pub mod udf;

pub use bulk_insert::graph_bulk_insert;
pub use config_cmd::graph_config;
pub use debug::graph_debug;
pub use delete::graph_delete;
pub use effect::graph_effect;
pub use explain::graph_explain;
pub use list::graph_list;
pub use memory::graph_memory;
pub use profile::graph_profile;
pub use query::graph_query;
pub use record::graph_record;
pub use ro_query::graph_ro_query;
pub use udf::graph_udf;

pub const EMPTY_KEY_ERR: RedisResult =
    Err(RedisError::Str("ERR Invalid graph operation on empty key"));
