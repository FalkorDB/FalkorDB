//! Turning a finished query into the payload it replicates.
//!
//! The buffer itself is built during execution — `CommitOp` for data records,
//! [`crate::runtime::index_ddl`] for index DDL — and this is where it is handed
//! over, or found to be empty.
//!
//! It lives in this crate rather than beside the command handlers because the
//! buffer and the counters belong to `Runtime`. The host owns transport alone:
//! when a payload is sent, under which key, over which context. What the bytes
//! are is [`crate::effects::EffectsFormat`]'s.

use crate::effects::{Current, EffectsFormat};
use crate::runtime::runtime::Runtime;

/// Decide whether to use effects replication and get the pre-built buffer.
/// The buffer was built in `CommitOp` before pending was cleared.
/// Returns Some(buffer) if effects should be sent, None for verbatim replication.
/// The payload a finished write should replicate, or `None` when it produced
/// nothing.
///
/// There is no query-replay alternative any more: a write either ships as
/// effects or is not replicated at all. That removes the size heuristic this
/// used to run — `EFFECTS_THRESHOLD` only ever chose between effects and
/// replaying the query — and with it the possibility of the two engines
/// disagreeing about which mechanism carried a given write.
pub fn take_effects_buffer(runtime: &Runtime) -> Option<Vec<u8>> {
    let buf = runtime.effects_buffer.borrow_mut().take()?;
    // A payload holding nothing but its header carries no records. How long a
    // header is belongs to the format, not to this function.
    (!Current::is_empty(&buf)).then_some(buf)
}
