//! Turning a finished query into the payload it replicates.
//!
//! The buffer itself is built during execution — `CommitOp` for data records,
//! [`crate::runtime::index_ddl`] for index DDL — and this is where it is decided
//! what becomes of it: which wire version it is in, whether it is worth sending
//! at all, and, under v2 only, the index DDL that never went through `Pending`.
//!
//! It lives in this crate rather than beside the command handlers because every
//! input is here: the buffer and the counters belong to `Runtime`, the version
//! to this module's parent, and the v2 index records are written with this
//! crate's own writers. The one thing the host still owns is transport —
//! compressing the payload and handing it to `RM_Replicate`.
//!
//! A sibling of the wire-format modules rather than a second `effects` module
//! under `runtime`: everything named "effects" belongs in one tree, and the
//! difference from [`super`] is that this half needs a `Runtime` to ask, while
//! that half only needs bytes.

use orx_tree::Collection;
use std::sync::atomic::Ordering;

use crate::{
    effects::EFFECTS_THRESHOLD,
    effects::v2::{
        EFFECT_CREATE_INDEX, EFFECT_DROP_INDEX, EFFECTS_VERSION, write_string, write_u16,
    },
    planner::IR,
    runtime::runtime::Runtime,
};

/// Decide whether to use effects replication and get the pre-built buffer.
/// The buffer was built in `CommitOp` before pending was cleared.
/// Returns Some(buffer) if effects should be sent, None for verbatim replication.
/// Bytes of header before the first record, for whichever version wrote `buf`.
fn header_len(buf: &[u8]) -> usize {
    match buf.first() {
        Some(&v) if v >= crate::effects::v3::EFFECTS_VERSION => 2,
        _ => 1,
    }
}

/// The payload a finished write should replicate, or `None` to replicate the
/// query verbatim instead.
///
/// Every caller that finishes a write goes through here — `GRAPH.QUERY` and
/// `GRAPH.RECORD`, which does real writes and replicates exactly like it. The
/// two used to carry separate copies of this, differing only in which arguments
/// they had to hand, and the copies had to be kept in step by hand every time
/// the format decision changed.
pub fn take_effects_buffer(
    runtime: &Runtime,
    is_non_deterministic: bool,
    exec_time_ms: f64,
) -> Option<Vec<u8>> {
    // Which format this query's payload is in. Keyed off the buffer the runtime
    // actually built, not the config, which can change mid-query.
    let emit_v3 =
        crate::effects::emit_v3_for(runtime.effects_buffer.borrow().as_deref().unwrap_or(&[]));

    // v2 cannot round-trip `OPTIONS {...}`, so a CreateIndex carrying them falls
    // back to verbatim GRAPH.QUERY replication by skipping the effects buffer
    // entirely. v3 puts the evaluated map on the wire, so the fallback is lifted
    // there and the statement replicates as an effect like anything else.
    let has_unencodable_index = !emit_v3
        && runtime
            .plan
            .iter()
            .any(|node| matches!(node, IR::CreateIndex { options, .. } if options.is_some()));
    if has_unencodable_index {
        return None;
    }

    let buf = should_use_effects(is_non_deterministic, runtime, exec_time_ms);
    if emit_v3 {
        // The runtime appended index DDL itself, during execution and in plan
        // order — see `runtime::index_ddl`.
        buf
    } else {
        // v2 keeps index DDL out of `Pending`, so it is collected here by
        // scanning the plan.
        build_index_effects(runtime, buf)
    }
}

fn should_use_effects(
    is_non_deterministic: bool,
    runtime: &Runtime,
    exec_time_ms: f64,
) -> Option<Vec<u8>> {
    let threshold = EFFECTS_THRESHOLD.load(Ordering::Relaxed);

    let buf = runtime.effects_buffer.borrow_mut().take();
    let buf = match buf {
        // A payload holding nothing but its header carries no records. The
        // header is one byte in v2 and two in v3 — the flags byte — so this
        // cannot be a constant.
        Some(b) if b.len() > header_len(&b) => b,
        _ => return None,
    };

    let n_effects = runtime.effects_count.get();

    let use_effects = if is_non_deterministic || threshold == 0 || runtime.force_effects.get() {
        true
    } else if n_effects == 0 {
        false
    } else {
        let avg_mod_time_us = (exec_time_ms / n_effects as f64) * 1000.0;
        avg_mod_time_us > threshold as f64
    };

    if use_effects { Some(buf) } else { None }
}

/// Encode IndexType as u8 tag for effects buffer.
const fn index_type_tag(it: &crate::index::IndexType) -> u8 {
    use crate::index::IndexType;
    match it {
        IndexType::Range => 0,
        IndexType::Fulltext => 1,
        IndexType::Vector => 2,
    }
}

/// Encode EntityType as u8 tag for effects buffer.
const fn entity_type_tag(et: &crate::entity_type::EntityType) -> u8 {
    use crate::entity_type::EntityType;
    match et {
        EntityType::Node => 0,
        EntityType::Relationship => 1,
    }
}

/// Scan the plan for CreateIndex / DropIndex IR nodes and append their **v2**
/// effects to the buffer. Returns the (possibly new) effects buffer.
///
/// v2 only, and the caller must check: it writes v2 records and seeds a v2
/// header, so calling it while the emit version is 3 would put v2 records
/// behind a v3 header and the replica would refuse the whole payload. Under v3
/// the runtime emits index DDL itself, from the arm that has the evaluated
/// `OPTIONS` map — see `Runtime::emit_index_effect`.
///
/// Caller must also ensure no CreateIndex carries OPTIONS: v2 cannot round-trip
/// them, so those statements replicate verbatim.
fn build_index_effects(
    runtime: &Runtime,
    mut effects_buffer: Option<Vec<u8>>,
) -> Option<Vec<u8>> {
    for node in runtime.plan.iter() {
        match node {
            IR::CreateIndex {
                label,
                attrs,
                index_type,
                entity_type,
                options: _,
            } => {
                let buf = effects_buffer.get_or_insert_with(|| vec![EFFECTS_VERSION]);
                buf.push(EFFECT_CREATE_INDEX);
                buf.push(index_type_tag(index_type));
                buf.push(entity_type_tag(entity_type));
                write_string(buf, label);
                write_u16(buf, attrs.len() as u16);
                for attr in attrs {
                    write_string(buf, attr);
                }
            }
            IR::DropIndex {
                label,
                attrs,
                index_type,
                entity_type,
            } => {
                let buf = effects_buffer.get_or_insert_with(|| vec![EFFECTS_VERSION]);
                buf.push(EFFECT_DROP_INDEX);
                buf.push(index_type_tag(index_type));
                buf.push(entity_type_tag(entity_type));
                write_string(buf, label);
                write_u16(buf, attrs.len() as u16);
                for attr in attrs {
                    write_string(buf, attr);
                }
            }
            _ => {}
        }
    }
    effects_buffer
}
