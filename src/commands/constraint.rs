use crate::dispatch::must_run_inline;
use crate::query_session::{QuerySession, WriteAbort, WriteFacts};
use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{
        BlockedClient, CtxSink, ThreadedGraph, admit, c_graph_key, c_graph_name, ffi,
        register_graph, registered_graph_name,
    },
    redis_type::GRAPH_TYPE,
};
use graph::effects::EffectsBuffer;
use graph::effects::announce::{AnnouncedConstraint, SchemaBaseline};
use graph::entity_type::EntityType;
use graph::graph::constraint::{ConstraintStatus, ConstraintType};
use graph::graph::graph::Graph;
use graph::identifier_limits::validate_identifier_len;
use parking_lot::RwLock;
use redis_module::{Context, ContextFlags, NextArg, RedisResult, RedisString, RedisValue, raw};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Validate that a name is a valid identifier: starts with a letter or underscore,
/// followed by letters, digits, or underscores.
fn is_valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The status of a constraint that is still there, or `None` if it is not.
///
/// `None` is a real answer, not a missing one, and both callers need it as such.
/// A create record carries the status this node reached — `create_constraint`
/// has already decided Operational or Failed for a small label, and leaves a
/// large one under construction for the settling thread — while a drop record
/// carries no status field at all. So a `None` on the announce path means the
/// constraint is gone and there is nothing to announce, rather than a value to
/// substitute a default for.
/// Intern the constraint's label and property names.
///
/// Deliberately here and not in `Graph::create_constraint`, which is a state
/// mutation and stores names rather than ids. Three consumers need the ids to
/// exist, and none of them can create one: the RDB encoder takes `&self`,
/// `build_constraint_buffer` runs after the commit, and C's `Constraint_New`
/// takes `AttributeID *fields` outright. So it has to happen on the write path,
/// and this is the only caller of `create_constraint`.
///
/// **After** the create succeeds, not before: `create_constraint` refuses a
/// duplicate and a UNIQUE without its supporting index, and neither should leave
/// a name interned. Dropping the fork would discard it either way — everything
/// here runs on the private version `MvccGraph::fork` handed out, and a refused
/// create never commits it — but not interning on a refused create keeps that
/// independent of the MVCC model rather than reliant on it.
///
/// **Before** the announcement, and after the baseline: the effect diffs against
/// the baseline to decide what to announce.
///
/// Without this, a constraint on a not-yet-seen property is persisted with
/// attribute id 0 — `encode_constraint_block` resolves each property with
/// `position(..).unwrap_or(0)` — and reads back as a different property across
/// any RDB save or replica sync. Measured on `main`: `MANDATORY Q [z]` becomes
/// `MANDATORY Q [a]` after `DEBUG RELOAD`. That is #2749.
fn register_constraint_schema(
    g: &mut graph::graph::graph::Graph,
    entity_type: EntityType,
    label: &Arc<String>,
    properties: &[Arc<String>],
) {
    match entity_type {
        EntityType::Node => {
            g.get_label_id_mut(label);
        }
        EntityType::Relationship => {
            g.get_type_id_mut(label);
        }
    }
    for property in properties {
        g.add_node_attribute_name(property);
    }
}

fn find_status(
    g: &graph::graph::graph::Graph,
    ct: ConstraintType,
    entity_type: EntityType,
    label: &Arc<String>,
    properties: &[Arc<String>],
) -> Option<ConstraintStatus> {
    g.constraints()
        .iter()
        .find(|c| c.matches(&ct, &entity_type, label, properties))
        .map(|c| c.status)
}

/// Drive a constraint left UNDER CONSTRUCTION to the status validation settles
/// on, then announce it.
///
/// Runs on a dedicated OS thread rather than the query threadpool: validation can
/// take seconds on a large label, and a query worker blocked here would also
/// block unrelated reads queued behind it on the shared MPMC dispatch.
///
/// Two phases under different locks. The scan runs as a *reader*, so concurrent
/// `db.constraints()` keeps seeing the constraint UNDER CONSTRUCTION; the write
/// lock is taken only to publish the outcome.
///
/// Retried while the refusal is transient, because giving up leaves the
/// constraint UNDER CONSTRUCTION: the master never enforces it and
/// `GRAPH.CONSTRAINT CREATE` answers `Constraint already exists` from then on,
/// since `create_constraint` only clears a FAILED one out of the way. A
/// `CLIENT PAUSE ... WRITE` or an in-flight `FAILOVER` refuses the escalation,
/// and both end on their own.
///
/// **Bounded by [`RETRY_BUDGET`], not unbounded.** Nothing bounds a
/// `CLIENT PAUSE` — its timeout is whatever the caller passed — so retrying
/// until it lifts means an OS thread that may never exit, one per pending
/// constraint. On exhaustion the thread logs what happened and stops.
///
/// Stranded is recoverable, which is what makes stopping safe:
/// [`Graph::drop_constraint`] has no status guard, so `GRAPH.CONSTRAINT DROP`
/// removes a constraint stuck UNDER CONSTRUCTION and it can then be created
/// again. The give-up log says exactly that, because nothing else will.
///
/// Publishing FAILED instead is not available here: that is a write, and a
/// write being refused is the reason we are giving up.
///
/// **C has no equivalent, because C has nothing here that can fail.** Its
/// enforcement thread sets the status in place under a *read* lock
/// (`Constraint_SetStatus`, "assuming under lock") and then calls
/// `RedisModule_Replicate` straight from that thread (`Constraint_Replicate`),
/// never asking for a write lock and never checking whether propagating is
/// currently allowed. Two things make that possible for it and not for us: it
/// has no MVCC, so publishing a status is a field assignment rather than a
/// version commit; and it has no pause check.
///
/// That second one is not a feature we are missing. Replicating from a worker
/// thread inside a `CLIENT PAUSE` window is what trips Redis's `propagateNow`
/// assertion — reproduced against the `edge-c` image, exit 133, twice out of
/// two. So C's answer to the situation this loop handles is to abort the
/// process. The pause check exists so we do not, and once a write is refused
/// rather than crashed through, something has to be done with the refusal:
/// abandoning it strands the constraint permanently. Hence retrying.
pub struct Settling {
    pub ct: ConstraintType,
    pub entity_type: EntityType,
    pub label: Arc<String>,
    pub properties: Vec<Arc<String>>,
}

/// How long a settling thread will keep trying before it gives up.
///
/// Long enough that no ordinary `CLIENT PAUSE` or failover outlasts it — both
/// are measured in seconds — and short enough that a thread cannot outlive the
/// reason it exists. There is no principled value, because there is no bound on
/// a pause; what matters is that one exists.
const RETRY_BUDGET: Duration = Duration::from_secs(300);

/// Between attempts. Flat rather than exponential: the wait is for an external
/// window to close, so backing off further only delays noticing that it has.
const BACKOFF: Duration = Duration::from_millis(100);

/// Take a constraint from UNDER CONSTRUCTION to its final status.
///
/// Blocking, and deliberately so: this is the body a spawned OS thread runs,
/// not an `async fn` and not something a runtime polls. It validates as a
/// reader, then publishes the status as a writer, retrying while the write is
/// refused because replication traffic is paused.
///
/// The caller runs it on a thread of its own because validation can take
/// seconds and must not occupy a query worker — reads queued behind it on the
/// shared dispatch would stall for reasons they have nothing to do with.
pub fn settle_constraint(
    graph: &Arc<RwLock<ThreadedGraph>>,
    announce: &[Settling],
    key: &[u8],
    was_replicated: bool,
) {
    // Bytes, because this is what the announcement is addressed to. A Redis key
    // is binary and need not be UTF-8 — `test_binary_key_name_scan_skip` keeps a
    // graph under `\xc3\x28` — so rendering it through a `String` first and
    // replicating `as_bytes()` of that would announce the settled constraint
    // under a key holding U+FFFD instead, which is a key nothing else addresses.
    // Rendered lossily below for the log lines, where it is only being read.
    let key_display = String::from_utf8_lossy(key);
    // A detached thread-safe context, made once for this thread and freed at the
    // end — the same shape as `telemetry`'s flusher, which is the other
    // background thread in the process that has to reach the dataset with no
    // client behind it. A worker serving a query builds its context from the
    // blocked client instead (`ffi::get_thread_safe_context(bc.inner)`); this
    // thread has no client, which is the only reason it differs.
    //
    // It is a handle, not a lock: the GIL is taken by `upgrade_to_write` inside
    // each attempt and released with the session, and this context is never
    // locked or unlocked. That is what keeps #2371 shut — the crash there came
    // from a *second* GIL acquire whose `RM_ThreadSafeContextUnlock` ran
    // `postExecutionUnitOperations` outside the pause check, not from the
    // existence of a second context.
    //
    // SAFETY: a null blocked-client yields a detached context, which is what is
    // wanted here — the GIL and the dataset, not a client to reply to.
    let ctx = unsafe { ffi::get_thread_safe_context(std::ptr::null_mut()) };
    let ctx = Context::new(ctx);

    let deadline = Instant::now() + RETRY_BUDGET;
    let mut attempts = 0_u32;

    loop {
        attempts += 1;
        match attempt_settle(graph, &ctx, announce, key, was_replicated) {
            Ok(()) => break,
            // Transient: the window closes on its own. Nothing is held here —
            // `attempt_settle` has dropped its session, and with it the GIL and
            // the write lock — so sleeping is safe. A pause or failover ends, and
            // a stale version is rebuilt on top of the commit that overtook it.
            Err(reason @ (WriteAbort::ReplicaTrafficPaused | WriteAbort::Stale)) => {
                if Instant::now() >= deadline {
                    for c in announce {
                        redis_module::logging::log_warning(format!(
                            "constraint on {:?} {} ({}) is stuck UNDER CONSTRUCTION: writes were \
                             refused for {}s across {attempts} attempts, so validation could not \
                             publish its result. It is not being enforced. Recover with \
                             GRAPH.CONSTRAINT DROP on graph '{key_display}' followed by CREATE.",
                            c.entity_type,
                            c.label,
                            c.properties
                                .iter()
                                .map(|p| p.as_str())
                                .collect::<Vec<_>>()
                                .join(", "),
                            RETRY_BUDGET.as_secs(),
                        ));
                    }
                    break;
                }
                // Once, on the first deferral. Not per iteration — the budget
                // allows three thousand of them — and not silent either: a
                // constraint that validated but cannot yet say so is worth a
                // line, and it is the only signal that this path ran at all,
                // which is what lets a test assert it was exercised rather
                // than pass because the window was never hit.
                if attempts == 1 {
                    redis_module::logging::log_notice(format!(
                        "constraint announcement on graph '{key_display}' is deferred ({reason}); \
                         retrying for up to {}s",
                        RETRY_BUDGET.as_secs(),
                    ));
                }
                std::thread::sleep(BACKOFF);
            }
            // The graph is gone, or this node is no longer a master. Neither
            // resolves by waiting: on a demoted node the constraint is the new
            // master's to finish, and `enforce_pending_constraints_after_promotion`
            // is what picks it up.
            Err(WriteAbort::GraphUnregistered | WriteAbort::NotAMaster) => break,
            // The version failed its own validation, so it was never published.
            // Permanent: retrying re-runs the same arithmetic on the same graph.
            // Logged loudly because the constraint is left unenforced and the id
            // space it was refused over is an engine fault, not a user error.
            Err(WriteAbort::Invalid(e)) => {
                redis_module::logging::log_warning(format!(
                    "constraint announcement on graph '{key_display}' was refused: {e}. \
                     The constraint is not being enforced."
                ));
                break;
            }
        }
    }

    // Every exit above `break`s rather than returns, so this runs on all of
    // them. SAFETY: the context was made by this thread at the top and has not
    // been freed; nothing borrows it past here.
    unsafe { ffi::free_thread_safe_context(ctx.ctx) };
}

/// One attempt at the two-phase settle. Takes a fresh session each time.
///
/// A fresh one is required, not merely tidy: `QuerySession::escalate` records
/// writer mode *before* `reauthorize_write` runs, so calling `upgrade_to_write`
/// again on a session that already failed takes the already-a-writer early
/// return and reports success — skipping the pause check that refused it. It also
/// means a failed session is still holding the GIL and the write lock, so it has
/// to be dropped before anything sleeps.
fn attempt_settle(
    graph: &Arc<RwLock<ThreadedGraph>>,
    ctx: &Context,
    announce: &[Settling],
    key: &[u8],
    was_replicated: bool,
) -> Result<(), WriteAbort> {
    let key_display = String::from_utf8_lossy(key);
    // `replicates: !was_replicated` — this thread re-announces the settled
    // constraint, so escalation has to run the pause check that makes
    // propagating safe. `originated_here: false` — it also runs on a replica
    // applying the master's `GRAPH.CONSTRAINT`, where a READONLY rejection would
    // strand the constraint and diverge.
    let session = QuerySession::begin_with(
        graph,
        WriteFacts {
            replicates: !was_replicated,
            originated_here: false,
        },
    );
    let results = session.with_graph(|tg| tg.graph.read().compute_pending_constraint_results());
    if results.is_empty() {
        return Ok(());
    }

    // Escalation takes the global lock before the write lock (#726) and keeps the
    // commit Arc-swap under it, so it cannot race a BGSAVE fork (#452) — the same
    // shape as bulk_insert's Phase 2.
    session.upgrade_to_write(None)?;
    // A refusal from `commit` comes back as the `WriteAbort` it maps to, not
    // flattened: the retry loop treats a stale version (retry) and an invalid
    // one (permanent) as opposites. Answering one bool for refusals once made
    // a permanent engine fault retry for the full five-minute budget, and an
    // early return that answered `Ok(())` once reported a constraint settled
    // that was still UNDER CONSTRUCTION, with nothing to re-drive it.
    session
        .with_graph_mut(|tg| -> Result<(), WriteAbort> {
            let mut fork = tg.graph.fork();
            fork.apply_constraint_validation_results(results);
            let g = tg.graph.commit(fork)?;

            if was_replicated {
                return Ok(());
            }
            for c in announce {
                // The constraint may have been dropped while validation ran:
                // `drop_constraint` has no UNDER CONSTRUCTION guard and
                // `apply_constraint_validation_results` matches on the constraint
                // id, so it silently did nothing. Announcing anyway would send a
                // CREATE — this path always builds one — and the replica would
                // install and enforce a constraint the master no longer has.
                let Some(status) = find_status(&g, c.ct, c.entity_type, &c.label, &c.properties)
                else {
                    continue;
                };
                let mut buf = EffectsBuffer::new();
                // Re-announce with the status validation settled on. The GIL is
                // still held here, from `upgrade_to_write`'s pause check
                // through this replicate, which is what keeps the check sound:
                // the propagation is flushed when the session's guard releases
                // it, inside the window the check validated.
                match buf.build_constraint(
                    &g,
                    true,
                    &AnnouncedConstraint {
                        ct: c.ct,
                        entity_type: c.entity_type,
                        status: Some(status),
                        label: &c.label,
                        properties: &c.properties,
                    },
                    &SchemaBaseline::of(&g),
                ) {
                    Ok(()) => {
                        let lossy_key = String::from_utf8_lossy(key);
                        let current_key_bytes = match registered_graph_name(graph) {
                            Some(name) if name != lossy_key => name.into_bytes(),
                            _ => key.to_vec(),
                        };
                        buf.replicate(&CtxSink(ctx), &current_key_bytes);
                    }
                    // Swallowing this left the replica's constraint UNDER
                    // CONSTRUCTION permanently: the master has settled it, the
                    // announcement that would carry that never went out, and
                    // nothing can re-drive it because `GRAPH.CONSTRAINT CREATE`
                    // answers "already exists". Silent, and only visible as a
                    // replica that enforces a constraint the master does not.
                    Err(e) => redis_module::logging::log_warning(format!(
                        "constraint on {:?} {} ({}) settled locally but could not be \
                         announced ({e}); replicas will keep reporting it UNDER \
                         CONSTRUCTION. Recover with GRAPH.CONSTRAINT DROP on graph \
                         '{key_display}' followed by CREATE.",
                        c.entity_type,
                        c.label,
                        c.properties
                            .iter()
                            .map(|p| p.as_str())
                            .collect::<Vec<_>>()
                            .join(", "),
                    )),
                }
            }
            Ok(())
        })
        .expect("writer mode after upgrade_to_write")
}

pub fn graph_constraint(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    // GRAPH.CONSTRAINT CREATE|DROP <key> UNIQUE|MANDATORY NODE|RELATIONSHIP <label> PROPERTIES <count> <prop1>...
    if args.len() < 8 {
        return Err(redis_module::RedisError::WrongArity);
    }
    let mut args = args.into_iter().skip(1);

    // Operation: CREATE or DROP
    let op_str = args.next_str()?;
    let is_create = match op_str.to_ascii_uppercase().as_str() {
        "CREATE" => true,
        "DROP" => false,
        _ => {
            return Err(redis_module::RedisError::String(
                "Invalid constraint operation".into(),
            ));
        }
    };

    // Graph key
    let key_str = args.next_arg()?;

    // Constraint type
    let ct_str = args.next_str()?;
    let ct = match ct_str.to_ascii_uppercase().as_str() {
        "UNIQUE" => ConstraintType::Unique,
        "MANDATORY" => ConstraintType::Mandatory,
        _ => {
            return Err(redis_module::RedisError::String(
                "Invalid constraint type".into(),
            ));
        }
    };

    // Entity type
    let et_str = args.next_str()?;
    let entity_type = match et_str.to_ascii_uppercase().as_str() {
        "NODE" => EntityType::Node,
        "RELATIONSHIP" => EntityType::Relationship,
        _ => {
            return Err(redis_module::RedisError::String(
                "Invalid constraint entity type".into(),
            ));
        }
    };

    // Label name
    let label_str = args.next_str()?;
    validate_identifier_len(label_str, "Label name").map_err(redis_module::RedisError::String)?;
    if !is_valid_identifier(label_str) {
        return Err(redis_module::RedisError::String(format!(
            "Label name {label_str} is invalid"
        )));
    }
    let label = Arc::new(label_str.to_string());

    // PROPERTIES keyword
    let props_kw = args.next_str()?;
    if !props_kw.eq_ignore_ascii_case("PROPERTIES") {
        return Err(redis_module::RedisError::String(
            "Expected PROPERTIES keyword".into(),
        ));
    }

    // Property count: Redis `string2ll` as in C (canonical decimal, no `+1` / `01`)
    let prop_count: i64 = args.next_arg()?.parse_integer().map_err(|_| {
        redis_module::RedisError::String(
            "Number of properties must be an integer between 1 and 255".into(),
        )
    })?;
    if !(1..=255).contains(&prop_count) {
        return Err(redis_module::RedisError::String(
            "Number of properties must be an integer between 1 and 255".into(),
        ));
    }

    // Property names
    let mut properties = Vec::with_capacity(prop_count as usize);
    for _ in 0..prop_count {
        let prop_str = args.next_str()?;
        validate_identifier_len(prop_str, "Property name")
            .map_err(redis_module::RedisError::String)?;
        if !is_valid_identifier(prop_str) {
            return Err(redis_module::RedisError::String(format!(
                "Property name {prop_str} is invalid"
            )));
        }
        let prop = Arc::new(prop_str.to_string());
        if properties.contains(&prop) {
            return Err(redis_module::RedisError::String(
                "Properties cannot contain duplicates".into(),
            ));
        }
        properties.push(prop);
    }

    // Reject trailing tokens
    if args.next().is_some() {
        return Err(redis_module::RedisError::String(
            "Unexpected extra arguments".into(),
        ));
    }

    let slot = if !must_run_inline(ctx) {
        let Some(slot) = admit(ctx) else {
            return Ok(RedisValue::NoReply);
        };
        Some(slot)
    } else {
        None
    };

    // Open or create graph
    let key = ctx.open_key_writable(&key_str);
    let graph = if let Some(g) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        g.clone()
    } else {
        if !is_create {
            return Err(redis_module::RedisError::String(
                "Unable to drop constraint, no such constraint.".into(),
            ));
        }
        let name = c_graph_name(&key_str);
        let g = Arc::new(RwLock::new(ThreadedGraph::new(
            *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize,
            &name,
        )));
        // Created under the name C derives from the key, and stored at the key
        // rebuilt from that name (`GraphContext_SetKey`) — not at the key the
        // command addressed, which differs once it holds a NUL.
        let create_key = ctx.open_key_writable(&c_graph_key(ctx, &key_str));
        create_key.set_value(&GRAPH_TYPE, g.clone())?;
        register_graph(name, g.clone());
        g
    };

    let is_replayed = ctx.get_flags().contains(ContextFlags::REPLICATED);
    let key_bytes: Vec<u8> = key_str.as_slice().to_vec();

    if let Some(slot) = slot {
        let bc = unsafe { BlockedClient::new(ctx.ctx) };
        graph::threadpool::spawn(
            move || {
                drop(slot);
                execute_constraint_async(
                    bc,
                    graph,
                    is_create,
                    ct,
                    entity_type,
                    label,
                    properties,
                    key_bytes,
                );
            },
            None,
        );
        return Ok(RedisValue::NoReply);
    }

    let tg = graph.write();
    let mut fork = tg.graph.fork();

    // Captured before the registration below, because the announcement is a
    // diff against it: an entry interned after this point is new, and travels
    // as `ADD_SCHEMA`/`ADD_ATTRIBUTE` ahead of the record whose ids depend on
    // it. Taking the baseline *after* registering would hide exactly the
    // entries that need announcing.
    let baseline = SchemaBaseline::of(&fork);

    let result =
        apply_constraint_to_fork(&mut fork, is_create, ct, entity_type, &label, &properties);

    match result {
        Ok(needs_background_validation) => {
            let mutated = tg.graph.commit(fork).map_err(WriteAbort::from)?;

            if needs_background_validation {
                spawn_constraint_validation(
                    graph.clone(),
                    key_str.as_slice().to_vec(),
                    ct,
                    entity_type,
                    Arc::clone(&label),
                    properties.clone(),
                    is_replayed,
                );
            }

            if !is_replayed {
                announce_constraint(
                    ctx,
                    &mutated,
                    &baseline,
                    is_create,
                    ct,
                    entity_type,
                    &label,
                    &properties,
                    key_str.as_slice(),
                    needs_background_validation,
                );
            }
            if is_create {
                Ok(RedisValue::SimpleStringStatic("PENDING"))
            } else {
                Ok(RedisValue::SimpleStringStatic("OK"))
            }
        }
        Err(e) if is_replayed && e == "Constraint already exists" => {
            // The second half of a v2-era AOF's verbatim pair: the constraint
            // was created by the first copy, so this one has nothing to do.
            // Erroring here would fail the load of an otherwise valid file.
            Ok(RedisValue::SimpleStringStatic("OK"))
        }
        Err(e) => Err(redis_module::RedisError::String(e)),
    }
}

fn apply_constraint_to_fork(
    fork: &mut Graph,
    is_create: bool,
    ct: ConstraintType,
    entity_type: EntityType,
    label: &Arc<String>,
    properties: &[Arc<String>],
) -> Result<bool, String> {
    if is_create {
        let needs_background_validation =
            fork.create_constraint(ct, entity_type, label, properties)?;
        register_constraint_schema(fork, entity_type, label, properties);
        Ok(needs_background_validation)
    } else {
        fork.drop_constraint(&ct, &entity_type, label, properties)
            .map(|()| false)
    }
}

fn spawn_constraint_validation(
    graph: Arc<RwLock<ThreadedGraph>>,
    key: Vec<u8>,
    ct: ConstraintType,
    entity_type: EntityType,
    label: Arc<String>,
    properties: Vec<Arc<String>>,
    was_replicated: bool,
) {
    std::thread::spawn(move || {
        settle_constraint(
            &graph,
            &[Settling {
                ct,
                entity_type,
                label,
                properties,
            }],
            &key,
            was_replicated,
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn announce_constraint(
    ctx: &Context,
    mutated: &Arc<Graph>,
    baseline: &SchemaBaseline,
    is_create: bool,
    ct: ConstraintType,
    entity_type: EntityType,
    label: &Arc<String>,
    properties: &[Arc<String>],
    key: &[u8],
    needs_background_validation: bool,
) {
    let status = if !is_create {
        Some(None)
    } else if needs_background_validation {
        Some(Some(ConstraintStatus::UnderConstruction))
    } else {
        find_status(mutated, ct, entity_type, label, properties).map(Some)
    };
    let Some(status) = status else {
        redis_module::logging::log_warning(format!(
            "constraint CREATE on {entity_type:?} '{label}' was applied but its \
             status could not be read back, so it was not announced; it will not \
             reach replicas or the AOF. Recover with GRAPH.CONSTRAINT DROP on \
             graph '{}' followed by CREATE.",
            String::from_utf8_lossy(key),
        ));
        return;
    };
    let mut buf = EffectsBuffer::new();
    match buf.build_constraint(
        mutated,
        is_create,
        &AnnouncedConstraint {
            ct,
            entity_type,
            status,
            label,
            properties,
        },
        baseline,
    ) {
        Ok(()) => {
            buf.replicate(&CtxSink(ctx), key);
        }
        Err(e) => redis_module::logging::log_warning(format!(
            "constraint {} on {entity_type:?} '{label}' was applied but could \
             not be announced ({e}); it will not reach replicas or the AOF",
            if is_create { "CREATE" } else { "DROP" },
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_constraint_async(
    bc: BlockedClient,
    graph: Arc<RwLock<ThreadedGraph>>,
    is_create: bool,
    ct: ConstraintType,
    entity_type: EntityType,
    label: Arc<String>,
    properties: Vec<Arc<String>>,
    key_bytes: Vec<u8>,
) {
    let ts_ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
    let ctx = Context::new(ts_ctx);

    let facts = WriteFacts {
        replicates: true,
        originated_here: true,
    };

    'attempt: loop {
        let session = QuerySession::begin_with(&graph, facts);
        let mut fork = session.write_version();
        let baseline = SchemaBaseline::of(&fork);

        let precheck_result =
            apply_constraint_to_fork(&mut fork, is_create, ct, entity_type, &label, &properties);

        let needs_background_validation = match precheck_result {
            Ok(nbv) => nbv,
            Err(e) => {
                let cerr = ffi::sanitise_error(e);
                unsafe { ffi::reply_error(ctx.ctx, cerr.as_ptr()) };
                drop(bc);
                unsafe { ffi::free_thread_safe_context(ts_ctx) };
                return;
            }
        };

        match session.upgrade_to_write(Some(&mut fork)) {
            Ok(()) => {}
            Err(WriteAbort::Stale) => {
                continue 'attempt;
            }
            Err(e) => {
                let cerr = ffi::sanitise_error(e.to_string());
                unsafe { ffi::reply_error(ctx.ctx, cerr.as_ptr()) };
                drop(bc);
                unsafe { ffi::free_thread_safe_context(ts_ctx) };
                return;
            }
        }

        let mutated =
            match session.with_graph_mut(|tg| tg.graph.commit(fork).map_err(WriteAbort::from)) {
                Some(Ok(m)) => m,
                Some(Err(WriteAbort::Stale)) => {
                    continue 'attempt;
                }
                Some(Err(e)) => {
                    let cerr = ffi::sanitise_error(e.to_string());
                    unsafe { ffi::reply_error(ctx.ctx, cerr.as_ptr()) };
                    drop(bc);
                    unsafe { ffi::free_thread_safe_context(ts_ctx) };
                    return;
                }
                None => {
                    let cerr = ffi::sanitise_error("Failed to commit constraint".to_string());
                    unsafe { ffi::reply_error(ctx.ctx, cerr.as_ptr()) };
                    drop(bc);
                    unsafe { ffi::free_thread_safe_context(ts_ctx) };
                    return;
                }
            };

        // A Redis RENAME may have moved the graph while we were waiting for the write
        // lock. Resolve the current key name from the registry so announcements and
        // background validation replicate under the graph's live key. Keep the addressed
        // binary key bytes when the graph was not renamed to avoid lossy UTF-8 conversion.
        let lossy_key = String::from_utf8_lossy(&key_bytes);
        let current_key_bytes = match registered_graph_name(&graph) {
            Some(name) if name != lossy_key => name.into_bytes(),
            Some(_) => key_bytes,
            None => {
                let cerr = ffi::sanitise_error(WriteAbort::GraphUnregistered.to_string());
                unsafe { ffi::reply_error(ctx.ctx, cerr.as_ptr()) };
                drop(bc);
                unsafe { ffi::free_thread_safe_context(ts_ctx) };
                return;
            }
        };

        if needs_background_validation {
            spawn_constraint_validation(
                graph.clone(),
                current_key_bytes.clone(),
                ct,
                entity_type,
                Arc::clone(&label),
                properties.clone(),
                false,
            );
        }

        announce_constraint(
            &ctx,
            &mutated,
            &baseline,
            is_create,
            ct,
            entity_type,
            &label,
            &properties,
            &current_key_bytes,
            needs_background_validation,
        );

        session.release_locks();

        if is_create {
            raw::reply_with_simple_string(ctx.ctx, c"PENDING".as_ptr());
        } else {
            raw::reply_with_simple_string(ctx.ctx, c"OK".as_ptr());
        }
        drop(bc);
        unsafe { ffi::free_thread_safe_context(ts_ctx) };
        return;
    }
}
