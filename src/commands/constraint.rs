use crate::query_session::{QuerySession, WriteAbort, WriteFacts, gil_context};
use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{ThreadedGraph, c_graph_key, c_graph_name, register_graph},
    redis_type::GRAPH_TYPE,
};
use graph::effects::emit_v3;
use graph::effects::v3::emit::{AnnouncedConstraint, SchemaBaseline, build_constraint_buffer};
use graph::entity_type::EntityType;
use graph::graph::constraint::{ConstraintStatus, ConstraintType};
use graph::identifier_limits::validate_identifier_len;
use parking_lot::RwLock;
use redis_module::{Context, ContextFlags, NextArg, RedisResult, RedisString, RedisValue};
use std::sync::Arc;
use std::time::Duration;

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
/// Retried until it succeeds or the work stops being this node's, because
/// giving up strands the constraint permanently: the master keeps it UNDER
/// CONSTRUCTION, never enforces it, accepts writes it should reject, and nothing
/// can re-drive it — `GRAPH.CONSTRAINT CREATE` returns `Constraint already
/// exists` from then on. A `CLIENT PAUSE ... WRITE` or an in-flight `FAILOVER`
/// refuses the escalation, and both are transient by nature.
///
/// Deliberately unbounded rather than capped at some number of attempts. Nothing
/// bounds a `CLIENT PAUSE` — its timeout is whatever the caller passed — so any
/// cap is a guess, and the case it would fire in is a long pause, which is
/// exactly when silently abandoning the constraint is worst. The two `return`
/// arms below are the real exit conditions: this loop ends when the graph is
/// gone or this node is no longer the one that should finish the work.
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

pub fn settle_async_constraint(
    graph: &Arc<RwLock<ThreadedGraph>>,
    announce: &[Settling],
    key: &str,
    was_replicated: bool,
) {
    const BACKOFF: Duration = Duration::from_millis(100);

    loop {
        match attempt_settle(graph, announce, key, was_replicated) {
            Ok(()) => return,
            // Transient: the window closes on its own. Nothing is held here —
            // `attempt_settle` has dropped its session, and with it the GIL and
            // the write lock — so sleeping is safe.
            Err(WriteAbort::ReplicaTrafficPaused) => std::thread::sleep(BACKOFF),
            // The graph is gone, or this node is no longer a master. Neither
            // resolves by waiting: on a demoted node the constraint is the new
            // master's to finish, and `enforce_pending_constraints_after_promotion`
            // is what picks it up.
            Err(WriteAbort::GraphUnregistered | WriteAbort::NotAMaster) => return,
        }
    }
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
    announce: &[Settling],
    key: &str,
    was_replicated: bool,
) -> Result<(), WriteAbort> {
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
    let results = session.with_graph(|tg| {
        tg.graph
            .read()
            .borrow()
            .compute_pending_constraint_results()
    });
    if results.is_empty() {
        return Ok(());
    }

    // Escalation takes the global lock before the write lock (#726) and keeps the
    // commit Arc-swap under it, so it cannot race a BGSAVE fork (#452) — the same
    // shape as bulk_insert's Phase 2.
    session.upgrade_to_write()?;
    session
        .with_graph_mut(|tg| {
            let Some(g_arc) = tg.graph.write() else {
                return;
            };
            g_arc
                .borrow_mut()
                .apply_constraint_validation_results(results);
            let settled = Arc::clone(&g_arc);
            tg.graph.commit(g_arc);

            if was_replicated || !emit_v3() {
                return;
            }
            let g = settled.borrow();
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
                let mut buf = Vec::new();
                // Re-announce with the status validation settled on. Inside this
                // closure the GIL is still held from `upgrade_to_write`'s pause
                // check, and replicating through that same context is what keeps
                // the check sound — a second context would unlock separately and
                // propagate outside it (#2371).
                if build_constraint_buffer(
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
                    &mut buf,
                )
                .is_ok()
                    && let Some(raw_ctx) = gil_context()
                {
                    // SAFETY: borrowed for the life of the GIL guard this thread
                    // holds; `Context` has no `Drop`, so nothing frees it.
                    let ctx = Context::new(raw_ctx.as_ptr());
                    ctx.replicate("GRAPH.EFFECT", &[key.as_bytes(), buf.as_slice()]);
                }
            }
        })
        .expect("writer mode after upgrade_to_write");
    Ok(())
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
    let is_create = match op_str.to_uppercase().as_str() {
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
    let ct = match ct_str.to_uppercase().as_str() {
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
    let entity_type = match et_str.to_uppercase().as_str() {
        "NODE" | "LABEL" => EntityType::Node,
        "RELATIONSHIP" | "EDGE" => EntityType::Relationship,
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
    if props_kw.to_uppercase() != "PROPERTIES" {
        return Err(redis_module::RedisError::String(
            "Expected PROPERTIES keyword".into(),
        ));
    }

    // Property count
    let prop_count_str = args.next_str()?;
    let prop_count: i64 = prop_count_str.parse().map_err(|_| {
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

    let mut tg = graph.write();
    let Some(g_arc) = tg.graph.write() else {
        return Err(redis_module::RedisError::String(
            "ERR another write is in progress, retry the query".into(),
        ));
    };

    let is_replicated = ctx.get_flags().contains(ContextFlags::REPLICATED);

    // Before the mutation: `create_constraint` registers the label and the
    // property names, and those registrations have to be announced ahead of the
    // record whose ids depend on them.
    let baseline = SchemaBaseline::of(&g_arc.borrow());

    let result: Result<bool, String> = {
        let mut g = g_arc.borrow_mut();
        if is_create {
            g.create_constraint(ct, entity_type, label.clone(), properties.clone())
        } else {
            g.drop_constraint(&ct, &entity_type, &label, &properties)
                .map(|()| false)
        }
    };

    match result {
        Ok(needs_async_validation) => {
            // The effect is built from the graph this write mutated, so keep a
            // handle before `commit` consumes the one it swaps in.
            let mutated = Arc::clone(&g_arc);
            tg.graph.commit(g_arc);

            // Spawn background validation for large datasets on a dedicated
            // OS thread (not the query threadpool). Validation can take
            // seconds for large datasets, and we don't want it to occupy a
            // query worker — that would block unrelated reads queued behind
            // it on the shared MPMC dispatch.
            //
            // Two-phase under different locks: the long-running validation
            // runs under a *read* lock on the outer `RwLock<ThreadedGraph>`,
            // so concurrent `db.constraints()` queries continue to see the
            // constraint in UNDER CONSTRUCTION state. The outer write lock is
            // taken only briefly at the end to commit the status update.
            if needs_async_validation {
                std::thread::spawn({
                    let graph = graph.clone();
                    let label = Arc::clone(&label);
                    let properties = properties.clone();
                    let key = key_str.to_string();
                    let was_replicated = is_replicated;
                    move || {
                        settle_async_constraint(
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
                    }
                });
            }

            if !is_replicated {
                // Announce the outcome, not the command. The replica installs
                // the status this node reached rather than validating on its
                // own — it would scan at a different time against different
                // interleavings and could legitimately disagree.
                //
                // Under v3 that is one GRAPH.EFFECT. A second one follows from
                // the validation thread below when there is validation to wait
                // for, carrying the final status; the apply side upserts, so it
                // converges on this one instead of duplicating it.
                //
                // Verbatim is the v2 path, and it needs the command twice: the
                // replica treats the second copy as the activation signal and
                // its handler hits the `Constraint already exists` branch.
                // `replicate_verbatim` rather than the parameterized
                // `RM_Replicate`, which in this Redis version returns OK from a
                // module command handler without actually propagating.
                // A drop carries no status. A create carries the one this node
                // reached: still building if the caller spawned validation,
                // otherwise whatever `create_constraint` decided inline — read
                // back rather than assumed, and present because the graph write
                // lock has been held since it was created.
                let status = if !is_create {
                    None
                } else if needs_async_validation {
                    Some(ConstraintStatus::UnderConstruction)
                } else {
                    Some(
                        find_status(&mutated.borrow(), ct, entity_type, &label, &properties)
                            .expect("a create leaves its constraint in place"),
                    )
                };
                let mut buf = Vec::new();
                let encoded = emit_v3()
                    && build_constraint_buffer(
                        &mutated.borrow(),
                        is_create,
                        &AnnouncedConstraint {
                            ct,
                            entity_type,
                            status,
                            label: &label,
                            properties: &properties,
                        },
                        &baseline,
                        &mut buf,
                    )
                    .is_ok();
                if encoded {
                    ctx.replicate("GRAPH.EFFECT", &[key_str.as_slice(), buf.as_slice()]);
                } else {
                    ctx.replicate_verbatim();
                    if is_create {
                        ctx.replicate_verbatim();
                    }
                }
            }
            if is_create {
                Ok(RedisValue::SimpleStringStatic("PENDING"))
            } else {
                Ok(RedisValue::SimpleStringStatic("OK"))
            }
        }
        Err(e) if is_replicated && e == "Constraint already exists" => {
            // Activation signal on replica — constraint already created by
            // the first replicated command, silently succeed.
            tg.graph.rollback();
            Ok(RedisValue::SimpleStringStatic("OK"))
        }
        Err(e) => {
            tg.graph.rollback();
            Err(redis_module::RedisError::String(e))
        }
    }
}
