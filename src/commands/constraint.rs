use crate::query_session::{QuerySession, WriteFacts};
use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{ThreadedGraph, c_graph_key, c_graph_name, register_graph},
    redis_type::GRAPH_TYPE,
};
use graph::entity_type::EntityType;
use graph::graph::constraint::ConstraintType;
use graph::identifier_limits::validate_identifier_len;
use parking_lot::RwLock;
use redis_module::{Context, ContextFlags, NextArg, RedisResult, RedisString, RedisValue};
use std::sync::Arc;

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

    let result: Result<bool, String> = {
        let mut g = g_arc.borrow_mut();
        if is_create {
            g.create_constraint(ct, entity_type, label, properties)
        } else {
            g.drop_constraint(&ct, &entity_type, &label, &properties)
                .map(|()| false)
        }
    };

    match result {
        Ok(needs_async_validation) => {
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
                let graph_clone = graph.clone();
                std::thread::spawn(move || {
                    // Phase 1: the long-running validation runs as a reader, so
                    // concurrent `db.constraints()` still sees the constraint UNDER
                    // CONSTRUCTION.
                    //
                    // `replicates: false` — this thread emits no replication of its
                    // own, so no pause window can be propagated into. #2419 adds a
                    // GRAPH.EFFECT re-announce here, and must flip this to `true` in
                    // the same commit.
                    //
                    // `originated_here: false` — it also runs on a replica, applying
                    // the master's replicated GRAPH.CONSTRAINT, where a READONLY
                    // rejection would strand the constraint UNDER CONSTRUCTION and
                    // diverge. Covered by testConstraintReplication::
                    // test_02_async_validation_reaches_operational_on_replica.
                    let session = QuerySession::begin_with(
                        &graph_clone,
                        WriteFacts {
                            replicates: false,
                            originated_here: false,
                        },
                    );
                    let results = session.with_graph(|tg| {
                        tg.graph
                            .read()
                            .borrow()
                            .compute_pending_constraint_results()
                    });
                    // Phase 2: publish the status update as a writer. Escalation
                    // takes the global lock before the write lock (#726) and keeps the
                    // commit Arc-swap under it, so it cannot race a BGSAVE fork
                    // (#452) — same shape as bulk_insert's Phase 2.
                    if session.upgrade_to_write().is_ok() {
                        session
                            .with_graph_mut(|tg| {
                                if let Some(g_arc) = tg.graph.write() {
                                    g_arc
                                        .borrow_mut()
                                        .apply_constraint_validation_results(results);
                                    tg.graph.commit(g_arc);
                                }
                            })
                            .expect("writer mode after upgrade_to_write");
                    }
                    // `session` releases both locks here.
                });
            }

            if !is_replicated {
                // Two-phase replication protocol (matches C FalkorDB):
                //   1. The CREATE/DROP itself.
                //   2. A second copy of the same command, which the replica
                //      treats as the "activation" signal — its handler hits
                //      the `Constraint already exists` branch below and
                //      silently succeeds.
                // Calling `replicate_verbatim()` twice queues two separate
                // entries via `alsoPropagate`. We use verbatim (not the
                // parameterized `RM_Replicate`) because in this Redis
                // version `RM_Replicate` from a module command handler
                // returns OK but does not actually propagate to replicas.
                ctx.replicate_verbatim();
                if is_create {
                    ctx.replicate_verbatim();
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
